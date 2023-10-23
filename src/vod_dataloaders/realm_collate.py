import dataclasses
import time
import typing as typ
import warnings

import numpy as np
import torch
import transformers
import vod_configs
import vod_search
import vod_types as vt
from loguru import logger
from vod_tools.misc.exceptions import dump_exceptions_to_file
from vod_tools.misc.template import Template

from .core import (
    in_batch_negatives,
    sample,
    search,
    utils,
)
from .tokenizer_collate import render_template_and_tokenize

T = typ.TypeVar("T")
P = typ.ParamSpec("P")


ROW_IDX_COL_NAME: str = "__row_idx__"
SECTION_IDS = "retrieval_ids"
SECTION_ID = "id"
SUBSET_ID = "subset_id"
SUBSET_IDS = "subset_ids"
RETRIEVAL_IDS = "retrieval_ids"
RELEVANCE_SCORES = "retrieval_scores"


@dataclasses.dataclass
class RealmTemplates:
    """Templates for the retrieval-augmented language modeling tasks."""

    query: Template
    section: Template
    lm: Template


class RealmCollate(vt.Collate[typ.Any, torch.Tensor | list[int | float | str]]):
    """Collate function for retrieval-augmented language modeling tasks.

    This function is used to convert a list of queries into a batch.
    For each queries, the function will search the search engine for the top results, and sample a subset.

    This function implements the following steps:
        1. search & merge
        2. sample the sections
        3. (optional) flatten sections (in-batch negative)
        4. fetch the content of each section from the huggingface `datasets.Dataset`
        5. tokenize the sections & queries
        6. cast & return the batch (`torch.Tensor`)
    """

    templates: RealmTemplates
    tokenizer_encoder: transformers.PreTrainedTokenizerBase
    tokenizer_lm: None | transformers.PreTrainedTokenizerBase
    search_client: vod_search.HybridSearchClient
    config: vod_configs.RealmCollateConfig

    def __init__(
        self,
        *,
        search_client: vod_search.HybridSearchClient,
        config: vod_configs.RealmCollateConfig,
        parameters: None | typ.MutableMapping = None,
    ):
        self.search_client = search_client
        self.config = config
        self.parameters = _validate_parameters(parameters or {}, search_client)
        self.tokenizer_encoder = config.tokenizer_encoder.instantiate()
        self.tokenizer_encoder = config.tokenizer_encoder.instantiate()
        self.templates = RealmTemplates(
            query=Template(config.templates.query),
            section=Template(config.templates.section),
            lm=Template(config.templates.lm),
        )

        if config.lookup_engine not in search_client.clients:
            raise ValueError(
                f"The `{config.lookup_engine}` client is required to lookup positive sections. "
                f"Please add it to the `search_client` instance"
            )

    @property
    def sections(self) -> vt.DictsSequence:
        """Get all indexed sections."""
        return self.search_client.sections

    @dump_exceptions_to_file
    def __call__(self, inputs: list[dict[str, typ.Any]], **kws: typ.Any) -> vt.RealmBatch:
        """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch."""
        start_time = time.perf_counter()
        input_batch = utils.pack_examples(inputs)  # list[dict] -> dict[list]
        relevances_map: list[dict[str, float]] = _extract_relevances(input_batch)  # Extract relevance scores

        # Search within each client
        search_results, raw_scores = self.search(input_batch, top_k=self.config.prefetch_n_sections)
        diagnostics = {f"{key}": s for key, s in search_results.meta.items()}

        # Sample the sections given the positive ones and the pool of candidates
        with utils.BlockTimer(name="sample_sections_time", output=diagnostics):
            samples = sample.sample_search_results(
                search_results=search_results,
                raw_scores=raw_scores,
                total=self.config.n_sections,
                max_pos_sections=self.config.max_pos_sections,
                temperature=float(self.config.do_sample),
                max_support_size=self.config.support_size,  # <-- limit the candidate pool size
            )

        # Flatten sections (in-batch negative)
        if self.config.in_batch_negatives:
            # NOTE: padding is required to output the same number of sections
            #       so `torch.compile()` to compile a single graph.
            samples = in_batch_negatives.flatten_samples(
                samples,
                padding=True,
            )

        # Replace negative indices with random ones
        #    this is required because `datasets.Dataset` doesn't support negative indices
        # samples = numpy_ops.replace_negative_indices(samples, world_size=len(self.sections))

        # Fetch the content of each section from the huggingface `datasets.Dataset`
        sections_shape = samples.batch.indices.shape
        flat_ids = samples.batch.indices.flatten().tolist()
        flat_sections_content: dict[str, list[typ.Any]] = self.sections[flat_ids]

        # Tokenize the sections and add them to the output
        with utils.BlockTimer(name="tokenize_time", output=diagnostics):
            tokenized_queries = _tokenize(
                input_batch,
                tokenizer=self.tokenizer_encoder,
                template=self.templates.query,
                prefix="query__",
                **self.config.tokenizer_encoder.kwargs(),
            )
            tokenized_sections = _tokenize(
                flat_sections_content,
                tokenizer=self.tokenizer_encoder,
                template=self.templates.section,
                prefix="section__",
                output_shape=samples.batch.indices.shape,
                **self.config.tokenizer_encoder.kwargs(),
            )

        # Get query/section attributes (e.g., subset_id, retrieval_ids, language, etc.)
        extra_attributes = _get_extra_attributes(
            input_batch,
            flat_sections_content,
            sections_shape=sections_shape,  # type: ignore
            extras_keys=set(self.config.query_extras) | set(self.config.section_extras),
        )

        # Retrieve relevances based on the section ids
        relevances = [
            [rmap.get(sid, 0) for sid in sec_ids]
            for rmap, sec_ids in zip(relevances_map, extra_attributes["section__id"])  # type: ignore
        ]

        # Make the final batch and potentially cast attributes to `torch.Tensor`
        input_batch = vt.RealmBatch(
            **tokenized_queries,
            **tokenized_sections,
            **_samples_to_dict(
                samples,
                relevances=relevances,
                prefix="section__",
                as_torch=True,
            ),
            **extra_attributes,
            diagnostics=diagnostics,
        )
        # Append the total time for the collate function
        input_batch.diagnostics["collate_time"] = time.perf_counter() - start_time
        # Append the mean maximum index of the sampled sections
        #  This is used to monitor the sampling efficiency
        input_batch.diagnostics["max_sampling_id"] = np.mean(samples.max_sampling_id)
        return input_batch

    def search(
        self,
        batch: dict[str, typ.Any],
        top_k: int,
    ) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
        """Search the batch of queries and return the top `top_k` results."""
        # Get the query ids
        query_subset_ids = batch[SUBSET_IDS]
        # Get the query text
        query_text: list[str] = self.templates.query.render_batch(batch)
        # Get the query vectors
        if self.search_client.requires_vectors:
            try:
                query_vectors = batch[utils.VECTOR_KEY]
            except KeyError as exc:
                raise ValueError(
                    f"The search client `{type(self.search_client).__name__}` requires vectors. "
                    f"Please make sure indexing the provided dataset returns a dict with a key "
                    f"`{utils.VECTOR_KEY}` and value being a vector representation for that row. "
                    f"Found keys: `{list(batch.keys())}`."
                ) from exc
            if isinstance(query_vectors, list):
                query_vectors = np.stack(query_vectors)
        else:
            query_vectors = None

        # Async search the query text using `search_client.async_search`
        return search.async_hybrid_search(
            text=query_text,
            shards=batch[vod_configs.TARGET_SHARD_KEY],
            vector=query_vectors,
            subset_ids=query_subset_ids,
            section_ids=batch[SECTION_IDS],
            top_k=top_k,
            clients=self.search_client.clients,
            weights=dict(self.parameters),
        )


def _samples_to_dict(
    samples: sample.PrioritySampledSections,
    relevances: list[list[float]],
    prefix: str = "",
    as_torch: bool = False,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Convert the sampled sections to a dictionary."""
    if samples.batch.labels is None:
        raise ValueError("The sections must have labels.")

    output = {
        f"{prefix}idx": samples.batch.indices,
        f"{prefix}score": samples.batch.scores,
        f"{prefix}label": samples.batch.labels > 0,
        f"{prefix}relevance": relevances,
        f"{prefix}log_weight": samples.log_weights,
        f"{prefix}lse_pos": samples.lse_pos,
        f"{prefix}lse_neg": samples.lse_neg,
        **{f"{prefix}{k}": v for k, v in samples.raw_scores.items()},
    }

    if as_torch:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fns = {
                f"{prefix}score": lambda x: torch.from_numpy(x).to(torch.float32),
                f"{prefix}label": lambda x: torch.from_numpy(x).to(torch.bool),
                f"{prefix}relevance": lambda x: torch.tensor(x, dtype=torch.float32),
            }
            output = {k: fns.get(k, torch.from_numpy)(v) for k, v in output.items()}

    return output  # type: ignore


def _tokenize(
    batch: dict[str, typ.Any],
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    template: Template,
    prefix: str,
    output_shape: None | tuple[int, ...] = None,
    **tokenize_kws: typ.Any,
) -> dict[str, torch.Tensor]:
    # Tokenize the queries
    tokenized = render_template_and_tokenize(
        batch,
        template=template,
        prefix_key=prefix,
        tokenizer=tokenizer,
        **tokenize_kws,
    )
    if output_shape is not None:
        tokenized = {k: v.view(*output_shape, -1) for k, v in tokenized.items()}
    return tokenized


def _get_extra_attributes(
    batch: dict[str, typ.Any],
    flat_sections_content: dict[str, typ.Any],
    *,
    sections_shape: tuple[int, int] | tuple[int],
    extras_keys: None | set[str] = None,
) -> dict[str, None | dict[str, typ.Any]]:
    extras_keys = extras_keys or set()
    extras_keys |= {"id", "subset_id", "subset_ids"}  # Required keys
    if len(sections_shape) > 2:  # noqa: PLR2004
        raise ValueError(f"Expected a 1D or 2D shape. Found {sections_shape}")

    # Handle query attributes
    query_extras = {}
    query_extras["query__section_ids"] = batch[SECTION_IDS]
    for k in extras_keys:
        if k not in batch:
            continue
        v = batch[k]
        query_extras[f"query__{k}"] = v

    # Handle section attributes
    sections_extras = {}
    for k in extras_keys:
        if k not in flat_sections_content:
            continue
        v = flat_sections_content[k]
        if len(sections_shape) == 2:  # noqa: PLR2004
            if isinstance(v, torch.Tensor):
                v = v.view(sections_shape)
            elif isinstance(v, np.ndarray):
                v = v.reshape(sections_shape)
            elif isinstance(v, list):
                v = utils.reshape_flat_list(v, sections_shape)
        sections_extras[f"section__{k}"] = v

    return {**query_extras, **sections_extras}


def _validate_parameters(
    parameters: typ.MutableMapping,
    search_client: vod_search.HybridSearchClient,
    default_value: float = 1.0,
) -> typ.MutableMapping:
    """Validate the parameters against the client names."""
    client_names = set(search_client.clients.keys())
    missing_clients = client_names - set(parameters.keys())
    if len(missing_clients):
        logger.warning(
            f"Missing weights for clients: `{missing_clients}`. "
            f"Fix this by referencing them in the `parameters` argument. "
            f"Replacing missing values with constant weights `1.0`."
        )
        for client in missing_clients:
            parameters[client] = default_value

    return parameters


def _extract_relevances(batch: dict[str, typ.Any]) -> list[dict[str, float]]:
    retrieval_ids = batch[RETRIEVAL_IDS]
    retrieval_scores = batch.get(RELEVANCE_SCORES, None)
    if retrieval_scores is None:
        # Set default relevance scores to 1.0 when not provided
        return [{x: 1.0 for x in ids} for ids in retrieval_ids]

    relevances = []
    for ids, scores in zip(retrieval_ids, retrieval_scores):
        relevances.append(dict(zip(ids, scores)))

    return relevances

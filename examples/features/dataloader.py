import pathlib
import tempfile
import typing as typ

import dotenv
import lightning as L
import rich
import torch
import torch.nn.functional as F
import transformers
import vod_configs as vc
import vod_dataloaders as vdl
import vod_datasets as vds
import vod_search as vs
import vod_types as vt
from rich.progress import track
from vod_ops import Predict
from vod_tools import arguantic, pretty

dotenv.load_dotenv(str(pathlib.Path(__file__).parent / ".predict.env"))


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dset: str = "squad_v2"
    split: str = "validation"
    name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    num_workers: int = 1
    subset_size: int = 1_000


class Encoder(torch.nn.Module):
    """Transformer Encoder."""

    def __init__(self, name_or_path: str):
        super().__init__()
        self.backbone = transformers.AutoModel.from_pretrained(name_or_path)

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass of the model, returns the embeddings of the [CLS] token."""
        output = self.backbone(batch["input_ids"], batch["attention_mask"])
        pooled_output = _mean_pooling(output.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled_output, p=2, dim=-1)

    def get_encoding_shape(self, *args, **kwargs) -> tuple[int]:  # noqa: ANN002, ANN003
        """Return the output shape."""
        return (self.backbone.config.hidden_size,)


class CollateFn:
    """Extract questions and tokenize into a batch."""

    def __init__(self, text_field: str, tokenizer: transformers.PreTrainedTokenizerBase, template: str):
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.template = template

    def __call__(
        self,
        inputs: typ.Iterable[typ.Mapping[str, typ.Any]],
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Collate function for the `predict` tool."""
        texts = [x[self.text_field] for x in inputs]
        texts = [self.template.format(text) for text in texts]
        output = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return dict(output)


@torch.inference_mode()
def run(args: Args) -> None:
    """Embed a dataset, spin a hybrid search service, and build a retrieval dataloader."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.name_or_path)
    model = Encoder(args.name_or_path)

    # 1. Instantiate Fabric
    fabric = L.Fabric()
    fabric.launch()
    model = fabric.setup_module(model)

    # 2. Load the dataset
    queries = vds.load_dataset(
        vc.QueriesDatasetConfig(
            identifier="squad_queries",
            name_or_path=args.dset,
            split=args.split,
            link="squad_sections",
        )
    )
    sections = vds.load_dataset(
        vc.SectionsDatasetConfig(
            identifier="squad_sections",
            name_or_path=args.dset,
            split=args.split,
            search=None,
        )
    )
    rich.print(
        {
            "queries": queries,
            "sections": sections,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Compute the vectors for the questions and sections
        question_vectors = Predict(
            dataset=queries,
            save_dir=tmpdir,
            model=model,
            collate_fn=CollateFn(
                text_field="query",
                tokenizer=tokenizer,
                template="query: {}",
            ),
            model_output_key=None,
        )(
            fabric=fabric,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
            open_mode="x",
        )
        section_vectors = Predict(
            dataset=sections,
            save_dir=tmpdir,
            model=model,
            collate_fn=CollateFn(
                text_field="content",
                tokenizer=tokenizer,
                template="passage: {}",
            ),
            model_output_key=None,
        )(
            fabric=fabric,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
            open_mode="x",
        )

        # 4. Spin up a hybrid search engine
        with vs.build_hybrid_search_engine(
            sections={args.dset: sections},  # {shard_id : dataset}
            vectors={args.dset: vt.as_lazy_array(section_vectors)},  # {shard_id : vectors}
            configs={
                args.dset: vc.HybridSearchFactoryConfig(
                    engines={
                        "dense": vc.FaissFactoryConfig(factory="Flat"),
                        "sparse": vc.ElasticsearchFactoryConfig(persistent=False),
                    }
                )
            },
            cache_dir=tmpdir,
            dense_enabled=True,
            sparse_enabled=True,
        ) as master:
            search_client = master.get_client()
            rich.print(search_client)

            # 5. Setup the VOD Realm dataloader
            realm_dataloader = vdl.RealmDataloader.factory(
                queries={args.dset: (args.dset, queries)},  # {query_id : (shard_id, dataset)}
                vectors={args.dset: vt.as_lazy_array(question_vectors)},  # {query_id : vectors}
                search_client=search_client,
                collate_config=vc.RealmCollateConfig(
                    templates=vc.TemplatesConfig(
                        query="query: {{ query }}",
                        section="passage: {{ content }}",
                    ),
                    tokenizer_encoder=vc.TokenizerConfig(name_or_path=args.name_or_path),
                    tokenizer_lm=None,
                    n_sections=5,
                ),
            )

            # 6. Iterate through the questions
            for j, batch in enumerate(
                track(
                    realm_dataloader,
                    description="Making batches by retrieving documents dynamically",
                    total=len(realm_dataloader),
                )
            ):
                if j == 0:
                    pretty.pprint_retrieval_batch(batch, tokenizer=tokenizer, skip_special_tokens=True)
                    pretty.pprint_batch(batch, header="Batch")
                ...


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling - Take attention mask into account for correct averaging."""
    attention_mask_ = attention_mask.unsqueeze(-1).float()
    return torch.sum(last_hidden_state * attention_mask_, -2) / torch.clamp(attention_mask_.sum(-2), min=1e-9)


if __name__ == "__main__":
    args = Args()
    run(args)

import functools
import pathlib
import tempfile

import dotenv
import lightning as L
import rich
import torch
import torch.nn.functional as F
import transformers
import vod_configs
import vod_dataloaders
import vod_search
import vod_types as vt
from rich.progress import track
from torch.utils import data as torch_data
from vod_ops.utils import helpers
from vod_tools import arguantic, pipes, predict

from src import vod_datasets

dotenv.load_dotenv(str(pathlib.Path(__file__).parent / ".predict.env"))


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dset: str = "squad.en"
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

    def get_output_shape(self, *args, **kwargs) -> tuple[int]:  # noqa: ANN002, ANN003
        """Return the output shape."""
        return (self.backbone.config.hidden_size,)


@torch.inference_mode()
def run(args: Args) -> None:
    """Embed a dataset, spin a hybrid search service, and build a retrieval dataloader."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.name_or_path)
    model = Encoder(args.name_or_path)

    # 1. Instantiate Fabric
    fabric = L.Fabric()
    fabric.launch()
    model = fabric.setup_module(model)

    # 2. Load the dataset (we are working on extending this interface to external datasets)
    dset_factory = vod_datasets.RetrievalDatasetFactory.from_config({"name": args.dset, "split": args.split})
    questions = dset_factory.get_qa_split()
    sections = dset_factory.get_sections()
    if args.subset_size > 0:
        questions = questions.select(range(args.subset_size))
        sections = sections.select(range(args.subset_size))
    rich.print(
        {
            "factory": dset_factory,
            "questions": questions,
            "sections": sections,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Compute the vectors for the questions and sections
        predict_fn = functools.partial(
            predict.predict,
            fabric=fabric,
            model=model,
            loader_kwargs={"batch_size": args.batch_size, "num_workers": args.num_workers},
            cache_dir=tmpdir,
            model_output_key=None,
            collate_fn=functools.partial(
                pipes.torch_tokenize_collate,
                tokenizer=tokenizer,
                text_key="text",
                truncation=True,
            ),
        )
        question_vectors = predict_fn(dataset=questions)  # type: ignore
        section_vectors = predict_fn(dataset=sections)  # type: ignore

        # 4. Spin up a hybrid search engine
        with vod_search.build_hybrid_search_engine(
            sections=sections,  # type: ignore
            vectors=vt.as_lazy_array(section_vectors),
            config=vod_configs.SearchConfig(
                dense=vod_configs.FaissFactoryConfig(
                    factory="IVFauto,Flat",
                ),
                sparse=vod_configs.ElasticsearchFactoryConfig(
                    text_key="text",
                    persistent=False,
                ),
            ),
            cache_dir=tmpdir,
            dense_enabled=True,
            sparse_enabled=True,
        ) as master:
            search_client = master.get_client()
            rich.print(search_client)

            # 5. Setup the VOD retrieval dataloader
            collate_fn = vod_dataloaders.RealmCollate(
                tokenizer=tokenizer,
                sections=sections,  # type: ignore
                search_client=search_client,
                config=vod_configs.RetrievalCollateConfig(),
                parameters={"dense": 1.0, "sparse": 1.0},
            )
            dataset_with_vectors = helpers.IndexWithVectors(
                dataset=questions,
                vectors=vt.as_lazy_array(question_vectors),  # type: ignore
                vector_key="vector",
            )
            dataloader = torch_data.DataLoader(
                dataset=dataset_with_vectors,  # type: ignore
                collate_fn=collate_fn,
                num_workers=args.num_workers,
            )

            # 6. Iterate through the questions
            for j, batch in enumerate(
                track(
                    dataloader,
                    description="Making batches by retrieving documents dynamically",
                    total=len(dataloader),
                )
            ):
                if j == 0:
                    pipes.pprint_retrieval_batch(batch, tokenizer=tokenizer, skip_special_tokens=True)
                    pipes.pprint_batch(batch, header="Batch")
                ...


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling - Take attention mask into account for correct averaging."""
    attention_mask_ = attention_mask.unsqueeze(-1).float()
    return torch.sum(last_hidden_state * attention_mask_, -2) / torch.clamp(attention_mask_.sum(-2), min=1e-9)


if __name__ == "__main__":
    args = Args()
    run(args)

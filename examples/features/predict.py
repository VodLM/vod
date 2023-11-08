from __future__ import annotations

import tempfile
import typing as typ
from pathlib import Path

import datasets
import dotenv
import lightning as L
import rich
import torch
import transformers
import vod_types as vt
from tensorstore import _tensorstore as ts
from transformers import BertModel
from vod_ops import Predict
from vod_tools import arguantic
from vod_tools.ts_factory.ts_factory import TensorStoreFactory

dotenv.load_dotenv(str(Path(__file__).parent / ".predict.env"))


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    name_or_path: str = "google/bert_uncased_L-4_H-256_A-4"
    split: str = "train[:1%]"


class Encoder(torch.nn.Module):
    """Transformer Encoder."""

    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass of the model, returns the embeddings of the [CLS] token."""
        output = self.bert(batch["input_ids"], batch["attention_mask"])
        return {"poolet_output": output.pooler_output}


class CollateFn:
    """Extract questions and tokenize into a batch."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(
        self,
        inputs: typ.Iterable[typ.Mapping[str, typ.Any]],
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Collate function for the `predict` tool."""
        texts = [x["question"] for x in inputs]
        output = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return dict(output)


def run(args: Args) -> None:
    """Showcase the `predict` tool."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.name_or_path)
    bert = transformers.AutoModel.from_pretrained(args.name_or_path)
    model = Encoder(bert)
    model.eval()
    squad_: datasets.Dataset = datasets.load_dataset("squad", split=args.split)  # type: ignore
    squad: vt.DictsSequence = squad_
    rich.print(squad)

    # collate_fn
    collate_fn = CollateFn(tokenizer=tokenizer)

    # Init Lightning's fabric
    fabric = L.Fabric()
    fabric.launch()
    model = fabric.setup_module(model)

    # Compute the vectors (first create them and readthem, then only read them)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the predict function
        predict_fn = Predict(
            dataset=squad,
            save_dir=tmpdir,
            model=model,
            collate_fn=collate_fn,
            model_output_key="poolet_output",
        )

        # Step 1 - Write the vectors and save to cache
        store_factory: TensorStoreFactory = predict_fn(
            fabric=fabric,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
            open_mode="x",
        )

        # Step 2 - Call the function again, the function will read the vectors from the cache.
        store_factory: TensorStoreFactory = predict_fn(
            fabric=fabric,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
            open_mode="r",
        )

        # Step 3 - Open the TensorStore (on-disk memory structure)
        rich.print({"store_factory": store_factory})
        store: ts.TensorStore = store_factory.open()

        # Step 4 - Read the 10 first vectors as a numpy array
        np_vecs = store[:10].read().result()
        rich.print(
            {
                "np_vecs": np_vecs,
                "np_vecs.shape": np_vecs.shape,
            }
        )


if __name__ == "__main__":
    args = Args.parse()
    run(args)

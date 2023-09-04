from __future__ import annotations

import functools
import tempfile
from functools import partial
from pathlib import Path

import datasets
import dotenv
import lightning as L
import rich
import tensorstore as ts
import torch
import transformers
from transformers import BertModel
from vod_tools import arguantic, dstruct, pipes, predict

dotenv.load_dotenv(str(Path(__file__).parent / ".predict.env"))


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    model_name: str = "google/bert_uncased_L-4_H-256_A-4"
    split: str = "train[:1%]"


class Encoder(torch.nn.Module):
    """Transformer Encoder."""

    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass of the model, returns the embeddings of the [CLS] token."""
        output = self.bert(batch["input_ids"], batch["attention_mask"])
        return {"pooler_output": output.pooler_output}


def run(args: Args) -> None:
    """Showcase the `predict` tool."""
    bert = transformers.AutoModel.from_pretrained(args.model_name)
    model = Encoder(bert)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    squad: datasets.Dataset = datasets.load_dataset("squad", split=args.split)  # type: ignore
    rich.print(squad)
    collate_fn = partial(pipes.torch_tokenize_collate, tokenizer=tokenizer, text_key="question")

    # Init Lightning's fabric
    fabric = L.Fabric()
    fabric.launch()
    model = fabric.setup_module(model)

    # Compute the vectors (first create them and readthem, then only read them)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the predict function
        predict_fn = functools.partial(
            predict.predict,  # <- the actual function, below are the arguments
            fabric=fabric,
            cache_dir=tmpdir,
            model=model,
            model_output_key="pooler_output",
            collate_fn=collate_fn,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
        )

        # Step 1 - Write the vectors and save to cache
        store_factory: dstruct.TensorStoreFactory = predict_fn(squad, open_mode="x")  # type: ignore

        # Step 2 - Call the function again, this time in ready only mode `r`, the function will
        #         read the vectors from the cache without computing them again.
        store_factory: dstruct.TensorStoreFactory = predict_fn(squad, open_mode="r")  # type: ignore

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

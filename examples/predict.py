from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path

import datasets
import dotenv
import rich
import tensorstore as ts
import torch
import transformers
from transformers import BertModel

from src.vod_tools import pipes, predict
from src.vod_tools.misc.trainer import Trainer

dotenv.load_dotenv(str(Path(__file__).parent / ".predict.env"))


class Encoder(torch.nn.Module):
    """Transformer Encoder."""

    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass of the model, returns the embeddings of the [CLS] token."""
        output = self.bert(batch["input_ids"], batch["attention_mask"])
        return {"pooler_output": output.pooler_output}


def run() -> None:
    """Showcase the predict module."""
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    bert = transformers.AutoModel.from_pretrained(model_name)
    model = Encoder(bert)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    squad: datasets.Dataset = datasets.load_dataset("squad", split="train[:1%]")  # type: ignore
    rich.print(squad)
    collate_fn = partial(pipes.torch_tokenize_collate, tokenizer=tokenizer, text_key="question")

    # init the trainer and compute the vectors
    trainer = Trainer()
    with tempfile.TemporaryDirectory() as tmpdir:
        open_mode = {"valid": "x", "train": "r"}
        dsets = {"valid": squad, "train": squad}
        stores = {
            key: predict(
                dset,  # type: ignore
                trainer=trainer,
                cache_dir=tmpdir,
                model=model,
                model_output_key="pooler_output",
                collate_fn=collate_fn,
                loader_kwargs={"batch_size": 10, "num_workers": 0},
                open_mode=open_mode[key],  # type: ignore
            )
            for key, dset in dsets.items()
        }

        rich.print(stores)

        store: ts.TensorStore = stores["valid"].open()
        rich.print(store[:].read().result())


if __name__ == "__main__":
    run()

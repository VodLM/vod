from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path

import datasets
import dotenv
import rich
import tensorstore
import torch
import transformers
from transformers import BertModel

from raffle_ds_research.tools import pipes, predict
from raffle_ds_research.tools.utils.trainer import Trainer

dotenv.load_dotenv(Path(__file__).parent / ".predict.env")


class Encoder(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output = self.bert(batch["input_ids"], batch["attention_mask"])
        return {"pooler_output": output.pooler_output}


def run():
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    bert = transformers.AutoModel.from_pretrained(model_name)
    model = Encoder(bert)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    squad = datasets.load_dataset("squad", split="train[:1%]")
    rich.print(squad)
    collate_fn = partial(pipes.torch_tokenize_collate, tokenizer=tokenizer, field="question")

    # init the trainer and compute the vectors
    trainer = Trainer()
    with tempfile.TemporaryDirectory() as tmpdir:
        stores = predict(
            {"valid": squad, "train": squad},
            trainer=trainer,
            cache_dir=tmpdir,
            model=model,
            model_output_key="pooler_output",
            collate_fn=collate_fn,
            loader_kwargs={"batch_size": 10, "num_workers": 0},
        )

        store: tensorstore.TensorStore = stores["valid"].open()
        rich.print(store[:].read().result())


if __name__ == "__main__":
    run()

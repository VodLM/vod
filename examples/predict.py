from __future__ import annotations

import tempfile

import datasets
import rich
import tensorstore
import torch
import transformers
from transformers import BertModel

from raffle_ds_research import Trainer, predict


class CollateFn(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict]):
        encodings = self.tokenizer([r["question"] for r in rows], return_tensors="pt", padding=True)
        return dict(encodings)


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
    squad = datasets.load_dataset("squad", split="train[:10%]")
    rich.print(squad)
    collate_fn = CollateFn(tokenizer)

    # init the trainer
    trainer = Trainer()

    # compute the vectors
    with tempfile.TemporaryDirectory() as tmpdir:
        stores = predict(
            {"valid": squad, "train": squad},
            trainer=trainer,
            cache_dir=tmpdir,
            model=model,
            model_output_key="pooler_output",
            collate_fn=collate_fn,
            loader_kwargs={"batch_size": 10, "num_workers": 1},
        )
        rich.print(stores)

        store: tensorstore.TensorStore = stores["valid"].open()
        rich.print(store[:].read().result())


if __name__ == "__main__":
    # try tensorstore callback
    run()

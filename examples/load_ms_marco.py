import time

import numpy as np
import rich
import torch.utils.data
import transformers
from tqdm import tqdm

from raffle_ds_research.core.builders import MsMarcoBuilder
from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.pipes.utils.misc import iter_examples

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    builder = MsMarcoBuilder(
        load_kwargs=dict(keep_in_memory=False),
        prep_map_kwargs=dict(num_proc=4, batch_size=1000),
        tokenizer=tokenizer,
    )

    # Build the dataset
    dataset = builder()
    rich.print(dataset)

    # Tokenize and collate a batch
    collate_fn = builder.get_collate_fn()
    batch = collate_fn([dataset["train"][0], dataset["train"][1]])
    batch["text"] = ["hello world"] * len(list(iter_examples(batch)))
    pipes.pprint_batch(batch, header="ms_marco - batch")

    # Init the dataloader
    bs = 100
    loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=8,
    )

    # Iterate over the dataset
    _ = next(iter(loader))  # warmup
    t0 = time.time()
    for batch in tqdm(
        loader,
        desc="iterating over the train set",
        total=len(dataset["train"]) // bs - 1,
    ):
        pass

    rich.print(f"ms_marco - dataloader - train: {time.time() - t0:.2f}s")

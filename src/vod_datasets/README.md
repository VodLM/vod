# Raffle Datasets

A simple interface to download Raffle datasets as `datasets.Dataset` objects.

## HF-like interface

```python
import datasets
from raffle_ds_research.tools.raffle_datasets import load_raffle_dataset

frank: datasets.DatasetDict = load_raffle_dataset("frank", name="en.A.qa_splits")
```

## Frank

```python
from raffle_ds_research.tools.raffle_datasets import frank

dset: frank.HfFrankPart = frank.load_frank(language="en", split="A")
```
```
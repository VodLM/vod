"""Wrappers for Raffle datasets."""
from __future__ import annotations

from .base import RetrievalDataset  # noqa: D104
from .frank import HfFrankPart, load_frank
from .interface import load_raffle_dataset
from .loader import ConcatenatedDatasetLoader, DatasetLoader
from .msmarco import MsmarcoRetrievalDataset, load_msmarco
from .squad import SquadRetrievalDataset, load_squad

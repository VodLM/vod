"""Wrappers for Raffle datasets."""
from __future__ import annotations

from .base import RetrievalDataset
from .frank import HfFrankPart, load_frank
from .loader import ConcatenatedDatasetLoader, RetrievalDatasetLoader
from .msmarco import MsmarcoRetrievalDataset, load_msmarco
from .squad import SquadRetrievalDataset, load_squad

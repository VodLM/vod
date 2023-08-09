"""Wrappers for Raffle datasets."""
from __future__ import annotations

from .frank import HfFrankPart, load_frank
from .loader import load_dataset, load_queries, load_sections
from .msmarco import MsmarcoRetrievalDataset, load_msmarco
from .squad import SquadRetrievalDataset, load_squad

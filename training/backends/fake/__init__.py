"""Deterministic CPU backend for protocol tests (no Ray, no GPU)."""
from .backend import FakeBackend, FakeHandle

__all__ = ["FakeBackend", "FakeHandle"]

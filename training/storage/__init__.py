"""
Storage layer for the training API.

This module provides storage abstractions for futures and metadata management.
"""
from .futures import FuturesStorage
from .metadata import MetadataStorage

__all__ = ["FuturesStorage", "MetadataStorage"]
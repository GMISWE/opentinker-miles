"""
Core business logic modules for the training API.

This package contains managers and builders for training client lifecycle,
async operations, and Slime argument construction.
"""
from .slime_builder import SlimeArgumentBuilder

__all__ = ["SlimeArgumentBuilder"]
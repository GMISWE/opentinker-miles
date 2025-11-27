"""
Training API module for kgateway AI Extension.

Implements the Tinker API for RL training with Slime backend.
"""

from .api import app, health

__all__ = ["app", "health"]

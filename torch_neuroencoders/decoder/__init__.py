"""
Decoder submodule: Decoding models and utilities for neural data analysis.

Exposes:
                - Decoder (from decode)
"""

from importlib import import_module

__all__ = ["decode", "Decoder"]


def __getattr__(name):
        if name == "decode":
                module = import_module(".decode", __name__)
                globals()[name] = module
                return module
        if name == "Decoder":
                from .decode import Decoder as cls

                globals()[name] = cls
                return cls
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

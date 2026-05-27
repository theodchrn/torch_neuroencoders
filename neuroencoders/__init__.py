"""
Neuroencoders: Python package for neural data analysis.

Submodules:
        - decoder: Decoding models and utilities
        - importData: Data import and parsing
        - fullEncoder: Encoding models
        - openEphysExport: OpenEphys data export tools
        - resultAnalysis: Analysis and visualization of results
        - simpleBayes: Bayesian decoding tools
        - transformData: Data transformation utilities
        - utils: General utilities
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError

try:
    from importlib.metadata import version

    __version__ = version("neuroencoders")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback if not installed

__all__ = [
    "decoder",
    "fullEncoder",
    "importData",
    "openEphysExport",
    "resultAnalysis",
    "simpleBayes",
    "transformData",
    "utils",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

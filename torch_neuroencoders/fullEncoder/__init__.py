"""
FullEncoder submodule: Encoding models for neural data analysis.

Exposes:
                - LSTMandSpikeNetwork (from an_network)
                - an_network (module containing neural network architectures)
"""

from importlib import import_module

__all__ = ["an_network", "LSTMandSpikeNetwork"]


def __getattr__(name):
        if name == "an_network":
                module = import_module(".an_network", __name__)
                globals()[name] = module
                return module
        if name == "LSTMandSpikeNetwork":
                from .an_network import LSTMandSpikeNetwork as cls

                globals()[name] = cls
                return cls
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

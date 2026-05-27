"""
Utils submodule: General utility functions for neuroencoders package.

Exposes:
                - Project, DataHelper (from global_classes)
"""

from importlib import import_module

__all__ = ["MOBS_Functions", "global_classes", "Project", "DataHelper"]


def __getattr__(name):
        if name == "MOBS_Functions":
                module = import_module(".MOBS_Functions", __name__)
                globals()[name] = module
                return module
        if name == "global_classes":
                module = import_module(".global_classes", __name__)
                globals()[name] = module
                return module
        if name in {"Project", "DataHelper"}:
                from .global_classes import DataHelper as data_helper, Project as project

                globals()["Project"] = project
                globals()["DataHelper"] = data_helper
                return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

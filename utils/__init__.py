"""Geometry helpers load independently from optional ColoRadar/SciPy tools.

Named legacy re-exports remain available on demand.
"""
from importlib import import_module


def __getattr__(name):
    for module_name in ('.pose', '.ColoRadar_tools'):
        module = import_module(module_name, __name__)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

"""
Unified hardware acceleration backend for neuroencoders.
Dynamically routes computations to GPU (NVIDIA RAPIDS) or CPU (Pandas/NumPy/SciPy/Skimage).
"""

import logging
import os

logger = logging.getLogger("neuroencoders.backend")

USE_GPU_BACKENDS = os.environ.get("NEUROENCODERS_USE_GPU_BACKENDS", "0") == "1"

# 1. Dataframes: cuDF (GPU) vs Pandas (CPU)
if USE_GPU_BACKENDS:
    try:
        import cudf as pd
        import dask_cudf

        HAS_GPU_DF = True
    except Exception:
        import pandas as pd

        dask_cudf = None
        HAS_GPU_DF = False
else:
    import pandas as pd

    dask_cudf = None
    HAS_GPU_DF = False

# 2. Machine Learning: cuML (GPU) vs Scikit-Learn (CPU)
if USE_GPU_BACKENDS:
    try:
        import cuml as ml

        HAS_GPU_ML = True
    except Exception:
        import sklearn as ml

        HAS_GPU_ML = False
else:
    import sklearn as ml

    HAS_GPU_ML = False

# 3. Image/Signal Processing: cuCIM (GPU) vs Scikit-Image (CPU)
if USE_GPU_BACKENDS:
    try:
        import cucim.skimage as skimage

        HAS_GPU_IMG = True
    except Exception:
        import skimage

        HAS_GPU_IMG = False
else:
    import skimage

    HAS_GPU_IMG = False

# 4. Graph Network Analysis: cuGraph (GPU) vs NetworkX (CPU)
if USE_GPU_BACKENDS:
    try:
        import cugraph as graph
        import networkx as nx

        HAS_GPU_GRAPH = True
    except Exception:
        import networkx as nx

        graph = None
        HAS_GPU_GRAPH = False
else:
    import networkx as nx

    graph = None
    HAS_GPU_GRAPH = False


def get_backend_status():
    """Returns a dictionary indicating which components are GPU accelerated."""
    status = {
        "dataframes": "GPU (cuDF)" if HAS_GPU_DF else "CPU (Pandas)",
        "machine_learning": "GPU (cuML)" if HAS_GPU_ML else "CPU (Scikit-Learn)",
        "image_processing": "GPU (cuCIM)" if HAS_GPU_IMG else "CPU (Scikit-Image)",
        "graph_networks": "GPU (cuGraph)" if HAS_GPU_GRAPH else "CPU (NetworkX)",
    }
    return status


def log_backend_info():
    """Logs the current hardware acceleration status."""
    logger.info("--- Neuroencoders Backend Routing ---")
    for component, backend in get_backend_status().items():
        logger.info(f"{component.replace('_', ' ').title()}: {backend}")

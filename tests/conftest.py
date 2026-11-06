"""
tests/conftest.py - Pytest configuration and Windows OpenMP compatibility setup.
"""

import os
import sys

# Prevent OpenMP runtime collision between PyTorch (libiomp5md) and CasADi (libgomp-1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure CasADi DLL directory is registered for Windows DLL loader
try:
    import casadi as cs
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(cs.__file__))
except Exception:
    pass

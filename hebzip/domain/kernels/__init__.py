"""Pure math kernels: written ONCE, run as plain Python or compiled by numba.

Only ``services/engines.py`` imports these modules (after choosing the engine), so the decorator in
``jit.py`` knows whether to compile before any kernel is defined. Domain classes receive the
functions through a ``KernelSet`` and never import a kernel module themselves.
"""

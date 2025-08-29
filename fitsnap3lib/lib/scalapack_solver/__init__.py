# ScaLAPACK solver module for FitSNAP

try:
    # Try to import from the Python wrapper which wraps the compiled Cython module
    from .scalapack_wrapper import lstsq, dummy_lstsq
    SCALAPACK_AVAILABLE = True
except ImportError as e:
    # Module not compiled yet
    SCALAPACK_AVAILABLE = False
    import warnings
    warnings.warn(f"ScaLAPACK module not compiled: {e}. Please build the module first.")
    
    # Define dummy functions to prevent import errors
    def lstsq(*args, **kwargs):
        raise ImportError("ScaLAPACK module not compiled. Please build it first.")
    
    def dummy_lstsq(*args, **kwargs):
        raise ImportError("ScaLAPACK module not compiled. Please build it first.")

__all__ = ['lstsq', 'dummy_lstsq', 'SCALAPACK_AVAILABLE']
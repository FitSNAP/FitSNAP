"""SLATE ridge solver module"""
try:
    from .slate_wrapper import ridge_solve_qr
    __all__ = ['ridge_solve_qr']
except ImportError:
    ridge_solve_qr = None
    __all__ = []

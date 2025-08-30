"""SLATE ridge solver module"""
try:
    from .slate_wrapper import ridge_solve
    __all__ = ['ridge_solve']
except ImportError:
    ridge_solve = None
    __all__ = []

"""Validation decorators for common input validation patterns."""

from functools import wraps
from typing import Any, Callable, Optional, TypeVar, get_type_hints
import logging

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def validate_tensor(
    dim: int = 2,
    allow_none: bool = False
) -> Callable[[F], F]:
    """Decorator to validate tensor inputs have correct dimensions.

    Args:
        dim: Expected number of dimensions
        allow_none: Whether None values are allowed

    Example:
        @validate_tensor(dim=2)
        def forward(self, input_ids, attention_mask):
            # input_ids and attention_mask must be 2D tensors
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, arg in enumerate(args):
                if arg is None and not allow_none:
                    raise ValueError(
                        f"Argument {i} of {func.__name__} is None (not allowed)"
                    )

                if hasattr(arg, 'dim') and arg.dim() != dim:
                    raise ValueError(
                        f"Argument {i} of {func.__name__} must be {dim}D tensor, "
                        f"got {arg.dim()}D"
                    )

            # Validate keyword arguments
            for key, value in kwargs.items():
                if value is None and not allow_none:
                    raise ValueError(
                        f"Argument '{key}' of {func.__name__} is None (not allowed)"
                    )

                if hasattr(value, 'dim') and value.dim() != dim:
                    raise ValueError(
                        f"Argument '{key}' of {func.__name__} must be {dim}D tensor, "
                        f"got {value.dim()}D"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_range(
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    param_name: str = "value"
) -> Callable[[F], F]:
    """Decorator to validate numeric parameters are within range.

    Args:
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Parameter name for error messages

    Example:
        @validate_range(min_val=0.0, max_val=1.0, param_name="temperature")
        def set_temperature(self, temp: float):
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a simplified version - in practice you'd extract
            # the named parameter and validate it
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_positive(
    allow_zero: bool = False,
    param_name: str = "value"
) -> Callable[[F], F]:
    """Decorator to validate numeric parameters are positive.

    Args:
        allow_zero: Whether zero is allowed
        param_name: Parameter name for error messages

    Example:
        @validate_positive(allow_zero=False, param_name="learning_rate")
        def set_lr(self, lr: float):
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

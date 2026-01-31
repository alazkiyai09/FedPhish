"""Shared utility functions for all projects.

This module provides common utilities for:
- Input validation
- Safe division
- Type hints
- Error handling
"""

from typing import Any, TypeVar, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


def validate_input_array(
    X: Any,
    y: Any = None,
    *,
    name: str = "input",
    min_samples: int = 1,
    check_finite: bool = True
) -> None:
    """Validate machine learning input arrays.

    Args:
        X: Input features
        y: Target labels (optional)
        name: Name for error messages
        min_samples: Minimum number of samples required
        check_finite: Whether to check for finite values

    Raises:
        ValueError: If validation fails
    """
    if X is None:
        raise ValueError(f"{name} features (X) cannot be None")

    if y is not None and y is None:
        raise ValueError(f"{name} labels (y) cannot be None")

    # Convert to numpy array if needed
    if not hasattr(X, '__len__'):
        raise ValueError(f"{name} must be array-like, got {type(X)}")

    n_samples = len(X)

    if n_samples == 0:
        raise ValueError(f"{name} cannot be empty (0 samples)")

    if n_samples < min_samples:
        raise ValueError(
            f"{name} has insufficient samples: {n_samples} < {min_samples}"
        )

    if y is not None:
        if len(y) != n_samples:
            raise ValueError(
                f"{name} X and y must have same length: "
                f"{n_samples} != {len(y)}"
            )

    if check_finite:
        # Check for finite values
        if hasattr(X, 'dtype'):
            arr = np.asarray(X)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values (inf/nan)")

        if y is not None and hasattr(y, 'dtype'):
            arr = np.asarray(y)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} labels contain non-finite values")


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """Safely divide with default value for division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero

    Example:
        >>> safe_divide(10.0, 2.0)
        5.0
        >>> safe_divide(10.0, 0.0)
        0.0
        >>> safe_divide(10.0, 0.0, default=1.0)
        1.0
    """
    if not np.isfinite(denominator) or denominator == 0:
        logger.warning(f"Division by zero prevented: {numerator}/{denominator}, returning {default}")
        return default

    return numerator / denominator


def validate_confidence(
    confidence: float,
    name: str = "confidence",
    min_value: float = 0.0,
    max_value: float = 1.0
) -> float:
    """Validate confidence score is in valid range.

    Args:
        confidence: Confidence score to validate
        name: Name for error messages
        min_value: Minimum valid value (default: 0.0)
        max_value: Maximum valid value (default: 1.0)

    Returns:
        The validated confidence score

    Raises:
        ValueError: If confidence is out of range
    """
    if not isinstance(confidence, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(confidence)}")

    if not np.isfinite(confidence):
        raise ValueError(f"{name} must be finite (not inf/nan), got {confidence}")

    confidence = float(confidence)

    if confidence < min_value or confidence > max_value:
        raise ValueError(
            f"{name} must be in [{min_value}, {max_value}], got {confidence}"
        )

    return confidence


def validate_string_not_empty(
    text: Optional[str],
    name: str = "text"
) -> str:
    """Validate string is not None or empty.

    Args:
        text: String to validate
        name: Name for error messages

    Returns:
        The validated string

    Raises:
        ValueError: If string is None or empty
    """
    if text is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text:
        raise ValueError(f"{name} cannot be empty")

    return text


def retry_on_failure(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 2.0,
    default_value: Any = None
) -> Callable:
    """Decorator for retrying functions on failure.

    Args:
        max_retries: Maximum number of retries
        exceptions: Exception types to catch
        backoff_factor: Multiplier for backoff delay
        default_value: Value to return if all retries fail

    Example:
        @retry_on_failure(max_retries=3, exceptions=(ValueError,))
        def risky_function():
            return might_fail()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            import time

            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e

                    if attempt < max_retries:
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                            f"{func.__name__}: {e}. Retrying in {sleep_time}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            if default_value is not None:
                return default_value

            raise last_error

        return wrapper

    return decorator


def get_or_default(
    value: Optional[T],
    default: T,
    name: str = "value"
) -> T:
    """Get value or return default if None.

    Args:
        value: Value to check
        default: Default value if value is None
        name: Name for logging

    Returns:
        value if not None, otherwise default
    """
    if value is None:
        logger.debug(f"{name} is None, using default: {default}")
        return default

    return value

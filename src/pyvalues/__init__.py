from .values import (
    DEFAULT_LANGUAGE,
    Score,
    AttainmentScore,
    Values,
    ValuesWithoutAttainment,
    ValuesWithAttainment,
    RefinedValues,
    RefinedCoarseValues,
    OriginalValues,
    RefinedValuesWithAttainment,
    RefinedCoarseValuesWithAttainment,
    OriginalValuesWithAttainment,
)
from .document import (
    Document,
    ValuesAnnotatedDocument,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "Score",
    "AttainmentScore",
    "Document",
    "Values",
    "ValuesWithoutAttainment",
    "ValuesWithAttainment",
    "ValuesAnnotatedDocument",
    "RefinedValues",
    "RefinedCoarseValues",
    "OriginalValues",
    "RefinedValuesWithAttainment",
    "RefinedCoarseValuesWithAttainment",
    "OriginalValuesWithAttainment",
]

from importlib import metadata

__version__ = metadata.version(__package__)  # type: ignore

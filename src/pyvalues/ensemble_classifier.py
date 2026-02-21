from typing import Generator, Generic, Iterable, Tuple, TypeVar

from pydantic_extra_types.language_code import LanguageAlpha2
from .classifiers import (
    OriginalValuesClassifier,
    OriginalValuesWithAttainmentClassifier,
    RefinedCoarseValuesClassifier,
    RefinedCoarseValuesWithAttainmentClassifier,
    RefinedValuesClassifier,
    RefinedValuesWithAttainmentClassifier,
)
from .values import (
    DEFAULT_LANGUAGE,
    AttainmentScore,
    OriginalValues,
    OriginalValuesWithAttainment,
    RefinedCoarseValues,
    RefinedCoarseValuesWithAttainment,
    RefinedValues,
    RefinedValuesWithAttainment,
    Values,
    ValuesWithoutAttainment,
)


CLASSIFIER = TypeVar("CLASSIFIER", bound="OriginalValuesClassifier")


class LanguageEnsembleClassifier(Generic[CLASSIFIER]):
    """
    Abstract base class for a classifier that assigns values based on different
    classifiers for each language.
    """

    _classifiers: dict[LanguageAlpha2, CLASSIFIER]

    def __init__(
            self,
            classifiers: dict[LanguageAlpha2, CLASSIFIER] = {}
    ):
        self._classifiers = classifiers

    def __getitem__(self, language: LanguageAlpha2):
        if language not in self._classifiers:
            OriginalValuesClassifier._raise_unsupported_language(language)
        return self._classifiers[language]

    def __setitem__(self, language: LanguageAlpha2, classifier: CLASSIFIER):
        self._classifiers[language] = classifier

    def __delitem__(self, language: LanguageAlpha2):
        del self._classifiers[language]


class OriginalValuesLanguageEnsembleClassifier(
    OriginalValuesClassifier,
    LanguageEnsembleClassifier[OriginalValuesClassifier]
):
    """
    Classifier that assigns values based on different classifiers for each
    language.
    """

    def __init__(
            self,
            classifiers: dict[LanguageAlpha2, OriginalValuesClassifier] = {}
    ):
        super().__init__(classifiers=classifiers)

    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str], None, None]:
        return self[language].classify_document_for_original_values(
            segments=segments,
            language=language
        )

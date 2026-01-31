from abc import ABC, abstractmethod
from typing import Generator, Sequence

from .values import (
    RefinedValues,
    RefinedCoarseValues,
    OriginalValues,
    RefinedValuesWithAttainment,
    RefinedCoarseValuesWithAttainment,
    OriginalValuesWithAttainment
)


class OriginalValuesClassifier(ABC):
    """ Classifier for the ten values from Schwartz original system.
    """

    def classify_for_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_all_for_original_values([text], language).__next__()

    @abstractmethod
    def classify_all_for_original_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[OriginalValues, None, None]:
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """

    def classify_for_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_for_refined_coarse_values(text=text, language=language).original_values()

    def classify_for_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        return self.classify_all_for_refined_coarse_values([text], language).__next__()

    @abstractmethod
    def classify_all_for_original_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[OriginalValues, None, None]:
        for values in self.classify_all_for_refined_coarse_values(texts, language):
            yield values.original_values()

    @abstractmethod
    def classify_all_for_refined_coarse_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedCoarseValues, None, None]:
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system.
    """

    def classify_for_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        return self.classify_for_refined_values(text=text, language=language).coarse_values()

    def classify_for_refined_values(self, text: str, language: str = "EN") -> RefinedValues:
        return self.classify_all_for_refined_values([text], language).__next__()

    def classify_all_for_refined_coarse_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedCoarseValues, None, None]:
        for values in self.classify_all_for_refined_values(texts, language):
            yield values.coarse_values()

    @abstractmethod
    def classify_all_for_refined_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedValues, None, None]:
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    """ Classifier for the ten values from Schwartz original system with attainment.
    """

    def classify_for_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_for_original_values_with_attainment(
            text=text,
            language=language
        ).without_attainment()

    def classify_for_original_values_with_attainment(
        self,
        text: str,
        language: str = "EN"
    ) -> OriginalValuesWithAttainment:
        return self.classify_all_for_original_values_with_attainment([text], language).__next__()

    def classify_all_for_original_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[OriginalValues, None, None]:
        for values in self.classify_all_for_original_values_with_attainment(texts, language):
            yield values.without_attainment()

    @abstractmethod
    def classify_all_for_original_values_with_attainment(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[OriginalValuesWithAttainment, None, None]:
        pass


class RefinedCoarseValuesWithAttainmentClassifier(
        OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix and with attainment.
    """

    def classify_for_refined_coarse_values(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValues:
        return self.classify_for_refined_coarse_values_with_attainment(
            text=text, language=language).without_attainment()

    def classify_for_original_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> OriginalValuesWithAttainment:
        return self.classify_for_refined_coarse_values_with_attainment(
            text=text, language=language).original_values()

    def classify_for_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValuesWithAttainment:
        return self.classify_all_for_refined_coarse_values_with_attainment([text], language).__next__()

    def classify_all_for_refined_coarse_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedCoarseValues, None, None]:
        for values in self.classify_all_for_refined_coarse_values_with_attainment(texts, language):
            yield values.without_attainment()

    def classify_all_for_original_values_with_attainment(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[OriginalValuesWithAttainment, None, None]:
        for values in self.classify_all_for_refined_coarse_values_with_attainment(texts, language):
            yield values.original_values()

    @abstractmethod
    def classify_all_for_refined_coarse_values_with_attainment(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedCoarseValuesWithAttainment, None, None]:
        pass


class RefinedValuesWithAttainmentClassifier(
        RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system with attainment.
    """

    def classify_for_refined_values(
        self, text: str, language: str = "EN"
    ) -> RefinedValues:
        return self.classify_for_refined_values_with_attainment(
            text=text, language=language).without_attainment()

    def classify_for_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValuesWithAttainment:
        return self.classify_for_refined_values_with_attainment(
            text=text, language=language).coarse_values()

    def classify_for_refined_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedValuesWithAttainment:
        return self.classify_all_for_refined_values_with_attainment([text], language).__next__()

    def classify_all_for_refined_values(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedValues, None, None]:
        for values in self.classify_all_for_refined_values_with_attainment(texts, language):
            yield values.without_attainment()

    def classify_all_for_refined_coarse_values_with_attainment(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedCoarseValuesWithAttainment, None, None]:
        for values in self.classify_all_for_refined_values_with_attainment(texts, language):
            yield values.coarse_values()

    @abstractmethod
    def classify_all_for_refined_values_with_attainment(
        self,
        texts: Generator[str, None, None] | Sequence[str],
        language: str = "EN"
    ) -> Generator[RefinedValuesWithAttainment, None, None]:
        pass

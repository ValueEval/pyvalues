from abc import ABC, abstractmethod
from typing import Generator, Iterable, Tuple

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

    def classify_for_original_values(self, text: str, language: str = "EN") -> Tuple[OriginalValues, str, str]:
        return self.classify_all_for_original_values([text], language).__next__()

    @abstractmethod
    def classify_all_for_original_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """

    def classify_for_original_values(self, text: str, language: str = "EN") -> Tuple[OriginalValues, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values(text=text, language=language)
        return values.original_values(), t, lang

    def classify_for_refined_coarse_values(self, text: str, language: str = "EN") -> Tuple[RefinedCoarseValues, str, str]:
        return self.classify_all_for_refined_coarse_values([text], language).__next__()

    @abstractmethod
    def classify_all_for_original_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_coarse_values(texts, language):
            yield values.original_values(), t, lang

    @abstractmethod
    def classify_all_for_refined_coarse_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system.
    """

    def classify_for_refined_coarse_values(self, text: str, language: str = "EN") -> Tuple[RefinedCoarseValues, str, str]:
        values, t, lang = self.classify_for_refined_values(text=text, language=language)
        return values.coarse_values(), t, lang

    def classify_for_refined_values(self, text: str, language: str = "EN") -> Tuple[RefinedValues, str, str]:
        return self.classify_all_for_refined_values([text], language).__next__()

    def classify_all_for_refined_coarse_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_values(texts, language):
            yield values.coarse_values(), t, lang

    @abstractmethod
    def classify_all_for_refined_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedValues, str, str], None, None]:
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    """ Classifier for the ten values from Schwartz original system with attainment.
    """

    def classify_for_original_values(self, text: str, language: str = "EN") -> Tuple[OriginalValues, str, str]:
        values, t, lang = self.classify_for_original_values_with_attainment(
            text=text,
            language=language
        )
        return values.without_attainment(), t, lang

    def classify_for_original_values_with_attainment(
        self,
        text: str,
        language: str = "EN"
    ) -> Tuple[OriginalValuesWithAttainment, str, str]:
        return self.classify_all_for_original_values_with_attainment([text], language).__next__()

    def classify_all_for_original_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        for values, t, lang in self.classify_all_for_original_values_with_attainment(texts, language):
            yield values.without_attainment(), t, lang

    @abstractmethod
    def classify_all_for_original_values_with_attainment(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str, str], None, None]:
        pass


class RefinedCoarseValuesWithAttainmentClassifier(
        OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix and with attainment.
    """

    def classify_for_refined_coarse_values(
        self, text: str, language: str = "EN"
    ) -> Tuple[RefinedCoarseValues, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values_with_attainment(
            text=text, language=language)
        return values.without_attainment(), t, lang

    def classify_for_original_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> Tuple[OriginalValuesWithAttainment, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values_with_attainment(
            text=text, language=language)
        return values.original_values(), t, lang

    def classify_for_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> Tuple[RefinedCoarseValuesWithAttainment, str, str]:
        return self.classify_all_for_refined_coarse_values_with_attainment([text], language).__next__()

    def classify_all_for_refined_coarse_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_coarse_values_with_attainment(texts, language):
            yield values.without_attainment(), t, lang

    def classify_all_for_original_values_with_attainment(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_coarse_values_with_attainment(texts, language):
            yield values.original_values(), t, lang

    @abstractmethod
    def classify_all_for_refined_coarse_values_with_attainment(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str, str], None, None]:
        pass


class RefinedValuesWithAttainmentClassifier(
        RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system with attainment.
    """

    def classify_for_refined_values(
        self, text: str, language: str = "EN"
    ) -> Tuple[RefinedValues, str, str]:
        values, t, lang = self.classify_for_refined_values_with_attainment(
            text=text, language=language)
        return values.without_attainment(), t, lang

    def classify_for_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> Tuple[RefinedCoarseValuesWithAttainment, str, str]:
        values, t, lang = self.classify_for_refined_values_with_attainment(
            text=text, language=language)
        return values.coarse_values(), t, lang

    def classify_for_refined_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> Tuple[RefinedValuesWithAttainment, str, str]:
        return self.classify_all_for_refined_values_with_attainment([text], language).__next__()

    def classify_all_for_refined_values(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedValues, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_values_with_attainment(texts, language):
            yield values.without_attainment(), t, lang

    def classify_all_for_refined_coarse_values_with_attainment(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str, str], None, None]:
        for values, t, lang in self.classify_all_for_refined_values_with_attainment(texts, language):
            yield values.coarse_values(), t, lang

    @abstractmethod
    def classify_all_for_refined_values_with_attainment(
        self,
        texts: Iterable[str],
        language: str = "EN"
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str, str], None, None]:
        pass

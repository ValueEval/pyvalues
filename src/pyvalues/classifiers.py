from abc import ABC, abstractmethod
from typing import Generator, Iterable, Tuple
from pydantic_extra_types.language_code import LanguageAlpha2

from .values import (
    RefinedValues,
    RefinedCoarseValues,
    OriginalValues,
    RefinedValuesWithAttainment,
    RefinedCoarseValuesWithAttainment,
    OriginalValuesWithAttainment
)

DEFAULT_LANGUAGE = LanguageAlpha2("en")


class OriginalValuesClassifier(ABC):
    """
    Classifier for the ten values from Schwartz original system.
    """

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[OriginalValues, str, str]:
        return self.classify_document_for_original_values([segment], language).__next__()

    @abstractmethod
    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):
    """
    Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[OriginalValues, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values(segment=segment, language=language)
        return values.original_values(), t, lang

    def classify_for_refined_coarse_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedCoarseValues, str, str]:
        return self.classify_document_for_refined_coarse_values([segment], language).__next__()

    @abstractmethod
    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_coarse_values(segments, language):
            yield values.original_values(), t, lang

    @abstractmethod
    def classify_document_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):
    """
    Classifier for the 19 values from Schwartz refined system.
    """

    def classify_for_refined_coarse_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedCoarseValues, str, str]:
        values, t, lang = self.classify_for_refined_values(segment=segment, language=language)
        return values.coarse_values(), t, lang

    def classify_for_refined_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedValues, str, str]:
        return self.classify_document_for_refined_values([segment], language).__next__()

    def classify_document_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_values(segments, language):
            yield values.coarse_values(), t, lang

    @abstractmethod
    def classify_document_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValues, str, str], None, None]:
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    """
    Classifier for the ten values from Schwartz original system with attainment.
    """

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[OriginalValues, str, str]:
        values, t, lang = self.classify_for_original_values_with_attainment(
            segment=segment,
            language=language
        )
        return values.without_attainment(), t, lang

    def classify_for_original_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[OriginalValuesWithAttainment, str, str]:
        return self.classify_document_for_original_values_with_attainment([segment], language).__next__()

    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str, str], None, None]:
        for values, t, lang in self.classify_document_for_original_values_with_attainment(segments, language):
            yield values.without_attainment(), t, lang

    @abstractmethod
    def classify_document_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str, str], None, None]:
        pass


class RefinedCoarseValuesWithAttainmentClassifier(
        OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier):
    """
    Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix and with attainment.
    """

    def classify_for_refined_coarse_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedCoarseValues, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values_with_attainment(
            segment=segment, language=language)
        return values.without_attainment(), t, lang

    def classify_for_original_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[OriginalValuesWithAttainment, str, str]:
        values, t, lang = self.classify_for_refined_coarse_values_with_attainment(
            segment=segment, language=language)
        return values.original_values(), t, lang

    def classify_for_refined_coarse_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedCoarseValuesWithAttainment, str, str]:
        return self.classify_document_for_refined_coarse_values_with_attainment([segment], language).__next__()

    def classify_document_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValues, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_coarse_values_with_attainment(segments, language):
            yield values.without_attainment(), t, lang

    def classify_document_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_coarse_values_with_attainment(segments, language):
            yield values.original_values(), t, lang

    @abstractmethod
    def classify_document_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str, str], None, None]:
        pass


class RefinedValuesWithAttainmentClassifier(
        RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier):
    """
    Classifier for the 19 values from Schwartz refined system with attainment.
    """

    def classify_for_refined_values(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedValues, str, str]:
        values, t, lang = self.classify_for_refined_values_with_attainment(
            segment=segment, language=language)
        return values.without_attainment(), t, lang

    def classify_for_refined_coarse_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedCoarseValuesWithAttainment, str, str]:
        values, t, lang = self.classify_for_refined_values_with_attainment(
            segment=segment, language=language)
        return values.coarse_values(), t, lang

    def classify_for_refined_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Tuple[RefinedValuesWithAttainment, str, str]:
        return self.classify_document_for_refined_values_with_attainment([segment], language).__next__()

    def classify_document_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValues, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_values_with_attainment(segments, language):
            yield values.without_attainment(), t, lang

    def classify_document_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str, str], None, None]:
        for values, t, lang in self.classify_document_for_refined_values_with_attainment(segments, language):
            yield values.coarse_values(), t, lang

    @abstractmethod
    def classify_document_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str, str], None, None]:
        pass

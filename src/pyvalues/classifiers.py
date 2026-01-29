from abc import ABC, abstractmethod

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

    @abstractmethod
    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """

    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_refined_coarse_values(text=text, language=language).original_values()

    @abstractmethod
    def classify_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system.
    """

    def classify_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        return self.classify_refined_values(text=text, language=language).coarse_values()

    @abstractmethod
    def classify_refined_values(self, text: str, language: str = "EN") -> RefinedValues:
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    """ Classifier for the ten values from Schwartz original system with attainment.
    """

    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_original_values_with_attainment(text=text, language=language).without_attainment()

    @abstractmethod
    def classify_original_values_with_attainment(
        self,
        text: str,
        language: str = "EN"
    ) -> OriginalValuesWithAttainment:
        pass


class RefinedCoarseValuesWithAttainmentClassifier(
        OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier):
    """ Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix and with attainment.
    """

    def classify_refined_coarse_values(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValues:
        return self.classify_refined_coarse_values_with_attainment(
            text=text, language=language).without_attainment()

    def classify_original_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> OriginalValuesWithAttainment:
        return self.classify_refined_coarse_values_with_attainment(
            text=text, language=language).original_values()

    @abstractmethod
    def classify_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValuesWithAttainment:
        pass


class RefinedValuesWithAttainmentClassifier(
        RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier):
    """ Classifier for the 19 values from Schwartz refined system with attainment.
    """

    def classify_refined_values(
        self, text: str, language: str = "EN"
    ) -> RefinedValues:
        return self.classify_refined_values_with_attainment(
            text=text, language=language).without_attainment()

    def classify_refined_coarse_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedCoarseValuesWithAttainment:
        return self.classify_refined_values_with_attainment(
            text=text, language=language).coarse_values()

    @abstractmethod
    def classify_refined_values_with_attainment(
        self, text: str, language: str = "EN"
    ) -> RefinedValuesWithAttainment:
        pass

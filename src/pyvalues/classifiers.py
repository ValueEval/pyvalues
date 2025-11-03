from abc import ABC, abstractmethod

from pyvalues.values import OriginalValues

from .values import *


class OriginalValuesClassifier(ABC):

    @abstractmethod
    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):

    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_refined_coarse_values(text=text, language=language).original_values()

    @abstractmethod
    def classify_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):

    def classify_refined_coarse_values(self, text: str, language: str = "EN") -> RefinedCoarseValues:
        return self.classify_refined_values(text=text, language=language).coarse_values()

    @abstractmethod
    def classify_refined_values(self, text: str, language: str = "EN") -> RefinedValues:
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    
    def classify_original_values(self, text: str, language: str = "EN") -> OriginalValues:
        return self.classify_original_values_with_attainment(text=text, language=language).without_attainment()

    @abstractmethod
    def classify_original_values_with_attainment(
            self, text: str, language: str = "EN"
        ) -> OriginalValuesWithAttainment:
        pass


class RefinedCoarseValuesWithAttainmentClassifier(
    OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier):
    
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

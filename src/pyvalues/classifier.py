from abc import ABC, abstractmethod
from typing import Generator, Iterable
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.document import Document, ValuesAnnotatedDocument

from .values import (
    RefinedValues,
    RefinedCoarseValues,
    OriginalValues,
    RefinedValuesWithAttainment,
    RefinedCoarseValuesWithAttainment,
    OriginalValuesWithAttainment,
    DEFAULT_LANGUAGE
)


class OriginalValuesClassifier(ABC):
    """
    Classifier for the ten values from Schwartz original system.
    """

    @staticmethod
    def _raise_unsupported_language(language):
        """
        Raises the appropriate error when being asked for an unsupported
        language.

        :param self: Description
        :param language: Description
        """
        raise ValueError(
            f"Unsupported language: {language}."
        )

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> OriginalValues:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: OriginalValues
        """
        return self.classify_segments_for_original_values(
            [segment],
            language
        ).__next__()

    def classify_document_for_original_values(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[OriginalValues]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[OriginalValues]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_original_values(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[OriginalValues](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    @abstractmethod
    def classify_segments_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValues, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classifications for each segment
        :rtype: Generator[OriginalValues, None, None]
        """
        pass


class RefinedCoarseValuesClassifier(OriginalValuesClassifier):
    """
    Classifier for the the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> OriginalValues:
        values = self.classify_for_refined_coarse_values(
            segment=segment,
            language=language
        )
        return values.original_values()

    def classify_for_refined_coarse_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedCoarseValues:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: RefinedCoarseValues
        """
        return self.classify_segments_for_refined_coarse_values(
            [segment],
            language
        ).__next__()

    def classify_document_for_refined_coarse_values(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[RefinedCoarseValues]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[RefinedCoarseValues]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_refined_coarse_values(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[RefinedCoarseValues](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    def classify_segments_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValues, None, None]:
        for values in self.classify_segments_for_refined_coarse_values(segments, language):
            yield values.original_values()

    @abstractmethod
    def classify_segments_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValues, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classification for each segment
        :rtype: Generator[RefinedCoarseValues, None, None]
        """
        pass


class RefinedValuesClassifier(RefinedCoarseValuesClassifier):
    """
    Classifier for the 19 values from Schwartz refined system.
    """

    def classify_for_refined_coarse_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedCoarseValues:
        values = self.classify_for_refined_values(
            segment=segment,
            language=language
        )
        return values.coarse_values()

    def classify_for_refined_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedValues:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: RefinedValues
        """
        return self.classify_segments_for_refined_values(
            [segment],
            language
        ).__next__()

    def classify_document_for_refined_values(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[RefinedValues]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[RefinedValues]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_refined_values(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[RefinedValues](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    def classify_segments_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValues, None, None]:
        for values in self.classify_segments_for_refined_values(segments, language):
            yield values.coarse_values()

    @abstractmethod
    def classify_segments_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedValues, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classifications for each segment
        :rtype: Generator[RefinedValues, None, None]
        """
        pass


class OriginalValuesWithAttainmentClassifier(OriginalValuesClassifier):
    """
    Classifier for the ten values from Schwartz original system with attainment.
    """

    def classify_for_original_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> OriginalValues:
        values = self.classify_for_original_values_with_attainment(
            segment=segment,
            language=language
        )
        return values.without_attainment()

    def classify_for_original_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> OriginalValuesWithAttainment:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: OriginalValuesWithAttainment
        """
        return self.classify_segments_for_original_values_with_attainment(
            [segment],
            language
        ).__next__()

    def classify_document_for_original_values_with_attainment(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[OriginalValuesWithAttainment]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[OriginalValuesWithAttainment]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_original_values_with_attainment(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[OriginalValuesWithAttainment](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    def classify_segments_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValues, None, None]:
        for values in self.classify_segments_for_original_values_with_attainment(segments, language):
            yield values.without_attainment()

    @abstractmethod
    def classify_segments_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValuesWithAttainment, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classification for each segment
        :rtype: Generator[OriginalValuesWithAttainment, None, None]
        """
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
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedCoarseValues:
        values = self.classify_for_refined_coarse_values_with_attainment(
            segment=segment,
            language=language
        )
        return values.without_attainment()

    def classify_for_original_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> OriginalValuesWithAttainment:
        values = self.classify_for_refined_coarse_values_with_attainment(
            segment=segment, language=language)
        return values.original_values()

    def classify_for_refined_coarse_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedCoarseValuesWithAttainment:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: RefinedCoarseValuesWithAttainment
        """
        return self.classify_segments_for_refined_coarse_values_with_attainment(
            [segment],
            language
        ).__next__()

    def classify_document_for_refined_coarse_values_with_attainment(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[RefinedCoarseValuesWithAttainment]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[RefinedCoarseValuesWithAttainment]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_refined_coarse_values_with_attainment(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[RefinedCoarseValuesWithAttainment](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    def classify_segments_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValues, None, None]:
        for values in self.classify_segments_for_refined_coarse_values_with_attainment(segments, language):
            yield values.without_attainment()

    def classify_segments_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValuesWithAttainment, None, None]:
        for values in self.classify_segments_for_refined_coarse_values_with_attainment(segments, language):
            yield values.original_values()

    @abstractmethod
    def classify_segments_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValuesWithAttainment, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classifications for each segment
        :rtype: Generator[RefinedCoarseValuesWithAttainment, None, None]
        """
        pass


class RefinedValuesWithAttainmentClassifier(
        RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier):
    """
    Classifier for the 19 values from Schwartz refined system with attainment.
    """

    def classify_for_refined_values(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedValues:
        values = self.classify_for_refined_values_with_attainment(
            segment=segment,
            language=language
        )
        return values.without_attainment()

    def classify_for_refined_coarse_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedCoarseValuesWithAttainment:
        values = self.classify_for_refined_values_with_attainment(
            segment=segment,
            language=language
        )
        return values.coarse_values()

    def classify_for_refined_values_with_attainment(
            self,
            segment: str,
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> RefinedValuesWithAttainment:
        """
        Classifies the segment.

        :param segment:
            The segment to classify
        :type segment: str
        :param language:
            The language of the segment
        :type language: LanguageAlpha2 | str
        :return:
            The classification
        :rtype: RefinedValuesWithAttainment
        """
        return self.classify_segments_for_refined_values_with_attainment(
            [segment],
            language
        ).__next__()

    def classify_document_for_refined_values_with_attainment(
            self,
            document: Document
    ) -> ValuesAnnotatedDocument[RefinedValuesWithAttainment]:
        """
        Classifies each segment.

        :param document:
            The document to classify
        :type document: Document
        :return:
            A copy of the document with the classified values
        :rtype: ValuesAnnotatedDocument[RefinedValuesWithAttainment]
        """
        if document.segments is None:
            raise ValueError("Segments is None")
        values = list(self.classify_segments_for_refined_values_with_attainment(
            document.segments,
            document.language
        ))
        return ValuesAnnotatedDocument[RefinedValuesWithAttainment](
            id=document.id,
            language=document.language,
            segments=document.segments,
            values=values
        )

    def classify_segments_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedValues, None, None]:
        for values in self.classify_segments_for_refined_values_with_attainment(segments, language):
            yield values.without_attainment()

    def classify_segments_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValuesWithAttainment, None, None]:
        for values in self.classify_segments_for_refined_values_with_attainment(segments, language):
            yield values.coarse_values()

    @abstractmethod
    def classify_segments_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedValuesWithAttainment, None, None]:
        """
        Classifies each segment.

        :param segments:
            The segments to classify
        :type segments: Iterable[str]
        :param language:
            The language of the segments
        :type language: LanguageAlpha2 | str
        :return:
            Classifications for each segment
        :rtype: Generator[RefinedValuesWithAttainment, None, None]
        """
        pass

from io import FileIO
import json
import math
from importlib.resources import files
from typing import Callable, Generator, Iterable, Tuple
import unicodedata
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.ensemble_classifier import OriginalValuesLanguageEnsembleClassifier
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


def normalize_dictionary_token(token: str, language: LanguageAlpha2) -> str:
    # return re.sub("[^a-z]", "", token.lower())
    return token.lower()


def simple_tokenize(text: str) -> list[str]:
    punctuation_stripped = "".join(
        character for character in text
        if not unicodedata.category(character).startswith("P")
    )
    return punctuation_stripped.split()


def get_dictionaries(
        input: dict[str, str | Tuple[str, float, float]] | FileIO,
        language: LanguageAlpha2,
        normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token
) -> dict[str, dict[str, AttainmentScore]]:
    dictionaries = {}
    if isinstance(input, dict):
        for token, value in input.items():
            normalized_token = normalize_token(token, language)
            if isinstance(value, str):
                value = (value, 1.0, 0.0)
            if normalized_token not in dictionaries:
                dictionaries[normalized_token] = {}
            dictionaries[normalized_token][value[0]] = AttainmentScore(
                attained=value[1],
                constrained=value[2]
            )
        return dictionaries
    else:
        for value, tokens in json.load(input).items():
            for token_entry in tokens:
                normalized_token = ""
                score_attained = 1.0
                score_constrained = 0.0
                if isinstance(token_entry, str):
                    normalized_token = normalize_token(token_entry, language)
                else:
                    normalized_token = normalize_token(token_entry["token"], language)
                    set_attained = False
                    if "score" in token_entry:
                        score_attained = token_entry["score"]
                        set_attained = True
                    elif "attained" in token_entry:
                        score_attained = token_entry["attained"]
                        set_attained = True
                    if "constrained" in token_entry:
                        score_constrained = token_entry["constrained"]
                        if not set_attained:
                            set_attained = True
                            score_attained = 0.0
                    if not set_attained:
                        raise ValueError(f"Neither 'attained' (or 'score') nor 'constrained' set for token '{token_entry['token']}'")
                if normalized_token not in dictionaries:
                    dictionaries[normalized_token] = {}
                dictionaries[normalized_token][value] = AttainmentScore(
                    attained=score_attained,
                    constrained=score_constrained
                )
        return dictionaries


class DictionaryClassifier():
    """
    Abstract base class for a classifier that assigns values based on a
    dictionary.
    """

    _language: LanguageAlpha2
    _dictionaries: dict[str, dict[str, AttainmentScore]]
    _tokenize: Callable[[str], list[str]]
    _normalize_token: Callable[[str, LanguageAlpha2], str]
    _score_threshold: float
    _max_values: int

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        self._language = language
        self._dictionaries = get_dictionaries(dictionaries, language, normalize_token)
        self._tokenize = tokenize
        self._normalize_token = normalize_token
        self._score_threshold = score_threshold
        self._max_values = max_values

    def _classify_document(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE,
            with_attainment: bool = False,
    ) -> Generator[Tuple[list[str], str], None, None]:
        if language != self._language:
            OriginalValuesClassifier._raise_unsupported_language(language)

        for segment in segments:
            value_scores_total = {}
            value_scores_attained = {}
            value_scores_constrained = {}
            for token in self._tokenize(segment):
                normalized_token = self._normalize_token(token, language)
                if normalized_token in self._dictionaries:
                    for value, scores in self._dictionaries[normalized_token].items():
                        if value in value_scores_total:
                            value_scores_total[value] += scores.total()
                        else:
                            value_scores_total[value] = scores.total()
                        if with_attainment:
                            if value in value_scores_attained:
                                value_scores_attained[value] += scores.attained
                            else:
                                value_scores_attained[value] = scores.attained
                            if value in value_scores_constrained:
                                value_scores_constrained[value] += scores.constrained
                            else:
                                value_scores_constrained[value] = scores.constrained
            value_scores_sorted = dict(sorted(
                value_scores_total.items(), key=lambda item: item[1], reverse=True
            ))
            labels = []
            for value, score in value_scores_sorted.items():
                if score >= self._score_threshold:
                    if with_attainment:
                        if value_scores_attained[value] >= value_scores_constrained[value]:
                            labels.append(value + " attained")
                        else:
                            labels.append(value + " constrained")
                    else:
                        labels.append(value)
                else:
                    break
            if self._max_values > 0 and len(labels) > self._max_values:
                labels = labels[0:self._max_values]
            yield labels, segment


class OriginalValuesDictionaryClassifier(DictionaryClassifier, OriginalValuesClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    @staticmethod
    def get_default(**kwargs):
        resources_dir = files("pyvalues.assets.dictionaries.20221007")
        classifiers = {}
        for dictionary_file in resources_dir.iterdir():
            if dictionary_file.is_file():
                language = LanguageAlpha2(
                    dictionary_file.name.removesuffix(".json")
                )
                with dictionary_file.open() as io:
                    classifiers[language] = OriginalValuesDictionaryClassifier(
                        language=language,
                        dictionaries=io,  # type: ignore
                        **kwargs
                    )
        return OriginalValuesLanguageEnsembleClassifier(classifiers=classifiers)

    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str], None, None]:
        with_attainment = False
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield OriginalValues.from_labels(labels), segment


class RefinedCoarseValuesDictionaryClassifier(DictionaryClassifier, RefinedCoarseValuesClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    def classify_document_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValues, str], None, None]:
        with_attainment = False
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield RefinedCoarseValues.from_labels(labels), segment


class RefinedValuesDictionaryClassifier(DictionaryClassifier, RefinedValuesClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    def classify_document_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValues, str], None, None]:
        with_attainment = False
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield RefinedValues.from_labels(labels), segment


class OriginalValuesWithAttainmentDictionaryClassifier(DictionaryClassifier, OriginalValuesWithAttainmentClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    def classify_document_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str], None, None]:
        with_attainment = True
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield OriginalValuesWithAttainment.from_labels(labels), segment


class RefinedCoarseValuesWithAttainmentDictionaryClassifier(DictionaryClassifier, RefinedCoarseValuesWithAttainmentClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    def classify_document_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str], None, None]:
        with_attainment = True
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield RefinedCoarseValuesWithAttainment.from_labels(labels), segment


class RefinedValuesWithAttainmentDictionaryClassifier(DictionaryClassifier, RefinedValuesWithAttainmentClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: LanguageAlpha2,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str, LanguageAlpha2], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language:
            The language of the dictionaries
        :type language: str
        :param dictionaries:
            The dictionaries that map from token to value label
            or a JSON file with the value label as keys and as value a list of
            either strings (the tokens) or objects with values for "token" (the
            token) and "score" (score associated with token)
        :type dictionaries: dict[str, str| Tuple[str, float, float]] | FileIO
        :param tokenize:
            Function to split a text into tokens (to be normalized
            and then looked up in the dictionaries; default: whitespace
            tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token:
            Function to normalize a token before looking it
            up in the dictionary, also used on the dictionaries (default: lowercase
            and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold:
            Threshold that needs to be reached by summing up
            the scores of all tokens for a value so that the value is assigned
            (default: minimum positive float value)
        :type score_threshold: float
        :param max_values:
            Maximum number of values to assign, starting from
            those with highest score (default: no maximum number)
        """
        super().__init__(
            language=language,
            dictionaries=dictionaries,
            tokenize=tokenize,
            normalize_token=normalize_token,
            score_threshold=score_threshold,
            max_values=max_values
        )

    def classify_document_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str], None, None]:
        with_attainment = True
        for labels, segment in self._classify_document(
                segments,
                language,
                with_attainment
        ):
            yield RefinedValuesWithAttainment.from_labels(labels), segment

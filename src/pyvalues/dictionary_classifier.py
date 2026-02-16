from io import FileIO
import json
import math
import re
from typing import Callable, Generator, Iterable, Tuple
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


def normalize_dictionary_token(token: str) -> str:
    return re.sub("[^a-z]", "", token.lower())


def simple_tokenize(text: str) -> list[str]:
    return text.split()


def get_dictionaries(
        input: dict[str, str | Tuple[str, float, float]] | FileIO,
        normalize_token: Callable[[str], str] = normalize_dictionary_token
) -> dict[str, Tuple[str, float, float]]:
    dictionaries = {}
    if isinstance(input, FileIO):
        for value, tokens in json.load(input):
            for token_entry in tokens:
                normalized_token = ""
                score_attained = 1.0
                score_constrained = 0.0
                if isinstance(token_entry, str):
                    normalized_token = normalize_token(token_entry)
                else:
                    normalized_token = normalize_token(token_entry["token"])
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
                        raise ValueError(f"Neither 'attained' (or 'score') nor \
                                         'constrained' set for token \
                                         '{token_entry['token']}'")
                if normalized_token in dictionaries:
                    raise ValueError(
                        f"Token '{normalized_token}' part of both \
                        '{dictionaries[normalized_token]}' and '{value}' \
                        dictionaries (before normalization: '{token_entry}')"
                    )
                dictionaries[normalized_token] = (value, score_attained, score_constrained)
        return dictionaries
    else:
        for token, value in input:
            normalized_token = normalize_token(token)
            if isinstance(value, str):
                value = (value, 1.0, 0.0)
            if normalized_token in dictionaries:
                raise ValueError(
                    f"Token '{normalized_token}' part of both \
                    '{dictionaries[normalized_token]}' and '{value[0]}' \
                    dictionaries (before normalization: '{token}')"
                )
            dictionaries[normalized_token] = value
        return dictionaries


class DictionaryClassifier():
    """
    Abstract base class for a classifier that assigns values based on a
    dictionary.
    """

    _language: str
    _dictionaries: dict[str, Tuple[str, float, float]]
    _tokenize: Callable[[str], list[str]]
    _normalize_token: Callable[[str], str]
    _score_threshold: float
    _max_values: int

    def __init__(
            self,
            language: str,
            dictionaries: dict[str, str | Tuple[str, float, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        self._language = language
        self._dictionaries = get_dictionaries(dictionaries, normalize_token)
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
                normalized_token = self._normalize_token(token)
                if normalized_token in self._dictionaries:
                    value = self._dictionaries[normalized_token]
                    if value[0] in value_scores_total:
                        value_scores_total[value[0]] += (value[1] + value[2])
                    else:
                        value_scores_total[value[0]] = (value[1] + value[2])
                    if with_attainment:
                        if value[0] in value_scores_attained:
                            value_scores_attained[value[0]] += value[1]
                        else:
                            value_scores_attained[value[0]] = value[1]
                        if value[0] in value_scores_constrained:
                            value_scores_constrained[value[0]] += value[2]
                        else:
                            value_scores_constrained[value[0]] = value[2]
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
            if self._max_values > 0 and len(labels) == self._max_values:
                labels = labels[0:self._max_values]
            yield labels, segment


class DictionaryOriginalValuesClassifier(DictionaryClassifier, OriginalValuesClassifier):
    """
    Classifier that assigns values based on a dictionary.
    """

    def __init__(
            self,
            language: str,
            dictionaries: dict[str, str | Tuple[str, float]] | FileIO,
            tokenize: Callable[[str], list[str]] = simple_tokenize,
            normalize_token: Callable[[str], str] = normalize_dictionary_token,
            score_threshold: float = math.ulp(0),
            max_values: int = 0
    ):
        """
        Creates a dictionary classifier for one language.

        :param language: The language of the dictionaries
        :type language: str
        :param dictionaries: The dictionaries that map from token to value label
        or a JSON file with the value label as keys and as value a list of
        either strings (the tokens) or objects with values for "token" (the
        token) and "score" (score associated with token)
        :type dictionaries: dict[str, str] | FileIO
        :param tokenize: Function to split a text into tokens (to be normalized
        and then looked up in the dictionaries; default: whitespace
        tokenization)
        :type tokenize: Callable[[str], list[str]]
        :param normalize_token: Function to normalize a token before looking it
        up in the dictionary, also used on the dictionaries (default: lowercase
        and strip non-letters (a-z))
        :type normalize_token: Callable[[str], str]
        :param score_threshold: Threshold that needs to be reached by summing up
        the scores of all tokens for a value so that the value is assigned
        (default: minimum positive float value)
        :type score_threshold: float
        :param max_values: Maximum number of values to assign, starting from
        those with highest score (default: no maximum number)
        """
        super(DictionaryClassifier, self).__init__(
            language=language,  # type: ignore
            dictionaries=dictionaries,  # type: ignore
            tokenize=tokenize,  # type: ignore
            normalize_token=normalize_token,  # type: ignore
            score_threshold=score_threshold,  # type: ignore
            max_values=max_values  # type: ignore
        )

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

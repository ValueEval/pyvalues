import random
from typing import Generator, Iterable, Tuple, TypeVar
from pydantic_extra_types.language_code import LanguageAlpha2
from .classifiers import OriginalValuesClassifier, OriginalValuesWithAttainmentClassifier, RefinedCoarseValuesClassifier, RefinedCoarseValuesWithAttainmentClassifier, RefinedValuesClassifier, RefinedValuesWithAttainmentClassifier
from .values import DEFAULT_LANGUAGE, AttainmentScore, OriginalValues, OriginalValuesWithAttainment, RefinedCoarseValues, RefinedCoarseValuesWithAttainment, RefinedValues, RefinedValuesWithAttainment, Values, ValuesWithoutAttainment


VALUES = TypeVar("VALUES", bound="Values")
VALUES_WITHOUT_ATTAINMENT = TypeVar("VALUES_WITHOUT_ATTAINMENT", bound="ValuesWithoutAttainment")


class AllAttainedClassifier(RefinedValuesWithAttainmentClassifier):
    """
    Classifier that assigns all values as attained to each text.
    """

    def classify_document_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str], None, None]:
        for segment in segments:
            yield RefinedValuesWithAttainment(
                self_direction_action=AttainmentScore(attained=1),
                self_direction_thought=AttainmentScore(attained=1),
                stimulation=AttainmentScore(attained=1),
                hedonism=AttainmentScore(attained=1),
                achievement=AttainmentScore(attained=1),
                power_dominance=AttainmentScore(attained=1),
                power_resources=AttainmentScore(attained=1),
                face=AttainmentScore(attained=1),
                security_personal=AttainmentScore(attained=1),
                security_societal=AttainmentScore(attained=1),
                tradition=AttainmentScore(attained=1),
                conformity_rules=AttainmentScore(attained=1),
                conformity_interpersonal=AttainmentScore(attained=1),
                humility=AttainmentScore(attained=1),
                benevolence_caring=AttainmentScore(attained=1),
                benevolence_dependability=AttainmentScore(attained=1),
                universalism_concern=AttainmentScore(attained=1),
                universalism_nature=AttainmentScore(attained=1),
                universalism_tolerance=AttainmentScore(attained=1),
            ), segment


class AllConstrainedClassifier(RefinedValuesWithAttainmentClassifier):
    """
    Classifier that assigns all values as constrained to each text.
    """

    def classify_document_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str], None, None]:
        for segment in segments:
            yield RefinedValuesWithAttainment(
                self_direction_action=AttainmentScore(constrained=1),
                self_direction_thought=AttainmentScore(constrained=1),
                stimulation=AttainmentScore(constrained=1),
                hedonism=AttainmentScore(constrained=1),
                achievement=AttainmentScore(constrained=1),
                power_dominance=AttainmentScore(constrained=1),
                power_resources=AttainmentScore(constrained=1),
                face=AttainmentScore(constrained=1),
                security_personal=AttainmentScore(constrained=1),
                security_societal=AttainmentScore(constrained=1),
                tradition=AttainmentScore(constrained=1),
                conformity_rules=AttainmentScore(constrained=1),
                conformity_interpersonal=AttainmentScore(constrained=1),
                humility=AttainmentScore(constrained=1),
                benevolence_caring=AttainmentScore(constrained=1),
                benevolence_dependability=AttainmentScore(constrained=1),
                universalism_concern=AttainmentScore(constrained=1),
                universalism_nature=AttainmentScore(constrained=1),
                universalism_tolerance=AttainmentScore(constrained=1),
            ), segment


class RandomOriginalValuesClassifier(OriginalValuesClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: OriginalValues | float = 0.5
    ):
        """
        Creates a random classifier.
        
        :param probabilities: The probabilities for each value
        :type probabilities: OriginalValues
        """
        if isinstance(probabilities, OriginalValues):
            self._probabilities = probabilities.to_list()
        else:
            self._probabilities = [ probabilities ] * 10

    def classify_document_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValues, str], None, None]:
        for segment in segments:
            draw = draw_list(self._probabilities)
            yield OriginalValues.from_list(draw), segment


class RandomRefinedCoarseValuesClassifier(RefinedCoarseValuesClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: RefinedCoarseValues | float = 0.5
    ):
        """
        Creates a random classifier.
        
        :param probabilities: The probabilities for each value
        :type probabilities: RefinedCoarseValues
        """
        if isinstance(probabilities, RefinedCoarseValues):
            self._probabilities = probabilities.to_list()
        else:
            self._probabilities = [ probabilities ] * 12

    def classify_document_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValues, str], None, None]:
        for segment in segments:
            draw = draw_list(self._probabilities)
            yield RefinedCoarseValues.from_list(draw), segment


class RandomRefinedValuesClassifier(RefinedValuesClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: RefinedValues | float = 0.5
    ):
        """
        Creates a random classifier.
        
        :param probabilities: The probabilities for each value
        :type probabilities: RefinedValues
        """
        if isinstance(probabilities, RefinedValues):
            self._probabilities = probabilities.to_list()
        else:
            self._probabilities = [ probabilities ] * 19

    def classify_document_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValues, str], None, None]:
        for segment in segments:
            draw = draw_list(self._probabilities)
            yield RefinedValues.from_list(draw), segment


class RandomOriginalValuesWithAttainmentClassifier(OriginalValuesWithAttainmentClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: OriginalValuesWithAttainment | AttainmentScore
            = AttainmentScore(attained=0.2929, constrained=0.2929)
    ):
        """
        Creates a random classifier.

        :param probabilities: The probabilities for each value (and attainment;
        default is 0.2929 so that at least one attainment is drawn at 50%)
        :type probabilities: RefinedCoarseValuesWithAttainment
        """
        if isinstance(probabilities, AttainmentScore):
            self._probabilities = [
                probabilities.attained, probabilities.constrained
            ] * 10
        else:
            self._probabilities = probabilities.to_list()

    def classify_document_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[OriginalValuesWithAttainment, str], None, None]:
        for segment in segments:
            draw = pick_one_attainment(draw_list(self._probabilities))
            yield OriginalValuesWithAttainment.from_list(draw), segment


class RandomRefinedCoarseValuesWithAttainmentClassifier(RefinedCoarseValuesWithAttainmentClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: RefinedCoarseValuesWithAttainment | AttainmentScore
            = AttainmentScore(attained=0.2929, constrained=0.2929)
    ):
        """
        Creates a random classifier.

        :param probabilities: The probabilities for each value (and attainment;
        default is 0.2929 so that at least one attainment is drawn at 50%)
        :type probabilities: RefinedCoarseValuesWithAttainment
        """
        if isinstance(probabilities, AttainmentScore):
            self._probabilities = [
                probabilities.attained, probabilities.constrained
            ] * 12
        else:
            self._probabilities = probabilities.to_list()

    def classify_document_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedCoarseValuesWithAttainment, str], None, None]:
        for segment in segments:
            draw = pick_one_attainment(draw_list(self._probabilities))
            yield RefinedCoarseValuesWithAttainment.from_list(draw), segment


class RandomRefinedValuesWithAttainmentClassifier(RefinedValuesWithAttainmentClassifier):
    """
    Classifier that assigns values at random.
    """

    _probabilities: list[float]

    def __init__(
            self,
            probabilities: RefinedValuesWithAttainment | AttainmentScore
            = AttainmentScore(attained=0.2929, constrained=0.2929)
    ):
        """
        Creates a random classifier.
        
        :param probabilities: The probabilities for each value (and attainment;
        default is 0.2929 so that at least one attainment is drawn at 50%)
        :type probabilities: RefinedValuesWithAttainment
        """
        if isinstance(probabilities, AttainmentScore):
            self._probabilities = [
                probabilities.attained, probabilities.constrained
            ] * 19
        else:
            self._probabilities = probabilities.to_list()

    def classify_document_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[Tuple[RefinedValuesWithAttainment, str], None, None]:
        for segment in segments:
            draw = pick_one_attainment(draw_list(self._probabilities))
            yield RefinedValuesWithAttainment.from_list(draw), segment


def draw_list(probabilities: list[float]):
    return [float(probability >= random.random()) for probability in probabilities]

def pick_one_attainment(draw):
    for i in range(int(len(draw) / 2)):
        if draw[i + 0] + draw[i + 1] > 1:
            draw[i + random.randint(0, 1)] = 0
    return draw

from typing import Generator, Iterable, Tuple
from pydantic_extra_types.language_code import LanguageAlpha2
from .classifiers import RefinedValuesWithAttainmentClassifier
from .values import DEFAULT_LANGUAGE, AttainmentScore, RefinedValuesWithAttainment


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

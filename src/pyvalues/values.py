from abc import ABC, abstractmethod
from typing import Annotated, Callable, Iterable, Self, Sequence, Tuple
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from .radarplot import plot_radar
import matplotlib.pyplot as plt

original_values = [
    "Self-direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Tradition",
    "Conformity",
    "Benevolence",
    "Universalism"
]

# Based on https://sashamaps.net/docs/resources/20-colors/
original_values_colors = [
    "#ffe119",  # self-direction
    "#fffac8",  # stimulation
    "#bfef45",  # hedonism
    "#3cb44b",  # achievement
    "#aaffc3",  # power
    "#000075",  # security
    "#911eb4",  # tradition
    "#f032e6",  # conformity
    "#e6194b",  # benevolence
    "#f58231",  # universalism
]

original_values_with_attainment = \
    [v + " attained" for v in original_values] + \
    [v + " constrained" for v in original_values]

refined_coarse_values = [
    "Self-direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Face",
    "Security",
    "Tradition",
    "Conformity",
    "Humility",
    "Benevolence",
    "Universalism"
]

# Based on https://sashamaps.net/docs/resources/20-colors/
refined_coarse_values_colors = [
    "#ffe119",  # self-direction
    "#fffac8",  # stimulation
    "#bfef45",  # hedonism
    "#3cb44b",  # achievement
    "#aaffc3",  # power
    "#42d4f4",  # face
    "#000075",  # security
    "#911eb4",  # tradition
    "#f032e6",  # conformity
    "#800000",  # humility
    "#e6194b",  # benevolence
    "#f58231",  # universalism
]

refined_coarse_values_with_attainment = \
    [v + " attained" for v in refined_coarse_values] + \
    [v + " constrained" for v in refined_coarse_values]

refined_values = [
    "Self-direction: action",
    "Self-direction: thought",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance"
]

# Based on https://sashamaps.net/docs/resources/20-colors/
refined_values_colors = [
    "#808000", "#ffe119",  # self-direction
    "#fffac8",  # stimulation
    "#bfef45",  # hedonism
    "#3cb44b",  # achievement
    "#aaffc3", "#469990",  # power
    "#42d4f4",  # face
    "#000075", "#4363d8",  # security
    "#911eb4",  # tradition
    "#dcbeff", "#f032e6",  # conformity
    "#800000",  # humility
    "#e6194b", "#fabed4",  # benevolence
    "#9a6324", "#f58231", "#ffd8b1"  # universalism
]

refined_values_with_attainment = \
    [v + " attained" for v in refined_values] + \
    [v + " constrained" for v in refined_values]


Score = Annotated[float, Field(ge=0, le=1)]


class AttainmentScore(BaseModel):
    attained: Score = 0.0
    constrained: Score = 0.0

    @model_validator(mode="after")
    def _check_total(self) -> Self:
        total = self.attained + self.constrained
        if total > 1:
            raise ValueError(
                f"Total > 1: {self.attained} + {self.constrained} = {total}")
        return self

    def total(self) -> Score:
        return self.attained + self.constrained


class ThresholdedDecision(BaseModel):
    threshold: Score
    is_true: bool


class Evaluation(BaseModel):
    value_evaluations: dict[str, list[ThresholdedDecision]] = {}

    def __init__(self, value_evaluations: dict[str, list[ThresholdedDecision]]):
        super().__init__(value_evaluations=value_evaluations)
        for thresholded_decisions in self.value_evaluations.values():
            thresholded_decisions.sort(key=lambda x: x.threshold)

    def __getitem__(self, key: str) -> list[ThresholdedDecision]:
        return self.value_evaluations[key]

    def _f(self, threshold: Score = 0.5, beta: float = 1) -> Tuple[dict[str, Score], dict[str, Score], dict[str, Score]]:
        beta_square = beta * beta
        fs = {}
        precisions = {}
        recalls = {}
        for value, thresholded_decisions in self.value_evaluations.items():
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for thresholded_decision in thresholded_decisions:
                if thresholded_decision.threshold >= threshold:
                    if thresholded_decision.is_true:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if thresholded_decision.is_true:
                        false_negatives += 1
                    else:
                        true_negatives += 1

            precision = 0
            recall = 0
            f = 0
            if true_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                f = (1 + beta_square) * precision * recall / ((beta_square * precision) + recall)
            precisions[value] = precision
            recalls[value] = recall
            fs[value] = f

        return fs, precisions, recalls

    def precision_recall_steps(self) -> dict[str, Tuple[list[float], list[float]]]:
        steps = {}
        for value, thresholded_decisions in self.value_evaluations.items():
            num_positive = sum([
                thresholded_decision.is_true for thresholded_decision
                in thresholded_decisions
            ])
            assert num_positive > 0
            true_positives = 0
            false_positives = 0
            xs = []
            ys = []
            last_threshold = 2
            for thresholded_decision in reversed(thresholded_decisions):
                if thresholded_decision.threshold < last_threshold:
                    if last_threshold <= 1:
                        xs.append(true_positives / num_positive)
                        if true_positives == 0:
                            ys.append(0)
                        else:
                            ys.append(true_positives / (true_positives + false_positives))
                    last_threshold = thresholded_decision.threshold
                    if thresholded_decision.is_true:
                        true_positives += 1
                    else:
                        false_positives += 1
            xs.append(1)
            if true_positives > 0:
                ys.append(true_positives / (true_positives + false_positives))
            else:
                ys.append(0)
            steps[value] = (xs, ys)
        return steps

    def plot_precision_recall_curves(self):
        num_values = len(self.value_evaluations.keys())
        colors = None
        if num_values == 10:
            colors = original_values_colors
        elif num_values == 12:
            colors = refined_coarse_values_colors
        elif num_values == 19:
            colors = refined_values_colors
        else:
            raise ValueError(f"Invalid number of values: {num_values}")

        fig = plt.figure()
        i = 0
        for value, steps in self.precision_recall_steps().items():
            plt.step(steps[0], steps[1], where="post", label=value, color=colors[i])
            i += 1

        axes = fig.get_axes()[0]
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel("Recall")
        axes.set_ylabel("Precision")
        plt.legend(loc="lower left")
        return plt


class OriginalValuesEvaluation(Evaluation):
    def f(self, threshold: Score = 0.5, beta: float = 1) -> Tuple["OriginalValues", "OriginalValues", "OriginalValues"]:
        fs, precisions, recalls = self._f(threshold, beta)
        return OriginalValues.model_validate(fs), \
            OriginalValues.model_validate(precisions), \
            OriginalValues.model_validate(recalls),


class RefinedCoarseValuesEvaluation(Evaluation):
    def f(self, threshold: Score = 0.5, beta: float = 1) \
            -> Tuple["RefinedCoarseValues", "RefinedCoarseValues", "RefinedCoarseValues"]:
        fs, precisions, recalls = self._f(threshold, beta)
        return RefinedCoarseValues.model_validate(fs), \
            RefinedCoarseValues.model_validate(precisions), \
            RefinedCoarseValues.model_validate(recalls),


class RefinedValuesEvaluation(Evaluation):
    def f(self, threshold: Score = 0.5, beta: float = 1) -> Tuple["RefinedValues", "RefinedValues", "RefinedValues"]:
        fs, precisions, recalls = self._f(threshold, beta)
        return RefinedValues.model_validate(fs), \
            RefinedValues.model_validate(precisions), \
            RefinedValues.model_validate(recalls),


class Values(ABC, BaseModel):
    """ Scores (with or without attainment) for any system of values.
    """

    @staticmethod
    def from_list(list: list[float]) -> "Values":
        match len(list):
            case 10:
                return OriginalValues.from_list(list)
            case 12:
                return RefinedCoarseValues.from_list(list)
            case 19:
                return RefinedValues.from_list(list)
            case 20:
                return OriginalValuesWithAttainment.from_list(list)
            case 24:
                return RefinedCoarseValuesWithAttainment.from_list(list)
            case 38:
                return RefinedValuesWithAttainment.from_list(list)
            case _:
                raise AssertionError(f"Invalid number of scores: {len(list)}")

    @abstractmethod
    def names(self) -> list[str]:
        pass

    @abstractmethod
    def to_list(self) -> list[float]:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Score | AttainmentScore:
        pass


class ValuesWithoutAttainment(Values):
    """ Scores without attainment for any system of values.
    """
    @staticmethod
    def plot_all(value_scores_list: Sequence["ValuesWithoutAttainment"], **kwargs):
        """ Plot scores in a radar plot.

        Returns the matplotlib module, so one can directly use `savefig(file)` or `show()`
        on the returned value.

        ::

            import pyvalues
            values = pyvalues.OriginalValues.from_list([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            pyvalues.plot_value_scores([values], labels=["my values"]).show()
        """
        assert len(value_scores_list) > 0
        assert all(
            type(value_scores) is type(value_scores_list[0])
            for value_scores in value_scores_list
        )
        return plot_radar(
            dim_names=value_scores_list[0].names(),
            valuess=[value_scores.to_list() for value_scores in value_scores_list],
            **kwargs
        )

    def __getitem__(self, key: str) -> Score:
        return getattr(self, normalize_value(key))

    def to_labels(self, threshold=0.5) -> list[str]:
        return [
            label for (label, score) in zip(self.names(), self.to_list())
            if score >= threshold
        ]

    def plot(self, linecolors=["black"], **kwargs):
        return ValuesWithoutAttainment.plot_all(
            [self], linecolors=linecolors, **kwargs)


class ValuesWithAttainment(Values):
    """ Scores with attainment for any system of values.
    """
    @abstractmethod
    def without_attainment(self) -> ValuesWithoutAttainment:
        pass

    def total(self) -> ValuesWithoutAttainment:
        return self.without_attainment()

    @abstractmethod
    def attained(self) -> ValuesWithoutAttainment:
        pass

    @abstractmethod
    def constrained(self) -> ValuesWithoutAttainment:
        pass

    def __getitem__(self, key: str) -> AttainmentScore:
        return getattr(self, normalize_value(key))

    def to_labels(self, threshold=0.5) -> list[str]:
        labels = []
        for label, attainment_score in self.model_dump().items():
            attained = attainment_score["attained"]
            constrained = attainment_score["constrained"]
            if attained + constrained >= threshold:
                if attained >= constrained:
                    labels.append(label + " attained")
                else:
                    labels.append(label + " constrained")
        return labels

    def plot(self, **kwargs):
        return ValuesWithoutAttainment.plot_all(
            [self.attained(), self.constrained(), self.total()],
            labels=["Attained", "Constrained", "Total"],
            linecolors=["green", "red", "black"],
            **kwargs
        )


class OriginalValues(ValuesWithoutAttainment):
    """ Scores for the ten values from Schwartz original system.
    """
    self_direction: Score = Field(
        default=0.0,
        serialization_alias="Self-direction",
        validation_alias=AliasChoices("self_direction", "Self-direction"),
    )
    stimulation: Score = Field(
        default=0.0,
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: Score = Field(
        default=0.0,
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: Score = Field(
        default=0.0,
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power: Score = Field(
        default=0.0,
        serialization_alias="Power",
        validation_alias=AliasChoices("power", "Power"),
    )
    security: Score = Field(
        default=0.0,
        serialization_alias="Security",
        validation_alias=AliasChoices("security", "Security"),
    )
    tradition: Score = Field(
        default=0.0,
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity: Score = Field(
        default=0.0,
        serialization_alias="Conformity",
        validation_alias=AliasChoices("conformity", "Conformity"),
    )
    benevolence: Score = Field(
        default=0.0,
        serialization_alias="Benevolence",
        validation_alias=AliasChoices("benevolence", "Benevolence"),
    )
    universalism: Score = Field(
        default=0.0,
        serialization_alias="Universalism",
        validation_alias=AliasChoices("universalism", "Universalism"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "OriginalValues":
        assert len(list) == 10
        return OriginalValues(
            self_direction=list[0],
            stimulation=list[1],
            hedonism=list[2],
            achievement=list[3],
            power=list[4],
            security=list[5],
            tradition=list[6],
            conformity=list[7],
            benevolence=list[8],
            universalism=list[9]
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "OriginalValues":
        return OriginalValues.model_validate({label: 1 for label in labels})

    @staticmethod
    def average(value_scores_list: list["OriginalValues"]) -> "OriginalValues":
        return OriginalValues.from_list(average_value_scores(value_scores_list))

    @staticmethod
    def evaluate_all(
        tested: list["OriginalValues"],
        truth: list["OriginalValues"]
    ) -> OriginalValuesEvaluation:
        instance_evaluations = [t1.evaluate(t2) for t1, t2 in zip(tested, truth)]
        return OriginalValuesEvaluation(value_evaluations={
            value: [instance_evaluation[value] for instance_evaluation in instance_evaluations] for value in original_values
        })

    def names(self) -> list[str]:
        return original_values

    def to_list(self) -> list[float]:
        return [
            self.self_direction,
            self.stimulation,
            self.hedonism,
            self.achievement,
            self.power,
            self.security,
            self.tradition,
            self.conformity,
            self.benevolence,
            self.universalism
        ]

    def evaluate(self, truth: "OriginalValues") -> dict[str, ThresholdedDecision]:
        return evaluate(self, truth)


class RefinedCoarseValues(ValuesWithoutAttainment):
    """ Scores for the twelve values from Schwartz refined system (19 values)
    when combining values with same name prefix.
    """
    self_direction: Score = Field(
        default=0.0,
        serialization_alias="Self-direction",
        validation_alias=AliasChoices("self_direction", "Self-direction"),
    )
    stimulation: Score = Field(
        default=0.0,
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: Score = Field(
        default=0.0,
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: Score = Field(
        default=0.0,
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power: Score = Field(
        default=0.0,
        serialization_alias="Power",
        validation_alias=AliasChoices("power", "Power"),
    )
    face: Score = Field(
        default=0.0,
        serialization_alias="Face",
        validation_alias=AliasChoices("face", "Face"),
    )
    security: Score = Field(
        default=0.0,
        serialization_alias="Security",
        validation_alias=AliasChoices("security", "Security"),
    )
    tradition: Score = Field(
        default=0.0,
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity: Score = Field(
        default=0.0,
        serialization_alias="Conformity",
        validation_alias=AliasChoices("conformity", "Conformity"),
    )
    humility: Score = Field(
        default=0.0,
        serialization_alias="Humility",
        validation_alias=AliasChoices("humility", "Humility"),
    )
    benevolence: Score = Field(
        default=0.0,
        serialization_alias="Benevolence",
        validation_alias=AliasChoices("benevolence", "Benevolence"),
    )
    universalism: Score = Field(
        default=0.0,
        serialization_alias="Universalism",
        validation_alias=AliasChoices("universalism", "Universalism"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "RefinedCoarseValues":
        assert len(list) == 12
        return RefinedCoarseValues(
            self_direction=list[0],
            stimulation=list[1],
            hedonism=list[2],
            achievement=list[3],
            power=list[4],
            face=list[5],
            security=list[6],
            tradition=list[7],
            conformity=list[8],
            humility=list[9],
            benevolence=list[10],
            universalism=list[11]
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "RefinedCoarseValues":
        return RefinedCoarseValues.model_validate({label: 1 for label in labels})

    @staticmethod
    def average(value_scores_list: list["RefinedCoarseValues"]) -> "RefinedCoarseValues":
        return RefinedCoarseValues.from_list(average_value_scores(value_scores_list))

    @staticmethod
    def evaluate_all(
        tested: list["RefinedCoarseValues"],
        truth: list["RefinedCoarseValues"]
    ) -> RefinedCoarseValuesEvaluation:
        instance_evaluations = [t1.evaluate(t2) for t1, t2 in zip(tested, truth)]
        return RefinedCoarseValuesEvaluation(value_evaluations={
            value: [instance_evaluation[value] for instance_evaluation in instance_evaluations] for value in original_values
        })

    def names(self) -> list[str]:
        return refined_coarse_values

    def to_list(self) -> list[float]:
        return [
            self.self_direction,
            self.stimulation,
            self.hedonism,
            self.achievement,
            self.power,
            self.face,
            self.security,
            self.tradition,
            self.conformity,
            self.humility,
            self.benevolence,
            self.universalism
        ]

    def evaluate(self, truth: "RefinedCoarseValues") -> dict[str, ThresholdedDecision]:
        return evaluate(self, truth)

    def original_values(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction,
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=self.power,
            security=self.security,
            tradition=self.tradition,
            conformity=self.conformity,
            benevolence=self.benevolence,
            universalism=self.universalism,
        )


class RefinedValues(ValuesWithoutAttainment):
    """ Scores for the 19 values from Schwartz refined system.
    """
    self_direction_thought: Score = Field(
        default=0.0,
        serialization_alias="Self-direction: thought",
        validation_alias=AliasChoices("self_direction_thought", "Self-direction: thought"),
    )
    self_direction_action: Score = Field(
        default=0.0,
        serialization_alias="Self-direction: action",
        validation_alias=AliasChoices("self_direction_action", "Self-direction: action"),
    )
    stimulation: Score = Field(
        default=0.0,
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: Score = Field(
        default=0.0,
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: Score = Field(
        default=0.0,
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power_dominance: Score = Field(
        default=0.0,
        serialization_alias="Power: dominance",
        validation_alias=AliasChoices("power_dominance", "Power: dominance"),
    )
    power_resources: Score = Field(
        default=0.0,
        serialization_alias="Power: resources",
        validation_alias=AliasChoices("power_resources", "Power: resources"),
    )
    face: Score = Field(
        default=0.0,
        serialization_alias="Face",
        validation_alias=AliasChoices("face", "Face"),
    )
    security_personal: Score = Field(
        default=0.0,
        serialization_alias="Security: personal",
        validation_alias=AliasChoices("security_personal", "Security: personal"),
    )
    security_societal: Score = Field(
        default=0.0,
        serialization_alias="Security: societal",
        validation_alias=AliasChoices("security_societal", "Security: societal"),
    )
    tradition: Score = Field(
        default=0.0,
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity_rules: Score = Field(
        default=0.0,
        serialization_alias="Conformity: rules",
        validation_alias=AliasChoices("conformity_rules", "Conformity: rules"),
    )
    conformity_interpersonal: Score = Field(
        default=0.0,
        serialization_alias="Conformity: interpersonal",
        validation_alias=AliasChoices("conformity_interpersonal", "Conformity: interpersonal"),
    )
    humility: Score = Field(
        default=0.0,
        serialization_alias="Humility",
        validation_alias=AliasChoices("humility", "Humility"),
    )
    benevolence_caring: Score = Field(
        default=0.0,
        serialization_alias="Benevolence: caring",
        validation_alias=AliasChoices("benevolence_caring", "Benevolence: caring"),
    )
    benevolence_dependability: Score = Field(
        default=0.0,
        serialization_alias="Benevolence: dependability",
        validation_alias=AliasChoices("benevolence_dependability", "Benevolence: dependability"),
    )
    universalism_concern: Score = Field(
        default=0.0,
        serialization_alias="Universalism: concern",
        validation_alias=AliasChoices("universalism_concern", "Universalism: concern"),
    )
    universalism_nature: Score = Field(
        default=0.0,
        serialization_alias="Universalism: nature",
        validation_alias=AliasChoices("universalism_nature", "Universalism: nature"),
    )
    universalism_tolerance: Score = Field(
        default=0.0,
        serialization_alias="Universalism: tolerance",
        validation_alias=AliasChoices("universalism_tolerance", "Universalism: tolerance"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "RefinedValues":
        assert len(list) == 19
        return RefinedValues(
            self_direction_thought=list[0],
            self_direction_action=list[1],
            stimulation=list[2],
            hedonism=list[3],
            achievement=list[4],
            power_dominance=list[5],
            power_resources=list[6],
            face=list[7],
            security_personal=list[8],
            security_societal=list[9],
            tradition=list[10],
            conformity_rules=list[11],
            conformity_interpersonal=list[12],
            humility=list[13],
            benevolence_caring=list[14],
            benevolence_dependability=list[15],
            universalism_concern=list[16],
            universalism_nature=list[17],
            universalism_tolerance=list[18]
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "RefinedValues":
        return RefinedValues.model_validate({label: 1 for label in labels})

    @staticmethod
    def average(value_scores_list: list["RefinedValues"]) -> "RefinedValues":
        return RefinedValues.from_list(average_value_scores(value_scores_list))

    @staticmethod
    def evaluate_all(
        tested: list["RefinedValues"],
        truth: list["RefinedValues"]
    ) -> RefinedValuesEvaluation:
        instance_evaluations = [t1.evaluate(t2) for t1, t2 in zip(tested, truth)]
        return RefinedValuesEvaluation(value_evaluations={
            value: [instance_evaluation[value] for instance_evaluation in instance_evaluations] for value in original_values
        })

    def names(self) -> list[str]:
        return refined_values

    def to_list(self) -> list[float]:
        return [
            self.self_direction_action,
            self.self_direction_thought,
            self.stimulation,
            self.hedonism,
            self.achievement,
            self.power_dominance,
            self.power_resources,
            self.face,
            self.security_personal,
            self.security_societal,
            self.tradition,
            self.conformity_rules,
            self.conformity_interpersonal,
            self.humility,
            self.benevolence_caring,
            self.benevolence_dependability,
            self.universalism_concern,
            self.universalism_nature,
            self.universalism_tolerance,
        ]

    def evaluate(self, truth: "RefinedValues") -> dict[str, ThresholdedDecision]:
        return evaluate(self, truth)

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=mode([self.self_direction_action, self.self_direction_thought]),
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=mode([self.power_dominance, self.power_resources]),
            face=self.face,
            security=mode([self.security_personal, self.security_societal]),
            tradition=self.tradition,
            conformity=mode([self.conformity_rules, self.conformity_interpersonal]),
            humility=self.humility,
            benevolence=mode([self.benevolence_caring, self.benevolence_dependability]),
            universalism=mode([self.universalism_concern, self.universalism_nature, self.universalism_tolerance])
        )

    def original_values(self, mode: Callable[[Iterable[float]], float] = max) -> OriginalValues:
        return self.coarse_values(mode=mode).original_values()


class OriginalValuesWithAttainment(ValuesWithAttainment):
    """ Scores with attainment for the ten values from Schwartz original system.
    """
    self_direction: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Self-direction",
        validation_alias=AliasChoices("self_direction", "Self-direction"),
    )
    stimulation: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Power",
        validation_alias=AliasChoices("power", "Power"),
    )
    security: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Security",
        validation_alias=AliasChoices("security", "Security"),
    )
    tradition: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Conformity",
        validation_alias=AliasChoices("conformity", "Conformity"),
    )
    benevolence: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Benevolence",
        validation_alias=AliasChoices("benevolence", "Benevolence"),
    )
    universalism: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Universalism",
        validation_alias=AliasChoices("universalism", "Universalism"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "OriginalValuesWithAttainment":
        assert len(list) == 20
        return OriginalValuesWithAttainment(
            self_direction=AttainmentScore(attained=list[0], constrained=list[1]),
            stimulation=AttainmentScore(attained=list[2], constrained=list[3]),
            hedonism=AttainmentScore(attained=list[4], constrained=list[5]),
            achievement=AttainmentScore(attained=list[6], constrained=list[7]),
            power=AttainmentScore(attained=list[8], constrained=list[9]),
            security=AttainmentScore(attained=list[10], constrained=list[11]),
            tradition=AttainmentScore(attained=list[12], constrained=list[13]),
            conformity=AttainmentScore(attained=list[14], constrained=list[15]),
            benevolence=AttainmentScore(attained=list[16], constrained=list[17]),
            universalism=AttainmentScore(attained=list[18], constrained=list[19])
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "OriginalValuesWithAttainment":
        return OriginalValuesWithAttainment.model_validate(
            labels_with_attainment_to_dict(labels)
        )

    @staticmethod
    def average(value_scores_list: list["OriginalValuesWithAttainment"]) -> "OriginalValuesWithAttainment":
        return OriginalValuesWithAttainment.from_list(average_value_scores(value_scores_list))

    def names(self) -> list[str]:
        return original_values_with_attainment

    def to_list(self) -> list[float]:
        return [
            self.self_direction.attained,
            self.self_direction.constrained,
            self.stimulation.attained,
            self.stimulation.constrained,
            self.hedonism.attained,
            self.hedonism.constrained,
            self.achievement.attained,
            self.achievement.constrained,
            self.power.attained,
            self.power.constrained,
            self.security.attained,
            self.security.constrained,
            self.tradition.attained,
            self.tradition.constrained,
            self.conformity.attained,
            self.conformity.constrained,
            self.benevolence.attained,
            self.benevolence.constrained,
            self.universalism.attained,
            self.universalism.constrained
        ]

    def without_attainment(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power=self.power.total(),
            security=self.security.total(),
            tradition=self.tradition.total(),
            conformity=self.conformity.total(),
            benevolence=self.benevolence.total(),
            universalism=self.universalism.total(),
        )

    def attained(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction.attained,
            stimulation=self.stimulation.attained,
            hedonism=self.hedonism.attained,
            achievement=self.achievement.attained,
            power=self.power.attained,
            security=self.security.attained,
            tradition=self.tradition.attained,
            conformity=self.conformity.attained,
            benevolence=self.benevolence.attained,
            universalism=self.universalism.attained,
        )

    def constrained(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction.constrained,
            stimulation=self.stimulation.constrained,
            hedonism=self.hedonism.constrained,
            achievement=self.achievement.constrained,
            power=self.power.constrained,
            security=self.security.constrained,
            tradition=self.tradition.constrained,
            conformity=self.conformity.constrained,
            benevolence=self.benevolence.constrained,
            universalism=self.universalism.constrained,
        )

    def majority_attainment(self) -> "OriginalValuesWithAttainment":
        return OriginalValuesWithAttainment.model_validate(
            majority_attainment(self)
        )


class RefinedCoarseValuesWithAttainment(ValuesWithAttainment):
    """ Scores with attainment for the twelve values from Schwartz refined
    system (19 values) when combining values with same name prefix.
    """
    self_direction: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Self-direction",
        validation_alias=AliasChoices("self_direction", "Self-direction"),
    )
    stimulation: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Power",
        validation_alias=AliasChoices("power", "Power"),
    )
    face: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Face",
        validation_alias=AliasChoices("face", "Face"),
    )
    security: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Security",
        validation_alias=AliasChoices("security", "Security"),
    )
    tradition: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Conformity",
        validation_alias=AliasChoices("conformity", "Conformity"),
    )
    humility: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Humility",
        validation_alias=AliasChoices("humility", "Humility"),
    )
    benevolence: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Benevolence",
        validation_alias=AliasChoices("benevolence", "Benevolence"),
    )
    universalism: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Universalism",
        validation_alias=AliasChoices("universalism", "Universalism"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "RefinedCoarseValuesWithAttainment":
        assert len(list) == 24
        return RefinedCoarseValuesWithAttainment(
            self_direction=AttainmentScore(attained=list[0], constrained=list[1]),
            stimulation=AttainmentScore(attained=list[2], constrained=list[3]),
            hedonism=AttainmentScore(attained=list[4], constrained=list[5]),
            achievement=AttainmentScore(attained=list[6], constrained=list[7]),
            power=AttainmentScore(attained=list[8], constrained=list[9]),
            face=AttainmentScore(attained=list[10], constrained=list[11]),
            security=AttainmentScore(attained=list[12], constrained=list[13]),
            tradition=AttainmentScore(attained=list[14], constrained=list[15]),
            conformity=AttainmentScore(attained=list[16], constrained=list[17]),
            humility=AttainmentScore(attained=list[18], constrained=list[19]),
            benevolence=AttainmentScore(attained=list[20], constrained=list[21]),
            universalism=AttainmentScore(attained=list[22], constrained=list[23])
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "RefinedCoarseValuesWithAttainment":
        return RefinedCoarseValuesWithAttainment.model_validate(
            labels_with_attainment_to_dict(labels)
        )

    @staticmethod
    def average(value_scores_list: list["RefinedCoarseValuesWithAttainment"]) -> "RefinedCoarseValuesWithAttainment":
        return RefinedCoarseValuesWithAttainment.from_list(average_value_scores(value_scores_list))

    def names(self) -> list[str]:
        return refined_coarse_values_with_attainment

    def to_list(self) -> list[float]:
        return [
            self.self_direction.attained,
            self.self_direction.constrained,
            self.stimulation.attained,
            self.stimulation.constrained,
            self.hedonism.attained,
            self.hedonism.constrained,
            self.achievement.attained,
            self.achievement.constrained,
            self.power.attained,
            self.power.constrained,
            self.face.attained,
            self.face.constrained,
            self.security.attained,
            self.security.constrained,
            self.tradition.attained,
            self.tradition.constrained,
            self.conformity.attained,
            self.conformity.constrained,
            self.humility.attained,
            self.humility.constrained,
            self.benevolence.attained,
            self.benevolence.constrained,
            self.universalism.attained,
            self.universalism.constrained
        ]

    def original_values(self) -> OriginalValuesWithAttainment:
        return OriginalValuesWithAttainment(
            self_direction=self.self_direction,
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=self.power,
            security=self.security,
            tradition=self.tradition,
            conformity=self.conformity,
            benevolence=self.benevolence,
            universalism=self.universalism,
        )

    def without_attainment(self) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=self.self_direction.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power=self.power.total(),
            face=self.face.total(),
            security=self.security.total(),
            tradition=self.tradition.total(),
            conformity=self.conformity.total(),
            humility=self.humility.total(),
            benevolence=self.benevolence.total(),
            universalism=self.universalism.total(),
        )

    def attained(self) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=self.self_direction.attained,
            stimulation=self.stimulation.attained,
            hedonism=self.hedonism.attained,
            achievement=self.achievement.attained,
            power=self.power.attained,
            face=self.face.attained,
            security=self.security.attained,
            tradition=self.tradition.attained,
            conformity=self.conformity.attained,
            humility=self.humility.attained,
            benevolence=self.benevolence.attained,
            universalism=self.universalism.attained,
        )

    def constrained(self) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=self.self_direction.constrained,
            stimulation=self.stimulation.constrained,
            hedonism=self.hedonism.constrained,
            achievement=self.achievement.constrained,
            power=self.power.constrained,
            face=self.face.constrained,
            security=self.security.constrained,
            tradition=self.tradition.constrained,
            conformity=self.conformity.constrained,
            humility=self.humility.constrained,
            benevolence=self.benevolence.constrained,
            universalism=self.universalism.constrained,
        )

    def majority_attainment(self) -> "RefinedCoarseValuesWithAttainment":
        return RefinedCoarseValuesWithAttainment.model_validate(
            majority_attainment(self)
        )


class RefinedValuesWithAttainment(ValuesWithAttainment):
    """ Scores with attainment for the 19 values from Schwartz refined system.
    """
    self_direction_thought: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Self-direction: thought",
        validation_alias=AliasChoices("self_direction_thought", "Self-direction: thought"),
    )
    self_direction_action: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Self-direction: action",
        validation_alias=AliasChoices("self_direction_action", "Self-direction: action"),
    )
    stimulation: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Stimulation",
        validation_alias=AliasChoices("stimulation", "Stimulation"),
    )
    hedonism: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Hedonism",
        validation_alias=AliasChoices("hedonism", "Hedonism"),
    )
    achievement: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Achievement",
        validation_alias=AliasChoices("achievement", "Achievement"),
    )
    power_dominance: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Power: dominance",
        validation_alias=AliasChoices("power_dominance", "Power: dominance"),
    )
    power_resources: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Power: resources",
        validation_alias=AliasChoices("power_resources", "Power: resources"),
    )
    face: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Face",
        validation_alias=AliasChoices("face", "Face"),
    )
    security_personal: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Security: personal",
        validation_alias=AliasChoices("security_personal", "Security: personal"),
    )
    security_societal: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Security: societal",
        validation_alias=AliasChoices("security_societal", "Security: societal"),
    )
    tradition: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Tradition",
        validation_alias=AliasChoices("tradition", "Tradition"),
    )
    conformity_rules: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Conformity: rules",
        validation_alias=AliasChoices("conformity_rules", "Conformity: rules"),
    )
    conformity_interpersonal: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Conformity: interpersonal",
        validation_alias=AliasChoices("conformity_interpersonal", "Conformity: interpersonal"),
    )
    humility: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Humility",
        validation_alias=AliasChoices("humility", "Humility"),
    )
    benevolence_caring: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Benevolence: caring",
        validation_alias=AliasChoices("benevolence_caring", "Benevolence: caring"),
    )
    benevolence_dependability: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Benevolence: dependability",
        validation_alias=AliasChoices("benevolence_dependability", "Benevolence: dependability"),
    )
    universalism_concern: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Universalism: concern",
        validation_alias=AliasChoices("universalism_concern", "Universalism: concern"),
    )
    universalism_nature: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Universalism: nature",
        validation_alias=AliasChoices("universalism_nature", "Universalism: nature"),
    )
    universalism_tolerance: AttainmentScore = Field(
        default=AttainmentScore(),
        serialization_alias="Universalism: tolerance",
        validation_alias=AliasChoices("universalism_tolerance", "Universalism: tolerance"),
    )

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    @staticmethod
    def from_list(list: list[float]) -> "RefinedValuesWithAttainment":
        assert len(list) == 38
        return RefinedValuesWithAttainment(
            self_direction_action=AttainmentScore(attained=list[0], constrained=list[1]),
            self_direction_thought=AttainmentScore(attained=list[2], constrained=list[3]),
            stimulation=AttainmentScore(attained=list[4], constrained=list[5]),
            hedonism=AttainmentScore(attained=list[6], constrained=list[7]),
            achievement=AttainmentScore(attained=list[8], constrained=list[9]),
            power_dominance=AttainmentScore(attained=list[10], constrained=list[11]),
            power_resources=AttainmentScore(attained=list[12], constrained=list[13]),
            face=AttainmentScore(attained=list[14], constrained=list[15]),
            security_personal=AttainmentScore(attained=list[16], constrained=list[17]),
            security_societal=AttainmentScore(attained=list[18], constrained=list[19]),
            tradition=AttainmentScore(attained=list[20], constrained=list[21]),
            conformity_rules=AttainmentScore(attained=list[22], constrained=list[23]),
            conformity_interpersonal=AttainmentScore(attained=list[24], constrained=list[25]),
            humility=AttainmentScore(attained=list[26], constrained=list[27]),
            benevolence_caring=AttainmentScore(attained=list[28], constrained=list[29]),
            benevolence_dependability=AttainmentScore(attained=list[30], constrained=list[31]),
            universalism_concern=AttainmentScore(attained=list[32], constrained=list[33]),
            universalism_nature=AttainmentScore(attained=list[34], constrained=list[35]),
            universalism_tolerance=AttainmentScore(attained=list[36], constrained=list[37])
        )

    @staticmethod
    def from_labels(labels: list[str]) -> "RefinedValuesWithAttainment":
        return RefinedValuesWithAttainment.model_validate(
            labels_with_attainment_to_dict(labels)
        )

    @staticmethod
    def average(value_scores_list: list["RefinedValuesWithAttainment"]) -> "RefinedValuesWithAttainment":
        return RefinedValuesWithAttainment.from_list(average_value_scores(value_scores_list))

    def names(self) -> list[str]:
        return refined_values_with_attainment

    def to_list(self) -> list[float]:
        return [
            self.self_direction_action.attained,
            self.self_direction_thought.attained,
            self.stimulation.attained,
            self.hedonism.attained,
            self.achievement.attained,
            self.power_dominance.attained,
            self.power_resources.attained,
            self.face.attained,
            self.security_personal.attained,
            self.security_societal.attained,
            self.tradition.attained,
            self.conformity_rules.attained,
            self.conformity_interpersonal.attained,
            self.humility.attained,
            self.benevolence_caring.attained,
            self.benevolence_dependability.attained,
            self.universalism_concern.attained,
            self.universalism_nature.attained,
            self.universalism_tolerance.attained,
            self.self_direction_action.constrained,
            self.self_direction_thought.constrained,
            self.stimulation.constrained,
            self.hedonism.constrained,
            self.achievement.constrained,
            self.power_dominance.constrained,
            self.power_resources.constrained,
            self.face.constrained,
            self.security_personal.constrained,
            self.security_societal.constrained,
            self.tradition.constrained,
            self.conformity_rules.constrained,
            self.conformity_interpersonal.constrained,
            self.humility.constrained,
            self.benevolence_caring.constrained,
            self.benevolence_dependability.constrained,
            self.universalism_concern.constrained,
            self.universalism_nature.constrained,
            self.universalism_tolerance.constrained
        ]

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValuesWithAttainment:
        return RefinedCoarseValuesWithAttainment(
            self_direction=combine_attainment_scores(
                [self.self_direction_action, self.self_direction_thought], mode=mode),
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=combine_attainment_scores(
                [self.power_dominance, self.power_resources], mode=mode),
            face=self.face,
            security=combine_attainment_scores(
                [self.security_personal, self.security_societal], mode=mode),
            tradition=self.tradition,
            conformity=combine_attainment_scores(
                [self.conformity_rules, self.conformity_interpersonal], mode=mode),
            humility=self.humility,
            benevolence=combine_attainment_scores(
                [self.benevolence_caring, self.benevolence_dependability], mode=mode),
            universalism=combine_attainment_scores(
                [self.universalism_concern, self.universalism_nature, self.universalism_tolerance], mode=mode)
        )

    def original_values(self, mode: Callable[[Iterable[float]], float] = max) -> OriginalValuesWithAttainment:
        return self.coarse_values(mode=mode).original_values()

    def without_attainment(self) -> RefinedValues:
        return RefinedValues(
            self_direction_action=self.self_direction_action.total(),
            self_direction_thought=self.self_direction_thought.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power_dominance=self.power_dominance.total(),
            power_resources=self.power_resources.total(),
            face=self.face.total(),
            security_personal=self.security_personal.total(),
            security_societal=self.security_societal.total(),
            tradition=self.tradition.total(),
            conformity_rules=self.conformity_rules.total(),
            conformity_interpersonal=self.conformity_interpersonal.total(),
            humility=self.humility.total(),
            benevolence_caring=self.benevolence_caring.total(),
            benevolence_dependability=self.benevolence_dependability.total(),
            universalism_concern=self.universalism_concern.total(),
            universalism_nature=self.universalism_nature.total(),
            universalism_tolerance=self.universalism_tolerance.total(),
        )

    def attained(self) -> RefinedValues:
        return RefinedValues(
            self_direction_action=self.self_direction_action.constrained,
            self_direction_thought=self.self_direction_thought.constrained,
            stimulation=self.stimulation.constrained,
            hedonism=self.hedonism.constrained,
            achievement=self.achievement.constrained,
            power_dominance=self.power_dominance.constrained,
            power_resources=self.power_resources.constrained,
            face=self.face.constrained,
            security_personal=self.security_personal.constrained,
            security_societal=self.security_societal.constrained,
            tradition=self.tradition.constrained,
            conformity_rules=self.conformity_rules.constrained,
            conformity_interpersonal=self.conformity_interpersonal.constrained,
            humility=self.humility.constrained,
            benevolence_caring=self.benevolence_caring.constrained,
            benevolence_dependability=self.benevolence_dependability.constrained,
            universalism_concern=self.universalism_concern.constrained,
            universalism_nature=self.universalism_nature.constrained,
            universalism_tolerance=self.universalism_tolerance.constrained,
        )

    def constrained(self) -> RefinedValues:
        return RefinedValues(
            self_direction_action=self.self_direction_action.constrained,
            self_direction_thought=self.self_direction_thought.constrained,
            stimulation=self.stimulation.constrained,
            hedonism=self.hedonism.constrained,
            achievement=self.achievement.constrained,
            power_dominance=self.power_dominance.constrained,
            power_resources=self.power_resources.constrained,
            face=self.face.constrained,
            security_personal=self.security_personal.constrained,
            security_societal=self.security_societal.constrained,
            tradition=self.tradition.constrained,
            conformity_rules=self.conformity_rules.constrained,
            conformity_interpersonal=self.conformity_interpersonal.constrained,
            humility=self.humility.constrained,
            benevolence_caring=self.benevolence_caring.constrained,
            benevolence_dependability=self.benevolence_dependability.constrained,
            universalism_concern=self.universalism_concern.constrained,
            universalism_nature=self.universalism_nature.constrained,
            universalism_tolerance=self.universalism_tolerance.constrained,
        )

    def majority_attainment(self) -> "RefinedValuesWithAttainment":
        return RefinedValuesWithAttainment.model_validate(
            majority_attainment(self)
        )


def normalize_value(value: str) -> str:
    return value.lower().replace("-", "_").replace(":", "").replace(" ", "_")


def combine_attainment_scores(
    scores: Iterable[AttainmentScore],
    mode: Callable[[Iterable[float]], float] = max
) -> AttainmentScore:
    totals = []
    attained = 0.0
    constrained = 0.0
    for score in scores:
        totals.append(score.total())
        attained += score.attained
        constrained += score.constrained
    total = mode(totals)
    assert total >= 0
    assert total <= 1
    if attained + constrained == 0:
        return AttainmentScore()
    else:
        weighted_attained = total * (attained / (attained + constrained))
        weighted_constrained = total - weighted_attained
        return AttainmentScore(
            attained=weighted_attained,
            constrained=weighted_constrained
        )


def average_value_scores(value_scores_list: Sequence[Values]) -> list[float]:
    num_scores = len(value_scores_list)
    if num_scores == 0:
        return []
    value_scores_matrix = [value_scores.to_list() for value_scores in value_scores_list]
    sums = map(sum, zip(*value_scores_matrix))
    means = [score_sum / num_scores for score_sum in sums]
    return means


def labels_with_attainment_to_dict(labels: list[str]) -> dict[str, float]:
    model = {}
    for label in labels:
        if label.endswith(" attained"):
            labelWithoutAttainment = label[:-9]
            assert labelWithoutAttainment not in model
            model[labelWithoutAttainment] = AttainmentScore(attained=1)
        elif label.endswith(" constrained"):
            labelWithoutAttainment = label[:-12]
            assert labelWithoutAttainment not in model
            model[labelWithoutAttainment] = AttainmentScore(constrained=1)
        else:
            assert label not in model
            model[label] = AttainmentScore(attained=1)
    return model


def majority_attainment(value_scores: ValuesWithAttainment) -> dict[str, AttainmentScore]:
    model = {}
    for value, attainment_score in value_scores.model_dump().items():
        attained = attainment_score["attained"]
        constrained = attainment_score["constrained"]
        if attained >= constrained:
            model[value] = AttainmentScore(attained=attained + constrained)
        else:
            model[value] = AttainmentScore(constrained=attained + constrained)
    return model


def evaluate(tested: ValuesWithoutAttainment, truth: ValuesWithoutAttainment) -> dict[str, ThresholdedDecision]:
    assert type(tested) is type(truth)
    decisions = {}
    for value in tested.names():
        decisions[value] = ThresholdedDecision(
            threshold=tested[value],
            is_true=truth[value] >= 0.5
        )
    return decisions

from abc import ABC, abstractmethod
import csv
from pathlib import Path
from typing import Annotated, Callable, ClassVar, Generator, Generic, Iterable, Self, Sequence, TextIO, Tuple, Type, TypeVar
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_extra_types.language_code import LanguageAlpha2
from .radarplot import plot_radar
import matplotlib.pyplot as plt

DEFAULT_LANGUAGE: LanguageAlpha2 = LanguageAlpha2("en")

ORIGINAL_VALUES = [
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
ORIGINAL_VALUES_COLORS = [
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

ORIGINAL_VALUES_WITH_ATTAINMENT = sum(
    [[v + " attained", v + "constrained"] for v in ORIGINAL_VALUES],
    []
)

REFINED_COARSE_VALUES = [
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
REFINED_COARSE_VALUES_COLORS = [
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

REFINED_COARSE_VALUES_WITH_ATTAINMENT = sum(
    [[v + " attained", v + "constrained"] for v in REFINED_COARSE_VALUES],
    []
)

REFINED_VALUES = [
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
REFINED_VALUES_COLORS = [
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

REFINED_VALUES_WITH_ATTAINMENT = sum(
    [[v + " attained", v + "constrained"] for v in REFINED_VALUES],
    []
)


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


VALUES = TypeVar("VALUES", bound="Values")
VALUES_WITHOUT_ATTAINMENT = TypeVar("VALUES_WITHOUT_ATTAINMENT", bound="ValuesWithoutAttainment")


class Evaluation(Generic[VALUES_WITHOUT_ATTAINMENT]):
    _cls: Type[VALUES_WITHOUT_ATTAINMENT]
    _value_evaluations: dict[str, list[ThresholdedDecision]] = {}

    def __init__(
        self,
        cls: Type[VALUES_WITHOUT_ATTAINMENT],
        value_evaluations: dict[str, list[ThresholdedDecision]]
    ):
        self._cls = cls
        self._value_evaluations = value_evaluations
        for thresholded_decisions in self._value_evaluations.values():
            thresholded_decisions.sort(key=lambda x: x.threshold)

    @classmethod
    def combine(cls, evaluations: Iterable[Self]) -> Self:
        clz = None
        combined = {}
        for evaluation in evaluations:
            for value, thresholded_decisions in evaluation._value_evaluations.items():
                if value not in combined:
                    combined[value] = thresholded_decisions.copy()
                else:
                    combined[value] += thresholded_decisions
            clz = evaluation._cls
        if clz is not None:
            return cls(clz, combined)
        raise ValueError("No evaluation given")

    def __getitem__(self, key: str) -> list[ThresholdedDecision]:
        return self._value_evaluations[key]

    def f(
            self,
            threshold: Score = 0.5,
            beta: float = 1
    ) -> Tuple["VALUES_WITHOUT_ATTAINMENT", "VALUES_WITHOUT_ATTAINMENT", "VALUES_WITHOUT_ATTAINMENT"]:
        beta_square = beta * beta
        fs = {}
        precisions = {}
        recalls = {}
        for value, thresholded_decisions in self._value_evaluations.items():
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

        return self._cls.model_validate(fs), \
            self._cls.model_validate(precisions), \
            self._cls.model_validate(recalls)

    def precision_recall_steps(self) -> dict[str, Tuple[list[float], list[float]]]:
        steps = {}
        for value, thresholded_decisions in self._value_evaluations.items():
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
        num_values = len(self._value_evaluations.keys())
        colors = None
        if num_values == 10:
            colors = ORIGINAL_VALUES_COLORS
        elif num_values == 12:
            colors = REFINED_COARSE_VALUES_COLORS
        elif num_values == 19:
            colors = REFINED_VALUES_COLORS
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


class Document(BaseModel, Generic[VALUES]):
    id: str | None = None
    language: LanguageAlpha2 = DEFAULT_LANGUAGE
    values: list[VALUES] | None = None
    segments: list[str] | None = None

    SEGMENT_FIELD: ClassVar[str] = "Text"
    ID_FIELD: ClassVar[str] = "ID"
    LANGUAGE_FIELD: ClassVar[str] = "Language"


class ValuesWriter(Generic[VALUES]):
    _writer: csv.DictWriter

    def __init__(
            self,
            cls: Type[VALUES],
            output_file: TextIO,
            delimiter: str = "\t"
    ):
        fieldnames = cls.names()
        self._writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter=delimiter
        )
        self._writer.writeheader()

    def write(self, values: VALUES):
        line: dict[str, float] = {
            value: score for (value, score) in zip(values.names(), values.to_list())
        }
        self._writer.writerow(line)

    def write_all(self, values: Iterable[VALUES]):
        for v in values:
            self.write(v)


class ValuesWithTextWriter(Generic[VALUES]):
    _writer: csv.DictWriter
    _write_document_id: bool
    _default_document_id: str | None
    _write_language: bool
    _default_language: LanguageAlpha2 | None

    def __init__(
            self,
            cls: Type[VALUES],
            output_file: TextIO,
            delimiter: str = "\t",
            write_document_id: bool = True,
            default_document_id: str | None = None,
            write_language: bool = True,
            default_language: LanguageAlpha2 | None = DEFAULT_LANGUAGE
    ):
        self._write_document_id = write_document_id
        self._default_document_id = default_document_id
        self._write_language = write_language
        self._default_language = default_language

        fieldnames = []
        if write_document_id:
            fieldnames += [Document.ID_FIELD]
        fieldnames += [Document.SEGMENT_FIELD]
        if write_language:
            fieldnames += [Document.LANGUAGE_FIELD]
        fieldnames += cls.names()

        self._writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter=delimiter
        )
        self._writer.writeheader()

    def write(
            self,
            values: VALUES,
            segment: str,
            document_id: str | None = None,
            language: LanguageAlpha2 | None = None
    ):
        line: dict[str, float | str] = {
            value: score for (value, score) in zip(values.names(), values.to_list())
        }
        line[Document.SEGMENT_FIELD] = segment
        if self._write_document_id:
            if document_id is not None:
                line[Document.ID_FIELD] = document_id
            elif self._default_document_id is not None:
                line[Document.ID_FIELD] = self._default_document_id
            else:
                raise ValueError("Missing document ID for writing and no default set")
        if self._write_language:
            if language is not None:
                line[Document.LANGUAGE_FIELD] = language
            elif self._default_language is not None:
                line[Document.LANGUAGE_FIELD] = self._default_language
            else:
                raise ValueError("Missing language for writing and no default set")
        self._writer.writerow(line)

    def write_all(
            self,
            values_with_segments: Iterable[Tuple[VALUES, str]],
            language: LanguageAlpha2,
            document_id: str | None = None
    ):
        for values, segment in values_with_segments:
            self.write(values=values, document_id=document_id, segment=segment, language=language)


class Values(ABC, BaseModel):
    """
    Scores (with or without attainment) for any system of values.
    """
    @classmethod
    @abstractmethod
    def from_list(cls, list: list[float]) -> Self:
        pass

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Self:
        value_scores: list[float] = [
            float(row.get(value_name, 0.0))
            for value_name in cls.names()
        ]
        return cls.from_list(value_scores)

    @classmethod
    def average(
        cls,
        value_scores_list: Iterable[Self]
    ) -> Self:
        value_scores_matrix = [value_scores.to_list() for value_scores in value_scores_list]
        num_scores = len(value_scores_matrix)
        if num_scores == 0:
            return cls()
        sums = map(sum, zip(*value_scores_matrix))
        means = [score_sum / num_scores for score_sum in sums]
        return cls.from_list(means)

    @classmethod
    def read_tsv(
        cls,
        input_file: str | Path,
        read_values: bool = True,
        document_id: str | None = None,
        language: LanguageAlpha2 = DEFAULT_LANGUAGE,
        delimiter: str = "\t",
        id_field: str | None = Document.ID_FIELD,
        language_field: str | None = Document.LANGUAGE_FIELD,
        segment_field: str | None = Document.SEGMENT_FIELD,
        **kwargs
    ) -> Generator[Document[Self], None, None]:
        current_document_id = document_id
        current_language: LanguageAlpha2 = language
        values: list[Self] | None = None
        segments: list[str] | None = None
        with open(input_file, newline='') as input_file_handle:
            reader = csv.DictReader(input_file_handle, delimiter=delimiter, **kwargs)
            for row in reader:
                row_document_id = None
                if id_field is not None:
                    row_document_id = row.get(id_field, document_id)
                if row_document_id is None or row_document_id != current_document_id:
                    if values is not None or segments is not None:
                        yield Document[Self](
                            id=current_document_id,
                            language=current_language,
                            values=values,
                            segments=segments
                        )
                        segments = None
                        values = None
                current_document_id = row_document_id
                if language_field is not None:
                    current_language = LanguageAlpha2(row.get(language_field, language))
                if read_values:
                    if values is None:
                        values = []
                    values.append(cls.from_row(row))
                if segment_field is not None:
                    segment = row.get(segment_field)
                    if segment is not None:
                        if segments is None:
                            segments = []
                        segments.append(segment)
            yield Document[Self](
                id=current_document_id,
                language=current_language,
                values=values,
                segments=segments
            )

    @classmethod
    def writer_tsv(
        cls,
        output_file: TextIO,
        delimiter: str = "\t"
    ) -> ValuesWriter[Self]:
        return ValuesWriter[Self](
            cls=cls,
            output_file=output_file,
            delimiter=delimiter
        )

    @classmethod
    def writer_tsv_with_text(
        cls,
        output_file: TextIO,
        delimiter: str = "\t",
        write_document_id: bool = True,
        default_document_id: str | None = None,
        write_language: bool = True,
        default_language: LanguageAlpha2 | None = DEFAULT_LANGUAGE
    ) -> ValuesWithTextWriter[Self]:
        return ValuesWithTextWriter[Self](
            cls=cls,
            output_file=output_file,
            delimiter=delimiter,
            write_document_id=write_document_id,
            default_document_id=default_document_id,
            write_language=write_language,
            default_language=default_language
        )

    @classmethod
    @abstractmethod
    def names(cls) -> list[str]:
        pass

    @abstractmethod
    def to_list(self) -> list[float]:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Score | AttainmentScore:
        pass


class ValuesWithoutAttainment(Values):
    """
    Scores without attainment for any system of values.
    """
    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> Self:
        return cls.model_validate({label: 1 for label in labels})

    @classmethod
    def evaluate_all(
        cls,
        tested: Iterable["Self"],
        truth: Iterable["Self"]
    ) -> Evaluation["Self"]:
        instance_evaluations = [t1.evaluate(t2) for t1, t2 in zip(tested, truth)]
        return Evaluation(
            cls=cls,
            value_evaluations={
                value: [
                    instance_evaluation[value] for instance_evaluation in instance_evaluations
                ] for value in ORIGINAL_VALUES
            }
        )

    @staticmethod
    def plot_all(value_scores_list: Sequence["ValuesWithoutAttainment"], **kwargs):
        """
        Plot scores in a radar plot.

        Returns the matplotlib module, so one can directly use `savefig(file)` or `show()`
        on the returned value.

        ::

            import pyvalues
            values = pyvalues.OriginalValues.from_list([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            pyvalues.plot_value_scores([values], labels=["my values"]).show()

        :param value_scores_list: The scores to plot
        :type value_scores_list: Sequence["ValuesWithoutAttainment"]
        :param kwargs: Arguments to pass on for plotting
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

    def evaluate(self, truth: "Self") -> dict[str, ThresholdedDecision]:
        decisions = {}
        for value in self.names():
            decisions[value] = ThresholdedDecision(
                threshold=self[value],
                is_true=truth[value] >= 0.5
            )
        return decisions

    def plot(self, linecolors=["black"], **kwargs):
        return ValuesWithoutAttainment.plot_all(
            [self], linecolors=linecolors, **kwargs)


class ValuesWithAttainment(Values):
    """
    Scores with attainment for any system of values.
    """
    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> Self:
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
        return cls.model_validate(model)

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

    def majority_attainment(self) -> Self:
        model = {}
        for value, attainment_score in self.model_dump().items():
            attained = attainment_score["attained"]
            constrained = attainment_score["constrained"]
            if attained >= constrained:
                model[value] = AttainmentScore(attained=attained + constrained)
            else:
                model[value] = AttainmentScore(constrained=attained + constrained)
        return self.model_validate(model)

    def plot(self, **kwargs):
        return ValuesWithoutAttainment.plot_all(
            [self.attained(), self.constrained(), self.total()],
            labels=["Attained", "Constrained", "Total"],
            linecolors=["green", "red", "black"],
            **kwargs
        )


class OriginalValues(ValuesWithoutAttainment):
    """
    Scores for the ten values from Schwartz original system.
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

    @classmethod
    def from_list(cls, list: list[float]) -> "OriginalValues":
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

    @classmethod
    def names(cls) -> list[str]:
        return ORIGINAL_VALUES

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


class RefinedCoarseValues(ValuesWithoutAttainment):
    """
    Scores for the twelve values from Schwartz refined system (19 values)
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

    @classmethod
    def from_list(cls, list: list[float]) -> "RefinedCoarseValues":
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

    @classmethod
    def names(cls) -> list[str]:
        return REFINED_COARSE_VALUES

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

    def original_values(self) -> OriginalValues:
        """
        Drops Face and Humility scores.

        :return: The reduced scores
        :rtype: OriginalValues
        """
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
    """
    Scores for the 19 values from Schwartz refined system.
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

    @classmethod
    def from_list(cls, list: list[float]) -> "RefinedValues":
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

    @classmethod
    def names(cls) -> list[str]:
        return REFINED_VALUES

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

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValues:
        """
        Combines the scores of the values with the same prefix.

        E.g., 'Universalism: concern', 'Universalism: nature', and
        'Universalism: tolerance' to 'Universalism'.

        :param mode: Function to combine the scores (default: max)
        :type mode: Callable[[Iterable[float]], float]
        :return: The combined scores
        :rtype: RefinedCoarseValues
        """
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
        """
        Combines the scores of the values with the same prefix and drops Face
        and Humility.

        E.g., 'Universalism: concern', 'Universalism: nature', and
        'Universalism: tolerance' to 'Universalism'.

        :param mode: Function to combine the scores (default: max)
        :type mode: Callable[[Iterable[float]], float]
        :return: The combined and reduced scores
        :rtype: OriginalValues
        """
        return self.coarse_values(mode=mode).original_values()


class OriginalValuesWithAttainment(ValuesWithAttainment):
    """
    Scores with attainment for the ten values from Schwartz original system.
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

    @classmethod
    def from_list(cls, list: list[float]) -> "OriginalValuesWithAttainment":
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

    @classmethod
    def names(cls) -> list[str]:
        return ORIGINAL_VALUES_WITH_ATTAINMENT

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
        """
        Combines the scores of the values with the same attainment by taking
        their sum.

        E.g., 'Achievement attained' and 'Achievement constrained' to
        'Achievement'.

        :return: The combined and scores
        :rtype: OriginalValues
        """
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
        """
        Takes the scores of the values for attained only, dropping for constrained.

        :return: The attained scores
        :rtype: OriginalValues
        """
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
        """
        Takes the scores of the values for constrained only, dropping for attained.

        :return: The constrained scores
        :rtype: OriginalValues
        """
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


class RefinedCoarseValuesWithAttainment(ValuesWithAttainment):
    """
    Scores with attainment for the twelve values from Schwartz refined
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

    @classmethod
    def from_list(cls, list: list[float]) -> "RefinedCoarseValuesWithAttainment":
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

    @classmethod
    def names(cls) -> list[str]:
        return REFINED_COARSE_VALUES_WITH_ATTAINMENT

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
        """
        Drops Face and Humility scores.

        :return: The reduced scores
        :rtype: OriginalValuesWithAttainment
        """
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
        """
        Combines the scores of the values with the same attainment by taking
        their sum.

        E.g., 'Achievement attained' and 'Achievement constrained' to
        'Achievement'.

        :return: The combined scores
        :rtype: RefinedCoarseValues
        """
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
        """
        Takes the scores of the values for attained only, dropping for constrained.

        :return: The attained scores
        :rtype: RefinedCoarseValues
        """
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
        """
        Takes the scores of the values for constrained only, dropping for attained.

        :return: The constrained scores
        :rtype: RefinedCoarseValues
        """
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


class RefinedValuesWithAttainment(ValuesWithAttainment):
    """
    Scores with attainment for the 19 values from Schwartz refined system.
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

    @classmethod
    def from_list(cls, list: list[float]) -> "RefinedValuesWithAttainment":
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

    @classmethod
    def names(cls) -> list[str]:
        return REFINED_VALUES_WITH_ATTAINMENT

    def to_list(self) -> list[float]:
        return [
            self.self_direction_action.attained,
            self.self_direction_action.constrained,
            self.self_direction_thought.attained,
            self.self_direction_thought.constrained,
            self.stimulation.attained,
            self.stimulation.constrained,
            self.hedonism.attained,
            self.hedonism.constrained,
            self.achievement.attained,
            self.achievement.constrained,
            self.power_dominance.attained,
            self.power_dominance.constrained,
            self.power_resources.attained,
            self.power_resources.constrained,
            self.face.attained,
            self.face.constrained,
            self.security_personal.attained,
            self.security_personal.constrained,
            self.security_societal.attained,
            self.security_societal.constrained,
            self.tradition.attained,
            self.tradition.constrained,
            self.conformity_rules.attained,
            self.conformity_rules.constrained,
            self.conformity_interpersonal.attained,
            self.conformity_interpersonal.constrained,
            self.humility.attained,
            self.humility.constrained,
            self.benevolence_caring.attained,
            self.benevolence_caring.constrained,
            self.benevolence_dependability.attained,
            self.benevolence_dependability.constrained,
            self.universalism_concern.attained,
            self.universalism_concern.constrained,
            self.universalism_nature.attained,
            self.universalism_nature.constrained,
            self.universalism_tolerance.attained,
            self.universalism_tolerance.constrained
        ]

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValuesWithAttainment:
        """
        Combines the scores of the values with the same prefix and attainment.

        E.g., 'Universalism: concern attained', 'Universalism: nature attained',
        and 'Universalism: tolerance attained' to 'Universalism attained'.

        :param mode: Function to combine the scores (default: max)
        :type mode: Callable[[Iterable[float]], float]
        :return: The combined scores
        :rtype: RefinedCoarseValuesWithAttainment
        """
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
        """
        Combines the scores of the values with the same prefix and attainment,
        and drops Face and Humility.

        E.g., 'Universalism: concern attained', 'Universalism: nature attained',
        and 'Universalism: tolerance attained' to 'Universalism attained'.

        :param mode: Function to combine the scores (default: max)
        :type mode: Callable[[Iterable[float]], float]
        :return: The combined and reduced scores
        :rtype: OriginalValuesWithAttainment
        """
        return self.coarse_values(mode=mode).original_values()

    def without_attainment(self) -> RefinedValues:
        """
        Combines the scores of the values with the same attainment by taking
        their sum.

        E.g., 'Achievement attained' and 'Achievement constrained' to
        'Achievement'.

        :return: The combined scores
        :rtype: RefinedValues
        """
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
        """
        Takes the scores of the values for attained only, dropping for constrained.

        :return: The attained scores
        :rtype: RefinedValues
        """
        return RefinedValues(
            self_direction_action=self.self_direction_action.attained,
            self_direction_thought=self.self_direction_thought.attained,
            stimulation=self.stimulation.attained,
            hedonism=self.hedonism.attained,
            achievement=self.achievement.attained,
            power_dominance=self.power_dominance.attained,
            power_resources=self.power_resources.attained,
            face=self.face.attained,
            security_personal=self.security_personal.attained,
            security_societal=self.security_societal.attained,
            tradition=self.tradition.attained,
            conformity_rules=self.conformity_rules.attained,
            conformity_interpersonal=self.conformity_interpersonal.attained,
            humility=self.humility.attained,
            benevolence_caring=self.benevolence_caring.attained,
            benevolence_dependability=self.benevolence_dependability.attained,
            universalism_concern=self.universalism_concern.attained,
            universalism_nature=self.universalism_nature.attained,
            universalism_tolerance=self.universalism_tolerance.attained,
        )

    def constrained(self) -> RefinedValues:
        """
        Takes the scores of the values for constrained only, dropping for attained.

        :return: The constrained scores
        :rtype: RefinedValues
        """
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

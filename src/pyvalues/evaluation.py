from typing import Generic, Iterable, Self, Tuple, Type
from pydantic import BaseModel
import matplotlib.pyplot as plt

from pyvalues.values import (
    ORIGINAL_VALUES_COLORS,
    REFINED_COARSE_VALUES_COLORS,
    REFINED_VALUES_COLORS,
    VALUES_WITHOUT_ATTAINMENT,
    Score,
)


class ThresholdedDecision(BaseModel):
    threshold: Score
    is_true: bool


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

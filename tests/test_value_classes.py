import unittest

from pydantic import ValidationError

from pyvalues import (
    OriginalValues,
)
from pyvalues.values import OriginalValuesWithAttainment, RefinedValues, RefinedValuesWithAttainment


class TestValueClasses(unittest.TestCase):

    def test_original_values(self):
        values = OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertEqual(0.0, values.self_direction)
        self.assertEqual(0.1, values.stimulation)
        self.assertEqual(0.2, values.hedonism)
        self.assertEqual(0.3, values.achievement)
        self.assertEqual(0.4, values.power)
        self.assertEqual(0.5, values.security)
        self.assertEqual(0.6, values.tradition)
        self.assertEqual(0.7, values.conformity)
        self.assertEqual(0.8, values.benevolence)
        self.assertEqual(0.9, values.universalism)

    def test_original_values_from_labels(self):
        values = OriginalValues.from_labels(["Conformity", "Self-Direction"])
        self.assertEqual(1.0, values.self_direction)
        self.assertEqual(0.0, values.stimulation)
        self.assertEqual(0.0, values.hedonism)
        self.assertEqual(0.0, values.achievement)
        self.assertEqual(0.0, values.power)
        self.assertEqual(0.0, values.security)
        self.assertEqual(0.0, values.tradition)
        self.assertEqual(1.0, values.conformity)
        self.assertEqual(0.0, values.benevolence)
        self.assertEqual(0.0, values.universalism)

    def test_error_on_too_few(self):
        with self.assertRaises(AssertionError):
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    def test_error_on_too_many(self):
        with self.assertRaises(AssertionError):
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def test_error_on_above_one(self):
        with self.assertRaises(ValidationError):
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 0.7, 0.8, 0.9])

    def test_error_on_negative(self):
        with self.assertRaises(ValidationError):
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8, 0.9])

    def test_convert(self):
        values = RefinedValues.from_labels(["Tradition", "Self-Direction: Action"]).convert(OriginalValues)
        self.assertEqual(1.0, values.self_direction)
        self.assertEqual(0.0, values.stimulation)
        self.assertEqual(0.0, values.hedonism)
        self.assertEqual(0.0, values.achievement)
        self.assertEqual(0.0, values.power)
        self.assertEqual(0.0, values.security)
        self.assertEqual(1.0, values.tradition)
        self.assertEqual(0.0, values.conformity)
        self.assertEqual(0.0, values.benevolence)
        self.assertEqual(0.0, values.universalism)

    def test_cap_at_one(self):
        scores = [0.0] * 38
        scores[0] = 0.40001
        scores[1] = 0.60001
        values = RefinedValuesWithAttainment.from_list(scores, cap_at_one=True)
        self.assertAlmostEqual(0.4, values.self_direction_action.attained, places=3)
        self.assertAlmostEqual(0.6, values.self_direction_action.constrained, places=3)

    def test_error_without_cap_at_one(self):
        scores = [0.0] * 38
        scores[0] = 0.40001
        scores[1] = 0.60001
        with self.assertRaises(ValidationError):
            RefinedValuesWithAttainment.from_list(scores)

    def test_error_on_invalid_convert(self):
        with self.assertRaises(ValueError):
            RefinedValues.from_labels(["Tradition", "Self-Direction: Action"]).convert(RefinedValuesWithAttainment)

    def test_binarize_values(self):
        values = OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        binarized = values.binarize(threshold=0.7)
        self.assertEqual(0.0, binarized.self_direction)
        self.assertEqual(0.0, binarized.stimulation)
        self.assertEqual(0.0, binarized.hedonism)
        self.assertEqual(0.0, binarized.achievement)
        self.assertEqual(0.0, binarized.power)
        self.assertEqual(0.0, binarized.security)
        self.assertEqual(0.0, binarized.tradition)
        self.assertEqual(1.0, binarized.conformity)
        self.assertEqual(1.0, binarized.benevolence)
        self.assertEqual(1.0, binarized.universalism)

    def test_binarize_values_with_attainment(self):
        values = OriginalValuesWithAttainment.from_list([
            0.0, 0.60,
            0.1, 0.55,
            0.2, 0.50,
            0.3, 0.35,
            0.4, 0.30,
            0.5, 0.25,
            0.6, 0.20,
            0.7, 0.15,
            0.8, 0.10,
            0.9, 0.05
        ])
        binarized = values.binarize(threshold=0.7)
        self.assertEqual(0.0, binarized.self_direction.total())
        self.assertEqual(0.0, binarized.stimulation.total())
        self.assertEqual(1.0, binarized.hedonism.total())
        self.assertEqual(0.0, binarized.achievement.total())
        self.assertEqual(1.0, binarized.power.total())
        self.assertEqual(1.0, binarized.security.total())
        self.assertEqual(1.0, binarized.tradition.total())
        self.assertEqual(1.0, binarized.conformity.total())
        self.assertEqual(1.0, binarized.benevolence.total())
        self.assertEqual(1.0, binarized.universalism.total())
        self.assertEqual(0.0, binarized.self_direction.attained)
        self.assertEqual(0.0, binarized.stimulation.attained)
        self.assertEqual(0.0, binarized.hedonism.attained)
        self.assertEqual(0.0, binarized.achievement.attained)
        self.assertEqual(1.0, binarized.power.attained)
        self.assertEqual(1.0, binarized.security.attained)
        self.assertEqual(1.0, binarized.tradition.attained)
        self.assertEqual(1.0, binarized.conformity.attained)
        self.assertEqual(1.0, binarized.benevolence.attained)
        self.assertEqual(1.0, binarized.universalism.attained)

    def test_top_values(self):
        values = OriginalValues.from_list([0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5])
        topped = values.top(k=3)
        self.assertEqual(0.0, topped.self_direction)
        self.assertEqual(0.9, topped.stimulation)
        self.assertEqual(0.0, topped.hedonism)
        self.assertEqual(0.8, topped.achievement)
        self.assertEqual(0.0, topped.power)
        self.assertEqual(0.7, topped.security)
        self.assertEqual(0.0, topped.tradition)
        self.assertEqual(0.0, topped.conformity)
        self.assertEqual(0.0, topped.benevolence)
        self.assertEqual(0.0, topped.universalism)

    def test_top_values_binarized(self):
        values = OriginalValues.from_list([0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5])
        topped = values.top(k=3, binarize=True)
        self.assertEqual(0.0, topped.self_direction)
        self.assertEqual(1.0, topped.stimulation)
        self.assertEqual(0.0, topped.hedonism)
        self.assertEqual(1.0, topped.achievement)
        self.assertEqual(0.0, topped.power)
        self.assertEqual(1.0, topped.security)
        self.assertEqual(0.0, topped.tradition)
        self.assertEqual(0.0, topped.conformity)
        self.assertEqual(0.0, topped.benevolence)
        self.assertEqual(0.0, topped.universalism)

    def test_top_values_with_attainment(self):
        values = OriginalValuesWithAttainment.from_list([
            0.0, 0.50,
            0.9, 0.00,
            0.1, 0.55,
            0.8, 0.10,
            0.2, 0.60,
            0.7, 0.10,
            0.3, 0.55,
            0.6, 0.20,
            0.4, 0.60,
            0.5, 0.30
        ])
        topped = values.top(k=3)
        self.assertEqual(0.0, topped.self_direction.total())
        self.assertEqual(0.9, topped.stimulation.total())
        self.assertEqual(0.0, topped.hedonism.total())
        self.assertEqual(0.9, topped.achievement.total())
        self.assertEqual(0.0, topped.power.total())
        self.assertEqual(0.0, topped.security.total())
        self.assertEqual(0.0, topped.tradition.total())
        self.assertEqual(0.0, topped.conformity.total())
        self.assertEqual(1.0, topped.benevolence.total())
        self.assertEqual(0.0, topped.universalism.total())
        self.assertEqual(0.0, topped.self_direction.attained)
        self.assertEqual(0.9, topped.stimulation.attained)
        self.assertEqual(0.0, topped.hedonism.attained)
        self.assertEqual(0.8, topped.achievement.attained)
        self.assertEqual(0.0, topped.power.attained)
        self.assertEqual(0.0, topped.security.attained)
        self.assertEqual(0.0, topped.tradition.attained)
        self.assertEqual(0.0, topped.conformity.attained)
        self.assertEqual(0.4, topped.benevolence.attained)
        self.assertEqual(0.0, topped.universalism.attained)

    def test_top_values_with_attainment_binarized(self):
        values = OriginalValuesWithAttainment.from_list([
            0.0, 0.50,
            0.9, 0.00,
            0.1, 0.55,
            0.8, 0.10,
            0.2, 0.60,
            0.7, 0.10,
            0.3, 0.55,
            0.6, 0.20,
            0.4, 0.60,
            0.5, 0.30
        ])
        topped = values.top(k=3, binarize=True)
        self.assertEqual(0.0, topped.self_direction.total())
        self.assertEqual(1.0, topped.stimulation.total())
        self.assertEqual(0.0, topped.hedonism.total())
        self.assertEqual(1.0, topped.achievement.total())
        self.assertEqual(0.0, topped.power.total())
        self.assertEqual(0.0, topped.security.total())
        self.assertEqual(0.0, topped.tradition.total())
        self.assertEqual(0.0, topped.conformity.total())
        self.assertEqual(1.0, topped.benevolence.total())
        self.assertEqual(0.0, topped.universalism.total())
        self.assertEqual(0.0, topped.self_direction.attained)
        self.assertEqual(1.0, topped.stimulation.attained)
        self.assertEqual(0.0, topped.hedonism.attained)
        self.assertEqual(1.0, topped.achievement.attained)
        self.assertEqual(0.0, topped.power.attained)
        self.assertEqual(0.0, topped.security.attained)
        self.assertEqual(0.0, topped.tradition.attained)
        self.assertEqual(0.0, topped.conformity.attained)
        self.assertEqual(0.0, topped.benevolence.attained)
        self.assertEqual(0.0, topped.universalism.attained)

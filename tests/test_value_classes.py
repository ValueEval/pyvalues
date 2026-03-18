import unittest

from pydantic import ValidationError

from pyvalues import (
    OriginalValues,
)
from pyvalues.values import RefinedValues, RefinedValuesWithAttainment


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

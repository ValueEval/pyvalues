import unittest

from pydantic import ValidationError

from pyvalues import (
    RefinedValues,
    RefinedCoarseValues,
    OriginalValues,
    RefinedValuesWithAttainment,
    RefinedCoarseValuesWithAttainment,
    OriginalValuesWithAttainment
)


class TestEvaluation(unittest.TestCase):

    def test_f(self):
        tested = [
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            OriginalValues.from_list([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ]
        truth = [
            OriginalValues.from_list([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            OriginalValues.from_list([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            OriginalValues.from_list([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        ]
        evaluation = OriginalValues.evaluate_all(tested, truth)
        f, precision, recall = evaluation.f()
        self.assertAlmostEqual(precision["self_direction"], 0)
        self.assertAlmostEqual(precision["stimulation"], 0)
        self.assertAlmostEqual(precision["hedonism"], 0)
        self.assertAlmostEqual(precision["achievement"], 0)
        self.assertAlmostEqual(precision["power"], 0)
        self.assertAlmostEqual(precision["security"], 2/3)
        self.assertAlmostEqual(precision["tradition"], 1)
        self.assertAlmostEqual(precision["conformity"], 1)
        self.assertAlmostEqual(precision["benevolence"], 1)
        self.assertAlmostEqual(precision["universalism"], 1)
        self.assertAlmostEqual(recall["self_direction"], 0)
        self.assertAlmostEqual(recall["stimulation"], 0)
        self.assertAlmostEqual(recall["hedonism"], 0)
        self.assertAlmostEqual(recall["achievement"], 0)
        self.assertAlmostEqual(recall["power"], 0)
        self.assertAlmostEqual(recall["security"], 1)
        self.assertAlmostEqual(recall["tradition"], 1)
        self.assertAlmostEqual(recall["conformity"], 1)
        self.assertAlmostEqual(recall["benevolence"], 1)
        self.assertAlmostEqual(recall["universalism"], 1)
        self.assertAlmostEqual(f["self_direction"], 0)
        self.assertAlmostEqual(f["stimulation"], 0)
        self.assertAlmostEqual(f["hedonism"], 0)
        self.assertAlmostEqual(f["achievement"], 0)
        self.assertAlmostEqual(f["power"], 0)
        self.assertAlmostEqual(f["security"], 0.8)
        self.assertAlmostEqual(f["tradition"], 1)
        self.assertAlmostEqual(f["conformity"], 1)
        self.assertAlmostEqual(f["benevolence"], 1)
        self.assertAlmostEqual(f["universalism"], 1)

    def test_steps(self):
        tested = [
            OriginalValues.from_list([x / 100 + 0.2 for x in range(10)]),
            OriginalValues.from_list([x / 100 + 0.3 for x in range(10)]),
            OriginalValues.from_list([x / 100 + 0.4 for x in range(10)]),
            OriginalValues.from_list([x / 100 + 0.5 for x in range(10)]),
            OriginalValues.from_list([x / 100 + 0.6 for x in range(10)]),
            OriginalValues.from_list([x / 100 + 0.7 for x in range(10)])
        ]
        truth = [
            OriginalValues.from_list([0.0 for _ in range(6)] + [1.0 for _ in range(4)]),
            OriginalValues.from_list([0.0 for _ in range(1)] + [1.0 for _ in range(9)]),
            OriginalValues.from_list([0.0 for _ in range(4)] + [1.0 for _ in range(6)]),
            OriginalValues.from_list([0.0 for _ in range(3)] + [1.0 for _ in range(7)]),
            OriginalValues.from_list([0.0 for _ in range(2)] + [1.0 for _ in range(8)]),
            OriginalValues.from_list([1.0 for _ in range(10)])
        ]
        evaluation = OriginalValues.evaluate_all(tested, truth)
        all_steps = evaluation.precision_recall_steps()

        steps = all_steps["Self-direction"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1.0, steps_x[0])
        self.assertAlmostEqual(1.0, steps_x[1])
        self.assertAlmostEqual(1.0, steps_x[2])
        self.assertAlmostEqual(1.0, steps_x[3])
        self.assertAlmostEqual(1.0, steps_x[4])
        self.assertAlmostEqual(1.0, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(1/2, steps_y[1])
        self.assertAlmostEqual(1/3, steps_y[2])
        self.assertAlmostEqual(1/4, steps_y[3])
        self.assertAlmostEqual(1/5, steps_y[4])
        self.assertAlmostEqual(1/6, steps_y[5])

        steps = all_steps["Stimulation"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1/2, steps_x[0])
        self.assertAlmostEqual(1/2, steps_x[1])
        self.assertAlmostEqual(1/2, steps_x[2])
        self.assertAlmostEqual(1/2, steps_x[3])
        self.assertAlmostEqual(2/2, steps_x[4])
        self.assertAlmostEqual(2/2, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(1/2, steps_y[1])
        self.assertAlmostEqual(1/3, steps_y[2])
        self.assertAlmostEqual(1/4, steps_y[3])
        self.assertAlmostEqual(2/5, steps_y[4])
        self.assertAlmostEqual(2/6, steps_y[5])

        steps = all_steps["Hedonism"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1/3, steps_x[0])
        self.assertAlmostEqual(2/3, steps_x[1])
        self.assertAlmostEqual(2/3, steps_x[2])
        self.assertAlmostEqual(2/3, steps_x[3])
        self.assertAlmostEqual(3/3, steps_x[4])
        self.assertAlmostEqual(3/3, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(2/2, steps_y[1])
        self.assertAlmostEqual(2/3, steps_y[2])
        self.assertAlmostEqual(2/4, steps_y[3])
        self.assertAlmostEqual(3/5, steps_y[4])
        self.assertAlmostEqual(3/6, steps_y[5])

        steps = all_steps["Achievement"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1/4, steps_x[0])
        self.assertAlmostEqual(2/4, steps_x[1])
        self.assertAlmostEqual(3/4, steps_x[2])
        self.assertAlmostEqual(3/4, steps_x[3])
        self.assertAlmostEqual(4/4, steps_x[4])
        self.assertAlmostEqual(4/4, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(2/2, steps_y[1])
        self.assertAlmostEqual(3/3, steps_y[2])
        self.assertAlmostEqual(3/4, steps_y[3])
        self.assertAlmostEqual(4/5, steps_y[4])
        self.assertAlmostEqual(4/6, steps_y[5])

        steps = all_steps["Power"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1/5, steps_x[0])
        self.assertAlmostEqual(2/5, steps_x[1])
        self.assertAlmostEqual(3/5, steps_x[2])
        self.assertAlmostEqual(4/5, steps_x[3])
        self.assertAlmostEqual(5/5, steps_x[4])
        self.assertAlmostEqual(5/5, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(2/2, steps_y[1])
        self.assertAlmostEqual(3/3, steps_y[2])
        self.assertAlmostEqual(4/4, steps_y[3])
        self.assertAlmostEqual(5/5, steps_y[4])
        self.assertAlmostEqual(5/6, steps_y[5])

        steps = all_steps["Security"]
        steps_x = steps[0]
        steps_y = steps[1]
        self.assertAlmostEqual(1/5, steps_x[0])
        self.assertAlmostEqual(2/5, steps_x[1])
        self.assertAlmostEqual(3/5, steps_x[2])
        self.assertAlmostEqual(4/5, steps_x[3])
        self.assertAlmostEqual(5/5, steps_x[4])
        self.assertAlmostEqual(5/5, steps_x[5])
        self.assertAlmostEqual(1/1, steps_y[0])
        self.assertAlmostEqual(2/2, steps_y[1])
        self.assertAlmostEqual(3/3, steps_y[2])
        self.assertAlmostEqual(4/4, steps_y[3])
        self.assertAlmostEqual(5/5, steps_y[4])
        self.assertAlmostEqual(5/6, steps_y[5])

        for value in ["Tradition", "Conformity", "Benevolence", "Universalism"]:
            steps = all_steps[value]
            steps_x = steps[0]
            steps_y = steps[1]
            self.assertAlmostEqual(1/6, steps_x[0])
            self.assertAlmostEqual(2/6, steps_x[1])
            self.assertAlmostEqual(3/6, steps_x[2])
            self.assertAlmostEqual(4/6, steps_x[3])
            self.assertAlmostEqual(5/6, steps_x[4])
            self.assertAlmostEqual(6/6, steps_x[5])
            self.assertAlmostEqual(1/1, steps_y[0])
            self.assertAlmostEqual(2/2, steps_y[1])
            self.assertAlmostEqual(3/3, steps_y[2])
            self.assertAlmostEqual(4/4, steps_y[3])
            self.assertAlmostEqual(5/5, steps_y[4])
            self.assertAlmostEqual(6/6, steps_y[5])

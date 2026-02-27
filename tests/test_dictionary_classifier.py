import unittest

from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.dictionary_classifier import (
    OriginalValuesDictionaryClassifier,
)


class TestDictionaryClassifier(unittest.TestCase):

    def test_original_values_default_classifier(self):
        classifier = OriginalValuesDictionaryClassifier.get_default()

        results = list(classifier.classify_segments_for_original_values([
            "Jobs! Jobs! Jobs! Less party and joy, more work and struggle!"
        ], LanguageAlpha2("en")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 1)
        self.assertEqual(results.hedonism, 1)
        self.assertEqual(results.achievement, 1)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 0)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

        results = list(classifier.classify_segments_for_original_values([
            "Die Behörden befolgen entschlossen die Anweisungen."
        ], LanguageAlpha2("de")))[0]
        self.assertEqual(results.self_direction, 1)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 0)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 1)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

    def test_original_values_default_classifier_threshold(self):
        classifier = OriginalValuesDictionaryClassifier.get_default(
            score_threshold=2.0
        )

        results = list(classifier.classify_segments_for_original_values([
            "Jobs! Jobs! Jobs! Less party and joy, more work and struggle!"
        ], LanguageAlpha2("en")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 1)
        self.assertEqual(results.achievement, 1)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 0)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

        results = list(classifier.classify_segments_for_original_values([
            "Die Behörden befolgen entschlossen die Anweisungen."
        ], LanguageAlpha2("de")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 0)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 1)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

    def test_original_values_default_classifier_threshold_too_high(self):
        classifier = OriginalValuesDictionaryClassifier.get_default(
            score_threshold=10.0
        )

        results = list(classifier.classify_segments_for_original_values([
            "Jobs! Jobs! Jobs! Less party and joy, more work and struggle!"
        ], LanguageAlpha2("en")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 0)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 0)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

        results = list(classifier.classify_segments_for_original_values([
            "Die Behörden befolgen entschlossen die Anweisungen."
        ], LanguageAlpha2("de")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 0)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 0)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

    def test_original_values_default_classifier_top_value(self):
        classifier = OriginalValuesDictionaryClassifier.get_default(
            max_values=1
        )

        results = list(classifier.classify_segments_for_original_values([
            "Jobs! Jobs! Jobs! Less party and joy, more work and struggle!"
        ], LanguageAlpha2("en")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 1)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 0)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

        results = list(classifier.classify_segments_for_original_values([
            "Die Behörden befolgen entschlossen die Anweisungen."
        ], LanguageAlpha2("de")))[0]
        self.assertEqual(results.self_direction, 0)
        self.assertEqual(results.stimulation, 0)
        self.assertEqual(results.hedonism, 0)
        self.assertEqual(results.achievement, 0)
        self.assertEqual(results.power, 0)
        self.assertEqual(results.security, 0)
        self.assertEqual(results.tradition, 0)
        self.assertEqual(results.conformity, 1)
        self.assertEqual(results.benevolence, 0)
        self.assertEqual(results.universalism, 0)

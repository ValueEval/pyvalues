import unittest

from pydantic import ValidationError
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues import (
    OriginalValues,
    OriginalValuesDictionaryClassifier
)


class TestDictionaryClassifier(unittest.TestCase):

    def test_original_values_default_classifier(self):
        classifier = OriginalValuesDictionaryClassifier.get_default()
        results = list(classifier.classify_document_for_original_values([
            "Die Behörden befolgen entschlossen die Anweisungen."
        ], LanguageAlpha2("de")))
        print(results[0])

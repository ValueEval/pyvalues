import unittest
import tempfile

from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues import (
    OriginalValues,
)
from pyvalues.document import Document, ValuesAnnotatedDocument


class TestDocument(unittest.TestCase):

    def test_write_read_annotated(self):
        documents = [
            ValuesAnnotatedDocument[OriginalValues](
                id="doc1",
                language=LanguageAlpha2("en"),
                segments=["foo", "bar"],
                values=[
                    OriginalValues.from_labels(["Achievement"]),
                    OriginalValues.from_labels(["Security", "Hedonism"])
                ]
            ),
            ValuesAnnotatedDocument[OriginalValues](
                id="doc2",
                language=LanguageAlpha2("de"),
                segments=["eins"],
                values=[
                    OriginalValues.from_labels([])
                ]
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_file_name = tmp + "/tmp.tsv"
            with open(tmp_file_name, "w") as tmp_file:
                writer = OriginalValues.writer_tsv_with_text(tmp_file)
                writer.write_documents(documents)
            
            documents_read = list(OriginalValues.read_tsv(
                tmp_file_name,
                document_id_field=Document.ID_FIELD
            ))

            self.assertEqual(2, len(documents_read))

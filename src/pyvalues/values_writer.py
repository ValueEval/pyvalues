import csv
from typing import Generic, Iterable, TextIO, Type
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.document import DEFAULT_LANGUAGE, Document
from pyvalues.values import VALUES, ValuesAnnotatedDocument


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
            default_language: LanguageAlpha2 | str | None = DEFAULT_LANGUAGE
    ):
        self._write_document_id = write_document_id
        self._default_document_id = default_document_id
        self._write_language = write_language
        if default_language is None:
            self._default_language = None
        else:
            self._default_language = LanguageAlpha2(default_language)

        fieldnames = []
        if write_document_id:
            fieldnames += [Document.ID_FIELD]
        fieldnames += [Document.TEXT_FIELD]
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
            language: LanguageAlpha2 | str | None = None
    ):
        line: dict[str, float | str] = {
            value: score for (value, score) in zip(values.names(), values.to_list())
        }
        line[Document.TEXT_FIELD] = segment
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

    def write_document(
            self,
            document: ValuesAnnotatedDocument[VALUES]
    ):
        if document.values is None:
            raise ValueError("Missing values")
        elif document.segments is None:
            raise ValueError("Missing segments")
        else:
            for values, segment in zip(document.values, document.segments):
                self.write(
                    values=values,
                    document_id=document.id,
                    segment=segment,
                    language=document.language
                )

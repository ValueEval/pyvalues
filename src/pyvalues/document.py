import csv
from pathlib import Path
from typing import Callable, ClassVar, Generator, Generic, Iterable
from pydantic import BaseModel
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.values import DEFAULT_LANGUAGE, VALUES


class Document(BaseModel):
    id: str | None = None
    language: LanguageAlpha2 = DEFAULT_LANGUAGE
    segments: list[str] | None = None

    TEXT_FIELD: ClassVar[str] = "Text"
    ID_FIELD: ClassVar[str] = "ID"
    LANGUAGE_FIELD: ClassVar[str] = "Language"

    @staticmethod
    def read_txt(
        input_file: str | Path,
        segmenter: Callable[[str], Iterable[str]] = lambda x: [x],
        document_id: str | None = None,
        language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> "Document":
        """
        Reads a text file.

        By default, the text file is read as one segment,
        but a ``segmenter`` can be provided to split the text content.

        :param input_file:
            The text file to read.
        :type input_file: str | Path

        :param segmenter:
            Segmenter used to split the text into segments.
            Default: use complete text as single segment.
        :type segmenter: Callable[[str], Iterable[str]]

        :param document_id:
            Default document ID to use when no ID is found in the row
            or when ``document_id_field`` is not specified.
        :type document_id: str | None

        :param language:
            Default language (ISO 639-1 / alpha-2) to use when no language is found in the row
            or when ``language_field`` is not specified.
        :type language: LanguageAlpha2 | str
        """
        with open(input_file) as file:
            content = file.read()
            segments = list(segmenter(content))
            return Document(
                id=document_id,
                language=LanguageAlpha2(language),
                segments=segments
            )

    @staticmethod
    def read_tsv(
        input_file: str | Path,
        segmenter: Callable[[str, LanguageAlpha2], Iterable[str]] = lambda x, _: [x],
        document_id: str | None = None,
        language: LanguageAlpha2 | str = DEFAULT_LANGUAGE,
        delimiter: str = "\t",
        document_id_field: str | None = None,
        language_field: str | None = None,
        text_field: str = TEXT_FIELD,
        **kwargs
    ) -> Generator["Document", None, None]:
        """
        Reads a tab-separated file (or one with a different delimiter).

        By default, each row is treated as its own document unless either
        (1) the ``document_id_field`` parameter is set and specifies a column name of the file,
        in which case consecutive rows with the same ID are treated as one document; or
        (2) the ``document_id`` parameter is set,
        in which case the set value is used for rows without ID.

        :param input_file:
            The tab-separated values file to read.
        :type input_file: str | Path

        :param segmenter:
            Segmenter used to split the text into segments.
            Takes a the text to split and the text language as parameters.
            Default: use complete text as single segment.
        :type segmenter: Callable[[str, LanguageAlpha2], Iterable[str]]

        :param document_id:
            Default document ID to use when no ID is found in the row
            or when ``document_id_field`` is not specified.
        :type document_id: str | None

        :param language:
            Default language (ISO 639-1 / alpha-2) to use when no language is found in the row
            or when ``language_field`` is not specified.
        :type language: LanguageAlpha2 | str

        :param delimiter:
            Field delimiter used in the file (defaults to tab).
        :type delimiter: str

        :param document_id_field:
            Name of the column containing document IDs. When provided, consecutive
            rows with the same ID are grouped into a single document;
            Default: None
        :type document_id_field: str | None

        :param language_field:
            Name of the column containing language codes. When provided, the value
            in this column overrides the default ``language`` for the current row
            (and thus the current document);
            Default: None
        :type language_field: str | None

        :param text_field:
            Name of the column containing segment text. Values from
            this column are collected into the ``segments`` attribute of the
            resulting document after running them through the ``segmenter``;
            Default: "Text"
        :type text_field: str

        :param kwargs:
            Additional keyword arguments passed to :class:`csv.DictReader`.

        :return:
            A generator yielding the read documents.
        :rtype: Generator[Document, None, None]
        """
        current_document_id = document_id
        current_language: LanguageAlpha2 = LanguageAlpha2(language)
        segments: list[str] = []
        with open(input_file, newline='') as input_file_handle:
            reader = csv.DictReader(input_file_handle, delimiter=delimiter, **kwargs)
            for row in reader:
                row_document_id = None
                if document_id_field is not None:
                    row_document_id = row.get(document_id_field, document_id)
                if row_document_id is None or row_document_id != current_document_id:
                    if len(segments) > 0:
                        yield Document(
                            id=current_document_id,
                            language=current_language,
                            segments=segments
                        )
                        segments: list[str] = []
                current_document_id = row_document_id
                if language_field is not None:
                    current_language = LanguageAlpha2(row.get(language_field, language))
                row_text = row.get(text_field)
                if row_text is None:
                    raise ValueError(f"Missing segment ({text_field})")
                else:
                    segments += segmenter(row_text, current_language)
            yield Document(
                id=current_document_id,
                language=current_language,
                segments=segments
            )


class ValuesAnnotatedDocument(Document, Generic[VALUES]):
    values: list[VALUES]

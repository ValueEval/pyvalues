import json
from typing import Generator, Generic, Iterable, Type
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pydantic_extra_types.language_code import LanguageAlpha2

from pyvalues.document import Document
from .classifier import (
    OriginalValuesClassifier,
    OriginalValuesWithAttainmentClassifier,
    RefinedCoarseValuesClassifier,
    RefinedCoarseValuesWithAttainmentClassifier,
    RefinedValuesClassifier,
    RefinedValuesWithAttainmentClassifier,
)
from .values import (
    DEFAULT_LANGUAGE,
    VALUES,
    OriginalValues,
    OriginalValuesWithAttainment,
    RefinedCoarseValues,
    RefinedCoarseValuesWithAttainment,
    RefinedValues,
    RefinedValuesWithAttainment,
)
from . import __version__

DEFAULT_METHOD: str = "POST"

DEFAULT_TIMEOUT_SECONDS: float = 30.0


class RemoteClassifier(Generic[VALUES]):
    """
    Abstract base class for a classifier that is deployed somewhere else and
    called through an API.
    """
    _response_class: Type[VALUES]
    _url: str
    _method: str
    _headers: dict[str, str]
    _timeout_seconds: float

    def __init__(
            self,
            response_class: Type[VALUES],
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        self._response_class = response_class
        self._url = url
        self._method = method
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"pyvalues/{__version__} (+https://github.com/ValueEval/pyvalues)"
        }
        if authorization_token is not None:
            self._headers["Authorization"] = "Bearer " + authorization_token
        self._timeout_seconds = timeout_seconds

    def _classify_document(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 = DEFAULT_LANGUAGE
    ) -> Generator[VALUES, None, None]:
        document = Document(language=language, segments=list(segments))
        request_data = document.model_dump_json().encode("utf-8")
        request = Request(
            self._url,
            data=request_data,
            headers=self._headers,
            method=self._method
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_data = response.read().decode("utf-8")
        except HTTPError as e:
            print("Status:", e.code)
            print("Headers:", e.headers)
            print("Body:")
            print(e.read().decode())
            return
        response_document = json.loads(response_data)
        for values in response_document["values"]:
            yield self._response_class.model_validate(values)


class OriginalValuesRemoteClassifier(RemoteClassifier[OriginalValues], OriginalValuesClassifier):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=OriginalValues,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_original_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValues, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )


class RefinedCoarseValuesRemoteClassifier(
    RemoteClassifier[RefinedCoarseValues], RefinedCoarseValuesClassifier
):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=RefinedCoarseValues,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_refined_coarse_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValues, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )


class RefinedValuesRemoteClassifier(
    RemoteClassifier[RefinedValues], RefinedValuesClassifier
):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=RefinedValues,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_refined_values(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedValues, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )


class OriginalValuesWithAttainmentRemoteClassifier(
    RemoteClassifier[OriginalValuesWithAttainment],
    OriginalValuesWithAttainmentClassifier
):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=OriginalValuesWithAttainment,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_original_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[OriginalValuesWithAttainment, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )


class RefinedCoarseValuesWithAttainmentRemoteClassifier(
    RemoteClassifier[RefinedCoarseValuesWithAttainment],
    RefinedCoarseValuesWithAttainmentClassifier
):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=RefinedCoarseValuesWithAttainment,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_refined_coarse_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedCoarseValuesWithAttainment, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )


class RefinedValuesWithAttainmentRemoteClassifier(
    RemoteClassifier[RefinedValuesWithAttainment],
    RefinedValuesWithAttainmentClassifier
):
    """
    Classifier that is deployed somewhere else and called through an API.
    """

    def __init__(
            self,
            url: str,
            method: str = DEFAULT_METHOD,
            authorization_token: str | None = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ):
        """
        Creates a classifier that just calls a remote classification API.

        :param url:
            The URL of the API to call (including protocol and port).
        :type url: str

        :param method:
            The HTTP method of the API, usually "POST" (default) or "GET".
        :type method: str

        :param authorization_token:
            A token to authorize for the API, if needed. The token will be
            placed in the "Authorization" header of the API request as
            "Bearer [token]".
        :type authorization_token: str | None

        :param timeout_seconds:
            Number of seconds until the call is aborted.
        :type timeout_seconds: float
        """
        super().__init__(
            response_class=RefinedValuesWithAttainment,
            url=url,
            method=method,
            authorization_token=authorization_token,
            timeout_seconds=timeout_seconds
        )

    def classify_segments_for_refined_values_with_attainment(
            self,
            segments: Iterable[str],
            language: LanguageAlpha2 | str = DEFAULT_LANGUAGE
    ) -> Generator[RefinedValuesWithAttainment, None, None]:
        return self._classify_document(
            segments,
            LanguageAlpha2(language)
        )

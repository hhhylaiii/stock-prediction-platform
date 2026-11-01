"""Errors a source can raise once it has answered (SPEC §5 failure handling)."""


class SourceError(Exception):
    """A source answered, but not with usable data."""


class SourceNotReadyError(SourceError):
    """The source has not published data for a trading day yet; retry later."""


class SchemaChangedError(SourceError):
    """The source response no longer matches the expected layout; raw data is kept."""


class SourceRequestError(SourceError):
    """The source refused the request (a 4xx other than 429); retrying will not help."""


class RejectedStocksError(Exception):
    """A per-stock run finished, or stopped, with stocks whose source data was unusable."""

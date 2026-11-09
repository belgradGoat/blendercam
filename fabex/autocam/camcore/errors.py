"""Errors raised by camcore, with the API's error codes (the glue turns them into
FabexAPIError)."""

import difflib
from dataclasses import fields


class CamcoreError(ValueError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def check_keys(cls, data: dict, what: str) -> None:
    """Refuse keys that are not fields of the dataclass, listing close matches."""
    if not isinstance(data, dict):
        raise CamcoreError("INVALID_VALUE", f"{what} must be a dict, not {type(data).__name__}")
    names = [f.name for f in fields(cls)]
    unknown = [k for k in data if k not in names]
    if unknown:
        raise CamcoreError(
            "INVALID_PROPERTY",
            f"{what} has no field(s) {unknown}",
            {
                "unknown": unknown,
                "close_matches": {k: difflib.get_close_matches(k, names, 3, 0.5) for k in unknown},
                "fields": names,
            },
        )

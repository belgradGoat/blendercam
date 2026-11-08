"""Fabex 'exception.py'

Generic CAM Exception class, plus structured errors for the scripting / agent API.
"""


class CamException(Exception):
    pass


class FabexAPIError(CamException):
    """An error with a stable machine-readable code, raised by the Fabex scripting API.

    Agents and scripts should branch on `code`, not on the message text.
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class BrepError(FabexAPIError):
    """An error related to B-rep (BRepKit) sources."""

"""Application-domain exceptions safe to translate into API responses."""

from collections.abc import Sequence


class DomainError(RuntimeError):
    """Base class for expected GrowthCrew domain failures."""


class ResourceNotFoundError(DomainError):
    """Requested resource does not exist."""


class ResourceConflictError(DomainError):
    """Requested operation conflicts with current resource state."""


class StaleResourceError(ResourceConflictError):
    """Client attempted to update an outdated resource version."""


class ApprovalRequiredError(ResourceConflictError):
    """A stage cannot be generated until the stages it builds on are approved.

    Carries the target stage and the prerequisite stages still awaiting
    approval as plain stage-name strings, so callers (the full-strategy
    orchestrator in particular) can report exactly what is blocking
    without parsing the message. This module deliberately stays free of
    schema imports.
    """

    def __init__(self, message: str, *, stage: str, unapproved: Sequence[str]) -> None:
        super().__init__(message)
        self.stage = stage
        self.unapproved: tuple[str, ...] = tuple(unapproved)

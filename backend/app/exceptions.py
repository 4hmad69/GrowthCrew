"""Application-domain exceptions safe to translate into API responses."""


class DomainError(RuntimeError):
    """Base class for expected GrowthCrew domain failures."""


class ResourceNotFoundError(DomainError):
    """Requested resource does not exist."""


class ResourceConflictError(DomainError):
    """Requested operation conflicts with current resource state."""


class StaleResourceError(ResourceConflictError):
    """Client attempted to update an outdated resource version."""

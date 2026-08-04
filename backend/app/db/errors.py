"""Database-specific application exceptions."""


class DatabaseError(RuntimeError):
    """Base class for controlled database failures."""


class DatabaseNotConfiguredError(DatabaseError):
    """Raised when a database-backed operation has no configured database."""


class DatabaseUnavailableError(DatabaseError):
    """Raised when PostgreSQL cannot be reached."""


class DatabaseNotReadyError(DatabaseError):
    """Raised when PostgreSQL is reachable but required schema features are missing."""
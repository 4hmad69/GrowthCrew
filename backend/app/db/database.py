"""SQLAlchemy engine and session lifecycle."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import Settings
from backend.app.db.errors import DatabaseNotConfiguredError


class Database:
    """Own the SQLAlchemy engine and create short-lived ORM sessions."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session_factory = sessionmaker(
            bind=engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        """Create a database runtime from validated application settings."""

        return cls(create_database_engine(settings))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session, rolling back failures and always closing it."""

        session = self._session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Close all pooled database connections owned by this process."""

        self.engine.dispose()


def create_database_engine(settings: Settings) -> Engine:
    """Build the PostgreSQL engine without opening a connection immediately."""

    database_url = settings.database_url_value
    if database_url is None:
        raise DatabaseNotConfiguredError("Database configuration is missing.")

    url = make_url(database_url)
    if url.get_backend_name() != "postgresql" or url.get_driver_name() != "psycopg":
        raise DatabaseNotConfiguredError(
            "GROWTHCREW_DATABASE_URL must use postgresql+psycopg."
        )

    return create_engine(
        url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_recycle=settings.database_pool_recycle_seconds,
        connect_args={
            "connect_timeout": settings.database_connect_timeout_seconds,
            "application_name": "growthcrew-api",
        },
    )
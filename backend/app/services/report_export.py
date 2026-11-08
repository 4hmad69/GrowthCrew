"""Turn a workspace's strategy report into a downloadable file.

Read-only, like the report it exports: it builds the report from the
stage records, their approvals and their history, and writes it out in the
requested format. It never calls an LLM, retrieval or web search, so it
keeps working when the model provider is down.

A file with nothing in it is not a useful download, so a workspace where no
stage has been generated is refused with a conflict. The report itself (as
JSON) is still available then, and says which stages are missing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.exceptions import ResourceConflictError
from backend.app.reports.delivery import MEDIA_TYPES, render_report, report_filename
from backend.app.schemas.report import ReportFormat
from backend.app.services.report_content import ReportContentService


@dataclass(frozen=True)
class ReportFile:
    """A report written out in one format, ready to send."""

    filename: str
    media_type: str
    content: bytes


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ReportExportService:
    """Build a workspace's report and write it out as a file."""

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._content = ReportContentService(session, clock=clock)

    def export(self, workspace_id: UUID, report_format: ReportFormat) -> ReportFile:
        """The workspace's report as a file.

        Raises ResourceNotFoundError if the workspace does not exist, and
        ResourceConflictError if no stage has been generated yet.
        """

        report = self._content.build(workspace_id)
        if not report.sections:
            raise ResourceConflictError(
                "No stage has been generated yet, so there is nothing to export."
            )
        return ReportFile(
            filename=report_filename(report, report_format),
            media_type=MEDIA_TYPES[report_format],
            content=render_report(report, report_format),
        )

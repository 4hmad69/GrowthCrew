"""Report REST API routes.

A report is the whole strategy chain written out: every stage's current
generation, whether it is approved, and whether the inputs it was built
from have since changed. These routes only read. One returns the report as
JSON, for a screen to show; the other returns it as a file to download
(PDF, Word or CSV).

Like the approval and history routes they never call an LLM, retrieval, or
web search, so they depend on the database session alone and keep working
when the model provider is down. Nothing is stored: a report is built from
the stage records every time, so it can never disagree with them.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db_session
from backend.app.reports.delivery import MEDIA_TYPES, content_disposition
from backend.app.schemas.report import ReportDocument, ReportFormat
from backend.app.services.report_content import ReportContentService
from backend.app.services.report_export import ReportExportService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/report",
    tags=["reports"],
)

DbSession = Annotated[Session, Depends(get_db_session)]

# A report is derived from records that can change at any moment, so a copy
# kept by a browser or a proxy could silently be out of date.
NO_STORE = "no-store"


@router.get(
    "",
    response_model=ReportDocument,
    status_code=status.HTTP_200_OK,
    summary="Read the strategy report as data",
    responses={404: {"description": "Workspace not found."}},
)
def get_report(
    workspace_id: UUID,
    session: DbSession,
    response: Response,
) -> ReportDocument:
    """Return the report: each stage's current generation and how far to trust it.

    Every stage appears exactly once, in generation order, either as a
    section or in omitted with the reason it is missing. A section carries
    its content, whether its current version is approved, and what it was
    built from; flags lists every reason it should not be presented as
    settled (draft, stale, unknown provenance). A workspace with nothing
    generated is a valid report in which every stage is omitted.
    """

    response.headers["Cache-Control"] = NO_STORE
    return ReportContentService(session).build(workspace_id)


@router.get(
    "/{report_format}",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    summary="Download the strategy report as a file",
    responses={
        200: {
            "description": "The report as a file, sent as an attachment.",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in MEDIA_TYPES.values()
            },
        },
        404: {"description": "Workspace not found."},
        409: {"description": "No stage has been generated yet, so there is nothing to export."},
    },
)
def download_report(
    workspace_id: UUID,
    report_format: ReportFormat,
    session: DbSession,
) -> Response:
    """Return the report as a PDF, Word document or CSV file.

    The file says on its face which stages are drafts, out of date, of
    unknown provenance, or not included, so a stage that has not earned
    trust is never presented as if it had.
    """

    report_file = ReportExportService(session).export(workspace_id, report_format)
    return Response(
        content=report_file.content,
        media_type=report_file.media_type,
        headers={
            "Content-Disposition": content_disposition(report_file.filename),
            "Cache-Control": NO_STORE,
            "X-Content-Type-Options": "nosniff",
        },
    )

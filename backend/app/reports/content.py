"""The order a stage's content is written in, shared by every exporter."""

from pydantic import JsonValue

from backend.app.schemas.report import ReportStageSection
from backend.app.schemas.stage_history import STAGE_CONTENT_FIELDS


def ordered_content(section: ReportStageSection) -> list[tuple[str, JsonValue]]:
    """The stage's fields in their documented order, then any others sorted.

    Stored JSON does not keep the order a stage declares its fields in, so
    the known fields are put back in that order. Fields a stage has since
    gained or lost still appear, after the known ones, in a stable order.
    """

    known = STAGE_CONTENT_FIELDS[section.stage]
    ordered = [(name, section.content[name]) for name in known if name in section.content]
    ordered.extend(
        (name, section.content[name]) for name in sorted(set(section.content) - set(known))
    )
    return ordered

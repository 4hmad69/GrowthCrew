# Step 19 — Report Generation (PDF, Word, CSV)

## Purpose

The seven-stage chain, its approvals and its history all existed, but the
only way to get the result out was to read seven JSON responses. Step 19
writes the strategy chain out as one report a person can read, hand on, or
open in a spreadsheet: as data for a screen, and as a PDF, a Word document
or a CSV file.

The point of the step is honesty, not formatting. A report that looks
finished is persuasive, so every stage in it says which generation it is,
whether that version has been approved, and whether the inputs it was built
from have since changed. A stage that has not earned trust is never
presented as if it had.

Branch: `feature/report-generation`. Backend only; the download buttons are
Step 20.

## Scoping decisions

- **A report shows the current generation, labelled. It does not show "the
  last approved generation".** This changes what the roadmap sketched, and
  the reason is in the data model. An approval is one row per stage, pinned
  to a record id and version. Regenerating changed content retires it, and
  once it is retired no route returns it (the approval route answers 404).
  Approving again overwrites the row, so nothing records which past
  generations were ever approved. Showing "the last approved one" would
  rest on a row that is hidden when it matters most and lost when it is
  replaced. Showing the current generation and saying plainly that it is a
  draft needs no new state and cannot be wrong.
- **A section names exactly one generation, or the stage is left out.** A
  stage's section is built from its newest history snapshot, and only when
  that snapshot still describes the live record by both `id` and `version`
  (the rule `strategy-status` already uses). Otherwise the stage is omitted
  and the reason is stated: `not_generated`, or `history_out_of_sync` when
  the record was changed outside history. Content that cannot be tied to a
  generation is never reported under a guessed one.
- **Approval is matched the way the rest of the system matches it.** A
  section is approved when the live record's current approval matches its
  `record_id` and `approved_version`. A forced regeneration of identical
  content leaves the version, and so the approval, in place; the newest of
  those generations is the one reported.
- **Unknown provenance is flagged, never called fresh and never called
  stale.** A stage built before history, or one whose inputs could not be
  identified, carries the flag `unknown_provenance`. It is not clean even if
  approved.
- **Three flags and one summary.** A section is `clean` only when it has no
  flag: not `draft` (not approved), not `stale` (built from inputs that have
  since changed), not `unknown_provenance`. A report is clean only when it
  is complete and every section is clean.
- **Whole-workspace report only.** No per-stage or partial export. The
  status line and every table say what is missing.
- **The caller cannot ask for an older generation.** The roadmap suggested
  reading "a specific generation through the history API". The report
  names the generation it shows, but selecting a past one is not offered:
  restoring and comparing generations were deliberately left out of Step 18
  and this step did not reopen them.
- **A report is derived, never stored.** It is built from the stage
  records, their approvals and their history on every request, so it cannot
  disagree with them. No table, no migration, no cache.
- **Read-only, and no model provider.** The routes depend on the database
  session alone, like the approval and history routes, so a report can be
  read while the LLM endpoint is down. A test makes the LLM gateway throw
  and checks that both routes still answer.
- **No gating changes.** A report does not approve, regenerate, revoke or
  block anything. Staleness stays report-only.
- **A file export with nothing to export is a 409.** When no stage has been
  generated, the JSON report still answers 200 (every stage listed as
  omitted), but a PDF, Word or CSV of nothing is refused with
  "No stage has been generated yet, so there is nothing to export."
- **Text is written as it was stored.** Each format keeps stored text
  exactly, except for what the format itself forces (see each format below).
- **Generated text is untrusted.** It comes from a model and from web
  pages, so each exporter treats it as data: spreadsheet formulas are
  neutralised in CSV, markup is shown literally in PDF and Word, and no
  generated text reaches an HTTP header.

## Runtime architecture

```text
GET /workspaces/{id}/report                -> ReportContentService.build()
GET /workspaces/{id}/report/{pdf|docx|csv} -> ReportExportService.export()
                                                |
                                                v
                                    ReportContentService.build()
        WorkspaceRepository, BusinessProfileRepository (title),
        StrategyRecordLoader (live records + current approvals),
        StageSnapshotRepository (newest snapshot per stage),
        StageStalenessService (staleness and provenance per stage)
                                                |
                                                v
                           ReportDocument (schemas/report.py)
                                                |
                                                v
        reports/delivery.py  ->  render_pdf | render_docx | render_csv
                                                |
                                                v
                  bytes + filename + media type + safe headers
```

- `ReportContentService` is the only place that decides what a report
  contains. The three exporters only write a finished `ReportDocument`
  out: they read nothing and decide nothing about trust.
- The reads happen in one request but not in one database snapshot. If a
  stage is regenerated while a report is being built, its record and its
  snapshot can disagree, and that shows up as `history_out_of_sync` for
  that stage, never as one generation's content labelled as another's.
- Shared pieces: `reports/labels.py` (every word a written report uses, so
  Word and PDF cannot drift apart), `reports/text.py` (which characters a
  document can carry, where a paragraph ends, what an empty value looks
  like, how a list item is named), `reports/content.py` (the order a
  stage's fields are written in).

## What was built

- `schemas/report.py`: `ReportFormat`, `ReportStageFlag`, `OmissionReason`,
  `OmittedStage`, `ReportStageSection` (with computed `flags` and
  `is_clean`), and `ReportDocument` (with computed `is_complete`,
  `is_clean` and `flagged_stages`). A document must account for every stage
  exactly once, in generation order, as a section or as omitted.
- `services/report_content.py`: `ReportContentService`, described above.
- `services/report_export.py`: `ReportExportService` and `ReportFile`; the
  409 for an empty workspace.
- `reports/csv_export.py`, `reports/docx_export.py`, `reports/pdf_export.py`:
  one `render_*` function each.
- `reports/delivery.py`: renderer dispatch, media types, filenames, and the
  `Content-Disposition` header.
- `api/reports.py`: the two routes, registered in `api/router.py`.
- Dependencies: `python-docx==1.2.0` and `reportlab==5.0.1` in
  `requirements.txt`; `pypdf==6.21.1` in `requirements-dev.txt`, used only
  by tests to read PDFs back.

## The API

```text
GET /api/v1/workspaces/{workspace_id}/report
GET /api/v1/workspaces/{workspace_id}/report/{report_format}   pdf | docx | csv
```

- The JSON report: `workspace_id`, `business_name`, `generated_at`,
  `sections`, `omitted`, and the computed `is_complete`, `is_clean` and
  `flagged_stages`. A section carries `stage`, `generation`, `record_id`,
  `record_version`, `approval` (`approved` or `draft`), `staleness`,
  `model_used`, `generated_at`, `content`, and the computed `flags` and
  `is_clean`. The title is the business profile's name, falling back to the
  workspace name.
- Errors: unknown workspace 404; malformed id or any format other than the
  three lowercase names 422 (`PDF` is rejected); file export with nothing
  generated 409; database failures are sanitised as everywhere else.
- Both routes send `Cache-Control: no-store`, because a report is derived
  from records that can change at any moment. Files also send
  `X-Content-Type-Options: nosniff` and
  `Content-Disposition: attachment; filename="..."`.
- Filenames are `<business-name>-strategy-report-<UTC date>.<ext>`, reduced
  to lowercase ASCII letters, digits and hyphens: accents are folded
  (`Café Müller & Söhne <GmbH>` becomes `cafe-muller-sohne-gmbh`), the stem
  is cut at 60 characters, and a name with nothing to keep (for example one
  written entirely in Japanese) falls back to `growthcrew`. The header value
  is sanitised a second time so it can never break out of its quotes.

## The three formats

**CSV** is a long table, one row per value, with columns `stage,
generation, record_version, approval, flags, path, value`. Every row carries
its stage's trust state, so a row is never separated from its flags when
the file is filtered or pasted. `path` locates a value (`personas[0].name`);
a stage that is not in the report gets one row flagged
`omitted:<reason>`. Fields come back in their documented order even though
stored JSON does not keep it. UTF-8 with a byte-order mark (Excel misreads
UTF-8 without one) and CRLF line endings. Text beginning with `=`, `+`,
`-`, `@`, a tab or a carriage return gets a leading `'` so a spreadsheet
reads it as text; numbers are left alone, but the *string* `"-5"` is
quoted. `null` and empty text are both a blank cell.

**Word (.docx)** is A4 with a cover (business name, report title, when it
was generated, and a shaded one-line status), a "Stages at a glance" table
with a repeating header row, then one Word heading per stage with a shaded
notice under any flagged stage, and a heading plus the reason for any stage
that is not in the report. Content is written generically: text becomes
paragraphs (a run of blank lines is one paragraph break, a single newline is
a line break), a list of plain values becomes bullets, a list of objects
becomes one labelled block per item headed by its `name` or `title`, and an
object becomes labelled lines. The footer has live page-number fields.
Line endings are normalised and characters XML cannot carry (control
characters and unpaired surrogates) are dropped, because they would
otherwise make the file unreadable.

**PDF** has the same layout and wording, plus a bookmark for every stage and
"Page N of M" in the footer. It is reproducible byte for byte and dated to
the report's own generation time, not the moment of export. It embeds the
Bitstream Vera fonts that ship inside reportlab, so it looks the same on
every machine. Vera covers English and most Western European text and the
usual punctuation, but only part of Latin Extended and no Greek, Cyrillic,
Vietnamese, emoji, arrows, rupee sign, Arabic or CJK. A character the font
cannot show is never printed as a blank or a box: a run of them becomes one
`?`, and the stage (or the cover) says how many places were affected and
that the Word and CSV exports keep the original text. A PDF re-flows text,
so runs of spaces collapse and leading spaces are not kept.

## Real bugs and findings caught while building this

- **The shared base schema trims whitespace from every string, including
  strings nested in stage content.** A report built on it would have quietly
  altered what a generation said. The first integration test compared the
  report with the history API's own copy, which had the same trimming, so it
  passed. `ReportStageSection` now switches trimming off, and the test
  compares with the database row. Fixed in this step (commit 3).
- **The same trimming exists in Step 18's merged history API, and is not
  fixed.** Checked directly: content stored as `'  padded\n\n'` is returned
  by `GET .../history/{stage}/{generation}` as `'padded'`, although that
  route is documented as returning content exactly as the generation
  produced it. See Known gaps; it belongs on its own `fix/...` branch.
- **A recreated stage record at the same version could be matched to the
  old history** if only `version` were compared. A deliberate break of the
  resolver survived until a test was added for a record deleted and
  recreated behind history's back.
- **PDF bookmarks all pointed at page 1.** The footer needs the page total,
  so pages are written in a second pass, and a bookmark made during layout
  did not yet know its page. Found by reading the outline back with an
  independent reader; the outline is now written during the second pass.
- **Non-embedded Helvetica was being selected.** reportlab's `Table`
  defaults to Helvetica 10 pt. Nothing was drawn in it, but a test that
  checks every font selected for drawing is an embedded subset caught it,
  and each table style now names the embedded font.
- **Word's own schema rejected two things python-docx produced.** Run
  through the XSD validator: a shaded paragraph's border and shading
  elements must come before its spacing and indent, and the default
  template's zoom setting lacks a required attribute. Both fixed and
  covered by tests.
- **Tables and notices overhung the text edge by 6 pt** because reportlab's
  default frame has padding; the frame is now padding-free and a test pins
  the alignment.
- **Wording that claimed too much.** The first PDF font note said Vera
  covers "Latin text"; probing the font showed it lacks `ń`, Greek and
  Cyrillic, and the note was corrected to say so.
- **Weak tests found by deliberate breaks.** Two "at most the maximum
  length" assertions compared against the constant itself and could never
  fail; a test of the report title compared the output with the same
  constant. All three now pin literals.

## Testing

427 new tests, 387 of them unit and 40 against real PostgreSQL:

| File | Tests |
|---|---|
| `test_report_schemas.py` | 34 |
| `integration/test_report_content_integration.py` | 22 |
| `test_report_text.py` | 68 |
| `test_report_labels.py` | 30 |
| `test_report_csv_export.py` | 36 |
| `test_report_docx_export.py` | 60 |
| `test_report_pdf_export.py` | 77 |
| `test_report_delivery.py` | 48 |
| `test_report_api.py` | 34 |
| `integration/test_report_api_integration.py` | 18 |

- The exporters are tested by writing a report and reading the file back
  with an independent library (python-docx, pypdf, the `csv` module), not by
  inspecting the exporter's internals.
- Service behaviour is tested on real PostgreSQL, per the established
  convention: a fresh approved chain, drafts, a stage forced upstream,
  an edited profile, a content change, a record changed behind history's
  back, a deleted stage, a recreated record, unknown provenance, and a
  business name full of special characters.
- The HTTP contract tests stub the services and the session: paths, the
  OpenAPI schema, status codes, headers, the 405 on every write method,
  error sanitisation, and that the routes depend on the database session
  alone.

## Verification performed

- Claude's sandbox (PostgreSQL 16 with pgvector): `ruff` clean; **1332
  passed, 23 skipped** with the integration gate on and **952 passed** with
  it off, up from 905 and 565 at the start of the step. The baseline was
  taken on the pushed `main` (`863ad79`) before any work and matched the
  Step 18 figures exactly.
- About 260 deliberate breaks across the step, each caught by a test, run
  only from a verified-green baseline. About 16 survived their first pass;
  each led to a new test, a deleted dead branch, or a re-targeted
  mutation. Runs made while the sandbox database had stopped were
  discarded and redone.
- **Independent validation of the files.** The Word output passes the
  OOXML schema validator and was rendered to images and inspected;
  `qpdf --check` reports no errors for the PDF, `pdffonts` shows only
  embedded subset fonts with Unicode maps, and the PDF was rendered and
  inspected. A report built from a genuinely generated chain (real agents on
  the local stubs, real PostgreSQL) was exported in all three formats and
  checked.
- **Confirmed on Ahmad's machine:** Ahmad reported, after the last code
  commit, that all tests pass on his machine on
  `feature/report-generation`. He did not report counts, so none are
  recorded here. That branch contains every Step 18 commit, so the
  full-run item carried over from Step 18 is covered by the same report
  (see the Step 18 document).
- **Not verified:** the Word file was not opened in Microsoft Word itself,
  nor the PDF in Adobe Reader; both were checked with libraries, a schema
  validator, LibreOffice rendering and poppler tools. Right-to-left text is
  carried through in the Word file but its layout is left-to-right and was
  not checked. Rendering is synchronous and was not load-tested: a report
  with 300 long items renders, but no timing was measured.

## Known gaps, deliberately out of scope for this step

- **Only the current generation can be reported.** There is no way to ask for
  an earlier generation, and no "last approved" view; see the scoping
  decision above.
- **No partial or per-stage report.**
- **The history detail route trims whitespace (Step 18 code, open).**
  `GET .../history/{stage}/{generation}` returns nested strings with
  surrounding whitespace removed. The fix is small (the detail schema should
  stop inheriting the whitespace-stripping config for its `content` field)
  but it is a bug in merged code, so it needs its own `fix/...` branch and a
  regression test against the API.
- **PDF font coverage is limited** (see above). Wider coverage needs a font
  file added to the repository.
- **CSV cannot tell `null` from empty text.** Both are blank.
- **Model output is shown literally.** Markdown in generated text appears as
  typed (`**bold**` stays `**bold**`) in Word and PDF.
- **Rendering is synchronous and in memory.** A very large report is built
  inside the request; revisit with the background execution deferred to
  Step 20.
- **No download buttons or report view** in the frontend (Step 20).
- Inherited and untouched: no approver identity until the app has users;
  two simultaneous full runs for one workspace are still not serialised; the
  pre-existing `alembic check` drift on `knowledge_chunks.chunk_metadata`.
- Inherited and untouched: `README.md` still says the repository "currently
  implements Step 3".
- Carried over from Step 17 and untouched: the Marketing Strategy
  real-embeddings test failed in two different ways on Ahmad's machine and is
  judged real-model nondeterminism but unproven.

## What Step 20 can rely on

- `GET .../report` is the single source for any report screen: every stage
  appears once, as a section or as omitted with a reason, and `flags`,
  `is_clean`, `is_complete` and `flagged_stages` are computed server-side, so
  the frontend need not re-derive trust.
- `GET .../report/{pdf|docx|csv}` is a plain attachment download. It answers
  409 for an empty workspace, so a download button should be enabled only
  when `sections` is not empty.
- Neither route needs the model provider or changes any state, so they are
  safe to call on every screen load, and `no-store` means a refetch always
  shows the current state.
- Whether to add a "regenerate stale" mode is still a Step 20 decision.
  A report only describes staleness; it does not act on it.

## Definition of done

- [x] Report schemas: every stage accounted for exactly once, in generation
      order; computed flags and summary; unknown provenance never clean
- [x] `ReportContentService` on real PostgreSQL: current generation per
      stage matched to the live record by id and version, approval matched
      as elsewhere, staleness and provenance carried through, and
      `history_out_of_sync` / `not_generated` omissions with reasons
- [x] CSV, Word and PDF exporters, each tested by reading the file back with
      an independent reader; Word passes the OOXML schema validator and the
      PDF passes `qpdf --check` with only embedded fonts
- [x] Text fidelity: stored text reaches CSV and Word exactly (a regression
      for the whitespace trimming found in this step); the PDF states what
      it could not show
- [x] Spreadsheet-formula neutralisation in CSV; markup shown literally in
      PDF and Word; filenames and headers built from safe characters only
- [x] Read-only report routes: JSON and file download, documented in
      OpenAPI, `no-store`, 404/409/422 and sanitised failures, database
      session as the only dependency, working with the LLM gateway broken
- [x] `ruff` clean; 1332 passing, 23 skipped with the integration gate on
      (952 passing with it off), in Claude's sandbox
- [x] Full test run on Ahmad's machine on this branch: reported as passing;
      no counts were reported, so none are recorded
- [x] No real-provider test was added or needed: the report path makes no
      LLM, retrieval or web-search call, and tests enforce that
- [ ] `GET .../history/{stage}/{generation}` still trims whitespace from
      stored content (found in this step, Step 18 code); needs its own
      `fix/...` branch and an API regression test
- [ ] Word file opened in Microsoft Word and PDF opened in Adobe Reader:
      checked with libraries, a schema validator, LibreOffice and poppler
      only
- [x] `test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings`
      (Marketing Strategy, real embeddings), carried over from Step 17 and
      untouched by this step: failed in two different ways, judged
      real-model nondeterminism, not the gate, but unproven; revisit if it
      recurs
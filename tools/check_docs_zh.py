#!/usr/bin/env python3
"""Check the Simplified Chinese translations under docs/zh_CN against docs/.

Each English page docs/FOO.md must have a counterpart docs/zh_CN/FOO.md that keeps
the structure of the original: same headings, fenced code blocks, table rows, links
and numbers, with the prose translated. The checker also verifies that in-page
anchors resolve (translated headings keep an English <a id="...">) and that links to
sibling docs point at files that exist next to the translation.

Usage:
    python tools/check_docs_zh.py                  # every page that has a translation
    python tools/check_docs_zh.py --only DETAILS   # one page
    python tools/check_docs_zh.py --all            # also list pages with no translation
    python tools/check_docs_zh.py --no-english     # skip the leftover-English check
    python tools/check_docs_zh.py --report out.txt # also write the report to a file

Exit code 1 when any page fails.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
ZH = DOCS / "zh_CN"

# Lines whose English text is expected to stay: UI strings, log lines and code are
# handled structurally (fenced blocks, indented blocks and inline-code-heavy lines).
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019-]{1,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
LINK_RE = re.compile(r"\]\(")
TARGET_RE = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
TABLE_RE = re.compile(r"^\s*\|")
ANCHOR_DEF_RE = re.compile(r"<a\s+(?:id|name)\s*=\s*[\"']([^\"']+)[\"']")
ANCHOR_LINK_RE = re.compile(r"\]\(#([^)\s]+)\)")
DOC_LINK_RE = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s#]+\.md)(?:#([^)\s]+))?\)")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
HTML_TAG_RE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$")


@dataclass
class Metrics:
    headings: int = 0
    fences: int = 0
    table_rows: int = 0
    list_items: int = 0
    links: int = 0
    numbers: int = 0
    inline_code: int = 0
    english_lines: list[int] = field(default_factory=list)


@dataclass
class Result:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    en: Metrics | None = None
    zh: Metrics | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub("", line)


def countable(text: str) -> str:
    """Drop HTML tags and link targets so only prose numbers are counted.

    A translated heading keeps its English slug in an <a id="..."> line, which would
    otherwise count the heading's numbers twice.
    """
    text = re.sub(r"<[^>\n]*>", " ", text)  # never across lines: a stray "<" would eat the file
    text = re.sub(r"\]\([^)\s]*\)", "](L)", text)
    return text


def scan(text: str, *, collect_english: bool, verbatim: frozenset[str] = frozenset()) -> Metrics:
    """Walk the markdown once: count structure, collect English-only prose lines.

    `verbatim` holds the English page's stripped lines: a translation line equal to
    one of them is intentionally kept as-is (data rows, program strings), not left
    over untranslated prose.
    """
    m = Metrics()
    text = countable(text)
    in_fence = False
    fence_marker = ""
    prev_blank = True

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            if not in_fence:
                in_fence = True
                m.fences += 1
            elif line.strip().startswith(fence_marker or marker):
                in_fence = False
            continue
        if in_fence:
            continue

        # indented code block (4 spaces or a tab after a blank line) is code too
        indented = raw.startswith("    ") or raw.startswith("\t")
        if indented and (prev_blank or indented):
            prev_blank = line.strip() == ""
            continue
        prev_blank = line.strip() == ""

        if not line.strip():
            continue

        m.links += len(LINK_RE.findall(line))
        m.numbers += len(NUMBER_RE.findall(line))
        m.inline_code += len(INLINE_CODE_RE.findall(line))

        heading = HEADING_RE.match(line)
        if heading:
            m.headings += 1
        if TABLE_RE.match(line):
            m.table_rows += 1
        if LIST_RE.match(line):
            m.list_items += 1

        if not collect_english:
            continue
        if raw.strip() in verbatim and (
            # a data row kept verbatim, a quoted model output, or a bullet/continuation
            # whose text is a verbatim program string, is kept on purpose; copied
            # prose is not
            TABLE_RE.match(raw) or HTML_TAG_RE.match(raw) or line.lstrip().startswith(">")
            or re.match(r"^\s*(?:[-*+]\s+)?`", raw)
        ):
            continue
        if not re.sub(r"<[^>]*>", "", line).strip():  # pure HTML markup (e.g. an anchor line)
            continue
        if line.lstrip().startswith("|") and line.count("|") > 0:
            cells = [strip_inline_code(c) for c in line.strip().strip("|").split("|")]
            prose = " ".join(cells)
        else:
            prose = strip_inline_code(line)
        if CJK_RE.search(prose):
            continue
        # paths, URLs and identifiers are not prose (a line of only links is not English either)
        prose = re.sub(r"\]\([^)\s]*\)", "](x)", prose)
        prose = re.sub(r"\S*/\S*", " ", prose)
        words = WORD_RE.findall(prose)
        # A line that is mostly inline code or a bare URL is not prose.
        if len(words) < 5:
            continue
        if len(prose.strip()) and len(words) * 3 < len(prose.strip()) * 0.25:
            continue
        if re.match(r"^\s*[-*+]\s+`", line) and words and len(words) <= 8 and line.count("`") >= 2:
            continue
        m.english_lines.append(lineno)

    return m


def slugify(text: str) -> str:
    """GitHub heading anchor slug, for headings that stay in English."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]+", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def compare(name: str, en_text: str, zh_text: str, *, english_check: bool, ignore: set[str], max_number_drift: float) -> Result:
    res = Result(name=name)
    en = scan(en_text, collect_english=False)
    zh = scan(zh_text, collect_english=english_check,
              verbatim=frozenset(l.strip() for l in countable(en_text).splitlines()))
    res.en, res.zh = en, zh

    for label, a, b in (
        ("headings", en.headings, zh.headings),
        ("code blocks", en.fences, zh.fences),
        ("table rows", en.table_rows, zh.table_rows),
        ("links", en.links, zh.links),
        ("list items", en.list_items, zh.list_items),
    ):
        if a != b:
            res.errors.append(f"{label}: {b} in translation, {a} in English")

    drift = abs(en.numbers - zh.numbers)
    allowed = max(5, int(en.numbers * max_number_drift))
    if drift > allowed:
        res.errors.append(f"numbers: {zh.numbers} in translation, {en.numbers} in English")

    if zh_text.startswith("\ufeff"):
        res.errors.append("file starts with a UTF-8 BOM")
    if "\r\n" in zh_text:
        res.warnings.append("CRLF line endings (repo markdown is LF)")

    # anchors: every in-page link needs an <a id=...> or a matching English heading
    defined = set(ANCHOR_DEF_RE.findall(zh_text))
    english_headings = {slugify(h) for h in (HEADING_RE.match(l).group(2) for l in en_text.splitlines() if HEADING_RE.match(l))}
    for target in sorted(set(ANCHOR_LINK_RE.findall(zh_text))):
        if target in defined:
            continue
        if target in english_headings or target in set(ANCHOR_DEF_RE.findall(en_text)):
            res.errors.append(f"in-page anchor #{target} has no <a id=\"{target}\"></a>")
        else:
            res.warnings.append(f"in-page anchor #{target} resolves nowhere (English page has the same problem)")

    # Document links. A translation keeps the English page's links in the same order
    # (the link counts above must match), so pair them by position and compare what
    # each one resolves to. docs/zh_CN/X.md is the translation of docs/X.md, so a link
    # to a docs page becomes the bare sibling name, and a link out of docs/ gains one
    # more "../". Links the English page already breaks stay warnings.
    def repo_of(base: Path, target: str) -> Path:
        return (base / target).resolve()

    en_order = [t.split("#")[0] for t in TARGET_RE.findall(en_text) if t.split("#")[0]]
    zh_order = [t.split("#")[0] for t in TARGET_RE.findall(zh_text) if t.split("#")[0]]

    def anchors_in(path: Path) -> set[str]:
        text = read(path)
        found = set(ANCHOR_DEF_RE.findall(text))
        found |= {slugify(HEADING_RE.match(l).group(2)) for l in text.splitlines() if HEADING_RE.match(l)}
        return found

    paired = list(zip(zh_order, en_order)) if len(zh_order) == len(en_order) else []
    if not paired and en_order:
        res.warnings.append("link order cannot be paired with the English page")

    for raw in TARGET_RE.findall(zh_text):
        target, _, anchor = raw.partition("#")
        if not target or target.startswith(("http", "/", "mailto:")):
            continue
        link_name = Path(target).name  # not `name`: that is this page's file name (see .check-ignore)
        is_docs_page = target.endswith(".md") and link_name != "README.md" and (DOCS / link_name).exists()
        if is_docs_page:
            if target != link_name:
                res.errors.append(f"link {target} should be the sibling translation {link_name}")
                continue
            resolved = ZH / link_name
            if not resolved.exists():
                res.warnings.append(f"link {link_name} is not translated yet")
                continue
        else:
            resolved = repo_of(ZH, target)
            if not resolved.exists():
                if (ZH / ("../" + target)).exists():
                    res.errors.append(f"link {target} needs one more ../ (the translation is one level deeper)")
                    continue
                english = next((e for z, e in paired if z == target), None)
                if english is None or not repo_of(DOCS, english).exists():
                    res.warnings.append(f"link target {target} is missing in the English page too")
                else:
                    res.errors.append(f"link target {target} does not exist")
                continue
        if not anchor or not target.endswith(".md"):
            continue
        if anchor in anchors_in(resolved):
            continue
        english_source = DOCS / Path(target).name
        if english_source.exists() and anchor in anchors_in(english_source):
            res.errors.append(f"link {target}#{anchor} is not defined in the target page")
        else:
            res.warnings.append(f"link {target}#{anchor} resolves nowhere (English page has the same problem)")

    for lineno in zh.english_lines:
        if f"{name}:{lineno}" in ignore:
            continue
        res.errors.append(f"line {lineno} still looks like English prose")

    return res


def fix_links(zh_text: str) -> tuple[str, int]:
    """Repair the two link depth mistakes the flat docs/ layout invites.

    A link to a translated docs page becomes a bare sibling name; a link out of docs/
    gains the missing "../". README.md is never rewritten: docs/zh_CN/README.md is the
    index, not the repository README.
    """
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        raw = match.group(1)
        target, _, anchor = raw.partition("#")
        if not target or target.startswith(("http", "/", "mailto:")):
            return match.group(0)
        suffix = "#" + anchor if anchor else ""
        name = Path(target).name
        if target.endswith(".md") and (DOCS / name).exists() and name != "README.md":
            if target != name:
                changed += 1
            return "](" + name + suffix + ")"
        if (ZH / target).exists():
            return match.group(0)
        if (ZH / ("../" + target)).exists():
            changed += 1
            return "](../" + target + suffix + ")"
        return match.group(0)

    return TARGET_RE.sub(repl, zh_text), changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], help="substring of the page name (repeatable)")
    parser.add_argument("--all", action="store_true", help="also list English pages without a translation")
    parser.add_argument("--no-english", action="store_true", help="skip the leftover-English prose check")
    parser.add_argument("--report", help="also write the report to this path")
    parser.add_argument("--number-drift", type=float, default=0.02, help="allowed number-count drift (default 0.02)")
    parser.add_argument("--fix-links", action="store_true", help='rewrite "](../x)" to "](../../x)" where only the deeper path exists')
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ignore: set[str] = set()
    ignore_file = ZH / ".check-ignore"
    if ignore_file.exists():
        for line in read(ignore_file).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore.add(line.split()[0])

    pages = sorted(p for p in DOCS.glob("*.md"))
    if args.only:
        pages = [p for p in pages if any(o.lower() in p.name.lower() for o in args.only)]

    results: list[Result] = []
    missing: list[str] = []
    fixed = 0
    for page in pages:
        target = ZH / page.name
        if not target.exists():
            missing.append(page.name)
            continue
        zh_text = read(target)
        if args.fix_links:
            zh_text, n = fix_links(zh_text)
            if n:
                target.write_text(zh_text, encoding="utf-8", newline="\n")
                fixed += n
        results.append(
            compare(
                page.name,
                read(page),
                zh_text,
                english_check=not args.no_english,
                ignore=ignore,
                max_number_drift=args.number_drift,
            )
        )
    if args.fix_links:
        print(f"fixed {fixed} link(s)")

    lines: list[str] = []
    failed = 0
    for res in results:
        status = "ok" if res.ok else "FAIL"
        if not res.ok:
            failed += 1
        zh, en = res.zh, res.en
        assert zh and en
        lines.append(
            f"[{status}] {res.name}  headings {zh.headings}/{en.headings}"
            f"  code {zh.fences}/{en.fences}"
            f"  table {zh.table_rows}/{en.table_rows}"
            f"  links {zh.links}/{en.links}"
            f"  list {zh.list_items}/{en.list_items}"
            f"  numbers {zh.numbers}/{en.numbers}"
        )
        for err in res.errors:
            lines.append(f"         - {err}")
        for warn in res.warnings:
            lines.append(f"         ~ {warn}")

    if missing:
        lines.append("")
        lines.append(f"no translation yet ({len(missing)}): " + ", ".join(missing) if args.all else "")

    translated = len(results)
    lines.append("")
    lines.append(
        f"{translated} translated page(s), {failed} failing, "
        f"{translated - failed} passing, {len(missing)} page(s) without a translation "
        f"out of {len(list(DOCS.glob('*.md')))} English pages"
    )

    report = "\n".join(l for l in lines if l != "" or True)
    print(report)
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

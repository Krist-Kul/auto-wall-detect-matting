#!/usr/bin/env python3
"""Markdown -> .docx converter tuned for this repo's docs.

Handles: ATX headings, fenced code blocks (monospace, shaded, no wrapping),
pipe tables (header row shaded + bold), bullet/numbered lists incl. nesting,
blockquotes, horizontal rules, and inline `code` / **bold** / *italic* /
[link](url).
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"
CODE_SHADE = "F2F2F2"
HEAD_SHADE = "E8E8E8"
LINK_COLOR = RGBColor(0x0B, 0x4F, 0x9E)
CODE_COLOR = RGBColor(0xB0, 0x30, 0x60)


# ---------------------------------------------------------------- xml helpers
def shade(element, fill):
    """Apply a solid background fill to a paragraph or table cell."""
    pr = element.get_or_add_pPr() if element.tag.endswith("}p") else element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def add_hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0B4F9E")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(color)
    rPr.append(u)
    run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


# ------------------------------------------------------------ inline markdown
TOKEN = re.compile(
    r"(`[^`]+`)"                       # code span
    r"|(\*\*\*.+?\*\*\*)"              # bold italic
    r"|(\*\*.+?\*\*)"                  # bold
    r"|(\*[^*\n]+?\*)"                 # italic
    r"|(\[[^\]]+\]\([^)]+\))"          # link
)


def add_inline(paragraph, text, base_bold=False, mono=False):
    """Render inline markdown into `paragraph`."""
    for part in (p for p in TOKEN.split(text) if p):
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            r = paragraph.add_run(part[1:-1])
            r.font.name = MONO_FONT
            r.font.size = Pt(9.5)
            r.font.color.rgb = CODE_COLOR
            r.bold = base_bold
            continue
        if part.startswith("***") and part.endswith("***"):
            r = paragraph.add_run(part[3:-3]); r.bold = True; r.italic = True
        elif part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1]); r.italic = True; r.bold = base_bold
        elif part.startswith("[") and "](" in part:
            label, url = part[1:-1].split("](", 1)
            if url.startswith(("http://", "https://")):
                add_hyperlink(paragraph, url, label)
                continue
            r = paragraph.add_run(label)          # relative repo path: no link
            r.italic = True
            r.bold = base_bold
        else:
            r = paragraph.add_run(part); r.bold = base_bold
        if mono:
            r.font.name = MONO_FONT


def plain(text):
    """Strip inline markdown to bare text (for table width estimation)."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"[`*]", "", text)


# ----------------------------------------------------------------- block bits
def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_divider(line):
    return bool(re.fullmatch(r"\|?[\s:|-]+\|[\s:|-]*", line.strip())) and "-" in line


def add_code_block(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.left_indent = Inches(0.2)
        pf.line_spacing = 1.0
        r = p.add_run(line if line.strip() else "")
        r.font.name = MONO_FONT
        r.font.size = Pt(8.5)
        shade(p._p, CODE_SHADE)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = True
    for j, cell_text in enumerate(header):
        cell = t.cell(0, j)
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        add_inline(p, cell_text, base_bold=True)
        shade(cell._tc, HEAD_SHADE)
    for i, row in enumerate(body, start=1):
        for j in range(len(header)):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            add_inline(p, row[j] if j < len(row) else "")
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    if r.font.size is None:
                        r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ---------------------------------------------------------------- main render
def convert(md_path, out_path, title=None, subtitle=None):
    text = Path(md_path).read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 18), ("Heading 2", 14), ("Heading 3", 12),
                       ("Heading 4", 11)):
        st = doc.styles[name]
        st.font.name = BODY_FONT
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

    if title:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(26)
        if subtitle:
            p2 = doc.add_paragraph()
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r2 = p2.add_run(subtitle)
            r2.italic = True
            r2.font.size = Pt(12)
            r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        doc.add_paragraph()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, buf)
            continue

        # blank
        if not stripped:
            i += 1
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            bdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "BBBBBB")
            bdr.append(bottom)
            pPr.append(bdr)
            i += 1
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and is_divider(lines[i + 1]):
            rows = [split_row(stripped)]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            p = doc.add_paragraph(style=f"Heading {min(level, 4)}")
            p.paragraph_format.space_before = Pt(14 if level <= 2 else 10)
            p.paragraph_format.space_after = Pt(4)
            add_inline(p, m.group(2), base_bold=True)
            i += 1
            continue

        # blockquote (may span lines)
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Inches(0.3)
            pf.space_before = Pt(4)
            pf.space_after = Pt(8)
            pPr = p._p.get_or_add_pPr()
            bdr = OxmlElement("w:pBdr")
            left = OxmlElement("w:left")
            left.set(qn("w:val"), "single")
            left.set(qn("w:sz"), "18")
            left.set(qn("w:space"), "8")
            left.set(qn("w:color"), "888888")
            bdr.append(left)
            pPr.append(bdr)
            add_inline(p, " ".join(x.strip() for x in buf if x.strip()))
            i += 1 if not buf else 0
            continue

        # list item (bullet or ordered), with continuation lines
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if m:
            indent, marker, content = m.group(1), m.group(2), m.group(3)
            depth = min(len(indent) // 2, 2)
            ordered = marker[0].isdigit()
            i += 1
            # soft-wrapped continuation lines belong to this item
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    break
                if re.match(r"^\s*([-*+]|\d+\.)\s+", nxt) or nxt.strip().startswith(("#", "|", "```", ">")):
                    break
                if len(nxt) - len(nxt.lstrip()) <= len(indent) and not nxt.startswith(" "):
                    break
                content += " " + nxt.strip()
                i += 1
            style = "List Number" if ordered else "List Bullet"
            if depth:
                style = f"{style} {depth + 1}"
            try:
                p = doc.add_paragraph(style=style)
            except KeyError:
                p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
            if ordered:
                add_inline(p, content)
            else:
                add_inline(p, content)
            continue

        # plain paragraph (join soft-wrapped lines)
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i]
            s = nxt.strip()
            if not s or s.startswith(("#", "|", "```", ">", "---")):
                break
            if re.match(r"^\s*([-*+]|\d+\.)\s+", nxt):
                break
            buf.append(s)
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        add_inline(p, " ".join(buf))

    doc.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2],
            title=sys.argv[3] if len(sys.argv) > 3 else None,
            subtitle=sys.argv[4] if len(sys.argv) > 4 else None)

# tools/

Document generators. Source of truth for the prose is `docs/SUMMARY.md`; the Word file
is a conversion of it, the PDF is a separate visual redesign of the same material.

## summary.docx

    pip install python-docx
    python tools/md2docx.py docs/SUMMARY.md summary.docx \
        "auto-wall-detect-matting" "Project summary — a technical walkthrough"

`md2docx.py` converts the markdown directly: headings, fenced code blocks, pipe tables,
lists, blockquotes and inline formatting.

## summary.pdf

    pip install weasyprint
    python -c "import weasyprint; weasyprint.HTML(filename='tools/summary_pdf.html').write_pdf('summary.pdf')"

`summary_pdf.html` is a hand-written, print-styled A4 document — cover with contents,
colour-coded stage flow, model cards, callouts and comparison panels. It is **not**
generated from the markdown, so content changes must be made in both places.

Preview a page while editing:

    python -c "import pypdfium2 as p; p.PdfDocument('summary.pdf')[0].render(scale=1.4).to_pil().save('/tmp/pg1.png')"

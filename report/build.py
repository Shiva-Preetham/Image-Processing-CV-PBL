"""
Builds the project report: report/MalariaVision_Project_Report.docx and .pdf

    python report/build.py

1. node build_report.js writes the Word file from the results in outputs/.
2. LibreOffice converts it to PDF.
3. The PDF is searched for every heading and figure caption, and their page
   numbers are written to pages.json; steps 1-2 are repeated until the
   contents and the list of figures show the right page numbers.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
DOCX = HERE / "MalariaVision_Project_Report.docx"
PDF = DOCX.with_suffix(".pdf")
PAGES = HERE / "pages.json"
SOFFICE = shutil.which("soffice") or r"C:\Program Files\LibreOffice\program\soffice.exe"


def roman(n):
    out = ""
    for value, sym in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def build_pdf():
    subprocess.run(["node", "build_report.js"], cwd=HERE, check=True)
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(DOCX)],
                   check=True, capture_output=True)


def find_pages():
    entries = json.loads((HERE / "toc_entries.json").read_text())
    doc = pymupdf.open(PDF)
    pages = [[" ".join(l.split()) for l in pg.get_text().splitlines() if l.strip()] for pg in doc]
    lof = next(i for i, lines in enumerate(pages) if "LIST OF FIGURES" in lines)
    start = next(i for i in range(lof + 1, len(pages)) if "1. INTRODUCTION" in pages[i])

    def locate(prefix):
        key = prefix[:45]
        for i in range(start, len(pages)):
            if any(l.startswith(key) or (len(l) > 15 and key.startswith(l)) for l in pages[i]):
                return i - start + 1
        return None

    found = {"__lof__": roman(lof)}
    missing = []
    for text in entries["headings"] + entries["figures"]:
        pageno = locate(text)
        if pageno is None:
            missing.append(text)
        else:
            found[text] = pageno
    return found, missing, len(doc)


def main():
    previous = None
    for attempt in range(1, 5):
        build_pdf()
        found, missing, total = find_pages()
        if missing:
            print("Not found in PDF:", missing)
        doc = pymupdf.open(PDF)
        blank = [i + 1 for i, pg in enumerate(doc)
                 if i > 0 and len([l for l in pg.get_text().splitlines() if l.strip()]) <= 1
                 and not pg.get_images()]
        if blank:
            print("Warning: blank pages", blank)
        print(f"pass {attempt}: {total} pages")
        if found == previous:
            break
        PAGES.write_text(json.dumps(found, indent=2))
        previous = found
    else:
        sys.exit("Page numbers did not settle")
    print(f"Done: {DOCX.name} and {PDF.name} ({total} pages)")


if __name__ == "__main__":
    main()

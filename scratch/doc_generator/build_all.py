"""
Master Builder for ABHERA Complete Project Guide & Viva Preparation
Generates:
1. D:\\Abhera(Mini)\\documentation\\ABHERA_Complete_Project_Guide_and_Viva.md
2. D:\\Abhera(Mini)\\documentation\\ABHERA_Complete_Project_Guide_and_Viva.pdf
"""

import os
import sys
from pathlib import Path
import re
import pymupdf
import markdown_it

# Add current directory to sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from ch01_intro import get_chapter_01
from ch02_tech import get_chapter_02
from ch03_arch import get_chapter_03
from ch04_folder import get_chapter_04
from ch05_frontend import get_chapter_05
from ch06_backend import get_chapter_06
from ch07_database import get_chapter_07
from ch08_ml import get_chapter_08
from ch09_ner import get_chapter_09
from ch10_workflow import get_chapter_10
from ch11_legal import get_chapter_11
from ch12_testing import get_chapter_12
from ch13_deployment import get_chapter_13
from ch14_viva import get_chapter_14
from ch15_presentation import get_chapter_15
from ch16_glossary import get_chapter_16

PROJECT_ROOT = Path("D:/Abhera(Mini)")
DOCS_DIR = PROJECT_ROOT / "documentation"
MD_OUTPUT_PATH = DOCS_DIR / "ABHERA_Complete_Project_Guide_and_Viva.md"
PDF_OUTPUT_PATH = DOCS_DIR / "ABHERA_Complete_Project_Guide_and_Viva.pdf"

def assemble_markdown():
    print("[1/5] Assembling master markdown document...")
    
    header = """# ABHERA: Complete Project Documentation & University Viva Preparation Guide
## Adaptive BERT-based Harassment Entity Recognition & Assistance
### Comprehensive System Manual, Engineering Architecture, Codebase Walkthrough, and 170-Question Defense Guide

---

### Document Overview & Academic Metadata
- **Project Name**: ABHERA (Adaptive BERT-based Harassment Entity Recognition & Assistance)
- **Domain**: Artificial Intelligence, Natural Language Processing, Machine Learning, Women's Safety & Legal Informatics
- **Jurisdictional Scope**: Republic of India (Bharatiya Nyaya Sanhita 2023, IT Act 2000, POSH Act 2013, PWDVA 2005)
- **Author / Candidate**: Department of Computer Science & Engineering
- **Version**: 2.0 (Audited & Production-Promoted Release)
- **Champion ML Architecture**: Fine-Tuned `bert-base-uncased` with Positive-Weight BCE Loss and Per-Class Optimal Thresholds
- **Token Classification Architecture**: Fine-Tuned `bert-base-uncased` Token Classifier with 13-Tag BIO Ontology & Sub-Word Offset Alignment
- **Knowledge Retrieval Engine**: Deterministic Relational Knowledge Base (PostgreSQL 16) with 32 Curated Legal Provisions & 12 Emergency Services
- **Automated Validation**: 154 Passing Automated Unit Tests across 11 Backend Modules
- **Document Status**: Complete Comprehensive Master Edition (Chapters 1 through 16 + Appendix)

---

## Master Table of Contents
1. [CHAPTER 1: Introduction to ABHERA](#chapter-1-introduction-to-abhera)
2. [CHAPTER 2: Complete Technology Explanation](#chapter-2-complete-technology-explanation)
3. [CHAPTER 3: Complete Project Architecture](#chapter-3-complete-project-architecture)
4. [CHAPTER 4: Complete Folder Structure](#chapter-4-complete-folder-structure)
5. [CHAPTER 5: Frontend Development](#chapter-5-frontend-development)
6. [CHAPTER 6: Backend Development](#chapter-6-backend-development)
7. [CHAPTER 7: Database Design](#chapter-7-database-design)
8. [CHAPTER 8: Dataset and Machine Learning](#chapter-8-dataset-and-machine-learning)
9. [CHAPTER 9: Named Entity Recognition (NER)](#chapter-9-named-entity-recognition-ner)
10. [CHAPTER 10: Complete ABHERA Workflow](#chapter-10-complete-abhera-workflow)
11. [CHAPTER 11: Legal Guidance and Report Generation](#chapter-11-legal-guidance-and-report-generation)
12. [CHAPTER 12: Testing and Evaluation](#chapter-12-testing-and-evaluation)
13. [CHAPTER 13: Deployment](#chapter-13-deployment)
14. [CHAPTER 14: Complete Viva Preparation (170 Master Q&As)](#chapter-14-complete-viva-preparation-master-qa-bank)
    - [Section A: Basic Project Questions (Q1 - Q15)](#section-a-basic-project-questions-q1--q15)
    - [Section B: Programming Fundamentals (Q16 - Q27)](#section-b-programming-fundamentals-q16--q27)
    - [Section C: Frontend Questions (Q28 - Q39)](#section-c-frontend-questions-q28--q39)
    - [Section D: Backend and API Questions (Q40 - Q54)](#section-d-backend-and-api-questions-q40--q54)
    - [Section E: Database and SQL Questions (Q55 - Q66)](#section-e-database-and-sql-questions-q55--q66)
    - [Section F: Machine Learning Fundamentals (Q67 - Q81)](#section-f-machine-learning-fundamentals-q67--q81)
    - [Section G: NLP and BERT Questions (Q82 - Q96)](#section-g-nlp-and-bert-questions-q82--q96)
    - [Section H: NER Questions (Q97 - Q108)](#section-h-ner-questions-q97--q108)
    - [Section I: Project Architecture Questions (Q109 - Q120)](#section-i-project-architecture-questions-q109--q120)
    - [Section J: Testing Questions (Q121 - Q132)](#section-j-testing-questions-q121--q132)
    - [Section K: Deployment Questions (Q133 - Q144)](#section-k-deployment-questions-q133--q144)
    - [Section L: Limitations and Future Scope (Q145 - Q154)](#section-l-limitations-and-future-scope-q145--q154)
    - [Section M: Difficult External Examiner Questions (Q155 - Q170)](#section-m-difficult-external-examiner-questions-q155--q170)
15. [CHAPTER 15: Interview and Viva Presentation Guide](#chapter-15-interview-and-viva-presentation-guide)
16. [CHAPTER 16: Comprehensive Glossary & Quick Revision Sheet](#chapter-16-comprehensive-glossary--quick-revision-sheet)

---
"""

    chapters = [
        header,
        get_chapter_01(),
        get_chapter_02(),
        get_chapter_03(),
        get_chapter_04(),
        get_chapter_05(),
        get_chapter_06(),
        get_chapter_07(),
        get_chapter_08(),
        get_chapter_09(),
        get_chapter_10(),
        get_chapter_11(),
        get_chapter_12(),
        get_chapter_13(),
        get_chapter_14(),
        get_chapter_15(),
        get_chapter_16(),
    ]
    
    full_markdown = "\n\n".join(chapters)
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        
    print(f" -> Master Markdown written successfully: {MD_OUTPUT_PATH} ({len(full_markdown):,} bytes, {full_markdown.count(chr(10)):,} lines)")
    return full_markdown


def convert_markdown_to_html(md_text):
    print("[2/5] Converting Markdown to formatted semantic HTML...")
    
    md_parser = markdown_it.MarkdownIt("commonmark").enable("table").enable("strikethrough")
    
    # Pre-process raw mermaid blocks and ASCII diagrams to clean <pre> blocks
    processed_md = re.sub(r'```mermaid(.*?)```', r'```\1```', md_text, flags=re.DOTALL)
    
    raw_html = md_parser.render(processed_md)
    
    # Embed comprehensive executive academic CSS styling
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4 portrait;
    margin: 20mm 15mm 20mm 15mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.5;
    color: #1e293b;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    color: #1e3a8a;
    font-size: 18pt;
    font-weight: bold;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}
  h2 {{
    color: #0f172a;
    font-size: 13pt;
    font-weight: bold;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  h3 {{
    color: #0369a1;
    font-size: 11pt;
    font-weight: bold;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  h4 {{
    color: #334155;
    font-size: 10pt;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
  }}
  li {{
    margin-bottom: 4px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    margin-bottom: 12px;
    font-size: 8pt;
    page-break-inside: auto;
  }}
  tr {{
    page-break-inside: avoid;
    page-break-after: auto;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 5px 7px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: bold;
  }}
  tbody tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  code {{
    font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 8pt;
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 1px 3px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
  }}
  pre {{
    font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.35;
    background-color: #0f172a;
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 4px;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-break: break-all;
    margin-top: 6px;
    margin-bottom: 10px;
    page-break-inside: avoid;
  }}
  pre code {{
    background-color: transparent;
    color: inherit;
    padding: 0;
    border: none;
    font-size: inherit;
  }}
  blockquote {{
    border-left: 4px solid #2563eb;
    background-color: #eff6ff;
    color: #1e3a8a;
    margin: 8px 0;
    padding: 6px 12px;
    border-radius: 0 4px 4px 0;
    font-style: normal;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
  }}
  strong {{
    color: #0f172a;
  }}
</style>
</head>
<body>
{raw_html}
</body>
</html>
"""
    return styled_html


def compile_pdf_story(html_content):
    print("[3/5] Rendering multi-page document with PyMuPDF Story...")
    
    temp_pdf_path = DOCS_DIR / "temp_raw_story.pdf"
    writer = pymupdf.DocumentWriter(str(temp_pdf_path))
    story = pymupdf.Story(html_content)
    
    def rectfn(rect_num, filled):
        mediabox = pymupdf.Rect(0, 0, 595, 842) # A4 Portrait
        rect = pymupdf.Rect(40, 48, 555, 794)    # Printable bounding area
        return mediabox, rect, pymupdf.Matrix(1, 1)

    story.write(writer, rectfn)
    writer.close()
    
    print(f" -> Raw PDF story generated: {temp_pdf_path}")
    return temp_pdf_path


def post_process_pdf(temp_pdf_path):
    print("[4/5] Post-processing PDF: headers, footers, page numbering, metadata, and bookmarks...")
    
    doc = pymupdf.open(str(temp_pdf_path))
    total_pages = len(doc)
    print(f" -> Document contains {total_pages} pages.")
    
    # 1. Identify headings and build PDF table of contents / bookmarks
    toc = []
    chapter_pnos = {}
    
    for pno in range(total_pages):
        page = doc[pno]
        text = page.get_text()
        lines = text.split("\n")
        
        for line in lines:
            line_str = line.strip()
            # Match Chapter titles: "CHAPTER 1: Introduction to ABHERA"
            if re.match(r"^CHAPTER\s+\d+:", line_str, re.IGNORECASE):
                chap_match = re.match(r"^(CHAPTER\s+\d+:[^.]+)", line_str, re.IGNORECASE)
                title = chap_match.group(1).strip() if chap_match else line_str
                toc.append([1, title, pno + 1])
                chap_key = title.split(":")[0].strip().upper()
                if chap_key not in chapter_pnos:
                    chapter_pnos[chap_key] = pno + 1
            # Match major sections: "## 1.1 Meaning and Definition..." or "SECTION A: ..."
            elif re.match(r"^\d+\.\d+\s+[A-Za-z]", line_str):
                sec_title = line_str[:60].strip()
                toc.append([2, sec_title, pno + 1])
            elif re.match(r"^SECTION\s+[A-M]:", line_str, re.IGNORECASE):
                sec_match = re.match(r"^(SECTION\s+[A-M]:[^.]+)", line_str, re.IGNORECASE)
                title = sec_match.group(1).strip() if sec_match else line_str
                toc.append([2, title, pno + 1])
                
        # 2. Draw running headers and footers on every page
        header_text = "ABHERA: Complete Project Guide & University Viva Defense Manual"
        footer_text = f"Page {pno + 1} of {total_pages}  |  Department of Computer Science & Engineering  |  Confidential & Academic Reference"
        
        # Don't draw header on page 1 if it's the title cover
        if pno > 0:
            # Header line and text
            page.draw_line(pymupdf.Point(40, 40), pymupdf.Point(555, 40), color=(0.8, 0.85, 0.9), width=0.6)
            page.insert_text(
                pymupdf.Point(40, 34),
                header_text,
                fontsize=7.5,
                color=(0.3, 0.4, 0.55),
                fontname="helv",
            )
            
        # Footer line and text
        page.draw_line(pymupdf.Point(40, 804), pymupdf.Point(555, 804), color=(0.8, 0.85, 0.9), width=0.6)
        page.insert_text(
            pymupdf.Point(40, 816),
            footer_text,
            fontsize=7,
            color=(0.4, 0.45, 0.55),
            fontname="helv",
        )
        
    print(f" -> Extracted {len(toc)} bookmark outline entries.")
    if toc:
        doc.set_toc(toc)
        
    # 3. Set PDF Document Metadata
    doc.set_metadata({
        "title": "ABHERA: Complete Project Documentation & University Viva Preparation Guide",
        "author": "Gunda Sai Deepthi (Department of Computer Science & Engineering)",
        "subject": "Adaptive BERT-based Harassment Entity Recognition & Assistance (ABHERA)",
        "keywords": "BERT, NLP, Machine Learning, Named Entity Recognition, FastAPI, PostgreSQL, BNS 2023, Women Safety, Viva Defense",
        "creator": "ABHERA Technical Documentation Engine",
        "producer": "PyMuPDF 1.28 & ABHERA Compiler",
    })
    
    # 4. Save Final Output
    print(f"[5/5] Saving final optimized PDF to: {PDF_OUTPUT_PATH}...")
    doc.save(
        str(PDF_OUTPUT_PATH),
        garbage=4,
        deflate=True,
        clean=True,
    )
    doc.close()
    
    # Remove temporary raw PDF
    if temp_pdf_path.exists():
        temp_pdf_path.unlink()
        
    print("================================================================================")
    print("SUCCESS: Complete Project Documentation & Viva Guide Generated Successfully!")
    print(f"  Markdown Source : {MD_OUTPUT_PATH} ({MD_OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"  Final PDF Output: {PDF_OUTPUT_PATH} ({PDF_OUTPUT_PATH.stat().st_size:,} bytes, {total_pages} pages)")
    print("================================================================================")


if __name__ == "__main__":
    md_text = assemble_markdown()
    html_content = convert_markdown_to_html(md_text)
    temp_pdf = compile_pdf_story(html_content)
    post_process_pdf(temp_pdf)

import os
import fitz # PyMuPDF

pdf_path = r"D:\Abhera(Mini)\scratch\ABHERA_Incident_Report.pdf"
out_dir = r"C:\Users\Shankara\.gemini\antigravity\brain\88e239af-f11d-4879-8bee-7ec472662557\scratch"

os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"Opened PDF with {len(doc)} pages.")

for idx, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    png_name = f"pdf-page-{idx + 1}.png"
    png_path = os.path.join(out_dir, png_name)
    pix.save(png_path)
    print(f"Saved page {idx + 1} to {png_path}")

print("All PDF pages rendered to PNG images!")

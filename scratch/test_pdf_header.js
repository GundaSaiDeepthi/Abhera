const { jsPDF } = require('jspdf');
const fs = require('fs');
const path = require('path');

const doc = new jsPDF({
  orientation: 'portrait',
  unit: 'mm',
  format: 'a4',
});

const pWidth = doc.internal.pageSize.getWidth();
const centerX = pWidth / 2;

// Dark Navy Banner
doc.setFillColor(15, 23, 42);
doc.rect(0, 0, pWidth, 36, 'F');

// Gold Accent Strip
doc.setFillColor(176, 141, 87);
doc.rect(0, 35.2, pWidth, 0.8, 'F');

// 1. ABHERA
doc.setTextColor(255, 255, 255);
doc.setFont('helvetica', 'bold');
doc.setFontSize(16);
doc.text('ABHERA', centerX, 11, { align: 'center' });

// 2. Tagline
doc.setFont('helvetica', 'normal');
doc.setFontSize(8);
doc.setTextColor(226, 232, 240);
doc.text('Your Story. Understood.', centerX, 16.8, { align: 'center' });
doc.text('Your Rights. Empowered.', centerX, 20.8, { align: 'center' });

// 3. Title
doc.setFont('helvetica', 'bold');
doc.setFontSize(10.5);
doc.setTextColor(255, 255, 255);
doc.text('INCIDENT ANALYSIS REPORT', centerX, 29.5, { align: 'center' });

// Save test PDF artifact
const outPath = path.join(__dirname, 'test_header.pdf');
const pdfBuffer = Buffer.from(doc.output('arraybuffer'));
fs.writeFileSync(outPath, pdfBuffer);

console.log('PDF rendered successfully to:', outPath);

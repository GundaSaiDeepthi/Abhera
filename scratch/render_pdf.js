const { jsPDF } = require('../frontend/node_modules/jspdf');
const fs = require('fs');
const path = require('path');

const sampleReportPath = path.join(__dirname, 'sample_report.json');
const report = JSON.parse(fs.readFileSync(sampleReportPath, 'utf8'));

const ENTITY_LABEL_MAP = {
  PERP_REL: 'Person Involved / Relationship',
  TIME_FREQ: 'Frequency / Duration',
  LOCATION: 'Location / Workplace',
  PLATFORM: 'Platform / Channel',
  EVIDENCE: 'Evidence Mentioned',
  LAW_SEC: 'Statutory Section Referenced',
};

const formatEntityLabel = (key) => {
  if (!key) return 'Information';
  const upper = String(key).toUpperCase();
  if (ENTITY_LABEL_MAP[upper]) {
    return ENTITY_LABEL_MAP[upper];
  }
  return String(key)
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const formatActName = (rawAct) => {
  if (!rawAct) return 'STATUTORY ACT';
  const clean = String(rawAct).trim().toUpperCase();
  if (clean === 'DV_ACT' || clean === 'DV ACT' || clean === 'DOMESTIC_VIOLENCE_ACT') {
    return 'PROTECTION OF WOMEN FROM DOMESTIC VIOLENCE ACT, 2005';
  }
  if (clean === 'DOWRY_ACT' || clean === 'DOWRY PROHIBITION ACT') {
    return 'DOWRY PROHIBITION ACT, 1961';
  }
  if (clean === 'IPC') return 'INDIAN PENAL CODE (IPC)';
  if (clean === 'BNS') return 'BHARATIYA NYAYA SANHITA (BNS)';
  if (clean === 'IT_ACT') return 'INFORMATION TECHNOLOGY ACT, 2000';
  if (clean === 'POSH_ACT') return 'POSH ACT, 2013';
  return clean.replace(/_/g, ' ');
};

const safeStr = (val, fallback = '') => {
  if (val === null || val === undefined) return fallback;
  if (typeof val === 'object') {
    return val.text || val.name || val.label || val.value || fallback;
  }
  const str = String(val).trim();
  return str.length > 0 ? str : fallback;
};

const normalizeReportData = (reportData) => {
  if (!reportData) return {};

  let incidentLabels = [];
  let incidentMsg = null;
  const rawType = reportData.identified_incident_type;
  if (Array.isArray(rawType)) {
    incidentLabels = rawType.map((item) => safeStr(item)).filter(Boolean);
  } else if (rawType && typeof rawType === 'object') {
    const rawList = rawType.labels || rawType.categories || rawType.incident_types || rawType.predicted_labels || [];
    if (Array.isArray(rawList)) {
      incidentLabels = rawList.map((item) => safeStr(item)).filter(Boolean);
    }
    incidentMsg = safeStr(rawType.message || rawType.msg, null);
  } else if (typeof rawType === 'string' && rawType.trim()) {
    incidentLabels = [rawType.trim()];
  }

  let extractedMap = {};
  const rawExtracted = reportData.extracted_information;
  if (Array.isArray(rawExtracted)) {
    rawExtracted.forEach((item) => {
      if (item && typeof item === 'object') {
        const lbl = item.label || item.entity_group || item.type || item.entity || 'ENTITY';
        const val = safeStr(item.text || item.word || item.value);
        if (val) {
          if (!extractedMap[lbl]) extractedMap[lbl] = [];
          if (!extractedMap[lbl].includes(val)) extractedMap[lbl].push(val);
        }
      } else if (typeof item === 'string' && item.trim()) {
        if (!extractedMap['ENTITY']) extractedMap['ENTITY'] = [];
        extractedMap['ENTITY'].push(item.trim());
      }
    });
  } else if (rawExtracted && typeof rawExtracted === 'object') {
    Object.entries(rawExtracted).forEach(([lbl, vals]) => {
      if (Array.isArray(vals)) {
        extractedMap[lbl] = vals.map((v) => safeStr(v)).filter(Boolean);
      } else if (typeof vals === 'string' && vals.trim()) {
        extractedMap[lbl] = [vals.trim()];
      }
    });
  }

  let userDetails = [];
  const rawDetails = reportData.user_provided_details || reportData.user_answers;
  if (Array.isArray(rawDetails)) {
    userDetails = rawDetails.map((item, idx) => {
      if (typeof item === 'object' && item !== null) {
        return {
          question_id: item.question_id || `Q_${idx + 1}`,
          question: safeStr(item.question || item.question_text, `Question ${idx + 1}`),
          answer: safeStr(item.answer || item.answer_text || item.user_answer, 'Not provided'),
        };
      }
      return {
        question_id: `Q_${idx + 1}`,
        question: `Question ${idx + 1}`,
        answer: safeStr(item, 'Not provided'),
      };
    });
  } else if (rawDetails && typeof rawDetails === 'object') {
    userDetails = Object.entries(rawDetails).map(([q_id, ans], idx) => ({
      question_id: q_id,
      question: `Question ${q_id}`,
      answer: safeStr(ans, 'Not provided'),
    }));
  }

  let legalList = [];
  let legalMsg = null;
  const rawLegal = reportData.relevant_legal_information;
  if (Array.isArray(rawLegal)) {
    legalList = rawLegal;
  } else if (rawLegal && typeof rawLegal === 'object') {
    const rawProvisions =
      rawLegal.legal_information ||
      rawLegal.legal_provisions ||
      rawLegal.results ||
      rawLegal.provisions ||
      [];
    if (Array.isArray(rawProvisions)) {
      legalList = rawProvisions;
    }
    if (rawLegal.message) legalMsg = safeStr(rawLegal.message, null);
  } else if (typeof rawLegal === 'string') {
    legalMsg = rawLegal;
  }

  let supportList = [];
  let supportMsg = null;
  const rawSupport = reportData.available_support_services;
  if (Array.isArray(rawSupport)) {
    supportList = rawSupport;
  } else if (rawSupport && typeof rawSupport === 'object') {
    const rawServices = rawSupport.support_services || rawSupport.services || rawSupport.results || [];
    if (Array.isArray(rawServices)) {
      supportList = rawServices;
    }
    if (rawSupport.message) supportMsg = safeStr(rawSupport.message, null);
  } else if (typeof rawSupport === 'string') {
    supportMsg = rawSupport;
  }

  let nextSteps = [];
  if (Array.isArray(reportData.suggested_next_steps)) {
    nextSteps = reportData.suggested_next_steps.map((step) => safeStr(step)).filter(Boolean);
  }

  let evidenceNotes = [];
  if (Array.isArray(reportData.evidence_information_notes)) {
    evidenceNotes = reportData.evidence_information_notes.map((note) => safeStr(note)).filter(Boolean);
  }

  let rawNarrative = safeStr(reportData.incident_summary || reportData.narrative_text, 'Incident summary is not available.');
  let initialNarrative = rawNarrative;
  if (rawNarrative.includes('\n')) {
    const parts = rawNarrative.split('\n').map((p) => p.trim()).filter(Boolean);
    if (parts.length > 0) {
      const initialParts = [];
      for (const part of parts) {
        if (initialParts.length === 0) {
          initialParts.push(part);
        } else if (part.toLowerCase() === 'yes' || part.toLowerCase() === 'no') {
          break;
        } else {
          const matchesAnswer = userDetails.some((ud) => ud.answer && ud.answer.toLowerCase() === part.toLowerCase());
          if (matchesAnswer) break;
          initialParts.push(part);
        }
      }
      initialNarrative = initialParts.join('\n');
    }
  }

  return {
    incidentLabels,
    incidentMsg,
    extractedMap,
    userDetails,
    legalList,
    legalMsg,
    supportList,
    supportMsg,
    nextSteps,
    evidenceNotes,
    initialNarrative,
  };
};

const ABHERA_COLORS = {
  page: '#F7F5F0',
  section: '#F4F8FC',
  card: '#FFFFFF',
  softBlue: '#EAF2FB',
  primary: '#1E4F8A',
  darkBlue: '#163A63',
  navy: '#101820',
  charcoal: '#20262E',
  slate: '#667085',
  border: '#D5DCE5',
  gold: '#B89A5A',
};

const hexToRgb = (hex) => {
  if (!hex || typeof hex !== 'string') return [0, 0, 0];
  const clean = hex.replace('#', '');
  const num = parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
};

const setFillColorHex = (targetDoc, hex) => {
  const [r, g, b] = hexToRgb(hex);
  targetDoc.setFillColor(r, g, b);
};

const setDrawColorHex = (targetDoc, hex) => {
  const [r, g, b] = hexToRgb(hex);
  targetDoc.setDrawColor(r, g, b);
};

const setTextColorHex = (targetDoc, hex) => {
  const [r, g, b] = hexToRgb(hex);
  targetDoc.setTextColor(r, g, b);
};

function generatePDF() {
  const norm = normalizeReportData(report);

  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 14;
  const contentWidth = pageWidth - margin * 2;
  const maxY = pageHeight - 16;
  let y = 42;

  const drawCheckmark = (targetDoc, cx, cy, size = 3, colorHex = ABHERA_COLORS.primary) => {
    setDrawColorHex(targetDoc, colorHex);
    targetDoc.setLineWidth(0.65);
    targetDoc.line(cx, cy + size * 0.5, cx + size * 0.35, cy + size * 0.85);
    targetDoc.line(cx + size * 0.35, cy + size * 0.85, cx + size * 0.9, cy + size * 0.15);
  };

  const drawPageBackground = (targetDoc) => {
    const pdf = targetDoc || doc;
    const pWidth = pdf.internal.pageSize.getWidth();
    const pHeight = pdf.internal.pageSize.getHeight();
    setFillColorHex(pdf, ABHERA_COLORS.page);
    pdf.rect(0, 0, pWidth, pHeight, 'F');
  };

  const drawReportHeader = (targetDoc, isFirstPage = true) => {
    const pdf = targetDoc || doc;
    const pWidth = pdf.internal.pageSize.getWidth();
    const centerX = pWidth / 2;

    drawPageBackground(pdf);

    if (isFirstPage) {
      setFillColorHex(pdf, ABHERA_COLORS.navy);
      pdf.rect(0, 0, pWidth, 38, 'F');

      setFillColorHex(pdf, ABHERA_COLORS.gold);
      pdf.rect(0, 37.2, pWidth, 0.8, 'F');

      pdf.setTextColor(255, 255, 255);
      pdf.setFont('times', 'bold');
      pdf.setFontSize(26);
      pdf.text('ABHERA', centerX, 12, { align: 'center' });

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(8.5);
      setTextColorHex(pdf, ABHERA_COLORS.page);
      pdf.text('Your Story. Understood. Your Rights. Empowered.', centerX, 18, { align: 'center' });

      pdf.setFont('times', 'bold');
      pdf.setFontSize(12.5);
      pdf.setTextColor(255, 255, 255);
      pdf.text('INCIDENT ASSISTANCE REPORT', centerX, 25.5, { align: 'center' });

      const subBadgeW = 68;
      setFillColorHex(pdf, ABHERA_COLORS.primary);
      pdf.roundedRect(centerX - subBadgeW / 2, 28.5, subBadgeW, 5.5, 1, 1, 'F');
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(7.5);
      pdf.setTextColor(255, 255, 255);
      pdf.text('CONFIDENTIAL ASSISTANCE DOCUMENT', centerX, 32.2, { align: 'center' });
    } else {
      setFillColorHex(pdf, ABHERA_COLORS.navy);
      pdf.rect(0, 0, pWidth, 12, 'F');

      setFillColorHex(pdf, ABHERA_COLORS.gold);
      pdf.rect(0, 11.4, pWidth, 0.6, 'F');

      pdf.setTextColor(255, 255, 255);
      pdf.setFont('times', 'bold');
      pdf.setFontSize(10);
      pdf.text('ABHERA', margin, 8);

      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(8.5);
      setTextColorHex(pdf, ABHERA_COLORS.page);
      pdf.text('INCIDENT ASSISTANCE REPORT', pWidth - margin, 8, { align: 'right' });
    }
  };

  const checkAddPage = (neededHeight = 14) => {
    if (y + neededHeight > maxY) {
      doc.addPage();
      drawReportHeader(doc, false);
      y = 18;
    }
  };

  const drawSectionHeader = (numStr, titleStr, noteStr, isContinued = false) => {
    const headerNeeded = noteStr && !isContinued ? 22 : 16;
    checkAddPage(headerNeeded);

    setFillColorHex(doc, ABHERA_COLORS.section);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin, y, contentWidth, 13, 2, 2, 'FD');

    setFillColorHex(doc, ABHERA_COLORS.primary);
    doc.roundedRect(margin + 1.5, y + 1.5, 11, 10, 1.5, 1.5, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.text(numStr, margin + 7, y + 8.2, { align: 'center' });

    const fullTitle = isContinued ? `${titleStr.toUpperCase()} (CONTINUED)` : titleStr.toUpperCase();
    setTextColorHex(doc, ABHERA_COLORS.navy);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text(fullTitle, margin + 15, y + 8.2);

    y += 16;
    if (noteStr && !isContinued) {
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8.5);
      setTextColorHex(doc, ABHERA_COLORS.slate);
      doc.text(noteStr, margin + 1, y);
      y += 7;
    }
    y += 2;
  };

  drawReportHeader(doc, true);
  y = 44;

  const metaColW = (contentWidth - 8) / 3;
  const metaH = 19;

  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(margin, y, metaColW, metaH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(margin, y, 2.5, metaH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('CASE NO.', margin + 6, y + 6.5);
  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  doc.text(safeStr(report.submission_id, 'SUB-79741a9d'), margin + 6, y + 13.5);

  const x2 = margin + metaColW + 4;
  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(x2, y, metaColW, metaH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(x2, y, 2.5, metaH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('DATE & TIME', x2 + 6, y + 6.5);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(new Date().toLocaleString(), x2 + 6, y + 13.5);

  const x3 = margin + (metaColW + 4) * 2;
  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(x3, y, metaColW, metaH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(x3, y, 2.5, metaH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('VERIFICATION', x3 + 6, y + 6.5);

  drawCheckmark(doc, x3 + 6, y + 10, 2.8, ABHERA_COLORS.primary);

  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  doc.text('PostgreSQL Verified', x3 + 10.5, y + 13.5);

  y += metaH + 7;

  const sitContainerH = 48;
  checkAddPage(sitContainerH);

  setFillColorHex(doc, ABHERA_COLORS.section);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(margin, y, contentWidth, sitContainerH, 3, 3, 'FD');

  doc.setFontSize(10.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  doc.text('YOUR SITUATION AT A GLANCE', margin + 6, y + 7.5);

  const conc = norm.incidentLabels.join(' + ') || 'Incident Under Analysis';
  const rawPerp = norm.extractedMap['PERP_REL']?.join(', ') || 'Not provided';
  const perp = rawPerp !== 'Not provided' ? rawPerp.charAt(0).toUpperCase() + rawPerp.slice(1) : 'Not provided';
  const loc = [...(norm.extractedMap['LOCATION'] || []), ...(norm.extractedMap['PLATFORM'] || [])].join(', ') || 'Not provided';

  const safeItem = (norm.userDetails || []).find(
    (d) => d.question_id === 'Q_SAFETY_01' || (d.question && String(d.question).toLowerCase().includes('danger'))
  );
  const safeAns = safeItem ? safeItem.answer : 'Not provided';

  const sitCardW = (contentWidth - 16) / 2;
  const sitCardH = 14.5;

  const sitY1 = y + 11.5;

  const sitX1 = margin + 5;
  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(sitX1, sitY1, sitCardW, sitCardH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(sitX1, sitY1, 3.5, sitCardH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('IDENTIFIED CONCERN', sitX1 + 6, sitY1 + 5);

  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  const concTrunc = doc.splitTextToSize(conc.toUpperCase(), sitCardW - 10)[0];
  doc.text(concTrunc || 'INCIDENT UNDER ANALYSIS', sitX1 + 6, sitY1 + 10.5);

  const sitX2 = margin + 5 + sitCardW + 6;
  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(sitX2, sitY1, sitCardW, sitCardH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(sitX2, sitY1, 2, sitCardH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('PERSON INVOLVED', sitX2 + 6, sitY1 + 5);

  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(doc.splitTextToSize(perp, sitCardW - 10)[0], sitX2 + 6, sitY1 + 10.5);

  const sitY2 = sitY1 + sitCardH + 4.5;

  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(sitX1, sitY2, sitCardW, sitCardH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(sitX1, sitY2, 2, sitCardH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('LOCATION / PLATFORM', sitX1 + 6, sitY2 + 5);

  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'normal');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(doc.splitTextToSize(loc, sitCardW - 10)[0], sitX1 + 6, sitY2 + 10.5);

  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(sitX2, sitY2, sitCardW, sitCardH, 2, 2, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(sitX2, sitY2, 2, sitCardH, 'F');

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text('SAFETY RESPONSE', sitX2 + 6, sitY2 + 5);

  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'normal');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(doc.splitTextToSize(safeAns, sitCardW - 10)[0], sitX2 + 6, sitY2 + 10.5);

  y += sitContainerH + 10;

  // SECTION 01
  drawSectionHeader('01', 'Incident Summary', 'Your original description is shown below.');

  const summaryText = safeStr(norm.initialNarrative || report.incident_summary, 'Incident summary is not available.');
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(10);
  const summaryLines = doc.splitTextToSize(`“${summaryText}”`, contentWidth - 14);
  const quoteBoxH = Math.max(20, summaryLines.length * 5.0 + 12);
  checkAddPage(quoteBoxH + 6);

  setFillColorHex(doc, ABHERA_COLORS.card);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(margin, y, contentWidth, quoteBoxH, 2.5, 2.5, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(margin, y, 3.5, quoteBoxH, 'F');

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  doc.text('YOUR ORIGINAL DESCRIPTION', margin + 7, y + 7);

  doc.setFontSize(10);
  doc.setFont('helvetica', 'italic');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(summaryLines, margin + 7, y + 13.5);
  y += quoteBoxH + 8;

  // SECTION 02
  drawSectionHeader('02', 'Identified Incident Type', 'Automated classification derived from multi-label BERT model inference.');
  const labels = norm.incidentLabels;
  if (labels.length > 0) {
    checkAddPage(34);
    setFillColorHex(doc, ABHERA_COLORS.card);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin, y, contentWidth, 32, 2.5, 2.5, 'FD');

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('IDENTIFIED CATEGORIES', margin + 6, y + 6);

    let chipX = margin + 6;
    let chipY = y + 8.5;
    labels.forEach((lbl) => {
      const txt = String(lbl).toUpperCase();
      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      const textWidth = doc.getTextWidth(txt);
      const chipW = textWidth + 8;
      if (chipX + chipW > margin + contentWidth - 6) {
        chipX = margin + 6;
        chipY += 7;
      }
      setFillColorHex(doc, ABHERA_COLORS.softBlue);
      setDrawColorHex(doc, ABHERA_COLORS.primary);
      doc.roundedRect(chipX, chipY, chipW, 5.5, 1, 1, 'FD');

      setTextColorHex(doc, ABHERA_COLORS.primary);
      doc.text(txt, chipX + 4, chipY + 4);
      chipX += chipW + 4;
    });

    const noteBoxY = y + 20;
    setFillColorHex(doc, ABHERA_COLORS.section);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin + 4, noteBoxY, contentWidth - 8, 8, 1, 1, 'FD');
    setFillColorHex(doc, ABHERA_COLORS.primary);
    doc.rect(margin + 4, noteBoxY, 2.5, 8, 'F');

    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.primary);
    doc.text('MODEL-BASED CLASSIFICATION:', margin + 8, noteBoxY + 5);

    doc.setFont('helvetica', 'italic');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('Used for querying verified statutory database records; not a formal legal determination.', margin + 52, noteBoxY + 5);

    y += 38;
  } else {
    checkAddPage(10);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'italic');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text(norm.incidentMsg || 'Incident type could not be determined from the available model output.', margin, y);
    y += 12;
  }

  // SECTION 03 (Keep together with header)
  const extractedEntries = Object.entries(norm.extractedMap).filter(([_, vals]) => Array.isArray(vals) && vals.length > 0);
  if (extractedEntries.length > 0) {
    const colW3 = (contentWidth - 6) / 2;
    const cardH3 = 19;
    const rowCount = Math.ceil(extractedEntries.length / 2);
    const totalSec3Height = 22 + rowCount * (cardH3 + 5);

    checkAddPage(totalSec3Height);
    drawSectionHeader('03', 'Extracted Information', 'Information identified from your description');

    for (let i = 0; i < extractedEntries.length; i += 2) {
      const item1 = extractedEntries[i];
      const item2 = extractedEntries[i + 1];

      const lbl1 = formatEntityLabel(item1[0]).toUpperCase();
      const valStr1 = item1[1].join(', ');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9.5);
      const val1Lines = doc.splitTextToSize(valStr1, colW3 - 10);

      setFillColorHex(doc, ABHERA_COLORS.card);
      setDrawColorHex(doc, ABHERA_COLORS.border);
      doc.roundedRect(margin, y, colW3, cardH3, 2, 2, 'FD');
      setFillColorHex(doc, ABHERA_COLORS.primary);
      doc.rect(margin, y, 2.5, cardH3, 'F');

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.slate);
      doc.text(lbl1, margin + 6, y + 6);

      doc.setFontSize(9.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.navy);
      doc.text(val1Lines[0] || '', margin + 6, y + 13.5);

      if (item2) {
        const lbl2 = formatEntityLabel(item2[0]).toUpperCase();
        const valStr2 = item2[1].join(', ');
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(9.5);
        const val2Lines = doc.splitTextToSize(valStr2, colW3 - 10);
        const x2 = margin + colW3 + 6;

        setFillColorHex(doc, ABHERA_COLORS.card);
        setDrawColorHex(doc, ABHERA_COLORS.border);
        doc.roundedRect(x2, y, colW3, cardH3, 2, 2, 'FD');
        setFillColorHex(doc, ABHERA_COLORS.primary);
        doc.rect(x2, y, 2.5, cardH3, 'F');

        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.slate);
        doc.text(lbl2, x2 + 6, y + 6);

        doc.setFontSize(9.5);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.navy);
        doc.text(val2Lines[0] || '', x2 + 6, y + 13.5);
      }

      y += cardH3 + 5;
    }
  } else {
    drawSectionHeader('03', 'Extracted Information', 'Information identified from your description');
    checkAddPage(10);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('No entities extracted from user narrative.', margin, y);
    y += 12;
  }

  // SECTION 04
  drawSectionHeader('04', 'Details You Provided', 'Direct user questionnaire responses recorded during the session.');
  if (Array.isArray(norm.userDetails) && norm.userDetails.length > 0) {
    norm.userDetails.forEach((item) => {
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9.5);
      const qLines = doc.splitTextToSize(`Q: ${item.question}`, contentWidth - 14);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      const aLines = doc.splitTextToSize(`A: ${item.answer}`, contentWidth - 14);

      const cardH = (qLines.length + aLines.length) * 5.0 + 8;
      checkAddPage(cardH + 4);

      setFillColorHex(doc, ABHERA_COLORS.card);
      setDrawColorHex(doc, ABHERA_COLORS.border);
      doc.roundedRect(margin, y, contentWidth, cardH, 2, 2, 'FD');

      doc.setFontSize(9.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.navy);
      doc.text(qLines, margin + 6, y + 6);
      const qH = qLines.length * 5.0;

      doc.setFont('helvetica', 'normal');
      setTextColorHex(doc, ABHERA_COLORS.charcoal);
      doc.text(aLines, margin + 6, y + 6 + qH);
      y += cardH + 5;
    });
  } else {
    checkAddPage(18);
    setFillColorHex(doc, ABHERA_COLORS.card);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin, y, contentWidth, 17, 2, 2, 'FD');
    setFillColorHex(doc, ABHERA_COLORS.slate);
    doc.rect(margin, y, 2.5, 17, 'F');

    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.navy);
    doc.text('NO ADDITIONAL RESPONSES RECORDED', margin + 6, y + 6.5);

    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('No additional user question responses were recorded during this session.', margin + 6, y + 12);
    y += 22;
  }

  // SECTION 05
  drawSectionHeader('05', 'Relevant Legal Information', 'Statutory provisions retrieved from ABHERA\'s verified PostgreSQL database.');

  if (norm.legalList.length > 0) {
    norm.legalList.forEach((law) => {
      const rawAct = safeStr(law.act || law.act_name, 'Statutory Act');
      const actName = formatActName(rawAct);
      const secNum = safeStr(law.section || law.section_number, 'N/A');
      const title = safeStr(law.title || law.heading || law.section_name, 'Statutory Provision');
      const desc = safeStr(law.description || law.content || law.section_text);

      const usableWidth = contentWidth - 16; // 166mm max text width inside card

      doc.setFont('times', 'bold');
      doc.setFontSize(11);
      const titleLines = doc.splitTextToSize(title, usableWidth);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      const descLines = doc.splitTextToSize(desc, usableWidth);

      const titleH = titleLines.length * 5.2;
      const descH = descLines.length * 4.6;

      const cardH = 40 + titleH + descH;

      if (y + cardH > maxY) {
        doc.addPage();
        drawReportHeader(doc, false);
        y = 18;
        drawSectionHeader('05', 'Relevant Legal Information', null, true);
      }

      setFillColorHex(doc, ABHERA_COLORS.card);
      setDrawColorHex(doc, ABHERA_COLORS.border);
      doc.roundedRect(margin, y, contentWidth, cardH, 2.5, 2.5, 'FD');

      setFillColorHex(doc, ABHERA_COLORS.primary);
      doc.rect(margin, y, 3.5, cardH, 'F');

      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.primary);
      doc.text(actName.toUpperCase(), margin + 8, y + 7.5);

      const badgeW = 24;
      const badgeX = margin + contentWidth - badgeW - 6;
      setFillColorHex(doc, ABHERA_COLORS.softBlue);
      setDrawColorHex(doc, ABHERA_COLORS.primary);
      doc.roundedRect(badgeX, y + 3.5, badgeW, 5.5, 1, 1, 'FD');

      drawCheckmark(doc, badgeX + 3, y + 4.5, 2.2, ABHERA_COLORS.primary);

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.primary);
      doc.text('VERIFIED', badgeX + 7.5, y + 7.5);

      doc.setFontSize(13);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.navy);
      doc.text(`SECTION ${secNum}`, margin + 8, y + 14.5);

      doc.setFontSize(11);
      doc.setFont('times', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.navy);
      doc.text(titleLines, margin + 8, y + 21);

      const descStartY = y + 21 + titleH + 2;
      doc.setFontSize(9.5);
      doc.setFont('helvetica', 'normal');
      setTextColorHex(doc, ABHERA_COLORS.charcoal);
      doc.text(descLines, margin + 8, descStartY);

      const footerY = y + cardH - 5.5;
      drawCheckmark(doc, margin + 8, footerY - 3, 2.4, ABHERA_COLORS.primary);

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.primary);
      doc.text('PostgreSQL Verified Database Record', margin + 12.5, footerY);

      y += cardH + 7;
    });
  } else {
    checkAddPage(10);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text(norm.legalMsg || 'Information not available in the provided knowledge base.', margin, y);
    y += 12;
  }

  // SECTION 06
  drawSectionHeader('06', 'Suggested Next Steps', 'General administrative and practical recommendations from backend report output.');
  if (Array.isArray(norm.nextSteps) && norm.nextSteps.length > 0) {
    norm.nextSteps.forEach((step, idx) => {
      const isEmergency = step.toLowerCase().includes('immediate physical danger') || step.includes('112');
      doc.setFont('helvetica', isEmergency ? 'bold' : 'normal');
      doc.setFontSize(9.5);
      const stepLines = doc.splitTextToSize(step, contentWidth - 22);
      const cardH = Math.max(15, stepLines.length * 4.8 + 7);
      checkAddPage(cardH + 4);

      const cardFill = isEmergency ? ABHERA_COLORS.softBlue : ABHERA_COLORS.card;
      const borderCol = isEmergency ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.border;

      setFillColorHex(doc, cardFill);
      setDrawColorHex(doc, borderCol);
      doc.roundedRect(margin, y, contentWidth, cardH, 2, 2, 'FD');

      if (isEmergency) {
        setFillColorHex(doc, ABHERA_COLORS.darkBlue);
        doc.rect(margin, y, 3.5, cardH, 'F');
      }

      setFillColorHex(doc, isEmergency ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.primary);
      doc.roundedRect(margin + (isEmergency ? 5 : 4), y + (cardH - 8.5) / 2, 8.5, 8.5, 1, 1, 'F');
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(255, 255, 255);
      doc.text(String(idx + 1).padStart(2, '0'), margin + (isEmergency ? 9.25 : 8.25), y + (cardH - 8.5) / 2 + 6, { align: 'center' });

      doc.setFontSize(9.5);
      doc.setFont('helvetica', isEmergency ? 'bold' : 'normal');
      setTextColorHex(doc, isEmergency ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.charcoal);
      doc.text(stepLines, margin + 16, y + 6);

      y += cardH + 5;
    });
  } else {
    checkAddPage(10);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('No suggested next steps available.', margin, y);
    y += 12;
  }

  // SECTION 07
  drawSectionHeader('07', 'Available Support Services', 'Verified helpline and organization records retrieved from PostgreSQL database.');
  if (norm.supportList.length > 0) {
    norm.supportList.forEach((service) => {
      const name = safeStr(service.name || service.organization_name || service.service_type, 'Support Service');
      const srvType = safeStr(service.service_type, 'Helpline');
      const contact = safeStr(service.contact_number || service.phone || service.helpline, 'Not provided');
      const dist = safeStr(service.district);
      const st = safeStr(service.state, 'All India');
      const locStr = dist ? `${dist}, ${st}` : st;

      const is112 = contact.includes('112') || name.includes('112') || srvType.toLowerCase().includes('emergency');
      const cardFill = is112 ? ABHERA_COLORS.softBlue : ABHERA_COLORS.card;
      const borderCol = is112 ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.border;

      checkAddPage(25);
      setFillColorHex(doc, cardFill);
      setDrawColorHex(doc, borderCol);
      doc.roundedRect(margin, y, contentWidth, 23, 2, 2, 'FD');
      setFillColorHex(doc, is112 ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.primary);
      doc.rect(margin, y, 3.5, 23, 'F');

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.slate);
      doc.text(`TYPE: ${srvType.toUpperCase()}   |   COVERAGE: ${locStr.toUpperCase()}`, margin + 7, y + 6);

      doc.setFontSize(10.5);
      doc.setFont('times', 'bold');
      setTextColorHex(doc, is112 ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.primary);
      const nameLines = doc.splitTextToSize(name, 100);
      doc.text(nameLines[0] || name, margin + 7, y + 12.5);

      doc.setFontSize(9.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, is112 ? ABHERA_COLORS.darkBlue : ABHERA_COLORS.primary);
      const contactText = `HELPLINE: ${contact}`;
      const contactLines = doc.splitTextToSize(contactText, 70);
      doc.text(contactLines[0] || contactText, margin + contentWidth - 6, y + 12.5, { align: 'right' });

      drawCheckmark(doc, margin + 7, y + 15, 2.2, ABHERA_COLORS.primary);

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      setTextColorHex(doc, ABHERA_COLORS.primary);
      doc.text('PostgreSQL Verified Support Record', margin + 11, y + 18);

      y += 28;
    });
  } else {
    checkAddPage(10);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text(norm.supportMsg || 'No matching support service was found in the available database.', margin, y);
    y += 12;
  }

  // SECTION 08
  drawSectionHeader('08', 'Evidence / Information Notes', 'Contextual evidence mentions attached to this submission.');
  if (Array.isArray(norm.evidenceNotes) && norm.evidenceNotes.length > 0) {
    norm.evidenceNotes.forEach((note) => {
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      const noteLines = doc.splitTextToSize(`• ${note}`, contentWidth - 14);
      const cardH = Math.max(13, noteLines.length * 4.8 + 6);
      checkAddPage(cardH + 4);

      setFillColorHex(doc, ABHERA_COLORS.card);
      setDrawColorHex(doc, ABHERA_COLORS.border);
      doc.roundedRect(margin, y, contentWidth, cardH, 2, 2, 'FD');
      setFillColorHex(doc, ABHERA_COLORS.primary);
      doc.rect(margin, y, 2.5, cardH, 'F');

      doc.setFontSize(9.5);
      doc.setFont('helvetica', 'normal');
      setTextColorHex(doc, ABHERA_COLORS.charcoal);
      doc.text(noteLines, margin + 6, y + 6);
      y += cardH + 4.5;
    });
  } else {
    checkAddPage(18);
    setFillColorHex(doc, ABHERA_COLORS.card);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin, y, contentWidth, 17, 2, 2, 'FD');
    setFillColorHex(doc, ABHERA_COLORS.slate);
    doc.rect(margin, y, 2.5, 17, 'F');

    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.navy);
    doc.text('NO SPECIFIC EVIDENCE ATTACHED', margin + 6, y + 6.5);

    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('No additional specific evidence notes were attached to this report.', margin + 6, y + 12);
    y += 22;
  }

  // SECTION 09
  drawSectionHeader('09', 'Model Prediction Information', 'Technical model provenance and inference parameters.');
  const modelInfo = report.model_prediction_information;

  const bertTask = safeStr(modelInfo?.bert_classifier?.task, 'Multi-Label Incident Classification');
  const bertArch = safeStr(modelInfo?.bert_classifier?.model_name, 'bert-base-uncased');
  const nerTask = safeStr(modelInfo?.ner_extractor?.task, 'Token Classification Entity Extraction');
  const nerArch = safeStr(modelInfo?.ner_extractor?.model_name, 'bert-base-uncased-ner');

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  const provText = `DATA PROVENANCE: ${safeStr(modelInfo?.data_provenance, 'Strict Anti-Hallucination Verified Database Records')}`;
  const provLines = doc.splitTextToSize(provText, contentWidth - 14);

  const cardH9 = 28 + provLines.length * 4.5;

  checkAddPage(cardH9 + 2);
  setFillColorHex(doc, ABHERA_COLORS.section);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(margin, y, contentWidth, cardH9, 2.5, 2.5, 'FD');
  setFillColorHex(doc, ABHERA_COLORS.primary);
  doc.rect(margin, y, 3.5, cardH9, 'F');

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.navy);
  doc.text(`BERT CLASSIFIER: ${bertTask} (${bertArch})`, margin + 7, y + 7);

  doc.text(`NER EXTRACTOR: ${nerTask} (${nerArch})`, margin + 7, y + 13.5);

  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  setTextColorHex(doc, ABHERA_COLORS.slate);
  doc.text(provLines, margin + 7, y + 19.5);

  const noticeW = 126;
  const noticeX = margin + 7;
  const noticeY = y + 19.5 + provLines.length * 4.5 + 2;
  setFillColorHex(doc, ABHERA_COLORS.softBlue);
  setDrawColorHex(doc, ABHERA_COLORS.primary);
  doc.roundedRect(noticeX, noticeY, noticeW, 6, 1, 1, 'FD');

  drawCheckmark(doc, noticeX + 3, noticeY + 1.2, 2.2, ABHERA_COLORS.primary);

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.primary);
  doc.text('DATABASE VERIFIED — NOT GENERATED BY AI MODELS', noticeX + 8, noticeY + 4.2);

  y += cardH9 + 6;

  // SECTION 10
  drawSectionHeader('10', 'Disclaimer', 'System legal notice.');
  const discText = safeStr(
    report.disclaimer,
    'This system provides informational assistance based on available datasets, trained models, and verified database records. It does not replace professional legal advice, emergency services, law enforcement, or other qualified support.'
  );
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  const discLines = doc.splitTextToSize(discText, contentWidth - 14);
  const discH = discLines.length * 4.8 + 13;
  checkAddPage(discH + 4);

  setFillColorHex(doc, ABHERA_COLORS.page);
  setDrawColorHex(doc, ABHERA_COLORS.border);
  doc.roundedRect(margin, y, contentWidth, discH, 2.5, 2.5, 'FD');

  setFillColorHex(doc, ABHERA_COLORS.gold);
  doc.rect(margin, y, contentWidth, 0.8, 'F');

  doc.setFontSize(10);
  doc.setFont('helvetica', 'bold');
  setTextColorHex(doc, ABHERA_COLORS.navy);
  doc.text('IMPORTANT DISCLAIMER', margin + 7, y + 7.5);

  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'normal');
  setTextColorHex(doc, ABHERA_COLORS.charcoal);
  doc.text(discLines, margin + 7, y + 14);

  // Footer on every page
  const totalPages = doc.internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    setFillColorHex(doc, ABHERA_COLORS.navy);
    doc.rect(0, pageHeight - 11, pageWidth, 11, 'F');

    setFillColorHex(doc, ABHERA_COLORS.gold);
    doc.rect(0, pageHeight - 11, pageWidth, 0.4, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(255, 255, 255);
    doc.text('CONFIDENTIAL ASSISTANCE DOCUMENT', margin, pageHeight - 4.5);

    doc.setFont('helvetica', 'normal');
    setTextColorHex(doc, ABHERA_COLORS.border);
    doc.text(`CASE NO. ${safeStr(report.submission_id, 'N/A')}`, pageWidth / 2, pageHeight - 4.5, { align: 'center' });

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(255, 255, 255);
    doc.text(`PAGE ${i} OF ${totalPages}`, pageWidth - margin, pageHeight - 4.5, { align: 'right' });
  }

  const outPath = path.join(__dirname, 'ABHERA_Incident_Report.pdf');
  const pdfBuffer = Buffer.from(doc.output('arraybuffer'));
  fs.writeFileSync(outPath, pdfBuffer);
  console.log(`PDF rendered successfully! Total pages: ${totalPages}`);
  console.log(`Output saved to: ${outPath}`);
}

generatePDF();

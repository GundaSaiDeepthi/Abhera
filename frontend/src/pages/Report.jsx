import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';
import { useConversation } from '../context/ConversationContext';
import { getReport } from '../services/api';
import AbheraEmblem from '../components/AbheraEmblem';
import {
  Shield,
  FileText,
  Printer,
  Download,
  RotateCcw,
  BookOpen,
  ShieldCheck,
  Tag,
  MessageSquare,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Info,
  HelpCircle,
  ArrowRight,
  Loader2,
  Phone,
  ExternalLink,
  MapPin,
  ChevronDown,
  ChevronUp,
  Check
} from 'lucide-react';

// Friendly entity label mapping helper for frontend presentation
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

// Defensive helper to format values safely without [object Object], null, or undefined
const safeStr = (val, fallback = '') => {
  if (val === null || val === undefined) return fallback;
  if (typeof val === 'object') {
    return val.text || val.name || val.label || val.value || fallback;
  }
  const str = String(val).trim();
  return str.length > 0 ? str : fallback;
};

// Helper function to normalize diverse backend report data structures defensively
const normalizeReportData = (reportData) => {
  if (!reportData) return {};

  // 1. Identified Incident Type
  let incidentLabels = [];
  let incidentMsg = null;
  const rawType = reportData.identified_incident_type;
  if (Array.isArray(rawType)) {
    incidentLabels = rawType.map((item) => safeStr(item)).filter(Boolean);
  } else if (rawType && typeof rawType === 'object') {
    const rawList =
      rawType.labels || rawType.categories || rawType.incident_types || rawType.predicted_labels || [];
    if (Array.isArray(rawList)) {
      incidentLabels = rawList.map((item) => safeStr(item)).filter(Boolean);
    }
    incidentMsg = safeStr(rawType.message || rawType.msg, null);
  } else if (typeof rawType === 'string' && rawType.trim()) {
    incidentLabels = [rawType.trim()];
  }

  // 2. Extracted Information (NER)
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

  // 3. User Provided Details
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

  // 4. Relevant Legal Information
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

  // 5. Available Support Services
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

  // 6. Suggested Next Steps
  let nextSteps = [];
  if (Array.isArray(reportData.suggested_next_steps)) {
    nextSteps = reportData.suggested_next_steps.map((step) => safeStr(step)).filter(Boolean);
  }

  // 7. Evidence / Information Notes
  let evidenceNotes = [];
  if (Array.isArray(reportData.evidence_information_notes)) {
    evidenceNotes = reportData.evidence_information_notes.map((note) => safeStr(note)).filter(Boolean);
  }

  // Extract initial narrative safely (excluding concatenated follow-up lines if present in legacy records)
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
          break; // Stop at first follow-up answer line
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

// MASTER DESIGN TOKENS FOR EXACT COLOR PARITY (WEB UI & PDF EXPORT)
const ABHERA_COLORS = {
  page: '#F7F5F0',        // Warm Ivory Page & Document Surface Background
  section: '#F4F8FC',     // Pale Blue Section Container Fill
  card: '#FFFFFF',        // Pure White Inner Content Cards
  softBlue: '#EAF2FB',    // Soft Blue Highlight Fill & Verified Pills
  primary: '#1E4F8A',     // Primary Legal Navy Blue
  darkBlue: '#163A63',    // Dark Navy Secondary
  navy: '#101820',        // Navy Black Structural Headers & Footers
  charcoal: '#20262E',    // Main Body Text Charcoal
  slate: '#667085',       // Slate Subtext & Metadata Labels
  border: '#D5DCE5',      // Light Card & Container Borders
  gold: '#B89A5A',        // Premium Gold Separator Accent
};

// Convert hex color string (#RRGGBB) to [R, G, B] array for jsPDF methods
const hexToRgb = (hex) => {
  if (!hex || typeof hex !== 'string') return [0, 0, 0];
  const clean = hex.replace('#', '');
  const num = parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
};

// Convenience helpers to set PDF colors using master tokens directly
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

const Report = () => {
  const navigate = useNavigate();
  const { report, setReport, submissionId, setSubmissionId, resetConversation } = useConversation();

  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [expandedLegal, setExpandedLegal] = useState({});

  // Load report from backend API if not in context but submissionId exists or is in sessionStorage
  useEffect(() => {
    let isMounted = true;
    const fetchReportIfNeeded = async () => {
      const activeSubId = submissionId || (typeof sessionStorage !== 'undefined' ? sessionStorage.getItem('abhera_submission_id') : null);
      if (!report && activeSubId) {
        setLoading(true);
        setFetchError(null);
        try {
          const res = await getReport(activeSubId);
          if (isMounted && res) {
            setReport(res);
            if (!submissionId && setSubmissionId) {
              setSubmissionId(activeSubId);
            }
          }
        } catch (err) {
          if (isMounted) {
            setFetchError('Unable to load incident report. Please ensure your submission ID is valid.');
          }
        } finally {
          if (isMounted) setLoading(false);
        }
      }
    };
    fetchReportIfNeeded();
    return () => {
      isMounted = false;
    };
  }, [report, submissionId, setReport, setSubmissionId]);

  const toggleLegalExpand = (idx) => {
    setExpandedLegal((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handlePrint = () => {
    window.print();
  };

  const handleDownload = () => {
    if (!report) return;

    const norm = normalizeReportData(report);

    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const pageWidth = doc.internal.pageSize.getWidth(); // 210mm
    const pageHeight = doc.internal.pageSize.getHeight(); // 297mm
    const margin = 14; // 14mm margins
    const contentWidth = pageWidth - margin * 2; // 182mm content width
    const maxY = pageHeight - 16; // 281mm printable bottom boundary
    let y = 42;

    // Helper: Vector Checkmark Drawer (Guarantees zero broken ' glyphs!)
    const drawCheckmark = (targetDoc, cx, cy, size = 3, colorHex = ABHERA_COLORS.primary) => {
      setDrawColorHex(targetDoc, colorHex);
      targetDoc.setLineWidth(0.65);
      targetDoc.line(cx, cy + size * 0.5, cx + size * 0.35, cy + size * 0.85);
      targetDoc.line(cx + size * 0.35, cy + size * 0.85, cx + size * 0.9, cy + size * 0.15);
    };

    // 1. ISOLATED PAGE BACKGROUND DRAWER
    const drawPageBackground = (targetDoc) => {
      const pdf = targetDoc || doc;
      const pWidth = pdf.internal.pageSize.getWidth();
      const pHeight = pdf.internal.pageSize.getHeight();

      // Page background fill: Warm Ivory #F7F5F0
      setFillColorHex(pdf, ABHERA_COLORS.page);
      pdf.rect(0, 0, pWidth, pHeight, 'F');
    };

    // 2. REPORT HEADER DRAWER
    const drawReportHeader = (targetDoc, isFirstPage = true) => {
      const pdf = targetDoc || doc;
      const pWidth = pdf.internal.pageSize.getWidth();
      const centerX = pWidth / 2;

      drawPageBackground(pdf);

      if (isFirstPage) {
        // Navy Black Banner (38mm height)
        setFillColorHex(pdf, ABHERA_COLORS.navy);
        pdf.rect(0, 0, pWidth, 38, 'F');

        // Premium Gold Accent Line
        setFillColorHex(pdf, ABHERA_COLORS.gold);
        pdf.rect(0, 37.2, pWidth, 0.8, 'F');

        // 1. ABHERA
        pdf.setTextColor(255, 255, 255);
        pdf.setFont('times', 'bold');
        pdf.setFontSize(26);
        pdf.text('ABHERA', centerX, 12, { align: 'center' });

        // 2. TAGLINE
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(8.5);
        setTextColorHex(pdf, ABHERA_COLORS.page);
        pdf.text('Your Story. Understood. Your Rights. Empowered.', centerX, 18, { align: 'center' });

        // 3. INCIDENT ASSISTANCE REPORT
        pdf.setFont('times', 'bold');
        pdf.setFontSize(12.5);
        pdf.setTextColor(255, 255, 255);
        pdf.text('INCIDENT ASSISTANCE REPORT', centerX, 25.5, { align: 'center' });

        // 4. CONFIDENTIAL ASSISTANCE DOCUMENT BADGE
        const subBadgeW = 68;
        setFillColorHex(pdf, ABHERA_COLORS.primary);
        pdf.roundedRect(centerX - subBadgeW / 2, 28.5, subBadgeW, 5.5, 1, 1, 'F');
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(7.5);
        pdf.setTextColor(255, 255, 255);
        pdf.text('CONFIDENTIAL ASSISTANCE DOCUMENT', centerX, 32.2, { align: 'center' });
      } else {
        // Running Top Header for Pages 2+ (12mm height)
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

    // 3. PAGE ADDITION CHECK
    const checkAddPage = (neededHeight = 14) => {
      if (y + neededHeight > maxY) {
        doc.addPage();
        drawReportHeader(doc, false);
        y = 18; // Resets Y below top header bar
      }
    };

    // Initial page header setup
    drawReportHeader(doc, true);
    y = 44;

    // 4. COVER METADATA BLOCK (3 distinct white cards in a horizontal row)
    const metaColW = (contentWidth - 8) / 3; // ~58mm per card
    const metaH = 19;

    // Card 1: CASE NO.
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

    // Card 2: DATE & TIME
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

    // Card 3: VERIFICATION
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

    // 5. "YOUR SITUATION AT A GLANCE" (PAGE 1 SNAPSHOT PANEL WITH 2x2 GRID OF CARDS)
    const sitContainerH = 48;
    checkAddPage(sitContainerH);

    // Outer Pale Blue Container (#F4F8FC fill, #D5DCE5 stroke)
    setFillColorHex(doc, ABHERA_COLORS.section);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(margin, y, contentWidth, sitContainerH, 3, 3, 'FD');

    // Title Bar inside container
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

    const sitCardW = (contentWidth - 16) / 2; // ~83mm width per card
    const sitCardH = 14.5; // 14.5mm height

    // Row 1 Cards (Top)
    const sitY1 = y + 11.5;

    // Card 1: IDENTIFIED CONCERN (Primary Blue Focal Point)
    const sitX1 = margin + 5;
    setFillColorHex(doc, ABHERA_COLORS.card);
    setDrawColorHex(doc, ABHERA_COLORS.border);
    doc.roundedRect(sitX1, sitY1, sitCardW, sitCardH, 2, 2, 'FD');
    setFillColorHex(doc, ABHERA_COLORS.primary);
    doc.rect(sitX1, sitY1, 3.5, sitCardH, 'F'); // Stronger focal point blue accent bar

    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.slate);
    doc.text('IDENTIFIED CONCERN', sitX1 + 6, sitY1 + 5);

    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'bold');
    setTextColorHex(doc, ABHERA_COLORS.primary);
    const concTrunc = doc.splitTextToSize(conc.toUpperCase(), sitCardW - 10)[0];
    doc.text(concTrunc || 'INCIDENT UNDER ANALYSIS', sitX1 + 6, sitY1 + 10.5);

    // Card 2: PERSON INVOLVED
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

    // Row 2 Cards (Bottom)
    const sitY2 = sitY1 + sitCardH + 4.5;

    // Card 3: LOCATION / PLATFORM
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

    // Card 4: SAFETY RESPONSE
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

    // 6. SECTION HEADER DRAWER (Solid navy block number badge + title)
    const drawSectionHeader = (numStr, titleStr, noteStr, isContinued = false) => {
      const headerNeeded = noteStr && !isContinued ? 22 : 16;
      checkAddPage(headerNeeded);

      // Pale Blue container header strip (#F4F8FC fill, #D5DCE5 stroke)
      setFillColorHex(doc, ABHERA_COLORS.section);
      setDrawColorHex(doc, ABHERA_COLORS.border);
      doc.roundedRect(margin, y, contentWidth, 13, 2, 2, 'FD');

      // Solid Primary Blue number block (#1E4F8A fill)
      setFillColorHex(doc, ABHERA_COLORS.primary);
      doc.roundedRect(margin + 1.5, y + 1.5, 11, 10, 1.5, 1.5, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.text(numStr, margin + 7, y + 8.2, { align: 'center' });

      // Section Title (Navy Black #101820)
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

    // SECTION 01: INCIDENT SUMMARY (PDF)
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

    // SECTION 02: IDENTIFIED INCIDENT TYPE (PDF)
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

      // Render classification pill chips
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

      // Model Note inside Pale-Blue Info Box
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

    // SECTION 03: EXTRACTED INFORMATION (PDF) — KEEP TOGETHER WITH HEADER
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

        // Card 1
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

        // Card 2
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

    // SECTION 04: DETAILS YOU PROVIDED (PDF)
    drawSectionHeader('04', 'Details You Provided', 'Direct user questionnaire responses recorded during the session.');
    if (Array.isArray(norm.userDetails) && norm.userDetails.length > 0) {
      norm.userDetails.forEach((item, idx) => {
        const numStr = String(idx + 1).padStart(2, '0');

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(9);
        const qLines = doc.splitTextToSize(item.question, contentWidth - 16);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        const aLines = doc.splitTextToSize(item.answer, contentWidth - 16);

        const cardH = 7 + (qLines.length * 4.5) + 5 + 4 + (aLines.length * 4.5) + 5;
        checkAddPage(cardH + 4);

        setFillColorHex(doc, ABHERA_COLORS.card);
        setDrawColorHex(doc, ABHERA_COLORS.border);
        doc.roundedRect(margin, y, contentWidth, cardH, 2, 2, 'FD');

        setFillColorHex(doc, ABHERA_COLORS.softBlue);
        doc.roundedRect(margin + 5, y + 4.5, 9, 5, 1, 1, 'F');
        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.primary);
        doc.text(numStr, margin + 6.8, y + 8);

        doc.setFontSize(7);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.slate);
        doc.text('QUESTION', margin + 17, y + 8);

        let curY = y + 14;
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.navy);
        doc.text(qLines, margin + 6, curY);

        curY += (qLines.length * 4.5) + 2;

        setDrawColorHex(doc, ABHERA_COLORS.border);
        doc.setLineWidth(0.2);
        doc.line(margin + 6, curY, margin + contentWidth - 6, curY);
        curY += 4.5;

        doc.setFontSize(7);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.primary);
        doc.text('ANSWER', margin + 6, curY);
        curY += 4.5;

        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        setTextColorHex(doc, ABHERA_COLORS.charcoal);
        doc.text(aLines, margin + 6, curY);

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

    // SECTION 05: RELEVANT LEGAL INFORMATION (PDF) — NO TEXT OVERFLOW, EXACT WRAPPING
    drawSectionHeader('05', 'Relevant Legal Information', 'Statutory provisions retrieved from ABHERA\'s verified PostgreSQL database.');

    if (norm.legalList.length > 0) {
      norm.legalList.forEach((law) => {
        const rawAct = safeStr(law.act || law.act_name, 'Statutory Act');
        const actName = formatActName(rawAct);
        const secNum = safeStr(law.section || law.section_number, 'N/A');
        const title = safeStr(law.title || law.heading || law.section_name, 'Statutory Provision');
        const desc = safeStr(law.description || law.content || law.section_text);

        // Usable text width inside card (contentWidth - 16 = 166mm)
        const usableWidth = contentWidth - 16;

        doc.setFont('times', 'bold');
        doc.setFontSize(11);
        const titleLines = doc.splitTextToSize(title, usableWidth);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9.5);
        const descLines = doc.splitTextToSize(desc, usableWidth);

        const titleH = titleLines.length * 5.2; // 5.2mm line height for title
        const descH = descLines.length * 4.6;   // 4.6mm line height for description

        // Exact content-driven card height
        const cardH = 40 + titleH + descH;

        // Intelligent Page Break Check: Never split a legal card across pages!
        if (y + cardH > maxY) {
          doc.addPage();
          drawReportHeader(doc, false);
          y = 18;
          // Draw subtle Section 05 Continuation header if continuing
          drawSectionHeader('05', 'Relevant Legal Information', null, true);
        }

        // Full-width Card Container (#FFFFFF fill, #D5DCE5 stroke)
        setFillColorHex(doc, ABHERA_COLORS.card);
        setDrawColorHex(doc, ABHERA_COLORS.border);
        doc.roundedRect(margin, y, contentWidth, cardH, 2.5, 2.5, 'FD');

        // 3.5mm Solid Primary Blue Left Accent Bar (#1E4F8A fill)
        setFillColorHex(doc, ABHERA_COLORS.primary);
        doc.rect(margin, y, 3.5, cardH, 'F');

        // 1. Act Name (Primary Blue #1E4F8A, uppercase)
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.primary);
        doc.text(actName.toUpperCase(), margin + 8, y + 7.5);

        // Verified Badge Pill (top right with vector checkmark)
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

        // 2. Section Number (Navy Black #101820)
        doc.setFontSize(13);
        doc.setFont('helvetica', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.navy);
        doc.text(`SECTION ${secNum}`, margin + 8, y + 14.5);

        // 3. Title (Times Bold, Navy Black #101820)
        doc.setFontSize(11);
        doc.setFont('times', 'bold');
        setTextColorHex(doc, ABHERA_COLORS.navy);
        doc.text(titleLines, margin + 8, y + 21);

        // 4. Description (Helvetica Normal, Charcoal #20262E)
        const descStartY = y + 21 + titleH + 2;
        doc.setFontSize(9.5);
        doc.setFont('helvetica', 'normal');
        setTextColorHex(doc, ABHERA_COLORS.charcoal);
        doc.text(descLines, margin + 8, descStartY);

        // 5. PostgreSQL Verified Database Record (anchored cleanly 5.5mm above card bottom)
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

    // SECTION 06: SUGGESTED NEXT STEPS (PDF)
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

        // Blue Numbered Badge Block
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

    // SECTION 07: AVAILABLE SUPPORT SERVICES (PDF)
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

    // SECTION 08: EVIDENCE / INFORMATION NOTES (PDF)
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

    // SECTION 09: MODEL PREDICTION INFORMATION (PDF)
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

    // Prominent non-generative notice pill with vector checkmark
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

    // SECTION 10: DISCLAIMER (PDF)
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

    // Dark Navy Footer on every page (Multi-pass dynamic page numbering)
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

    doc.save(`ABHERA_Incident_Report_${safeStr(report.submission_id, 'summary')}.pdf`);
  };

  const handleNewConversation = () => {
    resetConversation();
    navigate('/chatbot');
  };

  // Loading State
  if (loading) {
    return (
      <div className="report-loading-container">
        <Loader2 size={36} className="spin-icon" />
        <h2>Fetching Incident Assessment...</h2>
        <p>Retrieving verified report from backend database.</p>
      </div>
    );
  }

  // No Report State
  if (!report) {
    return (
      <div className="no-report-container">
        <div className="no-report-card">
          <FileText size={48} className="no-report-icon" />
          <h2>No Incident Report Available Yet</h2>
          <p>
            {fetchError ||
              'You have not completed an incident assessment conversation. Please start a conversation with the assistant.'}
          </p>
          <button className="btn-primary btn-large" onClick={handleNewConversation}>
            <span>BEGIN YOUR STORY</span>
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
    );
  }

  const norm = normalizeReportData(report);

  const incidentTypes = norm.incidentLabels || [];
  const extractedInfo = norm.extractedMap || {};
  const userDetails = norm.userDetails || [];
  const legalInfo = norm.legalList || [];
  const supportServices = norm.supportList || [];
  const nextSteps = norm.nextSteps || [];
  const evidenceNotes = norm.evidenceNotes || [];
  const modelInfo = report.model_prediction_information;

  // Extract Summary Card Values Safely
  const summaryConcern = incidentTypes.length > 0 ? incidentTypes.join(' + ') : 'Incident Under Analysis';
  const rawPerson = extractedInfo['PERP_REL'] && extractedInfo['PERP_REL'].length > 0 ? extractedInfo['PERP_REL'].join(', ') : 'Not provided';
  const summaryPerson = rawPerson !== 'Not provided' ? rawPerson.charAt(0).toUpperCase() + rawPerson.slice(1) : 'Not provided';
  const summaryLocation = [
    ...(extractedInfo['LOCATION'] || []),
    ...(extractedInfo['PLATFORM'] || [])
  ].filter(Boolean).join(', ') || 'Not provided';

  const safetyDetail = userDetails.find(
    (d) => d.question_id === 'Q_SAFETY_01' || (d.question && String(d.question).toLowerCase().includes('danger'))
  );
  const summarySafety = safetyDetail ? safetyDetail.answer : 'Not provided';

  return (
    <div className="report-page-container">
      {/* Report Actions Toolbar */}
      <div className="report-toolbar">
        <div className="toolbar-left">
          <div className="report-badge">
            <Shield size={14} />
            <span>CONFIDENTIAL CASE RECORD</span>
          </div>
          {report.submission_id && (
            <span className="submission-id-tag font-mono">
              CASE NO. <strong>{safeStr(report.submission_id)}</strong>
            </span>
          )}
        </div>

        <div className="toolbar-right">
          <button className="action-btn" onClick={handlePrint} title="Print report document">
            <Printer size={15} />
            <span>Print Report</span>
          </button>

          <button className="action-btn" onClick={handleDownload} title="Download PDF report document">
            <Download size={15} />
            <span>Download PDF</span>
          </button>

          <button className="action-btn reset-btn" onClick={handleNewConversation} title="Start new session">
            <RotateCcw size={15} />
            <span>New Session</span>
          </button>
        </div>
      </div>

      {/* Printable Official Legal Case Document */}
      <div className="report-document official-legal-document">
        {/* FAINT BACKGROUND LEGAL EMBLEM WATERMARK (~3-5% opacity) */}
        <div className="document-background-watermark" aria-hidden="true">
          <AbheraEmblem size={450} />
        </div>

        {/* DOCUMENT HEADER: STRICT VERTICAL CENTER STACK */}
        <header className="document-header centered-header">
          <div className="document-brand-centered">
            <div className="brand-logo-icon-wrap">
              <AbheraEmblem size={38} className="brand-logo" />
            </div>
            <h1 className="document-title-centered serif-heading">ABHERA</h1>
            <span className="document-subtitle-centered">
              Your Story. Understood. Your Rights. Empowered.
            </span>
          </div>

          <div className="document-title-centered-block">
            <h2 className="document-kicker-centered">INCIDENT ASSISTANCE REPORT</h2>
            <span className="case-record-subkicker">CONFIDENTIAL ASSISTANCE DOCUMENT</span>
          </div>

          <div className="document-header-divider"></div>

          <div className="document-meta-grid">
            <div className="meta-card">
              <span className="meta-card-label">SUBMISSION ID</span>
              <span className="meta-card-value font-mono">{safeStr(report.submission_id, 'N/A')}</span>
            </div>
            <div className="meta-card">
              <span className="meta-card-label">DATE & TIME</span>
              <span className="meta-card-value">{new Date().toLocaleString()}</span>
            </div>
            <div className="meta-card">
              <span className="meta-card-label">VERIFICATION</span>
              <span className="meta-card-value verified-badge">
                <CheckCircle2 size={14} />
                <span>DATABASE VERIFIED</span>
              </span>
            </div>
          </div>
        </header>

        {/* PROMINENT SUMMARY DASHBOARD: YOUR SITUATION AT A GLANCE */}
        <section className="situation-glance-section">
          <div className="glance-card-container">
            <div className="glance-header">
              <ShieldCheck size={20} className="glance-icon" />
              <div>
                <h2>YOUR SITUATION AT A GLANCE</h2>
                <p className="glance-subtitle">Key parameters summarized from your incident analysis</p>
              </div>
            </div>
            <div className="glance-grid">
              <div className="glance-card">
                <span className="glance-label font-mono">01 IDENTIFIED CONCERN</span>
                <span className="glance-value concern-highlight">{summaryConcern}</span>
              </div>
              <div className="glance-card">
                <span className="glance-label font-mono">02 PERSON INVOLVED</span>
                <span className="glance-value">{summaryPerson}</span>
              </div>
              <div className="glance-card">
                <span className="glance-label font-mono">03 LOCATION / PLATFORM</span>
                <span className="glance-value">{summaryLocation}</span>
              </div>
              <div className="glance-card">
                <span className="glance-label font-mono">04 SAFETY RESPONSE</span>
                <span className="glance-value">{summarySafety}</span>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 01: INCIDENT SUMMARY */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">01</div>
            <div className="section-header-text">
              <h2>INCIDENT SUMMARY</h2>
              <p className="section-subtitle-note">Your original description and follow-up responses are shown below.</p>
            </div>
          </div>

          <div className="section-content-area">
            <div className="narrative-quote-block report-white-card">
              <span className="block-kicker">YOUR ORIGINAL DESCRIPTION</span>
              <blockquote className="narrative-text">
                “{safeStr(report.incident_summary, 'Incident summary is not available.')}”
              </blockquote>
            </div>

            {userDetails.length > 0 && (
              <div className="followup-responses-block">
                <span className="block-kicker">FOLLOW-UP RESPONSES</span>
                <div className="followup-cards-list">
                  {userDetails.map((item, idx) => (
                    <div key={idx} className="followup-card report-white-card">
                      <div className="followup-question">
                        <span className="blue-icon-circle"><HelpCircle size={14} /></span>
                        <span>{item.question}</span>
                      </div>
                      <div className="followup-answer">{item.answer}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 02: IDENTIFIED INCIDENT TYPE */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">02</div>
            <div className="section-header-text">
              <h2>IDENTIFIED INCIDENT TYPE</h2>
              <p className="section-subtitle-note">Automated classification derived from multi-label model inference.</p>
            </div>
          </div>

          <div className="section-content-area">
            {incidentTypes.length > 0 ? (
              <div className="classification-card report-white-card">
                <span className="block-kicker">IDENTIFIED INCIDENT CLASSIFICATIONS</span>
                <div className="incident-pills-row">
                  {incidentTypes.map((cat, idx) => (
                    <div key={idx} className="incident-type-pill-box">
                      <span className="pill-index-tag">0{idx + 1}</span>
                      <span className="pill-text-cat">{cat.toUpperCase()}</span>
                    </div>
                  ))}
                </div>
                <div className="category-explanation-note">
                  <Info size={14} />
                  <span>
                    Note: Model classification is used exclusively to query verified statutory records from the database and does not constitute a formal legal determination.
                  </span>
                </div>
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <p className="fallback-text">
                  {norm.incidentMsg || 'Incident type could not be determined from the available model output.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 03: EXTRACTED INFORMATION */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">03</div>
            <div className="section-header-text">
              <h2>EXTRACTED INFORMATION</h2>
              <p className="section-subtitle-note">Information identified from your description</p>
            </div>
          </div>

          <div className="section-content-area">
            {Object.keys(extractedInfo).length > 0 ? (
              <div className="extracted-grid">
                {Object.entries(extractedInfo).map(([entType, items]) => {
                  const friendlyLabel = formatEntityLabel(entType);
                  return (
                    <div key={entType} className="extracted-card report-white-card">
                      <span className="extracted-label-kicker">{friendlyLabel.toUpperCase()}</span>
                      <div className="extracted-values-list">
                        {Array.isArray(items) && items.length > 0 ? (
                          items.map((val, idx) => (
                            <span key={idx} className="extracted-value-tag">
                              {val}
                            </span>
                          ))
                        ) : (
                          <span className="extracted-value-not-provided">Not provided</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">NO ENTITIES EXTRACTED</span>
                <p className="empty-state-desc">No specific entities were identified from the description.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 04: DETAILS YOU PROVIDED */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">04</div>
            <div className="section-header-text">
              <h2>DETAILS YOU PROVIDED</h2>
              <p className="section-subtitle-note">Direct user questionnaire responses recorded during the session.</p>
            </div>
          </div>

          <div className="section-content-area">
            {userDetails.length > 0 ? (
              <div className="user-details-cards">
                {userDetails.map((item, idx) => (
                  <div key={idx} className="detail-card report-white-card">
                    <div className="detail-card-header">
                      <span className="detail-card-idx font-mono">{String(idx + 1).padStart(2, '0')}</span>
                      <span className="detail-label-tag">QUESTION</span>
                    </div>
                    <div className="detail-question-text">{item.question}</div>
                    <div className="detail-answer-block">
                      <span className="detail-label-tag answer-tag">ANSWER</span>
                      <div className="detail-answer-text">{item.answer}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">NO ADDITIONAL RESPONSES RECORDED</span>
                <p className="empty-state-desc">No additional user question responses were recorded.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 05: RELEVANT LEGAL INFORMATION */}
        <section className="report-section-container highlight-section">
          <div className="section-header-block">
            <div className="section-number-block font-mono">05</div>
            <div className="section-header-text">
              <h2>RELEVANT LEGAL INFORMATION</h2>
              <p className="section-subtitle-note">Verified provisions relevant to the identified incident</p>
            </div>
          </div>

          <div className="section-content-area">
            <div className="legal-section-intro">
              <span className="legal-section-note">
                Relevant provisions identified from the verified legal knowledge base.
              </span>
              {Array.isArray(legalInfo) && legalInfo.length > 0 && (
                <span className="legal-count-badge font-mono">
                  VERIFIED LEGAL PROVISIONS [{legalInfo.length}]
                </span>
              )}
            </div>

            {Array.isArray(legalInfo) && legalInfo.length > 0 ? (
              <div className="legal-citation-sheet-stack">
                {legalInfo.map((law, idx) => {
                  const isExpanded = !!expandedLegal[idx];
                  const actName = safeStr(law.act || law.act_name, 'Statutory Act');
                  const secNum = safeStr(law.section || law.section_number, 'N/A');
                  const title = safeStr(law.title || law.heading || law.section_name, 'Statutory Provision');
                  const desc = safeStr(law.description || law.content || law.section_text);
                  return (
                    <div key={idx} className="legal-citation-card report-white-card">
                      <div className="citation-header-bar">
                        <span className="citation-kicker">{actName.toUpperCase()}</span>
                        <span className="citation-verified-pill">
                          <Check size={12} />
                          <span>VERIFIED</span>
                        </span>
                      </div>

                      <div className="citation-sec-num font-mono">SECTION {secNum}</div>

                      <h3 className="citation-title serif-heading">{title}</h3>

                      {desc && (
                        <div className={`citation-text-content ${!isExpanded && desc.length > 280 ? 'clamp-text' : ''}`}>
                          {desc}
                        </div>
                      )}

                      {desc && desc.length > 280 && (
                        <button
                          className="btn-text-expand"
                          onClick={() => toggleLegalExpand(idx)}
                        >
                          <span>{isExpanded ? 'Collapse Provision' : 'Read Full Provision Text'}</span>
                          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        </button>
                      )}

                      <div className="citation-footer">
                        <Check size={14} className="check-green" />
                        <span>STATUTORY DATABASE RECORD VERIFIED</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">INFORMATION NOT AVAILABLE</span>
                <p className="empty-state-desc">
                  {norm.legalMsg || 'Information not available in the provided knowledge base.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 06: SUGGESTED NEXT STEPS */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">06</div>
            <div className="section-header-text">
              <h2>SUGGESTED NEXT STEPS</h2>
              <p className="section-subtitle-note">General administrative and practical recommendations from backend report output.</p>
            </div>
          </div>

          <div className="section-content-area">
            {nextSteps.length > 0 ? (
              <div className="action-cards-stack">
                {nextSteps.map((step, idx) => (
                  <div key={idx} className="action-step-card report-white-card">
                    <div className="step-blue-circle font-mono">{String(idx + 1).padStart(2, '0')}</div>
                    <div className="action-step-text">{step}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">NO SUGGESTED STEPS RECORDED</span>
                <p className="empty-state-desc">No suggested next steps available.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 07: AVAILABLE SUPPORT SERVICES */}
        <section className="report-section-container highlight-section">
          <div className="section-header-block">
            <div className="section-number-block font-mono">07</div>
            <div className="section-header-text">
              <h2>AVAILABLE SUPPORT SERVICES</h2>
              <p className="section-subtitle-note">Verified helpline and organization records retrieved from PostgreSQL database.</p>
            </div>
          </div>

          <div className="section-content-area">
            {Array.isArray(supportServices) && supportServices.length > 0 ? (
              <div className="support-cards-grid">
                {supportServices.map((srv, idx) => {
                  const srvName = safeStr(srv.name || srv.organization_name, 'Support Service');
                  const phone = safeStr(srv.contact_number || srv.phone || srv.helpline);
                  const hasPhone = phone && phone !== 'N/A' && phone.trim().length > 0;
                  const isCleanPhone = hasPhone && /^[0-9+\-\s]+$/.test(phone.trim());
                  const srvType = safeStr(srv.service_type, 'Helpline');
                  const dist = safeStr(srv.district);
                  const st = safeStr(srv.state, 'All India');
                  const url = safeStr(srv.url || srv.website);

                  return (
                    <div key={idx} className="support-service-card report-white-card">
                      <div className="support-card-top">
                        <span className="service-type-badge">{srvType}</span>
                        {hasPhone && (
                          isCleanPhone ? (
                            <a href={`tel:${phone.replace(/\s+/g, '')}`} className="contact-phone-btn" title="Call Helpline">
                              <Phone size={13} />
                              <span>{phone}</span>
                            </a>
                          ) : (
                            <span className="contact-phone-badge">{phone}</span>
                          )
                        )}
                      </div>
                      <h3 className="service-title-text serif-heading">{srvName}</h3>
                      {(st || dist) && (
                        <p className="service-location-text">
                          <MapPin size={13} />
                          <span>Coverage: {dist ? `${dist}, ` : ''}{st}</span>
                        </p>
                      )}
                      {url && (
                        <a href={url} target="_blank" rel="noopener noreferrer" className="service-link-btn">
                          <span>Visit Website</span>
                          <ExternalLink size={13} />
                        </a>
                      )}
                      <div className="service-card-footer">
                        <Check size={13} className="check-green" />
                        <span>VERIFIED SUPPORT SERVICE RECORD</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">NO MATCHING SERVICES FOUND</span>
                <p className="empty-state-desc">
                  {norm.supportMsg || 'No matching support service was found in the available database.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 08: EVIDENCE / INFORMATION NOTES */}
        <section className="report-section-container">
          <div className="section-header-block">
            <div className="section-number-block font-mono">08</div>
            <div className="section-header-text">
              <h2>EVIDENCE / INFORMATION NOTES</h2>
              <p className="section-subtitle-note">Contextual evidence mentions attached to this submission.</p>
            </div>
          </div>

          <div className="section-content-area">
            {evidenceNotes.length > 0 ? (
              <div className="evidence-notes-card report-white-card">
                <ul className="evidence-bullet-list">
                  {evidenceNotes.map((note, idx) => (
                    <li key={idx}>{note}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <span className="empty-state-title">NO ADDITIONAL EVIDENCE NOTES</span>
                <p className="empty-state-desc">No additional evidence notes were attached to this report.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 09: MODEL PREDICTION INFORMATION */}
        <section className="report-section-container tech-section">
          <div className="section-header-block">
            <div className="section-number-block font-mono">09</div>
            <div className="section-header-text">
              <h2>MODEL PREDICTION INFORMATION</h2>
              <p className="section-subtitle-note">Technical model provenance and inference parameters.</p>
            </div>
          </div>

          <div className="section-content-area">
            {modelInfo ? (
              <div className="tech-cards-grid">
                <div className="tech-card report-white-card">
                  <span className="tech-card-label">BERT CLASSIFIER</span>
                  <span className="tech-card-val">{safeStr(modelInfo.bert_classifier?.task, 'Multi-Label Incident Classification')}</span>
                  <span className="tech-card-sub font-mono">Model: {safeStr(modelInfo.bert_classifier?.model_name, 'bert-base-uncased')}</span>
                </div>
                <div className="tech-card report-white-card">
                  <span className="tech-card-label">NER EXTRACTOR</span>
                  <span className="tech-card-val">{safeStr(modelInfo.ner_extractor?.task, 'Token Classification Entity Extraction')}</span>
                  <span className="tech-card-sub font-mono">Model: {safeStr(modelInfo.ner_extractor?.model_name, 'bert-base-uncased-ner')}</span>
                </div>
                <div className="tech-card report-white-card">
                  <span className="tech-card-label">DATA PROVENANCE</span>
                  <span className="tech-card-val">{safeStr(modelInfo.data_provenance, 'Strict Anti-Hallucination Verified Database Records')}</span>
                </div>
              </div>
            ) : (
              <div className="empty-state-card report-white-card">
                <p className="fallback-text">Technical prediction details unavailable.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 10: DISCLAIMER */}
        <section className="report-section-container disclaimer-section">
          <div className="section-header-block">
            <div className="section-number-block font-mono">10</div>
            <div className="section-header-text">
              <h2>DISCLAIMER</h2>
              <p className="section-subtitle-note">System legal notice and operational scope.</p>
            </div>
          </div>

          <div className="section-content-area">
            <div className="disclaimer-legal-card report-white-card">
              <AlertTriangle size={24} className="disclaimer-icon" />
              <div>
                <p>{safeStr(report.disclaimer, 'This system provides informational assistance based on available datasets, trained models, and verified database records. It does not replace professional legal advice, emergency services, law enforcement, or other qualified support.')}</p>
              </div>
            </div>
          </div>
        </section>

        {/* REPORT DOCUMENT FOOTER */}
        <footer className="document-report-footer">
          <div className="doc-footer-left font-mono">
            <span>ABHERA Incident Analysis Report</span>
            {report.submission_id && (
              <span className="doc-footer-case"> • Case ID: {safeStr(report.submission_id)}</span>
            )}
          </div>
          <div className="doc-footer-right font-mono">
            <span>Strict Anti-Hallucination Verified Database Record</span>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Report;

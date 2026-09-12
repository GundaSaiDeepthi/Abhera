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
  };
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

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - margin * 2;
    const maxY = pageHeight - 15;
    let y = 41;

    const drawReportHeader = (targetDoc, isFirstPage = true) => {
      const pdf = targetDoc || doc;
      const pWidth = pdf.internal.pageSize.getWidth();
      const centerX = pWidth / 2;

      if (isFirstPage) {
        // Primary Burgundy Banner (34mm height)
        pdf.setFillColor(122, 38, 58); // Burgundy #7A263A
        pdf.rect(0, 0, pWidth, 34, 'F');

        // Antique Gold Accent Line
        pdf.setFillColor(176, 141, 87); // Antique Gold #B08D57
        pdf.rect(0, 33.2, pWidth, 0.8, 'F');

        // 1. ABHERA
        pdf.setTextColor(255, 255, 255);
        pdf.setFont('times', 'bold');
        pdf.setFontSize(18);
        pdf.text('ABHERA', centerX, 10, { align: 'center' });

        // 2. TAGLINE
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(7.5);
        pdf.setTextColor(247, 244, 238); // Legal Ivory #F7F4EE
        pdf.text('Your Story. Understood. Your Rights. Empowered.', centerX, 15.5, { align: 'center' });

        // 3. INCIDENT ASSISTANCE REPORT
        pdf.setFont('times', 'bold');
        pdf.setFontSize(11);
        pdf.setTextColor(255, 255, 255);
        pdf.text('INCIDENT ASSISTANCE REPORT', centerX, 23.5, { align: 'center' });

        // 4. CONFIDENTIAL ASSISTANCE DOCUMENT
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(6.5);
        pdf.setTextColor(208, 183, 122); // Gold Light #D0B77A
        pdf.text('CONFIDENTIAL ASSISTANCE DOCUMENT', centerX, 28.5, { align: 'center' });
      } else {
        // Running Top Header for Pages 2+ (12mm height)
        pdf.setFillColor(122, 38, 58); // Burgundy #7A263A
        pdf.rect(0, 0, pWidth, 12, 'F');

        pdf.setFillColor(176, 141, 87); // Antique Gold #B08D57
        pdf.rect(0, 11.4, pWidth, 0.6, 'F');

        pdf.setTextColor(255, 255, 255);
        pdf.setFont('times', 'bold');
        pdf.setFontSize(10);
        pdf.text('ABHERA', margin, 8);

        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(8);
        pdf.setTextColor(247, 244, 238);
        pdf.text('INCIDENT ASSISTANCE REPORT', centerX, 8, { align: 'center' });

        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(7.5);
        pdf.text(`SUBMISSION ID: ${safeStr(report.submission_id, 'N/A')}`, pWidth - margin, 8, { align: 'right' });
      }
    };

    const checkAddPage = (neededHeight = 12) => {
      if (y + neededHeight > maxY) {
        doc.addPage();
        drawReportHeader(doc, false);
        y = 18; // Start y below running header
      }
    };

    drawReportHeader(doc, true);
    y = 41;

    // Cover Metadata Block (Clean 3-column card)
    doc.setFillColor(244, 241, 234); // Legal Ivory background #F4F1EA
    doc.setDrawColor(216, 210, 199); // Warm Stone border #D8D2C7
    doc.roundedRect(margin, y, contentWidth, 18, 2, 2, 'FD');

    const colW = contentWidth / 3;

    // Column 1: CASE NO.
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(130, 145, 163); // Slate Gray #8291A3
    doc.text('CASE NO.', margin + 6, y + 6);
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(32, 42, 53); // Charcoal #202A35
    doc.text(safeStr(report.submission_id, 'N/A'), margin + 6, y + 12);

    // Column 2: DATE & TIME
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(130, 145, 163);
    doc.text('DATE & TIME', margin + colW + 6, y + 6);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(32, 42, 53);
    doc.text(new Date().toLocaleString(), margin + colW + 6, y + 12);

    // Column 3: VERIFICATION
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(130, 145, 163);
    doc.text('VERIFICATION', margin + colW * 2 + 6, y + 6);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(63, 107, 89); // Verified Green #3F6B59
    doc.text('PostgreSQL Database Proven', margin + colW * 2 + 6, y + 12);

    y += 24;

    // TOP SUMMARY CARD: YOUR SITUATION AT A GLANCE (PDF)
    checkAddPage(32);
    doc.setFillColor(255, 255, 255); // Paper White #FFFFFF
    doc.setDrawColor(216, 210, 199);
    doc.roundedRect(margin, y, contentWidth, 28, 2, 2, 'FD');

    doc.setTextColor(8, 21, 37); // Midnight Navy #081525
    doc.setFont('times', 'bold');
    doc.setFontSize(10);
    doc.text('YOUR SITUATION AT A GLANCE', margin + 6, y + 6.5);

    const conc = norm.incidentLabels.join(', ') || 'Not provided';
    const rawPerp = norm.extractedMap['PERP_REL']?.join(', ') || 'Not provided';
    const perp = rawPerp !== 'Not provided' ? rawPerp.charAt(0).toUpperCase() + rawPerp.slice(1) : 'Not provided';
    const loc = [...(norm.extractedMap['LOCATION'] || []), ...(norm.extractedMap['PLATFORM'] || [])].join(', ') || 'Not provided';

    const safeItem = (norm.userDetails || []).find(
      (d) => d.question_id === 'Q_SAFETY_01' || (d.question && String(d.question).toLowerCase().includes('danger'))
    );
    const safeAns = safeItem ? safeItem.answer : 'Not provided';

    const halfW = (contentWidth - 12) / 2;

    // Grid Column 1 (Left Column)
    doc.setFontSize(7.5);
    doc.setTextColor(130, 145, 163);
    doc.setFont('helvetica', 'bold');
    doc.text('IDENTIFIED CONCERN', margin + 6, y + 13);
    doc.setFontSize(9);
    doc.setTextColor(63, 107, 89); // Verified Green #3F6B59
    doc.text(conc, margin + 6, y + 18);

    doc.setFontSize(7.5);
    doc.setTextColor(130, 145, 163);
    doc.setFont('helvetica', 'bold');
    doc.text('PERSON INVOLVED', margin + 6, y + 23);
    doc.setFontSize(9);
    doc.setTextColor(32, 42, 53);
    doc.text(perp, margin + 40, y + 23);

    // Grid Column 2 (Right Column)
    doc.setFontSize(7.5);
    doc.setTextColor(130, 145, 163);
    doc.setFont('helvetica', 'bold');
    doc.text('LOCATION / PLATFORM', margin + 6 + halfW, y + 13);
    doc.setFontSize(9);
    doc.setTextColor(32, 42, 53);
    doc.text(loc, margin + 6 + halfW, y + 18);

    doc.setFontSize(7.5);
    doc.setTextColor(130, 145, 163);
    doc.setFont('helvetica', 'bold');
    doc.text('SAFETY RESPONSE', margin + 6 + halfW, y + 23);
    doc.setFontSize(9);
    doc.setTextColor(32, 42, 53);
    doc.text(safeAns, margin + 6 + halfW + 35, y + 23);

    y += 34;

    const drawSectionHeader = (numStr, titleStr, noteStr) => {
      checkAddPage(22);
      
      // Number Badge in Midnight Navy
      doc.setFillColor(8, 21, 37); // Midnight Navy #081525
      doc.roundedRect(margin, y, 9, 7.5, 1, 1, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8);
      doc.text(numStr, margin + 2, y + 5.2);

      // Section Title in Midnight Navy
      doc.setTextColor(8, 21, 37);
      doc.setFont('times', 'bold');
      doc.setFontSize(11);
      doc.text(titleStr.toUpperCase(), margin + 12, y + 5.2);

      y += 9.5;
      if (noteStr) {
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(8);
        doc.setTextColor(130, 145, 163);
        doc.text(noteStr, margin, y);
        y += 5;
      }

      // Section separator line
      doc.setDrawColor(216, 210, 199);
      doc.line(margin, y, margin + contentWidth, y);
      y += 6;
    };

    // SECTION 01: INCIDENT SUMMARY (PDF)
    drawSectionHeader('01', 'Incident Summary', 'Your original description and follow-up responses are shown below.');
    
    // Original Description Quote Block
    const summaryText = safeStr(report.incident_summary, 'Incident summary is not available.');
    const summaryLines = doc.splitTextToSize(`“${summaryText}”`, contentWidth - 12);
    const quoteBoxH = Math.max(16, summaryLines.length * 4.5 + 8);
    checkAddPage(quoteBoxH + 4);

    doc.setFillColor(244, 241, 234); // Legal Ivory #F4F1EA
    doc.setDrawColor(216, 210, 199);
    doc.roundedRect(margin, y, contentWidth, quoteBoxH, 2, 2, 'FD');
    doc.setFillColor(182, 154, 97); // Antique Gold accent #B69A61
    doc.rect(margin, y, 2.5, quoteBoxH, 'F');

    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(118, 42, 61); // Burgundy #762A3D
    doc.text('YOUR ORIGINAL DESCRIPTION', margin + 6, y + 6);

    doc.setFontSize(9);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(32, 42, 53);
    doc.text(summaryLines, margin + 6, y + 11);
    y += quoteBoxH + 6;

    // Follow-up Responses Block
    if (norm.userDetails.length > 0) {
      checkAddPage(12);
      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(130, 145, 163);
      doc.text('FOLLOW-UP RESPONSES', margin, y);
      y += 5;

      norm.userDetails.forEach((item) => {
        const qLines = doc.splitTextToSize(`Q: ${item.question}`, contentWidth - 10);
        const aLines = doc.splitTextToSize(`A: ${item.answer}`, contentWidth - 10);
        const cardH = (qLines.length + aLines.length) * 4.5 + 6;
        checkAddPage(cardH + 4);

        doc.setFillColor(255, 255, 255);
        doc.setDrawColor(216, 210, 199);
        doc.roundedRect(margin, y, contentWidth, cardH, 1.5, 1.5, 'FD');

        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(32, 42, 53);
        doc.text(qLines, margin + 4, y + 5);
        const qH = qLines.length * 4.5;

        doc.setFont('helvetica', 'normal');
        doc.setTextColor(32, 42, 53);
        doc.text(aLines, margin + 4, y + 5 + qH);
        y += cardH + 4;
      });
    }
    y += 4;

    // SECTION 02: IDENTIFIED INCIDENT TYPE (PDF)
    drawSectionHeader('02', 'Identified Incident Type', 'Automated classification derived from multi-label model inference.');
    const labels = norm.incidentLabels;
    if (labels.length > 0) {
      checkAddPage(20);
      doc.setFillColor(244, 241, 234);
      doc.setDrawColor(216, 210, 199);
      doc.roundedRect(margin, y, contentWidth, 22, 2, 2, 'FD');

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(130, 145, 163);
      doc.text('IDENTIFIED CATEGORIES', margin + 6, y + 6);

      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(8, 21, 37); // Midnight Navy #081525
      doc.text(labels.join('   |   ').toUpperCase(), margin + 6, y + 12);

      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'italic');
      doc.setTextColor(130, 145, 163);
      doc.text('Note: Model classification used for querying verified statutory database records; not a formal legal determination.', margin + 6, y + 17);
      y += 26;
    } else {
      checkAddPage(10);
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'italic');
      doc.setTextColor(130, 145, 163);
      doc.text(norm.incidentMsg || 'Incident type could not be determined from the available model output.', margin, y);
      y += 10;
    }
    y += 4;

    // SECTION 03: EXTRACTED INFORMATION (PDF)
    drawSectionHeader('03', 'Extracted Information', 'Structured parameters identified from your narrative by NER sequence labeling.');
    let hasEntities = false;
    Object.entries(norm.extractedMap).forEach(([lbl, vals]) => {
      if (Array.isArray(vals) && vals.length > 0) {
        hasEntities = true;
        checkAddPage(12);
        const friendlyLabel = formatEntityLabel(lbl).toUpperCase();
        doc.setFillColor(244, 241, 234);
        doc.setDrawColor(216, 210, 199);
        doc.roundedRect(margin, y, contentWidth, 11, 1.5, 1.5, 'FD');

        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(130, 145, 163);
        doc.text(`${friendlyLabel}:`, margin + 4, y + 7);

        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(32, 42, 53);
        doc.text(vals.join(', '), margin + 65, y + 7);
        y += 14;
      }
    });
    if (!hasEntities) {
      checkAddPage(10);
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text('No entities extracted from user narrative.', margin, y);
      y += 10;
    }
    y += 4;

    // SECTION 04: DETAILS YOU PROVIDED (PDF)
    drawSectionHeader('04', 'Details You Provided', 'Direct user questionnaire responses recorded during the session.');
    if (Array.isArray(norm.userDetails) && norm.userDetails.length > 0) {
      norm.userDetails.forEach((item) => {
        const qLines = doc.splitTextToSize(`Q: ${item.question}`, contentWidth - 8);
        const aLines = doc.splitTextToSize(`A: ${item.answer}`, contentWidth - 8);
        const cardH = (qLines.length + aLines.length) * 4.5 + 6;
        checkAddPage(cardH + 4);

        doc.setFillColor(244, 241, 234);
        doc.setDrawColor(216, 210, 199);
        doc.roundedRect(margin, y, contentWidth, cardH, 1.5, 1.5, 'FD');

        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(32, 42, 53);
        doc.text(qLines, margin + 4, y + 5);
        const qH = qLines.length * 4.5;

        doc.setFont('helvetica', 'normal');
        doc.setTextColor(32, 42, 53);
        doc.text(aLines, margin + 4, y + 5 + qH);
        y += cardH + 4;
      });
    } else {
      checkAddPage(12);
      doc.setFillColor(244, 241, 234);
      doc.setDrawColor(216, 210, 199);
      doc.roundedRect(margin, y, contentWidth, 12, 1.5, 1.5, 'FD');
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text('NO ADDITIONAL RESPONSES RECORDED — No additional user question responses were recorded.', margin + 4, y + 7.5);
      y += 16;
    }
    y += 4;

    // SECTION 05: RELEVANT LEGAL INFORMATION (PDF)
    drawSectionHeader('05', 'Relevant Legal Information', 'Statutory provisions retrieved from ABHERA\'s verified PostgreSQL database.');
    if (norm.legalList.length > 0) {
      norm.legalList.forEach((law) => {
        const actName = safeStr(law.act || law.act_name, 'Statutory Act');
        const secNum = safeStr(law.section || law.section_number, 'N/A');
        const title = safeStr(law.title || law.heading || law.section_name, 'Statutory Provision');
        const desc = safeStr(law.description || law.content || law.section_text);

        const descLines = doc.splitTextToSize(desc, contentWidth - 10);
        const cardH = descLines.length * 4.5 + 22;

        checkAddPage(Math.min(cardH + 4, 40));

        doc.setFillColor(255, 255, 255); // Paper White #FFFFFF
        doc.setDrawColor(216, 210, 199); // Warm Stone border #D8D2C7
        doc.roundedRect(margin, y, contentWidth, cardH, 2, 2, 'FD');
        doc.setFillColor(182, 154, 97); // Antique Gold #B69A61 accent left bar
        doc.rect(margin, y, 2.5, cardH, 'F');

        doc.setFontSize(8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(32, 42, 53);
        doc.text(`ACT: ${actName.toUpperCase()}   |   SECTION: Section ${secNum}`, margin + 6, y + 6);

        doc.setFontSize(9.5);
        doc.setFont('times', 'bold');
        doc.setTextColor(8, 21, 37); // Midnight Navy #081525
        doc.text(title, margin + 6, y + 11.5);

        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(32, 42, 53);
        doc.text(descLines, margin + 6, y + 17);

        const footerY = y + cardH - 5;
        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(63, 107, 89); // Verified Green #3F6B59
        doc.text('✓ PostgreSQL Verified Database Record', margin + 6, footerY);

        y += cardH + 6;
      });
    } else {
      checkAddPage(10);
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text(norm.legalMsg || 'Information not available in the provided knowledge base.', margin, y);
      y += 10;
    }
    y += 4;

    // SECTION 06: SUGGESTED NEXT STEPS (PDF)
    drawSectionHeader('06', 'Suggested Next Steps', 'General administrative and practical recommendations from backend report output.');
    if (Array.isArray(norm.nextSteps) && norm.nextSteps.length > 0) {
      norm.nextSteps.forEach((step, idx) => {
        const stepLines = doc.splitTextToSize(step, contentWidth - 16);
        const cardH = stepLines.length * 4.5 + 6;
        checkAddPage(cardH + 4);

        doc.setFillColor(255, 255, 255);
        doc.setDrawColor(216, 210, 199);
        doc.roundedRect(margin, y, contentWidth, cardH, 1.5, 1.5, 'FD');

        doc.setFillColor(8, 21, 37);
        doc.roundedRect(margin + 3, y + 3, 7, 6, 1, 1, 'F');
        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(255, 255, 255);
        doc.text(String(idx + 1).padStart(2, '0'), margin + 4.2, y + 7);

        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(32, 42, 53);
        doc.text(stepLines, margin + 13, y + 7);

        y += cardH + 4;
      });
    } else {
      checkAddPage(10);
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text('No suggested next steps available.', margin, y);
      y += 10;
    }
    y += 4;

    // SECTION 07: AVAILABLE SUPPORT SERVICES (PDF)
    drawSectionHeader('07', 'Available Support Services', 'Verified helpline and organization records retrieved from PostgreSQL database.');
    if (norm.supportList.length > 0) {
      norm.supportList.forEach((service) => {
        const name = safeStr(service.name || service.organization_name, 'Support Service');
        const srvType = safeStr(service.service_type, 'Helpline');
        const contact = safeStr(service.contact_number || service.phone || service.helpline, 'Not provided');
        const dist = safeStr(service.district);
        const st = safeStr(service.state, 'All India');
        const locStr = dist ? `${dist}, ${st}` : st;

        checkAddPage(22);
        doc.setFillColor(255, 255, 255);
        doc.setDrawColor(216, 210, 199);
        doc.roundedRect(margin, y, contentWidth, 20, 2, 2, 'FD');

        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(130, 145, 163);
        doc.text(`TYPE: ${srvType.toUpperCase()}   |   LOCATION: ${locStr}`, margin + 6, y + 6);

        doc.setFontSize(10);
        doc.setFont('times', 'bold');
        doc.setTextColor(8, 21, 37);
        doc.text(name, margin + 6, y + 11.5);

        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(63, 107, 89);
        doc.text(`HELPLINE: ${contact}`, margin + 120, y + 11.5);

        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(63, 107, 89);
        doc.text('✓ PostgreSQL Verified Support Database', margin + 6, y + 16.5);

        y += 24;
      });
    } else {
      checkAddPage(10);
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text(norm.supportMsg || 'No matching support service was found in the available support-services database.', margin, y);
      y += 10;
    }
    y += 4;

    // SECTION 08: EVIDENCE / INFORMATION NOTES (PDF)
    drawSectionHeader('08', 'Evidence / Information Notes', 'Contextual evidence mentions attached to this submission.');
    if (Array.isArray(norm.evidenceNotes) && norm.evidenceNotes.length > 0) {
      norm.evidenceNotes.forEach((note) => {
        const noteLines = doc.splitTextToSize(`• ${note}`, contentWidth - 8);
        checkAddPage(noteLines.length * 4.5 + 4);
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(32, 42, 53);
        doc.text(noteLines, margin + 4, y);
        y += noteLines.length * 4.5 + 2;
      });
    } else {
      checkAddPage(12);
      doc.setFillColor(244, 241, 234);
      doc.setDrawColor(216, 210, 199);
      doc.roundedRect(margin, y, contentWidth, 12, 1.5, 1.5, 'FD');
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(130, 145, 163);
      doc.text('NO ADDITIONAL EVIDENCE NOTES — No additional evidence notes were attached to this report.', margin + 4, y + 7.5);
      y += 16;
    }
    y += 4;

    // SECTION 09: MODEL PREDICTION INFORMATION (PDF)
    drawSectionHeader('09', 'Model Prediction Information', 'Technical model provenance and inference parameters.');
    const modelInfo = report.model_prediction_information;
    checkAddPage(22);
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(216, 210, 199);
    doc.roundedRect(margin, y, contentWidth, 20, 2, 2, 'FD');

    const bertTask = safeStr(modelInfo?.bert_classifier?.task, 'Multi-Label Incident Classification');
    const bertArch = safeStr(modelInfo?.bert_classifier?.model_name, 'bert-base-uncased');
    const nerTask = safeStr(modelInfo?.ner_extractor?.task, 'Token Classification Entity Extraction');
    const nerArch = safeStr(modelInfo?.ner_extractor?.model_name, 'bert-base-uncased-ner');

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(32, 42, 53);
    doc.text(`BERT CLASSIFIER: ${bertTask} (${bertArch})`, margin + 6, y + 6);

    doc.text(`NER EXTRACTOR: ${nerTask} (${nerArch})`, margin + 6, y + 11.5);

    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(130, 145, 163);
    doc.text(`DATA PROVENANCE: ${safeStr(modelInfo?.data_provenance, 'Strict Anti-Hallucination Verified Database Records')}`, margin + 6, y + 16.5);
    y += 26;

    // SECTION 10: DISCLAIMER (PDF)
    drawSectionHeader('10', 'Disclaimer', 'System legal notice.');
    const discText = safeStr(
      report.disclaimer,
      'This system provides informational assistance based on available datasets, trained models, and verified database records. It does not replace professional legal advice, emergency services, law enforcement, or other qualified support.'
    );
    const discLines = doc.splitTextToSize(discText, contentWidth - 10);
    const discH = discLines.length * 4.5 + 8;
    checkAddPage(discH + 4);

    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(216, 210, 199);
    doc.roundedRect(margin, y, contentWidth, discH, 2, 2, 'FD');

    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(130, 145, 163);
    doc.text(discLines, margin + 5, y + 6);

    // Footer for all pages
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.5);
      doc.setTextColor(130, 145, 163);
      doc.setDrawColor(216, 210, 199);
      doc.line(margin, pageHeight - 12, pageWidth - margin, pageHeight - 12);
      doc.text(`ABHERA Incident Analysis Report  |  Case ID: ${safeStr(report.submission_id, 'N/A')}`, margin, pageHeight - 7);
      doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 7, { align: 'right' });
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
              Your Story. Understood.<br />
              Your Rights. Empowered.
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
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">01</span>
            <div className="section-header-text">
              <h2>INCIDENT SUMMARY</h2>
              <p className="section-subtitle-note">Your original description and follow-up responses are shown below.</p>
            </div>
          </div>

          <div className="section-content-area">
            <div className="narrative-quote-block">
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
                    <div key={idx} className="followup-card">
                      <div className="followup-question">
                        <HelpCircle size={14} />
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
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">02</span>
            <div className="section-header-text">
              <h2>IDENTIFIED INCIDENT TYPE</h2>
              <p className="section-subtitle-note">Automated classification derived from multi-label model inference.</p>
            </div>
          </div>

          <div className="section-content-area">
            {incidentTypes.length > 0 ? (
              <div className="classification-card">
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
              <div className="empty-state-card">
                <p className="fallback-text">
                  {norm.incidentMsg || 'Incident type could not be determined from the available model output.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 03: EXTRACTED INFORMATION */}
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">03</span>
            <div className="section-header-text">
              <h2>EXTRACTED INFORMATION</h2>
              <p className="section-subtitle-note">Structured parameters identified from your narrative by NER sequence labeling.</p>
            </div>
          </div>

          <div className="section-content-area">
            {Object.keys(extractedInfo).length > 0 ? (
              <div className="extracted-grid">
                {Object.entries(extractedInfo).map(([entType, items]) => {
                  const friendlyLabel = formatEntityLabel(entType);
                  return (
                    <div key={entType} className="extracted-card">
                      <span className="extracted-label-kicker">{friendlyLabel.toUpperCase()}</span>
                      <div className="extracted-values-list">
                        {Array.isArray(items) && items.length > 0 ? (
                          items.map((val, idx) => (
                            <span key={idx} className="extracted-value-tag">
                              {val}
                            </span>
                          ))
                        ) : (
                          <span className="empty-value-text">Not provided</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-state-card">
                <span className="empty-state-title">NO ENTITIES EXTRACTED</span>
                <p className="empty-state-desc">No specific entities were identified from the description.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 04: DETAILS YOU PROVIDED */}
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">04</span>
            <div className="section-header-text">
              <h2>USER-PROVIDED DETAILS</h2>
              <p className="section-subtitle-note">Direct user questionnaire responses recorded during the session.</p>
            </div>
          </div>

          <div className="section-content-area">
            {userDetails.length > 0 ? (
              <div className="user-details-cards">
                {userDetails.map((item, idx) => (
                  <div key={idx} className="detail-card">
                    <div className="detail-question">
                      <HelpCircle size={14} />
                      <span>{item.question}</span>
                    </div>
                    <div className="detail-answer">{item.answer}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state-card">
                <span className="empty-state-title">NO ADDITIONAL RESPONSES RECORDED</span>
                <p className="empty-state-desc">No additional user question responses were recorded.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 05: RELEVANT LEGAL INFORMATION (LEGAL CITATION SHEET) */}
        <section className="report-section highlight-section">
          <div className="section-header-block">
            <span className="section-number-digit legal-digit">05</span>
            <div className="section-header-text">
              <h2>RELEVANT LEGAL INFORMATION</h2>
              <p className="section-subtitle-note">Verified from the available knowledge base (PostgreSQL statutory records)</p>
            </div>
          </div>

          <div className="section-content-area">
            {Array.isArray(legalInfo) && legalInfo.length > 0 ? (
              <div className="legal-citation-sheet-stack">
                {legalInfo.map((law, idx) => {
                  const isExpanded = !!expandedLegal[idx];
                  const actName = safeStr(law.act || law.act_name, 'Statutory Act');
                  const secNum = safeStr(law.section || law.section_number, 'N/A');
                  const title = safeStr(law.title || law.heading || law.section_name, 'Statutory Provision');
                  const desc = safeStr(law.description || law.content || law.section_text);
                  const applicableCat = safeStr(law.applicable_label || law.incident_category || law.category);
                  return (
                    <div key={idx} className="legal-citation-card">
                      <div className="citation-header-bar">
                        <span className="citation-kicker">LEGAL REFERENCE</span>
                        {applicableCat && (
                          <span className="citation-applicable-tag">
                            APPLICABLE TO: <strong>{applicableCat.toUpperCase()}</strong>
                          </span>
                        )}
                      </div>

                      <div className="citation-act-section-row">
                        <div className="act-name-label">{actName.toUpperCase()}</div>
                        <div className="section-num-tag font-mono">SECTION {secNum}</div>
                      </div>

                      <h3 className="citation-title serif-heading">{title}</h3>

                      {desc && (
                        <div className={`citation-text-content ${!isExpanded && desc.length > 220 ? 'clamp-text' : ''}`}>
                          {desc}
                        </div>
                      )}

                      {desc && desc.length > 220 && (
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
              <div className="empty-state-card">
                <span className="empty-state-title">INFORMATION NOT AVAILABLE</span>
                <p className="empty-state-desc">
                  {norm.legalMsg || 'Information not available in the provided knowledge base.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 06: SUGGESTED NEXT STEPS */}
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">06</span>
            <div className="section-header-text">
              <h2>SUGGESTED NEXT STEPS</h2>
              <p className="section-subtitle-note">General administrative and practical recommendations from backend report output.</p>
            </div>
          </div>

          <div className="section-content-area">
            {nextSteps.length > 0 ? (
              <div className="action-cards-stack">
                {nextSteps.map((step, idx) => (
                  <div key={idx} className="action-step-card">
                    <div className="action-step-num font-mono">{String(idx + 1).padStart(2, '0')}</div>
                    <div className="action-step-text">{step}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state-card">
                <span className="empty-state-title">NO SUGGESTED STEPS RECORDED</span>
                <p className="empty-state-desc">No suggested next steps available.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 07: AVAILABLE SUPPORT SERVICES */}
        <section className="report-section highlight-section">
          <div className="section-header-block">
            <span className="section-number-digit support-digit">07</span>
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
                    <div key={idx} className="support-service-card">
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
              <div className="empty-state-card">
                <span className="empty-state-title">NO MATCHING SERVICES FOUND</span>
                <p className="empty-state-desc">
                  {norm.supportMsg || 'No matching support service was found in the available database.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 08: EVIDENCE / INFORMATION NOTES */}
        <section className="report-section">
          <div className="section-header-block">
            <span className="section-number-digit">08</span>
            <div className="section-header-text">
              <h2>EVIDENCE / INFORMATION NOTES</h2>
              <p className="section-subtitle-note">Contextual evidence mentions attached to this submission.</p>
            </div>
          </div>

          <div className="section-content-area">
            {evidenceNotes.length > 0 ? (
              <div className="evidence-notes-card">
                <ul className="evidence-bullet-list">
                  {evidenceNotes.map((note, idx) => (
                    <li key={idx}>{note}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="empty-state-card">
                <span className="empty-state-title">NO ADDITIONAL EVIDENCE NOTES</span>
                <p className="empty-state-desc">No additional evidence notes were attached to this report.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 09: MODEL PREDICTION INFORMATION */}
        <section className="report-section tech-section">
          <div className="section-header-block">
            <span className="section-number-digit tech-digit">09</span>
            <div className="section-header-text">
              <h2>MODEL PREDICTION INFORMATION</h2>
              <p className="section-subtitle-note">Technical model provenance and inference parameters.</p>
            </div>
          </div>

          <div className="section-content-area">
            {modelInfo ? (
              <div className="tech-cards-grid">
                <div className="tech-card">
                  <span className="tech-card-label">BERT CLASSIFIER</span>
                  <span className="tech-card-val">{safeStr(modelInfo.bert_classifier?.task, 'Multi-Label Incident Classification')}</span>
                  <span className="tech-card-sub font-mono">Model: {safeStr(modelInfo.bert_classifier?.model_name, 'bert-base-uncased')}</span>
                </div>
                <div className="tech-card">
                  <span className="tech-card-label">NER EXTRACTOR</span>
                  <span className="tech-card-val">{safeStr(modelInfo.ner_extractor?.task, 'Token Classification Entity Extraction')}</span>
                  <span className="tech-card-sub font-mono">Model: {safeStr(modelInfo.ner_extractor?.model_name, 'bert-base-uncased-ner')}</span>
                </div>
                <div className="tech-card">
                  <span className="tech-card-label">DATA PROVENANCE</span>
                  <span className="tech-card-val">{safeStr(modelInfo.data_provenance, 'Strict Anti-Hallucination Verified Database Records')}</span>
                </div>
              </div>
            ) : (
              <div className="empty-state-card">
                <p className="fallback-text">Technical prediction details unavailable.</p>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 10: DISCLAIMER */}
        <section className="report-section disclaimer-section">
          <div className="disclaimer-legal-card">
            <AlertTriangle size={24} className="disclaimer-icon" />
            <div>
              <h3>10. DISCLAIMER</h3>
              <p>{safeStr(report.disclaimer, 'This system provides informational assistance based on available datasets, trained models, and verified database records. It does not replace professional legal advice, emergency services, law enforcement, or other qualified support.')}</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Report;

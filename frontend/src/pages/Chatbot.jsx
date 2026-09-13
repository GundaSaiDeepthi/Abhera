import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConversation } from '../context/ConversationContext';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import LoadingIndicator from '../components/LoadingIndicator';
import AbheraEmblem from '../components/AbheraEmblem';
import { startSession, sendChatMessage } from '../services/api';
import {
  RotateCcw,
  Info,
  FileText,
  AlertCircle,
  ArrowRight,
  Lock,
  CheckCircle2,
  HelpCircle,
  Check
} from 'lucide-react';

const INITIAL_ASSISTANT_MESSAGE = {
  id: 'init-msg',
  role: 'assistant',
  sender: 'assistant',
  content:
    'Hello. I am your ABHERA legal assistance assistant. Please describe what happened in your own words. I will help structure the incident details and connect applicable verified statutory information and support resources.',
  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
};

const Chatbot = () => {
  const navigate = useNavigate();
  const {
    sessionId,
    setSessionId,
    submissionId,
    setSubmissionId,
    messages,
    addMessage,
    predictedLabels,
    setPredictedLabels,
    entities,
    setEntities,
    currentQuestion,
    setCurrentQuestion,
    conversationStatus,
    setConversationStatus,
    report,
    setReport,
    resetConversation,
    loading,
    setLoading,
    error,
    setError,
  } = useConversation();

  const [initializingSession, setInitializingSession] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, error]);

  const ensureSession = useCallback(async () => {
    if (sessionId) return sessionId;

    setInitializingSession(true);
    try {
      const res = await startSession();
      setSessionId(res.session_id);
      setSubmissionId(res.submission_id);
      setConversationStatus(res.conversation_status || 'ACTIVE');
      setInitializingSession(false);
      return res.session_id;
    } catch (err) {
      setInitializingSession(false);
      setError('Unable to initialize session with backend. Please check network connection.');
      return null;
    }
  }, [sessionId, setSessionId, setSubmissionId, setConversationStatus, setError]);

  const handleSendMessage = async (text) => {
    if (!text.trim() || loading) return;

    setError(null);
    const cleanText = text.trim();

    // If previous conversation was COMPLETED, reset state to start a fresh intake/submission
    if (conversationStatus === 'COMPLETED') {
      resetConversation();
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      sender: 'user',
      content: cleanText,
      text: cleanText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    addMessage(userMessage);
    setLoading(true);

    try {
      let currentSessId = conversationStatus === 'COMPLETED' ? null : sessionId;
      if (!currentSessId) {
        currentSessId = await ensureSession();
      }

      if (!currentSessId) {
        setLoading(false);
        return;
      }

      const res = await sendChatMessage(currentSessId, cleanText);

      if (res.session_id) setSessionId(res.session_id);
      if (res.submission_id) setSubmissionId(res.submission_id);
      if (res.predicted_labels) setPredictedLabels(res.predicted_labels);
      if (res.entities) setEntities(res.entities);
      if (res.current_question !== undefined) setCurrentQuestion(res.current_question);
      if (res.conversation_status) setConversationStatus(res.conversation_status);
      if (res.report) setReport(res.report);

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        sender: 'assistant',
        content: res.assistant_message,
        text: res.assistant_message,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      addMessage(assistantMessage);
      setLoading(false);

      if (res.conversation_status === 'PROCESSING') {
        navigate('/processing');
      } else if (res.conversation_status === 'COMPLETED' && res.report) {
        navigate('/report');
      }
    } catch (err) {
      setLoading(false);
      setError('Unable to process your message right now. Please try again.');
    }
  };

  const handleReset = () => {
    resetConversation();
    setError(null);
  };

  const displayMessages = messages.length > 0 ? messages : [INITIAL_ASSISTANT_MESSAGE];
  const hasPredicted = predictedLabels && predictedLabels.length > 0;
  const hasEntities = entities && Object.keys(entities).length > 0;

  const isQuestioning = conversationStatus === 'QUESTIONING';
  const isProcessing = conversationStatus === 'PROCESSING';
  const isCompleted = conversationStatus === 'COMPLETED' || !!report;

  const currentStageText = isCompleted
    ? 'REPORT READY'
    : isProcessing
    ? 'ANALYZING INCIDENT'
    : isQuestioning
    ? 'INFORMATION COLLECTION'
    : 'NARRATIVE INTAKE';

  const workflowSteps = [
    { num: '01', title: 'NARRATIVE RECEIVED', state: messages.length > 0 ? 'completed' : 'active' },
    { num: '02', title: 'CATEGORIES IDENTIFIED', state: hasPredicted ? 'completed' : (messages.length > 0 ? 'active' : 'pending') },
    { num: '03', title: 'ENTITIES EXTRACTED', state: isQuestioning ? 'active' : (hasEntities || isProcessing || isCompleted ? 'completed' : 'pending') },
    { num: '04', title: 'STATUTORY MATCHING', state: isProcessing ? 'active' : (isCompleted ? 'completed' : 'pending') },
    { num: '05', title: 'SUPPORT SERVICES', state: isProcessing ? 'active' : (isCompleted ? 'completed' : 'pending') },
    { num: '06', title: 'INCIDENT REPORT', state: isCompleted ? 'completed' : 'pending' },
  ];

  return (
    <div className="workspace-page-container">
      {/* 1. INCIDENT ANALYSIS WORKSPACE HEADER */}
      <header className="workspace-header">
        <div className="header-brand-block">
          <AbheraEmblem size={24} />
          <div>
            <h1 className="header-main-title font-serif">ABHERA</h1>
            <span className="header-sub-tag font-sans">
               LEGAL ASSISTANCE SESSION
            </span>
          </div>
        </div>

        <div className="header-metadata-grid font-sans">
          <div className="meta-item">
            <span className="meta-label">SESSION</span>
            <span className="meta-val font-mono">{sessionId || 'INITIATING...'}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">STATUS</span>
            <span className="meta-val status-green">{conversationStatus || 'ACTIVE'}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">CURRENT STAGE</span>
            <span className="meta-val">{currentStageText}</span>
          </div>

          <button
            className="btn-reset-session"
            onClick={handleReset}
            title="Start a new conversation session"
          >
            <RotateCcw size={13} />
            <span>New Conversation</span>
          </button>
        </div>
      </header>

      {/* 2. MAIN WORKSPACE GRID */}
      <div className="workspace-body-grid">
        {/* LEFT / MAIN CONVERSATION DOCUMENT PANEL */}
        <div className="document-main-panel">
          {/* Emergency Disclaimer Banner */}
          <div className="doc-disclaimer-banner font-sans">
            <Info size={15} className="text-teal" />
            <span>
              ABHERA organizes incident information and matches verified statutory laws. For immediate emergencies, call 112 or 1091.
            </span>
          </div>

          {/* Report Ready Banner */}
          {(conversationStatus === 'COMPLETED' || report) && (
            <div className="report-ready-card">
              <div className="report-ready-content">
                <FileText size={20} className="text-navy" />
                <div>
                  <strong className="font-sans">Incident Analysis Report Prepared</strong>
                  <p>Your incident details have been mapped to verified legal provisions and support services.</p>
                </div>
              </div>
              <button className="btn-hero-primary" onClick={() => navigate('/report')}>
                <span>VIEW INCIDENT REPORT</span>
                <ArrowRight size={13} />
              </button>
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div className="error-doc-banner">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* DYNAMIC QUESTION VISUAL BLOCK */}
          {conversationStatus === 'QUESTIONING' && currentQuestion && (
            <div className="information-needed-block">
              <div className="block-header font-sans">
                <HelpCircle size={15} className="text-gold" />
                <span className="block-kicker">ABHERA NEEDS MORE CONTEXT</span>
              </div>
              <p className="block-intro font-serif">
                We need a little more information to complete the incident analysis.
              </p>
              <div className="question-quote-box">
                <p className="question-text font-sans">
                  "{typeof currentQuestion === 'object'
                    ? (currentQuestion.question_text || currentQuestion.text || currentQuestion.question || '')
                    : String(currentQuestion)}"
                </p>
              </div>
            </div>
          )}

          {/* Messages List Container */}
          <div className="messages-document-list">
            {displayMessages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            {(loading || initializingSession) && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>

          {/* Bottom Fixed Chat Input */}
          <ChatInput onSendMessage={handleSendMessage} disabled={loading || initializingSession} />
        </div>

        {/* RIGHT SIDEBAR: CASE UNDERSTANDING RAIL */}
        <aside className="report-progress-sidebar">
          <div className="sidebar-doc-card">
            <div className="card-top-header font-sans">
              <FileText size={15} className="text-navy" />
              <h3>CASE UNDERSTANDING</h3>
            </div>

            <div className="thin-divider"></div>

            {/* Workflow Steps Rail */}
            <div className="card-section">
              <span className="section-kicker font-sans">PROGRESS STAGES</span>
              <div className="rail-list font-sans">
                {workflowSteps.map((ws) => (
                  <div key={ws.num} className={`rail-item ${ws.state}`}>
                    <span className="rail-num font-mono">{ws.num}</span>
                    <span className="rail-title font-mono">{ws.title}</span>
                    <span className="rail-icon">
                      {ws.state === 'completed' ? (
                        <Check size={13} className="text-teal" />
                      ) : ws.state === 'active' ? (
                        '●'
                      ) : (
                        '○'
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="thin-divider"></div>

            {/* Identified Incident Categories */}
            <div className="card-section">
              <span className="section-kicker font-sans">IDENTIFIED CATEGORIES</span>
              {hasPredicted ? (
                <div className="category-tags-wrap font-sans">
                  {predictedLabels.map((lbl, idx) => {
                    const labelText = typeof lbl === 'object' ? (lbl.name || lbl.code || lbl.label || String(lbl)) : String(lbl);
                    return (
                      <div key={idx} className="category-badge">
                        <CheckCircle2 size={12} className="text-teal" />
                        <span>{labelText}</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="empty-rail-text font-serif">Awaiting narrative input...</p>
              )}
            </div>

            {/* Extracted Entity Parameters */}
            <div className="card-section">
              <span className="section-kicker font-sans">EXTRACTED PARAMETERS</span>
              {hasEntities ? (
                <div className="entities-list font-sans">
                  {Object.entries(entities).map(([key, vals]) => {
                    const displayVals = Array.isArray(vals)
                      ? vals.map(v => typeof v === 'object' ? (v.text || v.word || JSON.stringify(v)) : String(v)).join(', ')
                      : String(vals);
                    if (!displayVals) return null;
                    return (
                      <div key={key} className="entity-row">
                        <span className="entity-key">{key}:</span>
                        <span className="entity-val">{displayVals}</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="empty-rail-text font-serif">NER sequence labeling active...</p>
              )}
            </div>

            <div className="thin-divider"></div>

            <div className="sidebar-footer-note font-sans">
              <Lock size={12} />
              <span>CONFIDENTIAL SESSION • Information used only for report synthesis.</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default Chatbot;

import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConversation } from '../context/ConversationContext';
import AbheraEmblem from '../components/AbheraEmblem';
import {
  CheckCircle2,
  Tag,
  BookOpen,
  ShieldCheck,
  FileText,
  MessageSquare,
  Check
} from 'lucide-react';

const Processing = () => {
  const navigate = useNavigate();
  const {
    conversationStatus,
    report,
    predictedLabels,
    entities,
    loading
  } = useConversation();

  useEffect(() => {
    if (conversationStatus === 'COMPLETED' && report) {
      navigate('/report');
    }
    else if (conversationStatus === 'QUESTIONING' || (conversationStatus === 'ACTIVE' && !loading && !report)) {
      navigate('/chatbot');
    }
  }, [conversationStatus, report, loading, navigate]);

  const hasLabels = predictedLabels && predictedLabels.length > 0;
  const hasEntities = entities && Object.keys(entities).length > 0;
  const isCompleted = conversationStatus === 'COMPLETED' || !!report;

  return (
    <div className="processing-page-wrapper">
      <div className="processing-card-container">
        {/* Header Block */}
        <div className="processing-header center">
          <div className="processing-brand-badge">
            <AbheraEmblem size={20} />
            <span>ABHERA • Your Story. Understood. Your Rights. Empowered.</span>
          </div>

          <h1 className="processing-title serif-heading">
            UNDERSTANDING YOUR STORY
          </h1>
          
          <p className="processing-subtitle">
            ABHERA is organizing the information provided in your consultation.
          </p>
        </div>

        {/* Legal Progress Rule */}
        <div className="document-analysis-progress-bar">
          <div className={`progress-line-fill ${isCompleted ? 'full' : 'animating'}`}></div>
        </div>

        {/* Legal Case Workflow List — 6 Steps */}
        <div className="processing-timeline-list">
          <div className="timeline-step-card completed">
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">01</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <MessageSquare size={18} />
                  <h3 className="timeline-step-title">01 INCIDENT ANALYSIS</h3>
                </div>
                <span className="timeline-step-desc">Parsed narrative text and user statement.</span>
              </div>
            </div>
            <div className="timeline-right"><Check size={18} className="check-green-icon" /></div>
          </div>

          <div className={`timeline-step-card ${hasLabels || isCompleted ? 'completed' : 'active'}`}>
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">02</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <BookOpen size={18} />
                  <h3 className="timeline-step-title">02 ENTITY RECOGNITION</h3>
                </div>
                <span className="timeline-step-desc">
                  {hasLabels
                    ? `Categorized: ${predictedLabels.map(l => typeof l === 'object' ? (l.name || l.code || String(l)) : String(l)).join(' + ')}`
                    : 'Classified incident categories using multi-label BERT transformer.'}
                </span>
              </div>
            </div>
            <div className="timeline-right">
              {hasLabels || isCompleted ? <Check size={18} className="check-green-icon" /> : <div className="document-pulse-ring"><span className="pulse-core"></span></div>}
            </div>
          </div>

          <div className={`timeline-step-card ${hasEntities || isCompleted ? 'completed' : (hasLabels ? 'active' : 'pending')}`}>
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">03</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <Tag size={18} />
                  <h3 className="timeline-step-title">03 CONTEXT UNDERSTANDING</h3>
                </div>
                <span className="timeline-step-desc">
                  {hasEntities
                    ? `Extracted parameters: ${Object.keys(entities).join(', ')}`
                    : 'Extracted key situational entities using NER token classification.'}
                </span>
              </div>
            </div>
            <div className="timeline-right">
              {hasEntities || isCompleted ? <Check size={18} className="check-green-icon" /> : <div className="document-pulse-ring"><span className="pulse-core"></span></div>}
            </div>
          </div>

          <div className={`timeline-step-card ${isCompleted ? 'completed' : (hasEntities ? 'active' : 'pending')}`}>
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">04</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <CheckCircle2 size={18} />
                  <h3 className="timeline-step-title">04 LEGAL MAPPING</h3>
                </div>
                <span className="timeline-step-desc">Queried statutory legal provisions directly from PostgreSQL database.</span>
              </div>
            </div>
            <div className="timeline-right">
              {isCompleted ? <Check size={18} className="check-green-icon" /> : <div className="document-pulse-ring"><span className="pulse-core"></span></div>}
            </div>
          </div>

          <div className={`timeline-step-card ${isCompleted ? 'completed' : (hasEntities ? 'active' : 'pending')}`}>
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">05</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <ShieldCheck size={18} />
                  <h3 className="timeline-step-title">05 SUPPORT MAPPING</h3>
                </div>
                <span className="timeline-step-desc">Mapped verified state and national helpline directory records.</span>
              </div>
            </div>
            <div className="timeline-right">
              {isCompleted ? <Check size={18} className="check-green-icon" /> : <div className="document-pulse-ring"><span className="pulse-core"></span></div>}
            </div>
          </div>

          <div className={`timeline-step-card ${isCompleted ? 'completed' : 'pending'}`}>
            <div className="timeline-left">
              <span className="timeline-step-num font-mono">06</span>
              <div className="timeline-text">
                <div className="timeline-title-row">
                  <FileText size={18} />
                  <h3 className="timeline-step-title">06 REPORT PREPARATION</h3>
                </div>
                <span className="timeline-step-desc">Assembled structured, non-generative Incident Analysis Report.</span>
              </div>
            </div>
            <div className="timeline-right">
              {isCompleted ? <Check size={18} className="check-green-icon" /> : <div className="pending-dot" />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Processing;

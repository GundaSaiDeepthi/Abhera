import React from 'react';
import { ShieldCheck, CheckCircle2, Database, BookOpen, Lock } from 'lucide-react';
import AbheraEmblem from './AbheraEmblem';

/**
 * HeroIllustration Component
 * Confidential Legal Case File Document Visual
 * Paper white container, thin warm border, gold accent line, navy typography, burgundy status marker.
 */
const HeroIllustration = ({ className = '' }) => {
  return (
    <div className={`hero-case-file-wrapper ${className}`}>
      <div className="digital-case-file-card">
        {/* Confidential Stamp / Header Bar */}
        <div className="case-file-top-bar">
          <div className="case-file-brand-lockup">
            <AbheraEmblem size={24} />
            <div>
              <span className="brand-wordmark-mini font-serif">ABHERA</span>
              <span className="case-file-kicker">CONFIDENTIAL ASSISTANCE</span>
            </div>
          </div>
          <div className="status-burgundy-tag">
            <span className="status-dot-pulse"></span>
            <span>STATUS: ACTIVE</span>
          </div>
        </div>

        <div className="thin-gold-line"></div>

        {/* Metadata Header */}
        <div className="case-file-sub-header">
          <span className="meta-kicker font-mono">CONFIDENTIAL CASE FILE • REF 2026</span>
          <h3 className="case-file-headline font-serif">CASE UNDERSTANDING</h3>
        </div>

        {/* 4 Key Processing Status Cards */}
        <div className="case-file-items-list">
          <div className="case-file-item-row">
            <div className="item-row-left">
              <CheckCircle2 size={16} className="text-verified" />
              <span className="item-title font-sans">Incident Analysis</span>
            </div>
            <span className="item-badge-verified">Verified</span>
          </div>

          <div className="case-file-item-row">
            <div className="item-row-left">
              <CheckCircle2 size={16} className="text-verified" />
              <span className="item-title font-sans">Information Extraction</span>
            </div>
            <span className="item-badge-verified">Verified</span>
          </div>

          <div className="case-file-item-row">
            <div className="item-row-left">
              <Database size={16} className="text-navy" />
              <span className="item-title font-sans">Legal Information</span>
            </div>
            <span className="item-badge-db font-mono">Database-backed</span>
          </div>

          <div className="case-file-item-row">
            <div className="item-row-left">
              <BookOpen size={16} className="text-navy" />
              <span className="item-title font-sans">Support Services</span>
            </div>
            <span className="item-badge-available font-mono">Available</span>
          </div>
        </div>

        <div className="thin-gold-line"></div>

        {/* Case File Footer */}
        <div className="case-file-footer font-sans">
          <div className="seal-lockup">
            <Lock size={13} className="text-gold" />
            <span>SESSION ENCRYPTED & GROUNDED</span>
          </div>
          <span className="confidential-small-text font-mono">NON-GENERATIVE REPORT</span>
        </div>
      </div>
    </div>
  );
};

export default HeroIllustration;

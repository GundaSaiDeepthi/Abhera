import React, { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Shield,
  MessageSquare,
  Cpu,
  Tag,
  HelpCircle,
  BookOpen,
  ShieldCheck,
  Lock,
  FileText,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Check,
  Scale,
  Search,
  Database
} from 'lucide-react';
import AbheraEmblem from '../components/AbheraEmblem';
import HeroIllustration from '../components/HeroIllustration';

const Home = () => {
  const location = useLocation();

  useEffect(() => {
    if (location.state?.scrollTo) {
      const el = document.getElementById(location.state.scrollTo);
      if (el) {
        setTimeout(() => {
          el.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      }
    }
  }, [location.state]);

  const processStages = [
    {
      step: '01',
      title: 'DESCRIBE',
      description: 'Share your incident in your own words through our simple, guided narrative interface.',
      icon: <MessageSquare size={20} />,
    },
    {
      step: '02',
      title: 'UNDERSTAND',
      description: 'Multi-label BERT model categorizes incident types while NER extracts key situational details.',
      icon: <Cpu size={20} />,
    },
    {
      step: '03',
      title: 'CLARIFY',
      description: 'The dynamic question engine generates targeted follow-ups only for missing required details.',
      icon: <HelpCircle size={20} />,
    },
    {
      step: '04',
      title: 'MATCH',
      description: 'Strict anti-hallucination layer retrieves verified statutory provisions and helplines from PostgreSQL.',
      icon: <BookOpen size={20} />,
    },
    {
      step: '05',
      title: 'REPORT',
      description: 'Receive an official 10-section Incident Analysis Report available for view and vector PDF export.',
      icon: <FileText size={20} />,
    },
  ];

  const capabilityBlocks = [
    {
      num: '01',
      title: 'INCIDENT UNDERSTANDING',
      description: 'Natural language classification accurately identifies complex, multi-label harassment and safety incidents.',
      icon: <Search size={20} />,
    },
    {
      num: '02',
      title: 'DYNAMIC CLARIFICATION',
      description: 'Context-aware question engine asks only for critical details missing from your description.',
      icon: <HelpCircle size={20} />,
    },
    {
      num: '03',
      title: 'LEGAL INFORMATION',
      description: 'Grounds all legal references directly in statutory Acts (BNS, IPC, POSH, IT Act, DV Act) from our database.',
      icon: <Scale size={20} />,
    },
    {
      num: '04',
      title: 'SUPPORT RESOURCES',
      description: 'Maps relevant state and national emergency helplines, One Stop Centres, and support organizations.',
      icon: <ShieldCheck size={20} />,
    },
  ];

  return (
    <div className="home-page-container">
      {/* 1. INSTITUTIONAL HERO SECTION */}
      <section className="hero-section">
        <div className="hero-grid">
          {/* HERO LEFT: LEGAL INFORMATION & RIGHTS ASSISTANCE */}
          <div className="hero-content">
            <div className="hero-trust-badge">
              <Shield size={13} />
              <span>AI-ASSISTED LEGAL GUIDANCE</span>
            </div>

            <h1 className="hero-main-title">
              <span className="brand-name">ABHERA</span>
              <span className="tagline-part serif-heading">Your Story. Understood.</span>
              <span className="tagline-part serif-heading tagline-secondary">Your Rights. Empowered.</span>
            </h1>

            <p className="hero-description">
              Describe what happened in your own words. ABHERA helps structure your experience, identify relevant information, and connect it with verified legal and support resources.
            </p>

            <div className="hero-cta-group">
              <Link to="/chatbot" className="btn-primary">
                <span>BEGIN YOUR STORY</span>
                <ArrowRight size={16} />
              </Link>

              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' });
                }}
              >
                <span>EXPLORE HOW IT WORKS</span>
              </button>
            </div>

            <div className="hero-trust-indicators">
              <div className="trust-item">
                <CheckCircle2 size={16} className="check-teal" />
                <span>Verified Statutory Knowledge Base</span>
              </div>
              <div className="trust-item">
                <CheckCircle2 size={16} className="check-teal" />
                <span>Strict Anti-Hallucination Architecture</span>
              </div>
            </div>
          </div>

          {/* HERO RIGHT VISUAL: EDITORIAL HUMAN ILLUSTRATION PANEL */}
          <div className="hero-visual-panel">
            <HeroIllustration />
          </div>
        </div>
      </section>

      {/* 2. SECTION 01 — HOW IT WORKS */}
      <section className="home-section bg-soft-paper" id="how-it-works">
        <div className="home-section-inner">
          <div className="section-header center">
            <div className="legal-kicker">01 — HOW IT WORKS</div>
            <h2 className="section-title serif-heading">How ABHERA Understands Your Story</h2>
            <p className="section-subtitle">
              From initial narrative description to structured statutory information and support service identification.
            </p>
          </div>

          <div className="journey-timeline-container">
            <div className="journey-line"></div>
            <div className="journey-steps-grid five-steps">
              {processStages.map((item, idx) => (
                <div key={idx} className="journey-step-card">
                  <div className="journey-step-top">
                    <div className="journey-num-badge font-mono">{item.step}</div>
                    <div className="journey-icon-box">{item.icon}</div>
                  </div>
                  <h3 className="journey-step-title">{item.title}</h3>
                  <p className="journey-step-desc">{item.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 3. SECTION 02 — WHAT ABHERA CAN DO */}
      <section className="home-section bg-document-ivory" id="capabilities">
        <div className="home-section-inner">
          <div className="section-header center">
            <div className="legal-kicker">02 — WHAT ABHERA CAN DO</div>
            <h2 className="section-title serif-heading">Designed to Understand. Built to Empower.</h2>
            <p className="section-subtitle">
              Four core capabilities designed to help women understand their rights and identify relevant statutory provisions and support services.
            </p>
          </div>

          <div className="capabilities-grid four-grid">
            {capabilityBlocks.map((cap, idx) => (
              <div key={idx} className={`capability-card cap-variant-${idx + 1}`}>
                <div className="cap-top-bar">
                  <span className="cap-num font-mono">{cap.num}</span>
                  <div className="cap-icon-box">{cap.icon}</div>
                </div>
                <h3 className="capability-title font-sans">{cap.title}</h3>
                <p className="capability-description">{cap.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 4. SECTION 03 — WHY THE SYSTEM IS DIFFERENT */}
      <section className="home-section bg-soft-paper">
        <div className="home-section-inner">
          <div className="section-header center">
            <div className="legal-kicker">03 — SYSTEM ARCHITECTURE</div>
            <h2 className="section-title serif-heading">ABHERA Technology Foundation</h2>
            <p className="section-subtitle">
              Model intelligence predicts incident categories; database verification provides the legal and support information.
            </p>
          </div>

          <div className="architecture-strip">
            <div className="arch-node">
              <Cpu size={20} className="arch-icon" />
              <span className="arch-label font-mono">BERT</span>
              <span className="arch-sub">Classification</span>
            </div>
            <div className="arch-plus">+</div>
            <div className="arch-node">
              <Tag size={20} className="arch-icon" />
              <span className="arch-label font-mono">NER</span>
              <span className="arch-sub">Entities</span>
            </div>
            <div className="arch-plus">+</div>
            <div className="arch-node">
              <HelpCircle size={20} className="arch-icon" />
              <span className="arch-label font-mono">Dynamic Engine</span>
              <span className="arch-sub">Missing Context</span>
            </div>
            <div className="arch-plus">+</div>
            <div className="arch-node">
              <Database size={20} className="arch-icon" />
              <span className="arch-label font-mono">PostgreSQL</span>
              <span className="arch-sub">Knowledge Base</span>
            </div>
            <div className="arch-equals">=</div>
            <div className="arch-result-node">
              <FileText size={22} className="arch-icon-gold" />
              <span className="arch-result-label font-mono">Structured Incident Analysis</span>
            </div>
          </div>
        </div>
      </section>

      {/* 5. SECTION 04 — SAFETY & PRIVACY */}
      <section className="home-section bg-document-ivory" id="safety">
        <div className="home-section-inner">
          <div className="section-header center" style={{ marginBottom: '2rem' }}>
            <div className="legal-kicker">04 — SAFETY & PRIVACY</div>
            <h2 className="section-title serif-heading">Privacy, Clarity and Verified Information</h2>
          </div>

          <div className="principles-cards-grid">
            <div className="principle-card">
              <div className="principle-card-header">
                <span className="principle-num font-mono">01</span>
                <Lock size={18} className="principle-icon" />
                <h3 className="principle-title">USER-CONTROLLED DATA</h3>
              </div>
              <p className="principle-desc">
                Your incident details remain under your session control and are used strictly to structure information and retrieve relevant legal resources.
              </p>
            </div>

            <div className="principle-card">
              <div className="principle-card-header">
                <span className="principle-num font-mono">02</span>
                <BookOpen size={18} className="principle-icon" />
                <h3 className="principle-title">KNOWLEDGE-BASE GROUNDED</h3>
              </div>
              <p className="principle-desc">
                All statutory legal provisions and helpline contacts are drawn directly from our empirical PostgreSQL database without generative fabrication.
              </p>
            </div>

            <div className="principle-card">
              <div className="principle-card-header">
                <span className="principle-num font-mono">03</span>
                <ShieldCheck size={18} className="principle-icon" />
                <h3 className="principle-title">INFORMATIONAL ASSISTANCE</h3>
              </div>
              <p className="principle-desc">
                ABHERA provides legal information and rights assistance. It does not provide formal legal representation, legal advice from a lawyer, or establish an attorney-client relationship.
              </p>
            </div>
          </div>

          <div className="disclaimer-card" style={{ marginTop: '2.5rem' }}>
            <div className="disclaimer-icon-box">
              <AlertTriangle size={22} />
            </div>
            <div className="disclaimer-text">
              <h3>Important Legal Notice & Emergency Assistance</h3>
              <p>
                ABHERA is an AI-assisted legal safety information platform designed to help organize incident details and map verified statutory provisions. It does <strong>not</strong> provide legal advice from a lawyer, legal consultation, or emergency intervention. In situations of immediate danger, please contact local emergency helplines (e.g., Women Helpline 1091 / Emergency 112) immediately.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 6. SECTION 05 — READY TO BEGIN FINAL CTA */}
      <section className="home-section final-cta-section">
        <div className="home-section-inner center">
          <div className="final-cta-card">
            <div className="legal-kicker gold-kicker">05 — READY TO BEGIN?</div>
            <h2 className="section-title serif-heading">Describe what happened in your own words.</h2>
            <p className="section-subtitle" style={{ maxWidth: '620px', margin: '0.75rem auto 1.75rem' }}>
              ABHERA helps organize your situation and connect it with verified legal and support information.
            </p>
            <Link to="/chatbot" className="btn-primary">
              <span>BEGIN YOUR STORY</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </section>

      {/* 6. INSTITUTIONAL FOOTER */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-top">
            <div className="footer-brand-col">
              <div className="footer-logo">
                <AbheraEmblem size={26} />
                <span>ABHERA</span>
              </div>
              <p className="footer-tagline serif-heading">
                "Your Story. Understood. Your Rights. Empowered."
              </p>
              <span className="footer-subkicker font-mono">WOMEN'S SAFETY & LEGAL INFORMATION SYSTEM</span>
            </div>

            <div className="footer-links-col">
              <h4>Navigation</h4>
              <Link to="/">Home</Link>
              <button type="button" onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}>
                How It Works
              </button>
              <button type="button" onClick={() => document.getElementById('capabilities')?.scrollIntoView({ behavior: 'smooth' })}>
                Capabilities
              </button>
              <button type="button" onClick={() => document.getElementById('safety')?.scrollIntoView({ behavior: 'smooth' })}>
                Safety & Privacy
              </button>
              <Link to="/chatbot">Start Assessment</Link>
            </div>

            <div className="footer-info-col">
              <h4>System Assurance</h4>
              <span>Verified Statutory Knowledge Base</span>
              <span>Strict Anti-Hallucination Architecture</span>
              <span>Grounded Non-Generative Incident Reports</span>
            </div>
          </div>

          <div className="footer-bottom">
            <p>© ABHERA — Informational assistance and rights guidance. Not formal legal advice from a lawyer.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;

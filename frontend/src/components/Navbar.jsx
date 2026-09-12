import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home as HomeIcon, BookOpen, ShieldCheck, Lock, Menu, X, ArrowRight } from 'lucide-react';
import AbheraEmblem from './AbheraEmblem';

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 20) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMobileMenu = () => setMobileMenuOpen((prev) => !prev);
  const closeMobileMenu = () => setMobileMenuOpen(false);

  const scrollToSection = (id) => {
    closeMobileMenu();
    if (location.pathname !== '/') {
      navigate('/', { state: { scrollTo: id } });
    } else {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <header className={`navbar-header ${scrolled ? 'scrolled' : ''}`}>
      <div className="navbar-container">
        {/* LEFT: Institutional Seal + Wordmark */}
        <Link to="/" className="navbar-brand" onClick={closeMobileMenu}>
          <div className="navbar-brand-icon">
            <AbheraEmblem size={26} />
          </div>
          <div className="navbar-brand-text">
            <span className="brand-title">ABHERA</span>
            <span className="brand-tagline">
              Your Story. Understood. Your Rights. Empowered.
            </span>
          </div>
        </Link>

        {/* Mobile Toggle Button */}
        <button
          className="mobile-menu-toggle"
          onClick={toggleMobileMenu}
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        {/* CENTER LINKS & RIGHT CTA */}
        <nav className={`navbar-nav ${mobileMenuOpen ? 'mobile-open' : ''}`}>
          <Link
            to="/"
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
            onClick={closeMobileMenu}
          >
            <HomeIcon size={14} />
            <span>Home</span>
          </Link>

          <button
            type="button"
            className="nav-link"
            onClick={() => scrollToSection('how-it-works')}
          >
            <BookOpen size={14} />
            <span>How It Works</span>
          </button>

          <button
            type="button"
            className="nav-link"
            onClick={() => scrollToSection('capabilities')}
          >
            <ShieldCheck size={14} />
            <span>Capabilities</span>
          </button>

          <button
            type="button"
            className="nav-link"
            onClick={() => scrollToSection('safety')}
          >
            <Lock size={14} />
            <span>Safety & Privacy</span>
          </button>

          <Link
            to="/chatbot"
            className="nav-btn-primary"
            onClick={closeMobileMenu}
          >
            <span>BEGIN YOUR STORY</span>
            <ArrowRight size={14} />
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;

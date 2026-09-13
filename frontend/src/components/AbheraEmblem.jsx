import React from 'react';

/**
 * ABHERA Symmetrical Balanced Legal Scales Emblem
 * Custom institutional emblem combining:
 * - Protective architectural arch outer structure ('A' silhouette)
 * - Perfectly balanced weighing scale beam & suspended scale pans (Justice & Equilibrium)
 * - Central vertical pillar of stability & authority
 * - Antique Gold (#B69A61) balance beam & fulcrum accents
 * - Open document / statutory book pedestal base
 * - Clean 24px/48px vector resolution
 */
const AbheraEmblem = ({ size = 28, className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 48 48"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`abhera-legal-emblem ${className}`}
    aria-label="ABHERA Balanced Legal Scales Emblem"
    role="img"
  >
    {/* 1. Outer Architectural Protective Arch ('A' Silhouette) */}
    <path
      d="M24 4C14.5 4 8.5 11.5 8.5 24V43H39.5V24C39.5 11.5 33.5 4 24 4Z"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />

    {/* 2. Inner Fine Arch Guide Rule */}
    <path
      d="M24 8C17 8 11.5 13.5 11.5 23V39.5H36.5V23C36.5 13.5 31 8 24 8Z"
      stroke="currentColor"
      strokeWidth="0.9"
      strokeOpacity="0.25"
      strokeDasharray="2 2"
    />

    {/* 3. Central Vertical Pillar of Authority */}
    <line
      x1="24"
      y1="10"
      x2="24"
      y2="38"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />

    {/* 4. BALANCED WEIGHING SCALE BEAM (Antique Gold Accent #B69A61) */}
    <line
      x1="11"
      y1="19"
      x2="37"
      y2="19"
      stroke="var(--antique-gold, #B3955A)"
      strokeWidth="2.4"
      strokeLinecap="round"
    />

    {/* 5. Center Fulcrum Diamond Pivot (Antique Gold) */}
    <polygon
      points="24,15.5 26.5,19 24,22.5 21.5,19"
      fill="var(--antique-gold, #B3955A)"
    />

    {/* 6. Left Scale Pan & Suspension Cords */}
    <line x1="14" y1="19" x2="11" y2="28" stroke="currentColor" strokeWidth="1.2" />
    <line x1="14" y1="19" x2="17" y2="28" stroke="currentColor" strokeWidth="1.2" />
    <path
      d="M9.5 28.5C9.5 31.5 18.5 31.5 18.5 28.5H9.5Z"
      fill="currentColor"
      fillOpacity="0.15"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinejoin="round"
    />

    {/* 7. Right Scale Pan & Suspension Cords (Symmetrical Equilibrium) */}
    <line x1="34" y1="19" x2="31" y2="28" stroke="currentColor" strokeWidth="1.2" />
    <line x1="34" y1="19" x2="37" y2="28" stroke="currentColor" strokeWidth="1.2" />
    <path
      d="M29.5 28.5C29.5 31.5 38.5 31.5 38.5 28.5H29.5Z"
      fill="currentColor"
      fillOpacity="0.15"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinejoin="round"
    />

    {/* 8. Apex Statutory Diamond Seal (Antique Gold Accent) */}
    <polygon
      points="24,5 26,8 24,11 22,8"
      fill="var(--antique-gold, #B3955A)"
    />

    {/* 9. Open Statutory Book / Pedestal Base */}
    <path
      d="M15 39.5L24 36L33 39.5V42L24 38.5L15 42V39.5Z"
      fill="var(--antique-gold, #B3955A)"
      fillOpacity="0.9"
    />
  </svg>
);

export default AbheraEmblem;

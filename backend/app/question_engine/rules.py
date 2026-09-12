"""
Rules Module for Dynamic Question Engine.

Defines category-to-required-entity rules, category normalization,
and multi-label information requirement combining logic.
"""

from typing import Dict, List, Set

# Category Code and Label Normalization Map
CATEGORY_MAP = {
    "DV": "Domestic Violence",
    "DOMESTIC VIOLENCE": "Domestic Violence",
    "ST": "Stalking",
    "STALKING": "Stalking",
    "SH": "Sexual Harassment",
    "SEXUAL HARASSMENT": "Sexual Harassment",
    "CA": "Cyber Harassment",
    "CYBER HARASSMENT": "Cyber Harassment",
    "CYBER ABUSE": "Cyber Harassment",
    "WH": "Workplace Harassment",
    "WORKPLACE HARASSMENT": "Workplace Harassment",
    "OV": "Other Harassment",
    "OTHER HARASSMENT": "Other Harassment",
    "OTHER VIOLENCE": "Other Harassment",
    "OTHER / GENERAL HARASSMENT": "Other Harassment",
    "DOWRY": "Dowry Harassment",
    "DOWRY HARASSMENT": "Dowry Harassment",
    "FPM": "False Promise of Marriage",
    "FALSE PROMISE OF MARRIAGE": "False Promise of Marriage",
    "GENERAL": "General",
}

# Entity / Information Dimensions required per category
CATEGORY_REQUIRED_ENTITIES: Dict[str, List[str]] = {
    "Domestic Violence": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "Dowry Harassment": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "Stalking": ["PERP_REL", "TIME_FREQ", "PLATFORM", "SAFETY_STATUS"],
    "Sexual Harassment": ["PERP_REL", "LOCATION", "INCIDENT_DETAILS"],
    "Cyber Harassment": ["PLATFORM", "TIME_FREQ", "INCIDENT_DETAILS"],
    "Workplace Harassment": ["PERP_REL", "LOCATION", "INCIDENT_DETAILS"],
    "Other Harassment": ["INCIDENT_DETAILS", "SAFETY_STATUS"],
    "False Promise of Marriage": ["PERP_REL", "TIME_FREQ", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "General": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
}

# Supervised NER Entities currently supported
SUPPORTED_NER_ENTITIES = {"PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM"}

# Extensible placeholder for future supervised NER classes
FUTURE_EXTENSIBLE_ENTITIES = {"EVIDENCE", "LAW_SEC"}

# Canonical Entity Priority Order (for consistent ordering of missing requirements)
ENTITY_PRIORITY_ORDER = [
    "SAFETY_STATUS",
    "PERP_REL",
    "LOCATION",
    "TIME_FREQ",
    "PLATFORM",
    "INCIDENT_DETAILS",
]


def normalize_category(category_name: str) -> str:
    """
    Normalizes category codes (e.g. 'DV', 'ST') or variants into canonical names.
    Defaults to 'General' if unknown.
    """
    if not category_name:
        return "General"
    cleaned = str(category_name).strip().upper()
    return CATEGORY_MAP.get(cleaned, str(category_name).strip().title())


def get_required_information(categories: List[str]) -> List[str]:
    """
    Determines combined required information dimensions for single or multi-label incident categories.

    Combines requirements from all active categories without duplication,
    preserving canonical priority order.
    """
    if not categories:
        categories = ["General"]

    required_set: Set[str] = set()

    for cat in categories:
        norm_cat = normalize_category(cat)
        reqs = CATEGORY_REQUIRED_ENTITIES.get(norm_cat, CATEGORY_REQUIRED_ENTITIES["General"])
        required_set.update(reqs)

    # Sort according to canonical priority order
    ordered_required = [ent for ent in ENTITY_PRIORITY_ORDER if ent in required_set]
    
    # Append any extra requirements not in canonical list
    for ent in required_set:
        if ent not in ordered_required:
            ordered_required.append(ent)

    return ordered_required

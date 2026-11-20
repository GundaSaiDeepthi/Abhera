"""
ABHERA — Input Relevance Validation Layer

Runs BEFORE BERT classification, NER, the Dynamic Question Engine, legal/support
mapping and report generation. It decides whether a user message describes a
potential women's-safety / harassment incident at all.

Why this layer exists
---------------------
The production BERT model (models/bert_multilabel_experiment_v2) was trained only
on harassment narratives (six labels DV, SH, ST, CA, WH, OV) and has NO
"not-an-incident" class. Its independent sigmoid outputs are therefore not
calibrated for out-of-domain text, e.g. (measured):

    "Today I went to the market to buy vegetables."  -> DV 0.78 (thr 0.73), OV 0.77 (thr 0.63)
    "Hello."                                         -> ST 0.81 (thr 0.79)
    "I am facing a problem."                         -> DV 0.81 (thr 0.73)

So BERT confidence alone can NOT be used to decide relevance. This module adds a
conservative, context-based relevance decision. The BERT model, its labels and
thresholds are left untouched.

Statuses (these are NOT BERT labels):
    INVALID_INPUT, OUT_OF_SCOPE, INSUFFICIENT_INFORMATION,
    UNCERTAIN_CLASSIFICATION, VALID_INCIDENT

Decision logic (summary)
------------------------
1. INVALID_INPUT  - empty / whitespace / no alphabetic content.
2. Greeting-only messages -> OUT_OF_SCOPE (with a greeting reply).
3. Contextual signals are computed with regex families (stems + phrase
   patterns, not a flat keyword list):
     * harm      - harmful/unwanted ACTIONS or conduct (threaten*, harass*,
                   follow(ing) me, unwanted ..., hit/beat me, blackmail*, ...)
     * persistence/refusal - "despite my requests", "keeps ...", "won't stop"
     * involvement - the narrator is affected (me/my/I was ...).
     * actor     - another person is involved (husband, colleague, someone, he ...)
     * distress  - vague concern without a concrete act ("problem", "issue",
                   "something happened", "uncomfortable", "need help")
   Common context words (market, home, office, friend, person) carry NO weight
   on their own.
4. harm + involvement                      -> VALID_INCIDENT
   harm without clear involvement          -> BERT consulted: confident label
                                              (prob >= 0.85 and >= its threshold)
                                              -> VALID_INCIDENT, else UNCERTAIN
   distress/actor-only, enough context     -> UNCERTAIN_CLASSIFICATION
   distress, short/no actor                -> INSUFFICIENT_INFORMATION
   no relevant signal                      -> OUT_OF_SCOPE (BERT is ignored,
                                              because it fires on unrelated text)

Limitations: rule-based signals are English-only, can miss highly indirect or
euphemistic descriptions (these fall to UNCERTAIN/INSUFFICIENT, which asks for
clarification rather than dropping the user), and do not understand negation
or hypotheticals ("in the film he threatened her"). See the implementation
report for details.
"""

import re
from typing import Any, Callable, Dict, List, Optional

INVALID_INPUT = "INVALID_INPUT"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
UNCERTAIN_CLASSIFICATION = "UNCERTAIN_CLASSIFICATION"
VALID_INCIDENT = "VALID_INCIDENT"

ALL_STATUSES = (
    INVALID_INPUT,
    OUT_OF_SCOPE,
    INSUFFICIENT_INFORMATION,
    UNCERTAIN_CLASSIFICATION,
    VALID_INCIDENT,
)

# Statuses for which the incident pipeline must NOT run.
NON_INCIDENT_STATUSES = (INVALID_INPUT, OUT_OF_SCOPE)

STATUS_MESSAGES = {
    INVALID_INPUT: "The message appears to be empty or unreadable. Please describe what happened in your own words.",
    OUT_OF_SCOPE: (
        "The description does not appear to contain information related to harassment or women's safety. "
        "Please describe an incident so that ABHERA can analyse it."
    ),
    INSUFFICIENT_INFORMATION: (
        "Could you provide a little more information about the situation? "
        "This will help ABHERA understand whether assistance is relevant."
    ),
    UNCERTAIN_CLASSIFICATION: (
        "The information provided is not sufficient to identify the nature of the incident. "
        "Could you clarify what happened?"
    ),
    VALID_INCIDENT: "",
}

GREETING_MESSAGE = (
    "Hello. I am ABHERA. Please describe the incident or situation you are facing in your own words, "
    "and I will help analyse it."
)

# BERT confidence required to accept a harm-signal message whose narrator involvement is unclear.
BERT_CONFIDENT_PROB = 0.85

# ---------------------------------------------------------------------------
# Signal pattern families
# ---------------------------------------------------------------------------
_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hii+|good\s+(morning|afternoon|evening|night)|namaste|greetings|"
    r"thanks?|thank\s+you|ok(ay)?|bye|yo)\b[\s!.,?]*(there|abhera|team)?[\s!.,?]*$",
    re.IGNORECASE,
)

_HARM_PATTERNS = [
    # harassment / abuse / threat families (stem-based)
    r"\bharass\w*", r"\babus\w*", r"\bthreat\w*", r"\bstalk\w*", r"\bmolest\w*", r"\bgrop\w*",
    r"\brap(e|ed|ing|ist)\b", r"\bsexual(ly)?\b", r"\bassault\w*", r"\bblackmail\w*", r"\bextort\w*",
    r"\bintimidat\w*", r"\bbull(y|ied|ying)\b", r"\bhumiliat\w*", r"\btortur\w*", r"\bcruel\w*",
    r"\bviolen\w*", r"\bdowry\b", r"\bmisbehav\w*", r"\beve[\s-]?teas\w*", r"\bcatcall\w*",
    r"\bobscene\b", r"\blewd\b", r"\bnude\w*", r"\bmorph(ed|ing)\b", r"\bperv\w*",
    r"\bmanhandl\w*", r"\bstrangl\w*", r"\bchok(e|ed|ing)\b", r"\bkidnap\w*", r"\bacid\b",
    # physical actions directed at a person
    # (verb must be directed AT a person, so "I hit my head on the door" / "hit the ball" do not match)
    r"\b(hit|hits|hitting|beat|beats|beating|slap\w*|kick\w*|punch\w*|push(es|ed|ing)?|shov\w*|burn(s|ed|t)?|"
    r"injur\w*|hurt\w*|harm\w*|kill\w*|attack\w*|touch(ed|es|ing)?|grab\w*|forc(e|ed|es|ing)|lock(s|ed)?|chok\w*)\s+"
    r"(me|her|us|my\s+(face|body|chest|daughter|son|child|children|mother|sister))\b",
    r"\b(i|she|we)\s+(was|were|got|have\s+been|had\s+been|am\s+being)\s+(hit|beaten|slapped|kicked|punched|pushed|"
    r"attacked|injured|hurt|touched|grabbed|burned|burnt|locked\s+up|forced)\b",
    # unwanted / non-consensual conduct
    r"\bunwanted\b", r"\bwithout\s+(my|her)\s+(consent|permission)\b", r"\binappropriate(ly)?\b",
    r"\bagainst\s+my\s+will\b", r"\bnon[\s-]?consensual\b",
    # following / surveillance directed at the narrator
    r"\bfollow(s|ed|ing)?\s+(me|her|us)\b", r"\bwatch(es|ed|ing)?\s+(me|my)\s+(every|all|constantly)",
    r"\b(spread(ing)?|post(ed|ing)?|leak(ed|ing)?|shar(ed|ing))\b.{0,30}\b(photos?|pictures?|pics|videos?|rumou?rs?)\b",
    r"\bfake\s+(profile|account)\b", r"\bhack(ed|ing)?\s+my\b",
    # pressure / control
    r"\bpressur\w*\s+(me|her)\b", r"\bdemand(s|ed|ing)?\s+(money|sex|sexual|favou?rs?)\b",
    r"\bnot\s+(let|letting|allow\w*)\s+me\b", r"\bpromised\s+to\s+marry\b", r"\bfalse\s+promise\b",
    # feeling unsafe tied to a person
    r"\b(unsafe|in\s+danger|scared\s+of|afraid\s+of|fear\s+for\s+my)\b",
]

_PERSISTENCE_PATTERNS = [
    r"\bdespite\s+my\b", r"\b(asked|told|begged|requested)\s+(him|her|them)\s+to\s+stop\b",
    r"\bwon'?t\s+(stop|leave\s+me\s+alone)\b", r"\b(refuses?|refused)\s+to\s+stop\b",
    r"\bkeeps?\s+(calling|messaging|texting|sending|coming|following|touching|staring|commenting)\b",
]

_INVOLVEMENT_RE = re.compile(
    r"\b(me|my|myself|i\s+(was|am|have|had|got|get|feel|received|receive|keep|kept|faced|face)|i'm|i've|we|us|our)\b",
    re.IGNORECASE,
)

_ACTOR_RE = re.compile(
    r"\b(husband|ex-?husband|wife|boyfriend|ex-?boyfriend|partner|ex|fianc[eé]e?|in-?laws?|"
    r"father|mother|brother|sister|uncle|cousin|relative|neighbou?r|colleague|co-?worker|boss|manager|"
    r"supervisor|employer|senior|teacher|professor|classmate|landlord|driver|stranger|someone|somebody|"
    r"a\s+(man|guy|boy|person|group)|men|he|him|his|they|them)\b",
    re.IGNORECASE,
)

_DISTRESS_PATTERNS = [
    r"\bproblem\w*\b", r"\bissue\w*\b", r"\btrouble\w*\b", r"\bsomething\s+(bad|wrong|happened|terrible)\b",
    r"\bhappened\s+to\s+me\b", r"\buncomfortable\b", r"\bupset\b", r"\bworried\b", r"\bdisturb\w*\b",
    r"\bneed\s+help\b", r"\bhelp\s+me\b", r"\bplease\s+help\b", r"\bnot\s+ok(ay)?\b", r"\bweird\b",
    r"\bbother\w*\b", r"\bstrange\b", r"\bsuffer\w*\b", r"\bscared\b", r"\bafraid\b", r"\bfear\w*\b",
    r"\bcomplain\w*\b", r"\bbad\s+(behaviou?r|experience|situation)\b",
]

_HARM_RES = [re.compile(p, re.IGNORECASE) for p in _HARM_PATTERNS]
# Explicit offence verbs that describe an incident even without first-person involvement.
_STRONG_OFFENCE_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(harass(ed|es|ing)|abus(ed|es|ing)|stalk(ed|s|ing)|molest(ed|s|ing)|grop(ed|es|ing)|"
    r"blackmail(ed|s|ing)|tortur(ed|es|ing)|threaten(ed|s|ing))\s+(her|him|them|his|their|a|an|[a-z])\b",
    r"\b(raped|sexually\s+assaulted|sexually\s+harassed)\b",
    r"\bdowry\s+(harassment|demand\w*|death|torture)\b",
    r"\b(falsely|false)\s+promise[ds]?\b",
]]
_PERSIST_RES = [re.compile(p, re.IGNORECASE) for p in _PERSISTENCE_PATTERNS]
_DISTRESS_RES = [re.compile(p, re.IGNORECASE) for p in _DISTRESS_PATTERNS]


def _matches(regexes: List[re.Pattern], text: str) -> List[str]:
    hits = []
    for r in regexes:
        m = r.search(text)
        if m:
            hits.append(m.group(0).strip())
    return hits


def extract_signals(text: str) -> Dict[str, Any]:
    """Computes contextual relevance signals for a message (no ML involved)."""
    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", text)
    return {
        "word_count": len(words),
        "harm": _matches(_HARM_RES, lower),
        "persistence": _matches(_PERSIST_RES, lower),
        "involvement": bool(_INVOLVEMENT_RE.search(lower)),
        "actor": bool(_ACTOR_RE.search(lower)),
        "distress": _matches(_DISTRESS_RES, lower),
        "strong_offence": _matches(_STRONG_OFFENCE_RES, lower),
    }


def _result(status: str, signals: Dict[str, Any], reason: str, bert: Optional[Dict[str, Any]] = None,
            message: Optional[str] = None, is_greeting: bool = False) -> Dict[str, Any]:
    return {
        "status": status,
        "proceed_with_analysis": status == VALID_INCIDENT,
        "message": message if message is not None else STATUS_MESSAGES[status],
        "reason": reason,
        "is_greeting": is_greeting,
        "signals": signals,
        "bert_check": bert,
    }


def _bert_confident(bert_predict: Optional[Callable[[str], Dict[str, Any]]], text: str) -> Optional[Dict[str, Any]]:
    """Runs the existing BERT predictor (unchanged) and reports whether any label is confidently predicted."""
    if bert_predict is None:
        return None
    try:
        res = bert_predict(text) or {}
    except Exception as e:  # pragma: no cover - defensive
        return {"available": False, "error": str(e)}
    scores = res.get("all_scores", {}) or {}
    thresholds = res.get("thresholds", {}) or {}
    confident = [
        code for code, p in scores.items()
        if code in ("DV", "SH", "ST", "CA", "WH", "OV")
        and p >= BERT_CONFIDENT_PROB and p >= float(thresholds.get(code, 0.5))
    ]
    return {"available": True, "confident_labels": confident, "max_score": max(scores.values()) if scores else 0.0}


def validate_input(text: Optional[str],
                   bert_predict: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Classifies a message into one of the relevance statuses.

    Args:
        text: raw user message.
        bert_predict: optional callable (e.g. BERTIncidentPredictor.predict). Only consulted for
            messages that contain harm signals but unclear narrator involvement. Never used to
            upgrade a message without any incident signal (BERT fires on unrelated text).
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return _result(INVALID_INPUT, {}, "empty input")

    clean = text.strip()
    if not re.search(r"[A-Za-z\u0900-\u097F]", clean):
        return _result(INVALID_INPUT, {}, "no alphabetic content")

    if _GREETING_RE.match(clean):
        return _result(OUT_OF_SCOPE, {}, "greeting only", message=GREETING_MESSAGE, is_greeting=True)

    s = extract_signals(clean)
    has_harm = bool(s["harm"]) or bool(s["persistence"])

    if has_harm and s["involvement"]:
        return _result(VALID_INCIDENT, s, "harmful/unwanted conduct affecting the narrator")

    if s["strong_offence"]:
        # Explicit offence description reported in third person (e.g. on behalf of someone else).
        return _result(VALID_INCIDENT, s, "explicit offence described (third-person report)")

    if has_harm:
        bert = _bert_confident(bert_predict, clean)
        if bert and bert.get("confident_labels"):
            return _result(VALID_INCIDENT, s, "harm signal confirmed by confident BERT prediction", bert)
        return _result(UNCERTAIN_CLASSIFICATION, s, "harm signal but unclear who was affected", bert)

    # Clarification is only requested when the user expresses a concern. A sentence that merely
    # mentions a person and "my" (e.g. "My mother cooked dinner") is NOT treated as a concern.
    if s["distress"]:
        if s["word_count"] >= 6 and s["actor"]:
            return _result(UNCERTAIN_CLASSIFICATION, s, "possible concern involving another person, no concrete act")
        return _result(INSUFFICIENT_INFORMATION, s, "vague concern without incident details")

    return _result(OUT_OF_SCOPE, s, "no harassment or safety-related context detected")

import os
import re
import time
import unicodedata
import hmac
import hashlib
import secrets
import numpy as np
import logging
import threading
import math
import json
import html
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
import folium
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. Page configuration, API key, constants
# ---------------------------------------------------------
st.set_page_config(page_title="RAHAT AI — Disaster Triage", page_icon="🚨", layout="wide")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:   # when hosted online, the key comes from Streamlit "secrets" instead of the .env file
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found! Please create a `.env` file and add your key.")
    st.stop()

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))   # milliseconds: a stuck request fails after 60 s

def get_secret(name):
    """Read a setting from the .env file (local) or Streamlit secrets (hosted)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None

IST = timezone(timedelta(hours=5, minutes=30))   # hosted servers run on UTC; used only for times shown on screen

GEMINI_MAX_ATTEMPTS = 3   # the most Gemini calls ONE action can make (the library itself never retries on its own)

def is_daily_quota(text: str) -> bool:
    """Gemini names its daily quota like 'GenerateRequestsPerDayPerProjectPerModel', so match with or without spaces."""
    t = text.lower().replace(" ", "").replace("_", "").replace("-", "")
    return "perday" in t or "daily" in t

def gen(**kwargs):
    """Call Gemini. Retries only temporary errors, at most GEMINI_MAX_ATTEMPTS calls in total, then gives a friendly message."""
    last = None
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            last = e
            msg = str(e).lower()
            if is_daily_quota(msg):   # daily free quota used up: retrying cannot help
                raise RuntimeError("The daily free AI limit has been reached. Please try again later.") from e
            if any(k in msg for k in ("429", "503", "500", "unavailable", "overloaded", "resource_exhausted", "quota")):
                if attempt < GEMINI_MAX_ATTEMPTS - 1:
                    time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("The AI service is busy right now (free-tier limit reached). Please wait about a minute and try again.") from last

# Same model name you already had working. Change it here only if the API ever says "model not found".
log = logging.getLogger("rahat")   # technical details go to the terminal, not to visitors
MAX_PHOTO_MB = 10
MAX_PHOTO_PIXELS = 40_000_000
MODEL_NAME = "gemini-3.8-flash"
MERGE_RADIUS_KM = 25     # reports further apart than this are never matched ...
MERGE_WINDOW_HOURS = 24  # ... nor reports more than this many hours apart
MERGE_AUTO_KM = 3        # automatic merge needs reports this close (with precise locations) ...
MERGE_AUTO_HOURS = 6     # ... and this close in time, plus matching description or photo
AUTO_SCORE = 70          # evidence score (0-100) needed for an automatic merge
REVIEW_SCORE = 40        # similar but not certain: NOT merged, flagged for an authority to decide
AUDIO_MIME = {"wav": "audio/wav", "mp3": "audio/mp3", "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}
DB_PATH = "reports.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1b. Authority sign-in (role-based access control)
#     Public visitors  -> "Submit report" only, no login needed.
#     Signed-in authorities -> "Control room" (view, dispatch, resolve, reopen, delete, reports).
#     Passwords are never stored in plain text: .env holds a salted PBKDF2 hash (made by make_password_hash.py).
# ---------------------------------------------------------
AUTH_TIMEOUT_MIN = 60     # automatic sign-out after this many minutes without any activity
MAX_FAILED_USER = 5       # wrong passwords allowed per username ...
MAX_FAILED_ALL = 20       # ... and in total, within the window below
LOCK_WINDOW_S = 300       # after that, sign-in stays locked until the oldest failure is 5 minutes old
_DUMMY_HASH = "pbkdf2_sha256:200000:" + "00" * 16 + ":" + "00" * 32   # checked when the username does not exist, so timing looks the same

def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.strip().split(":")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex.strip().lower())
    except Exception:
        return False

def load_authorities() -> dict:
    """Authority accounts from .env / secrets as {username (lower-case): password hash}. Empty = nobody can sign in."""
    users = {}
    name, pw_hash = get_secret("AUTHORITY_USERNAME"), get_secret("AUTHORITY_PASSWORD_HASH")
    if name and pw_hash:
        users[str(name).strip().lower()] = str(pw_hash).strip()
    for item in str(get_secret("AUTHORITY_USERS") or "").split(","):   # optional extra accounts: name=hash,name2=hash2
        if "=" in item:
            n, h = item.split("=", 1)
            if n.strip() and h.strip():
                users[n.strip().lower()] = h.strip()
    return users

@st.cache_resource
def _login_tracker():
    return {"lock": threading.Lock(), "fails": {}}   # shared by every visitor, so opening a new browser tab does not reset the limit

def attempt_login(username: str, password: str):
    """Check credentials. Returns (ok, message). Failed attempts are rate-limited."""
    users = load_authorities()
    if not users:
        return False, "Authority sign-in is not configured on this server."
    name = (username or "").strip().lower()[:64]
    keys = ([name] if name in users else []) + ["*"]
    limits = {name: MAX_FAILED_USER, "*": MAX_FAILED_ALL}
    tracker, now = _login_tracker(), time.time()
    with tracker["lock"]:
        for key in keys:
            recent = [t for t in tracker["fails"].get(key, []) if now - t < LOCK_WINDOW_S]
            tracker["fails"][key] = recent
            if len(recent) >= limits[key]:
                wait = int(LOCK_WINDOW_S - (now - recent[0])) // 60 + 1
                return False, f"Too many failed sign-in attempts. Try again in about {wait} minute(s)."
    stored = users.get(name)
    password_ok = verify_password((password or "")[:256], stored or _DUMMY_HASH)
    ok = bool(stored) and password_ok
    with tracker["lock"]:
        if ok:
            tracker["fails"].pop(name, None)
        else:
            for key in keys:
                tracker["fails"].setdefault(key, []).append(now)
    return (True, "") if ok else (False, "Invalid username or password.")

def is_authenticated() -> bool:
    auth = st.session_state.get("auth")
    if not auth:
        return False
    if time.time() - auth["last"] > AUTH_TIMEOUT_MIN * 60:
        sign_out()
        st.session_state["auth_notice"] = "Your session timed out. Please sign in again."
        return False
    auth["last"] = time.time()
    return True

def current_user():
    return (st.session_state.get("auth") or {}).get("user")

def sign_out():
    for key in [k for k in st.session_state.keys() if k in ("auth", "sitrep", "flash") or str(k).startswith(("alert_", "txt_"))]:
        del st.session_state[key]

def require_authority():
    """Server-side guard. Every control-room function calls this first, in addition to the check in the page itself."""
    if not is_authenticated():
        raise PermissionError("Authority sign-in required.")

# ---------------------------------------------------------
# 2. Structured output schema (what Gemini must return)
# ---------------------------------------------------------
class DisasterTriage(BaseModel):
    hazard_type: str = Field(description="Short hazard label of 1-3 words, e.g. Landslide, Flash Flood, Building Fire, Building Collapse")
    severity: int = Field(description="Severity from 1 (minor) to 5 (immediate threat to life), following the rubric in the prompt")
    severity_reason: str = Field(description="One short sentence explaining why this severity was chosen")
    people_at_risk: str = Field(description="Short description of who is endangered, e.g. '25-30 passengers in a bus'")
    people_at_risk_count: int = Field(description="Best single-number estimate of people at risk (use the middle of any range; 0 if unknown)")
    people_count_known: bool = Field(description="True ONLY if the report or photo states or clearly shows how many people are at risk (a number or a range). False if that number is unknown")
    resources_needed: list[str] = Field(description="Key emergency teams or supplies needed")
    language_detected: str = Field(description="Language of the citizen's report: Hindi, English or Hinglish")
    english_summary: str = Field(description="Clear, concise English dispatch summary")
    voice_transcript: str = Field(description="Word-for-word transcript of the attached voice note, in its original language and script; empty string if there is no voice note")
    photo_report_match: str = Field(description="Does the photo support what the citizen wrote? Answer exactly one of: Consistent, Partly consistent, Inconsistent. Answer 'Not provided' if no photo was attached")
    credibility_note: str = Field(description="One short sentence explaining the photo/report match; mention anything that does not match")

# ---------------------------------------------------------
# 3. Database helpers (SQLite = a simple file called reports.db)
# ---------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, location TEXT, lat REAL, lon REAL,
                hazard_type TEXT, severity INTEGER, severity_reason TEXT,
                people_count INTEGER, people_text TEXT, resources TEXT,
                summary TEXT, language TEXT, image_path TEXT, is_demo INTEGER DEFAULT 0
            )""")
        for col in ("credibility TEXT", "credibility_note TEXT", "transcript TEXT", "status TEXT DEFAULT 'New'", "status_updated_at TEXT", "tracking_code TEXT", "prev_status TEXT", "prev_status_updated_at TEXT",
                    "report_text TEXT", "audio_path TEXT", "audio_mime TEXT", "ai_status TEXT", "ai_error TEXT", "ai_attempted_at TEXT", "ai_started_at TEXT"):   # adds new columns to an older reports.db
            try:
                con.execute(f"ALTER TABLE reports ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_tracking ON reports(tracking_code)")
        con.execute("""CREATE TABLE IF NOT EXISTS merge_decisions (a_id INTEGER, b_id INTEGER, decision TEXT, decided_by TEXT,
                       decided_at TEXT, batch TEXT, PRIMARY KEY (a_id, b_id))""")   # codes are unique (old rows have none)

CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"   # no 0/O, 1/I/L, so a code is easy to read out or write down

def new_tracking_code() -> str:
    """Random, unguessable code like RAH-7K2M9-QXD4F (10 characters, about 49 bits)."""
    part = "".join(secrets.choice(CODE_ALPHABET) for _ in range(10))
    return f"RAH-{part[:5]}-{part[5:]}"

def normalize_code(raw: str):
    """Accepts lower case, spaces, missing dashes. Returns the clean code, or None if it is not a valid code."""
    clean = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(clean) == 13 and clean.startswith("RAH"):
        clean = clean[3:]
    if len(clean) != 10 or any(c not in CODE_ALPHABET for c in clean):
        return None
    return f"RAH-{clean[:5]}-{clean[5:]}"

def save_report(r: dict):
    """Saves a report and returns its tracking code (citizen reports only; demo reports get none)."""
    r = {"credibility": "Not checked", "credibility_note": "", "transcript": "", "status": "New", "status_updated_at": None,
         "tracking_code": None, "report_text": "", "audio_path": "", "audio_mime": "", "ai_status": None, "ai_error": None,
         "ai_attempted_at": None, "ai_started_at": None, **r}
    if r["tracking_code"] is None and not r.get("is_demo"):
        r["tracking_code"] = new_tracking_code()
    for _ in range(5):
        try:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("""
                    INSERT INTO reports (created_at, location, lat, lon, hazard_type, severity, severity_reason,
                                         people_count, people_text, resources, summary, language, image_path, is_demo,
                                         credibility, credibility_note, transcript, status, status_updated_at, tracking_code,
                                         report_text, audio_path, audio_mime, ai_status, ai_error, ai_attempted_at, ai_started_at)
                    VALUES (:created_at, :location, :lat, :lon, :hazard_type, :severity, :severity_reason,
                            :people_count, :people_text, :resources, :summary, :language, :image_path, :is_demo,
                            :credibility, :credibility_note, :transcript, :status, :status_updated_at, :tracking_code,
                            :report_text, :audio_path, :audio_mime, :ai_status, :ai_error, :ai_attempted_at, :ai_started_at)""", r)
            return r["tracking_code"]
        except sqlite3.IntegrityError:   # the code already exists (extremely unlikely): make another one
            if r["tracking_code"] is None:
                raise
            r["tracking_code"] = new_tracking_code()
    raise RuntimeError("Could not create a tracking code. Please try again.")

def load_reports() -> pd.DataFrame:
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT * FROM reports", con)

def clear_reports():
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM reports")
        con.execute("DELETE FROM merge_decisions")

def set_status(ids, status: str):
    """Changes the status and remembers the previous one, so it can be restored. Reports already in that status are left alone."""
    require_authority()
    stamp = None if status == "New" else datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("""UPDATE reports SET prev_status = status, prev_status_updated_at = status_updated_at,
                           status = ?, status_updated_at = ? WHERE id = ? AND status IS NOT ?""",
                        [(status, stamp, int(i), status) for i in ids])

def restore_status(ids):
    """Reopen / undo: every report goes back to the status it had before its last change (New if unknown)."""
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        for i in ids:
            row = con.execute("SELECT prev_status, prev_status_updated_at FROM reports WHERE id = ?", (int(i),)).fetchone()
            prev, prev_ts = row if row else (None, None)
            if prev not in STATUS_ORDER:
                prev, prev_ts = "New", None
            con.execute("""UPDATE reports SET status = ?, status_updated_at = ?, prev_status = NULL, prev_status_updated_at = NULL
                           WHERE id = ?""", (prev, None if prev == "New" else prev_ts, int(i)))

def load_decisions() -> dict:
    """The authority's answers on possible duplicates: {(smaller report id, larger report id): 'merge' | 'separate'}."""
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT a_id, b_id, decision FROM merge_decisions").fetchall()
    return {(a, b): d for a, b, d in rows}

def save_decision(pairs, decision: str, user: str):
    """Remembers an answer for every report pair between two incidents. No report is changed or deleted."""
    require_authority()
    batch, now = secrets.token_hex(6), datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("INSERT OR REPLACE INTO merge_decisions (a_id, b_id, decision, decided_by, decided_at, batch) VALUES (?,?,?,?,?,?)",
                        [(min(int(a), int(b)), max(int(a), int(b)), decision, user, now, batch) for a, b in pairs])

def list_decision_batches() -> list:
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("""SELECT batch, MIN(decision), COUNT(*), MIN(decided_by), MIN(decided_at) FROM merge_decisions
                              GROUP BY batch ORDER BY MIN(decided_at) DESC""").fetchall()
    return [dict(batch=b, decision=d, pairs=n, by=u, at=t) for b, d, n, u, t in rows]

def undo_decision(batch: str):
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM merge_decisions WHERE batch = ?", (batch,))

init_db()

# ---------------------------------------------------------
# 4. Priority score (simple and explainable)
#    score = severity x people factor x waiting factor x confirmation factor
# ---------------------------------------------------------
UNKNOWN_PEOPLE = -1            # stored in the database when the number of people at risk is not known
UNKNOWN_PEOPLE_ASSUMED = 5     # unknown is scored as "about 5 people": missing information never lowers urgency
HIGH_SCORE, MEDIUM_SCORE = 12, 6
SEVERITY_MIN_SCORE = {5: HIGH_SCORE, 4: MEDIUM_SCORE}   # severe emergencies can never fall below High (5) or Medium (4)

def people_label(n) -> str:
    return "Unknown" if n is None or n < 0 else str(int(n))

def people_phrase(n) -> str:
    return "number of people unknown (do not state a number)" if n is None or n < 0 else f"about {int(n)} people"

def priority(severity: int, people: int, minutes_waiting: float, n_reports: int = 1):
    """Returns (score, people factor, waiting factor, confirmation factor, level).
    The level comes from the final score only, so score, label and ranking always agree."""
    severity = min(max(int(severity), 1), 5)
    unknown = people is None or people < 0
    assumed = UNKNOWN_PEOPLE_ASSUMED if unknown else people
    people_factor = 1 + min(max(assumed, 0), 100) / 10           # 1.0 to 11.0
    waiting_factor = 1 + min(max(minutes_waiting, 0), 60) / 30   # 1.0 to 3.0 (grows while the incident waits)
    confirm_factor = min(1 + 0.15 * (n_reports - 1), 1.6)        # more independent reports = more confidence
    score = max(severity * people_factor * waiting_factor * confirm_factor, SEVERITY_MIN_SCORE.get(severity, 0))
    level = "High" if score >= HIGH_SCORE else "Medium" if score >= MEDIUM_SCORE else "Low"
    return score, people_factor, waiting_factor, confirm_factor, level

LEVEL_COLOR = {"High": "red", "Medium": "orange", "Low": "green"}
LEVEL_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
CRED_ICON = {"Consistent": "✅", "Partly consistent": "⚠️", "Inconsistent": "❌", "Not provided": "➖"}

def minutes_since(ts: str) -> float:
    return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 60

# ---------------------------------------------------------
# 5. Geocoding: turn typed location into latitude/longitude
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode(location: str):
    geolocator = Nominatim(user_agent="rahat-ai-hackathon", timeout=10)
    parts = [p.strip() for p in location.split(",") if p.strip()]
    # drop words like "near", "opposite" that confuse the map search
    clean = [re.sub(r"^(near|opposite|behind|beside|next to|close to|at)\s+", "", p, flags=re.I) for p in parts]
    attempts = [location] + [", ".join(clean[-n:]) for n in (3, 2) if len(clean) >= n]
    if len(clean) >= 2:   # e.g. "Chamoli, Uttarakhand" from "NH-58, near Chamoli, Uttarakhand"
        attempts += [f"{clean[i]}, {clean[-1]}" for i in range(len(clean) - 2, -1, -1)]
    attempts += clean[-1:]
    for q in dict.fromkeys(attempts):
        try:
            hit = geolocator.geocode(q, country_codes="in")
            if hit:
                return hit.latitude, hit.longitude
        except Exception:
            continue
    return None


# ---------------------------------------------------------
# 5b. Fallback: approximate centre of well-known places (used only if the online geocoder fails)
# ---------------------------------------------------------
KNOWN_PLACES = {
    "delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777), "kolkata": (22.5726, 88.3639), "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946), "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867), "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714), "jaipur": (26.9124, 75.7873), "lucknow": (26.8467, 80.9462), "patna": (25.5941, 85.1376),
    "bhopal": (23.2599, 77.4126), "indore": (22.7196, 75.8577), "nagpur": (21.1458, 79.0882), "chandigarh": (30.7333, 76.7794),
    "dehradun": (30.3165, 78.0322), "haridwar": (29.9457, 78.1642), "rishikesh": (30.0869, 78.2676), "uttarkashi": (30.7268, 78.4354),
    "chamoli": (30.4000, 79.3200), "joshimath": (30.5550, 79.5640), "kedarnath": (30.7352, 79.0669), "pithoragarh": (29.5829, 80.2182),
    "nainital": (29.3919, 79.4542), "shimla": (31.1048, 77.1734), "manali": (32.2396, 77.1887), "srinagar": (34.0837, 74.7973),
    "jammu": (32.7266, 74.8570), "leh": (34.1526, 77.5771), "guwahati": (26.1445, 91.7362), "shillong": (25.5788, 91.8933),
    "gangtok": (27.3389, 88.6065), "imphal": (24.8170, 93.9368), "agartala": (23.8315, 91.2868), "bhubaneswar": (20.2961, 85.8245),
    "visakhapatnam": (17.6868, 83.2185), "thiruvananthapuram": (8.5241, 76.9366), "kochi": (9.9312, 76.2673), "wayanad": (11.6854, 76.1320),
    "mangaluru": (12.9141, 74.8560), "panaji": (15.4909, 73.8278), "surat": (21.1702, 72.8311), "vadodara": (22.3072, 73.1812),
    "ranchi": (23.3441, 85.3096), "raipur": (21.2514, 81.6296), "noida": (28.5355, 77.3910), "gurugram": (28.4595, 77.0266),
    "ghaziabad": (28.6692, 77.4538), "varanasi": (25.3176, 82.9739), "prayagraj": (25.4358, 81.8463), "kanpur": (26.4499, 80.3319),
    "agra": (27.1767, 78.0081), "amritsar": (31.6340, 74.8723), "ludhiana": (30.9010, 75.8573),
}

def fallback_place(location: str):
    low = location.lower()
    for name in sorted(KNOWN_PLACES, key=len, reverse=True):
        if name in low:
            return KNOWN_PLACES[name]
    return None

# ---------------------------------------------------------
# 6. AI triage call
# ---------------------------------------------------------
def run_triage(image, location: str, report_text: str, audio=None) -> dict:
    # image may be None (text / voice-only report); audio = (bytes, mime_type) for an optional voice note
    img = None
    if image is not None:
        img = image.convert("RGB")
        img.thumbnail((1280, 1280))   # smaller image = faster, cheaper API call
    intro = ("Analyze the incident photo and the citizen's report together." if img is not None else
             "No photo was provided. Analyze the citizen's report (text and/or voice note) only.")
    prompt = f"""
    You are an emergency disaster triage specialist for Indian control rooms.
    {intro}

    Location provided: {location}
    Citizen report (text): {report_text or '(none - see the voice note)'}
    A voice note {'IS' if audio else 'is NOT'} attached. If attached, listen to it carefully (Hindi, English or Hinglish),
    transcribe it word for word into voice_transcript, and use it together with the text and the photo.

    SEVERITY RUBRIC:
    5 = immediate threat to life happening now (people injured, trapped in danger, fire or water reaching people)
    4 = people stranded or at serious risk; urgent response needed within the hour
    3 = significant damage or blocked access, no confirmed danger to people
    2 = minor damage, no people at risk
    1 = negligible
    If people are trapped or stranded because of the hazard, severity must be at least 4.

    Rules: do not invent details that are not supported by the photo or the report.
    If no photo was provided, do not describe or assume anything visual, and answer 'Not provided' for photo_report_match.
    Set people_count_known to true ONLY when the report or photo gives a number or range of people at risk; otherwise
    false (and people_at_risk_count 0). Never guess a number. Missing information must never make an emergency look
    less serious: judge severity from what is known, using the rubric above.
    If the photo and the report disagree, mention it in english_summary.
    Also judge honestly whether the photo supports the citizen's report (photo_report_match).
    Detect the language of the report and write an actionable English dispatch summary.
    """
    response = gen(
        model=MODEL_NAME,
        contents=([img] if img is not None else []) + ([types.Part.from_bytes(data=audio[0], mime_type=audio[1])] if audio else []) + [prompt],
        config={"response_mime_type": "application/json", "response_schema": DisasterTriage},
    )
    if getattr(response, "parsed", None):
        p = response.parsed
        return p.model_dump() if hasattr(p, "model_dump") else dict(p)
    return json.loads(response.text)

def clamp_int(value, default, lo, hi):
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))

def clean_triage(data, has_photo: bool = True) -> dict:
    """Makes the AI answer safe to store and show: correct types, severity 1-5, sensible lengths."""
    if not isinstance(data, dict):
        raise ValueError("The AI returned an unexpected answer.")
    def text(key, limit, default=""):
        return str(data.get(key) or default)[:limit]
    resources = data.get("resources_needed") or []
    if not isinstance(resources, (list, tuple)):
        resources = [resources]
    match = {"consistent": "Consistent", "partly consistent": "Partly consistent",
             "inconsistent": "Inconsistent"}.get(text("photo_report_match", 40).strip().lower(), "Not checked")
    people_count = clamp_int(data.get("people_at_risk_count"), UNKNOWN_PEOPLE, UNKNOWN_PEOPLE, 100000)
    if str(data.get("people_count_known")).strip().lower() in ("false", "no", "0"):
        people_count = UNKNOWN_PEOPLE
    if not has_photo:   # nothing to verify: shown as "Not provided", never judged by the AI
        match = "Not provided"
    return {
        "hazard_type": text("hazard_type", 60, "Unknown"),
        "severity": clamp_int(data.get("severity"), 3, 1, 5),
        "severity_reason": text("severity_reason", 300),
        "people_at_risk": text("people_at_risk", 200, "N/A"),
        "people_at_risk_count": people_count,
        "resources_needed": [str(r)[:80] for r in list(resources)[:10]],
        "language_detected": text("language_detected", 40, "Unknown"),
        "english_summary": text("english_summary", 1500),
        "voice_transcript": text("voice_transcript", 3000),
        "photo_report_match": match,
        "credibility_note": text("credibility_note", 300) if has_photo else "No photo was provided.",
    }

def save_photo(image: Image.Image) -> str:
    path = os.path.join(UPLOAD_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
    img = image.convert("RGB")
    img.thumbnail((1280, 1280))
    img.save(path, quality=85)
    return path

AUDIO_EXT = {mime: ext for ext, mime in AUDIO_MIME.items()}

def save_audio(data: bytes, mime: str) -> str:
    path = os.path.join(UPLOAD_DIR, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{AUDIO_EXT.get(mime, 'wav')}")
    with open(path, "wb") as f:
        f.write(data)
    return path

PENDING_SEVERITY = 4   # placeholder for an unassessed report: the existing rules keep severity 4 at Medium priority or higher
AI_REASON = {"daily quota": "the free daily AI limit has been reached", "rate limit": "the AI service is busy right now",
             "timeout": "the AI service took too long to respond", "service unavailable": "the AI service is temporarily unavailable",
             "other error": "the AI service could not process this report"}

def ai_error_kind(exc) -> str:
    """Short, safe category for a failed Gemini call (never the raw error text)."""
    text, e = "", exc
    for _ in range(4):   # gen() wraps the original error, so look at the whole chain
        if e is None:
            break
        text += f" {type(e).__name__} {e}".lower()
        e = e.__cause__
    if is_daily_quota(text):
        return "daily quota"
    if "timeout" in text or "timed out" in text or "deadline" in text:
        return "timeout"
    if any(k in text for k in ("503", "502", "504", "unavailable", "overloaded", "internal")):
        return "service unavailable"
    if any(k in text for k in ("429", "resource_exhausted", "rate limit", "quota", "free-tier", "busy")):
        return "rate limit"
    return "other error"

# ---------------------------------------------------------
# 6e. Basic emergency keyword check. Used ONLY while an AI assessment is Pending (Gemini failed).
#     Simple, explainable rules: no AI, no dispatching. A match raises the report to High for urgent HUMAN review.
# ---------------------------------------------------------
FLAGGED_SEVERITY = 5   # a keyword match uses the existing rule "severity 5 = High"; it can raise a priority, never lower it
_DB, _DA = "(?<![\u0900-\u097f])", "(?![\u0900-\u097f])"   # word edges for Devanagari (\b does not work well with it)
# (label, [(pattern, negatable)]).  negatable=False for pleas and requests ("bachao", "help us", "ambulance"), which are never cancelled.
EMERGENCY_RULES = [
    ("fire", [
        (r"\bfire\b", True), (r"\b(on fire|burning|burnt|blaze|flames?)\b", True), (r"\baag(h)?\b", True),
        (r"\bjal\s+(raha|rahi|rahe|gaya|gayi)\b", True), (_DB + "आग" + _DA, True), (r"जल\s*रह[ाीे]", True), ("लपट", True),
        (r"\bfire\s+(brigade|engine|truck|tender|department|service)s?\b", False)]),
    ("people trapped or needing rescue", [
        (r"\b(trapped|buried|stranded)\b", True),
        (r"\bstuck\s+(inside|under|in the (building|house|lift|elevator|car|vehicle|water|mud|flood))\b", True),
        (r"\bunder (the )?(rubble|debris|collapsed|fallen)\b", True), (r"\brescue\b", True),
        (r"\b(cannot|can't|cant|unable to|not able to) (get out|escape|exit|leave|move|breathe)\b", False),
        (r"\b(no one|nobody|none of us) (can|could|is able to) (get out|escape|exit|leave|move)\b", False),
        (r"\b(help us|save us|sos)\b", False), (r"\bphans\w*", True), (r"\bfans(a|e|i)\b", True), (r"\bmalb[ae]\b", True),
        (r"\b(bachao|bachaao|bacha lo|bachalo)\b", False),
        (_DB + "फ[ंँ]स", True), ("बचाओ|बचा\\s*लो|बचाइए", False), (_DB + "मलब[ाे]", True)]),
    ("building or structure collapse", [
        (r"\bcollaps\w*", True), (r"\bcave[- ]?in\b|\bcaved in\b", True),
        (r"\b(building|house|home|wall|roof|ceiling|bridge|structure|flyover|tower)\b(?:\s+\w{1,10}){0,2}\s+(fell|fallen|came down|caved in|gave way)\b", True),
        (r"\b(building|makaan|ghar|imarat|diwar|deewar|dewar|chhat|pul|bridge|wall|roof)\b(?:\s+\w{1,10}){0,2}\s+(gir|gira|giri|gire|girne|dhah|dhaha|dhahi|dhahe|toot|toota|tooti|tut)\w*", True),
        ("(इमारत|मकान|घर|दीवार|छत|पुल|भवन)(?:\\s+\\S{1,10}){0,1}\\s+(गिर|ढह|टूट)", True), ("ढ\u093cह", True)]),
    ("serious injury or death", [
        (r"\b(injur\w*|wounded|bleeding|unconscious|casualt\w*|fatalit\w*|killed|dead|died|dying|deaths?)\b", True),
        (r"\b(not|stopped|stop) breathing\b", True), (r"\bcritical condition\b|\bheart attack\b|\bbroken (leg|arm|bone|back|neck)\b", True),
        (r"\bgha+yal\w*", True), (r"\bzakhm\w*", True), (r"\bkhoon\b", True), (r"\bbehosh\w*", True),
        (r"\bmar\s+(gaya|gayi|gaye|gai)\b", True), (r"\bmaut\w*", True), (r"\bs(aa|a)ns\s+n(ahi|hi)\b", True), (r"\bdam\s+tod\w*", True),
        ("घायल", True), ("ज\u093c?ख\u093c?्म", True), ("ख\u093c?ून", True), ("बेहोश", True), ("मौत|मृत", True),
        ("मर\\s*(गया|गई|गये|गए|गयी)", True), ("(साँस|सांस)\\s*नहीं", True), ("दम\\s*तोड़", True),
        (r"\bambulance\b", False), ("एम्बुलेंस|एंबुलेंस", False)]),
    ("drowning or being swept away", [
        (r"\b(drown\w*|swept away|washed away|carried away by (the )?(water|flood|current))\b", True),
        (r"\b(doob|dub|dhoob)\w*\s+(rahe|raha|rahi|gaye|gaya|gayi)\b", True), (r"\bbeh\s+(gaye|gaya|gayi|rahe|raha|rahi)\b", True),
        (_DB + "डूब", True), ("बह\\s*(गए|गये|गया|गई|गयी|रहे|रहा|रही)", True)]),
    ("explosion or gas leak", [
        (r"\b(explosion|exploded|explode|blast|gas leak\w*|cylinder (blast|burst))\b", True), (r"\b(dhamaka|visfot)\b", True),
        ("धमाका|विस्फोट", True), ("गैस\\s*(लीक|रिस)", True)]),
]
_COMPILED = [(label, [(re.compile(unicodedata.normalize("NFC", p), re.I), neg) for p, neg in pats]) for label, pats in EMERGENCY_RULES]
_NEG_BEFORE = {"no", "not", "without", "nobody", "none", "zero", "never", "neither", "nor", "nil", "isn't", "aren't", "wasn't", "weren't",
               "hasn't", "haven't", "didn't", "doesn't", "don't", "isnt", "arent", "wasnt", "hasnt", "didnt", "doesnt", "dont"}
_NEG_AFTER = {"nahi", "nahin", "nhi", "nahee", "नहीं", "नही", "नहि"}   # Hindi / Hinglish: the "no" comes after the word
_BENIGN = re.compile(r"\b(?:(?:fire|mock|evacuation|safety) drill|false alarm|fire (?:station|exit|extinguisher|safety|insurance)|dead (?:end|line|lock|battery))\b")
_CLAUSE_SPLIT = re.compile(r"[.,;:!?\n।|]+|\b(?:but|however|although|though|lekin|magar|parantu)\b|\s(?:लेकिन|मगर|परंतु|किन्तु)\s")

def _tokens(s: str) -> list:
    return [t.strip(".,;:!?()[]{}\"'…-") for t in s.split()]

def _negated(clause: str, m) -> bool:
    """'no injuries', 'there is no fire', 'ghayal nahi': a negative word just before (English) or just after (Hindi) the match."""
    return (any(t in _NEG_BEFORE for t in _tokens(clause[:m.start()])[-3:]) or
            any(t in _NEG_AFTER for t in _tokens(clause[m.end():])[:2]))

def emergency_flags(*texts) -> list:
    """Emergency types clearly mentioned and not negated: [{'label', 'match'}]. Empty list = nothing obvious found."""
    text = " ".join(t for t in texts if isinstance(t, str))
    text = _BENIGN.sub(" ", unicodedata.normalize("NFC", text).replace("\u2019", "'").lower())
    clauses = _CLAUSE_SPLIT.split(text)
    found = []
    for label, patterns in _COMPILED:
        hit = None
        for clause in clauses:
            for rx, negatable in patterns:
                for m in rx.finditer(clause):
                    if not (negatable and _negated(clause, m)):
                        hit = m.group(0).strip()
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            found.append({"label": label, "match": hit[:30]})
    return found

def apply_emergency_flag(code: str, flags: list, severity: int = FLAGGED_SEVERITY) -> bool:
    """Flags a report that is STILL PENDING for urgent human review. Raises severity (never lowers it) and records why.
    ai_status stays 'Pending' and the incident status is untouched: nothing is dispatched automatically."""
    reason = ("URGENT HUMAN REVIEW. A simple keyword check (NOT an AI assessment) found: "
              + "; ".join(f"{f['label']} (\"{f['match']}\")" for f in flags) + ". The AI assessment is still pending.")
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("UPDATE reports SET severity = MAX(COALESCE(severity, 0), ?), severity_reason = ? "
                          "WHERE tracking_code = ? AND ai_status = 'Pending'", (severity, reason, code))
    return cur.rowcount > 0
# (end of emergency check)

AI_LOCK_S = 240   # an attempt that started less than this long ago is treated as still running

def claim_assessment(report_id: int) -> bool:
    """Marks the start of ONE Gemini attempt for a pending report. False = another attempt is already running
    (a double-click, a second operator, or the original submission), so no second Gemini call must be made."""
    now = datetime.now()
    stale = (now - timedelta(seconds=AI_LOCK_S)).isoformat()   # a crashed attempt does not block the report forever
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("UPDATE reports SET ai_started_at = ? WHERE id = ? AND ai_status = 'Pending' "
                          "AND (ai_started_at IS NULL OR ai_started_at < ?)", (now.isoformat(), int(report_id), stale))
    return cur.rowcount > 0

def mark_ai_pending(code: str, kind: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE reports SET ai_status = 'Pending', ai_error = ?, ai_attempted_at = ?, ai_started_at = NULL WHERE tracking_code = ?",
                    (kind, datetime.now().isoformat(), code))

def complete_assessment(code: str, data: dict):
    """Writes Gemini's assessment into the SAME report (found by its tracking code)."""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("""UPDATE reports SET hazard_type = ?, severity = ?, severity_reason = ?, people_count = ?, people_text = ?,
                       resources = ?, summary = ?, language = ?, credibility = ?, credibility_note = ?, transcript = ?,
                       ai_status = 'Completed', ai_error = NULL, ai_attempted_at = ?, ai_started_at = NULL WHERE tracking_code = ? AND ai_status = 'Pending'""",
                    (data.get("hazard_type", "Unknown"), int(data.get("severity", 1)), data.get("severity_reason", ""),
                     int(data.get("people_at_risk_count", UNKNOWN_PEOPLE)), data.get("people_at_risk", ""),
                     json.dumps(data.get("resources_needed", []) or []), data.get("english_summary", ""),
                     data.get("language_detected", ""), data.get("photo_report_match", "Not checked"),
                     data.get("credibility_note", ""), data.get("voice_transcript", "") or "", datetime.now().isoformat(), code))
    return cur.rowcount > 0   # False = the report was no longer pending (already assessed), so nothing was changed

def intake_report(image, audio, location: str, report_text: str, coords):
    """SAVE FIRST, ASSESS SECOND. Returns (tracking_code, Gemini data or None, error kind or None).
    The report (text, location, photo, voice note, tracking code) is stored before Gemini is called, so a Gemini
    failure can never lose it: the report simply stays 'Pending'."""
    image_path, audio_path, audio_mime = "", "", ""
    if image is not None:
        try:
            image_path = save_photo(image)
        except Exception:
            log.exception("Could not save the photo")
    if audio is not None:
        try:
            audio_path, audio_mime = save_audio(audio[0], audio[1]), audio[1]
        except Exception:
            log.exception("Could not save the voice note")
    try:
        code = save_report(dict(
            created_at=datetime.now().isoformat(), location=location, lat=coords[0] if coords else None, lon=coords[1] if coords else None,
            hazard_type="Pending AI assessment", severity=PENDING_SEVERITY,
            severity_reason="AI assessment pending. The report is saved and held at Medium priority or higher until it has been assessed.",
            people_count=UNKNOWN_PEOPLE, people_text="Unknown", resources="[]",
            summary=report_text[:300] if report_text else "Voice note received. AI transcription pending.",
            language="Pending", image_path=image_path, is_demo=0,
            credibility="Not checked" if image is not None else "Not provided",
            credibility_note="AI assessment pending." if image is not None else "No photo was provided.",
            transcript="", report_text=report_text, audio_path=audio_path, audio_mime=audio_mime, ai_status="Pending", ai_started_at=datetime.now().isoformat()))   # the first attempt is running now
    except Exception as e:
        log.exception("Could not save the report")
        raise RuntimeError("We could not save your report. Please try again. If someone is in immediate danger, call 112.") from e
    try:
        data = clean_triage(run_triage(image, location, report_text, audio), has_photo=image is not None)
        complete_assessment(code, data)
    except Exception as e:
        log.exception("AI assessment failed; the report stays saved as Pending")
        kind = ai_error_kind(e)
        try:
            mark_ai_pending(code, kind)
        except Exception:
            log.exception("Could not record the failure reason")
        try:   # Gemini failed: look for obvious emergencies in the citizen's own words (the transcript only comes from Gemini, so it is empty here)
            flags = emergency_flags(report_text, "")
            if flags:
                apply_emergency_flag(code, flags)
        except Exception:
            log.exception("Emergency keyword check failed")
        return code, None, kind
    return code, data, None

def retry_assessment(report_id: int):
    """AUTHORITY ONLY. Retries the AI assessment of ONE pending report using its SAVED evidence (text, photo, voice note).
    Updates the same row on success. Never changes the incident status, never dispatches. Returns (ok, message)."""
    require_authority()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM reports WHERE id = ?", (int(report_id),)).fetchone()
    if row is None:
        return False, "That report no longer exists."
    if row["ai_status"] != "Pending":
        return False, "This report already has a completed AI assessment."
    code, text = row["tracking_code"], row["report_text"] or ""
    image, audio = None, None
    try:   # reuse the evidence that was saved at submission
        if row["image_path"]:
            image = Image.open(row["image_path"])
            image.load()
        if row["audio_path"]:
            with open(row["audio_path"], "rb") as f:
                audio = (f.read(), row["audio_mime"] or "audio/wav")
    except Exception:
        log.exception("Saved evidence could not be read for the retry")
        return False, "The saved photo or voice note could not be read, so the AI assessment was not retried. The report is unchanged and still pending."
    if not claim_assessment(row["id"]):
        return False, "An AI assessment of this report is already running. Please wait a moment, then check again."
    try:
        data = clean_triage(run_triage(image, row["location"], text, audio), has_photo=image is not None)
    except Exception as e:
        log.exception("Retry of the AI assessment failed; the report stays Pending")
        kind = ai_error_kind(e)
        try:
            mark_ai_pending(code, kind)   # only records the reason and time: evidence, flags and status are untouched
        except Exception:
            log.exception("Could not record the failure reason")
        return False, f"The AI assessment could not be completed again ({AI_REASON[kind]}). The report is unchanged and still pending."
    if str(row["severity_reason"] or "").startswith("URGENT HUMAN REVIEW") and data["severity"] < row["severity"]:
        # never silently lower a keyword-check flag: keep the higher severity and say so
        data["severity_reason"] += (f" (The AI estimated severity {data['severity']}, but this stays at {row['severity']} because an earlier "
                                    "keyword check flagged it for urgent human review. A person should confirm.)")
        data["severity"] = row["severity"]
    if not complete_assessment(code, data):
        return False, "This report was assessed by someone else in the meantime, so nothing was changed."
    return True, "AI assessment completed. The report, its tracking code and its status are unchanged."

# ---------------------------------------------------------
# 6b. Duplicate merging: several reports of one incident -> ONE incident
# ---------------------------------------------------------
HAZARD_FAMILIES = [
    ("landslide", ["landslide", "landslip", "rockfall", "rock fall", "rockslide", "rock slide", "mudslide", "debris"]),
    ("flood", ["flood", "waterlog", "inundat", "cloudburst", "river"]),
    ("fire", ["fire", "blaze"]),
    ("collapse", ["collapse", "building"]),
    ("earthquake", ["earthquake", "quake", "tremor"]),
    ("storm", ["cyclone", "storm", "hail", "lightning"]),
]

def hazard_family(label: str) -> str:
    low = (label or "").lower()
    for family, words in HAZARD_FAMILIES:
        if any(w in low for w in words):
            return family
    return low.strip()

def distance_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))

def has_coords(r) -> bool:
    return pd.notna(r["lat"]) and pd.notna(r["lon"])

STATUS_ORDER = {"New": 0, "Dispatched": 1, "Resolved": 2}

def status_of(r) -> str:
    v = r.get("status")
    return v if v in STATUS_ORDER else "New"

def prev_of(r) -> str:
    v = r.get("prev_status")
    return v if v in STATUS_ORDER else "New"

def is_pending(r) -> bool:
    return r.get("ai_status") == "Pending"

def cred_of(r) -> str:
    v = r.get("credibility")
    return v if isinstance(v, str) and v else "Not checked"

def merge_people(reports) -> int:
    """Largest known count. Unknown (-1) if none is known, or if a report is unknown and the known counts are small."""
    vals = [UNKNOWN_PEOPLE if pd.isna(r["people_count"]) else int(r["people_count"]) for r in reports]
    known = [v for v in vals if v >= 0]
    if not known or (len(known) < len(vals) and max(known) < UNKNOWN_PEOPLE_ASSUMED):
        return UNKNOWN_PEOPLE
    return max(known)

STOP_WORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "near", "opposite", "behind", "beside", "next", "close", "and", "is", "are", "was",
              "were", "has", "have", "had", "with", "for", "from", "by", "road", "area", "ward", "street", "this", "that", "it", "as", "due"}
GENERIC_HAZARDS = {"", "unknown", "other", "incident", "emergency", "disaster"}   # too vague to prove two reports are the same event
APPROX_COORDS = {(round(v[0], 3), round(v[1], 3)) for v in KNOWN_PLACES.values()}   # city-centre fallback pins are not exact

def word_set(text) -> set:
    t = re.sub(r"\b(nh|sh)[-\s]?(\d+)\b", r"\1\2", (text if isinstance(text, str) else "").lower())   # "NH-58" and "NH 58" match
    return {w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP_WORDS and len(w) > 1}

def dice(a: set, b: set) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0

def is_precise(r) -> bool:
    return has_coords(r) and (round(r["lat"], 3), round(r["lon"], 3)) not in APPROX_COORDS

def photo_file(r):
    p = r.get("image_path")
    return p if isinstance(p, str) and p and os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def photo_hash(path: str):
    """64-bit 'difference hash'. The same picture saved again gives a hash that differs in only a few bits."""
    try:
        px = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=int)
        if px.std() < 3:   # almost flat picture: a hash would be meaningless
            return None
        return int("".join("1" if bit else "0" for bit in (px[:, :-1] > px[:, 1:]).flatten()), 2)
    except Exception:
        return None

def match_info(a, b) -> dict:
    """How strongly two reports look like the same incident. level: 'auto' (clear), 'review' (possible) or 'none'.
    Evidence: hazard type (must agree), location (40 pts), time (20), description (30), photo (15)."""
    out = {"level": "none", "score": 0, "km": None, "hours": None, "text": 0.0, "photo": None, "precise": False, "reasons": []}
    family = hazard_family(a["hazard_type"])
    if family != hazard_family(b["hazard_type"]):
        return out
    hours = abs((datetime.fromisoformat(a["created_at"]) - datetime.fromisoformat(b["created_at"])).total_seconds()) / 3600
    if hours > MERGE_WINDOW_HOURS:
        return out
    km = distance_km(a["lat"], a["lon"], b["lat"], b["lon"]) if has_coords(a) and has_coords(b) else None
    if km is not None and km > MERGE_RADIUS_KM:
        return out
    place_a, place_b = word_set(a["location"]), word_set(b["location"])
    place_sim = dice(place_a, place_b)
    if km is None and place_sim < 0.4:
        return out   # nothing ties the two places together
    precise = km is not None and is_precise(a) and is_precise(b)
    if precise:
        loc = 40 if km <= 0.5 else 36 if km <= 1 else 30 if km <= MERGE_AUTO_KM else 18 if km <= 10 else 8
    else:
        loc = 24 if place_sim >= 0.8 else 14 if place_sim >= 0.5 else 0
        if km is not None and km < 0.05:
            loc = max(loc, 20)   # the same approximate point
    close_precise = precise and km <= MERGE_AUTO_KM
    same_place = close_precise or place_sim >= 0.8 or (km is not None and km < 0.05)
    shared_road = [w for w in place_a & place_b if re.fullmatch(r"(nh|sh)\d+", w)]
    if shared_road:
        loc = min(40, loc + 6)
    time_pts = 20 if hours <= 1 else 16 if hours <= 3 else 12 if hours <= 6 else 6 if hours <= 12 else 2
    sim = 0.0 if (is_pending(a) or is_pending(b)) else dice(word_set(a["summary"]), word_set(b["summary"]))
    text_pts = round(30 * min(1.0, sim / 0.7))
    pa, pb = photo_file(a), photo_file(b)
    dist = None
    if pa and pb:
        ha, hb = photo_hash(pa), photo_hash(pb)
        if ha is not None and hb is not None:
            dist = bin(ha ^ hb).count("1")
    img_same = dist is not None and dist <= 4
    img_similar = dist is not None and dist <= 10
    img_pts = 15 if img_same else 8 if img_similar else 0
    score = min(100, loc + time_pts + text_pts + img_pts)
    auto = (family not in GENERIC_HAZARDS and not (is_pending(a) or is_pending(b)) and hours <= MERGE_AUTO_HOURS and
            ((close_precise and score >= AUTO_SCORE and (sim >= 0.30 or img_similar)) or (img_same and same_place)))
    reasons = []
    if km is not None:
        reasons.append(f"{km:.1f} km apart" + ("" if precise else " (approximate location)"))
    reasons.append(f"reported {hours * 60:.0f} min apart" if hours < 1 else f"reported {hours:.1f} h apart")
    if sim >= 0.3:
        reasons.append(f"similar descriptions ({sim:.0%})")
    if shared_road:
        reasons.append("same road: " + ", ".join(w.upper() for w in shared_road))
    if img_same:
        reasons.append("near-identical photo")
    elif img_similar:
        reasons.append("similar photo")
    out.update(level="auto" if auto else "review" if score >= REVIEW_SCORE else "none", score=score, km=km, hours=hours,
               text=sim, photo=dist, precise=precise, reasons=reasons)
    return out

def same_incident(a, b) -> bool:
    return match_info(a, b)["level"] == "auto"

def pair_key(a, b):
    return (min(a["id"], b["id"]), max(a["id"], b["id"]))

def merge_reports(reports: list, decisions=None) -> list:
    """Groups reports into incidents. Only CLEAR matches merge automatically (and only if the new report matches every
    report already in the group). Uncertain ones stay separate (see find_review_pairs). Original reports are never
    changed, so each keeps its own tracking code and status."""
    decisions = decisions or {}
    ordered = sorted(reports, key=lambda r: r["created_at"])
    # 1. reports an authority confirmed as the same incident are joined first
    parent = {r["id"]: r["id"] for r in ordered}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (x, y), d in decisions.items():
        if d == "merge" and x in parent and y in parent:
            parent[find(x)] = find(y)
    units = {}
    for r in ordered:
        units.setdefault(find(r["id"]), []).append(r)
    forced = {r["id"] for u in units.values() if len(u) > 1 for r in u}
    # 2. a unit joins a group only if it clearly matches EVERY report in it (no chains of loosely linked reports)
    groups = []
    for unit in sorted(units.values(), key=lambda u: u[0]["created_at"]):
        best, best_score = None, -1
        for g in groups:
            scores = []
            for x in g:
                for y in unit:
                    info = match_info(x, y)
                    if decisions.get(pair_key(x, y)) == "separate" or info["level"] != "auto":
                        scores = None
                        break
                    scores.append(info["score"])
                if scores is None:
                    break
            if scores and min(scores) > best_score:
                best, best_score = g, min(scores)
        if best is not None:
            best.extend(unit)
        else:
            groups.append(list(unit))
    for g in groups:
        g.sort(key=lambda r: r["created_at"])
    incidents = []
    for g in groups:
        # reports that FAIL the photo check stay visible, but cannot raise severity, people count, confirmations or status
        trusted = [r for r in g if cred_of(r) != "Inconsistent"]
        pool = trusted or g   # if every report failed the check, the incident is shown as unverified
        primary = max(pool, key=lambda r: (r["severity"], r["people_count"]))
        located = next((r for r in pool if has_coords(r)), next((r for r in g if has_coords(r)), primary))
        status = min((status_of(r) for r in pool), key=lambda s: STATUS_ORDER[s])   # a new trusted report re-opens a dispatched incident
        stamps = [r["status_updated_at"] for r in pool if status_of(r) == status and isinstance(r.get("status_updated_at"), str)]
        ordered = pool + [r for r in g if r not in pool]
        inc = dict(primary)
        inc.update(lat=located["lat"], lon=located["lon"],
                   severity=max(r["severity"] for r in pool), people_count=merge_people(pool),
                   n_reports=len(g), n_confirm=len(trusted) if trusted else 1, first_reported=pool[0]["created_at"],
                   image_path=next((r["image_path"] for r in ordered if r["image_path"]), ""), members=g,
                   status=status, status_updated_at=max(stamps) if stamps else None, member_ids=[r["id"] for r in g],
                   forced_ids=[r["id"] for r in g if r["id"] in forced])
        incidents.append(inc)
    return incidents

def find_review_pairs(incidents: list, decisions=None) -> list:
    """Separate incidents that MIGHT be one emergency. Nothing here is merged: an authority decides."""
    decisions = decisions or {}
    found = []
    for i in range(len(incidents)):
        for j in range(i + 1, len(incidents)):
            A, B = incidents[i], incidents[j]
            if A["status"] == "Resolved" and B["status"] == "Resolved":
                continue
            best = None
            for x in A["members"]:
                for y in B["members"]:
                    if decisions.get(pair_key(x, y)) == "separate":
                        continue
                    info = match_info(x, y)
                    if info["level"] != "none" and (best is None or info["score"] > best[2]["score"]):
                        best = (x, y, info)
            if best:
                found.append({"a_inc": A, "b_inc": B, "a": best[0], "b": best[1], "info": best[2]})
    found.sort(key=lambda f: -f["info"]["score"])
    return found

# ---------------------------------------------------------
# 6c. Responder alert drafting (English + Hindi)
# ---------------------------------------------------------
def draft_alert(inc: dict) -> str:
    require_authority()
    resources = ", ".join(json.loads(inc["resources"] or "[]"))
    prompt = f"""
    Write a short emergency dispatch alert for field responders, using ONLY the facts below. Do not add facts.

    Hazard: {inc['hazard_type']}
    Location: {inc['location']}
    Priority: {inc['level']} (severity {inc['severity']}/5)
    People at risk: {inc['people_text']} ({people_phrase(inc['people_count'])})
    Resources needed: {resources}
    Situation: {inc['summary']}
    Citizen reports received: {inc['n_reports']}
    Minutes since first report: {int(inc['minutes'])}

    Output exactly three lines, plain text, no markdown:
    RAHAT ALERT | {inc['level']} priority
    EN: <max 280 characters, starts with hazard and location, ends with the resources to send>
    HI: <the same message in simple Hindi, Devanagari script, max 280 characters>
    """
    response = gen(model=MODEL_NAME, contents=prompt)
    return response.text.strip()

def draft_sitrep(rows: list) -> str:
    require_authority()
    lines = []
    for i, x in enumerate(rows, start=1):
        lines.append(f"{i}. {x['hazard_type']} at {x['location']} | {x['level']} priority (score {x['score']:.0f}) | severity {x['severity']}/5 | "
                     f"{people_phrase(x['people_count'])} | {x['n_reports']} report(s) | waiting {int(x['minutes'])} min | "
                     f"needs: {', '.join(json.loads(x['resources'] or '[]'))} | photo check: {cred_of(x)} | status: {x['status']} | {x['summary'][:200]}")
    prompt = f"""
    You are an operations assistant to a disaster control room commander.
    Using ONLY the incident list below, write a situation report in plain English, maximum 180 words, short bullet points, no invented facts:
    - Overall picture (one sentence)
    - Top 3 priorities among incidents with status New (not yet dispatched), each with the recommended first action
    - One line on incidents already Dispatched or Resolved (if any)
    - Any resource needed at several incidents (possible conflict)
    - Reports needing verification (only those whose photo check is "Partly consistent" or "Inconsistent"; ignore "Not checked" and "Not provided")

    Incidents (ranked by priority):
    """ + "\n".join(lines)
    response = gen(model=MODEL_NAME, contents=prompt)
    return response.text.strip()

# ---------------------------------------------------------
# 6d. Track my report (PUBLIC: no sign-in; shows only safe details of ONE report, found by its secret code)
# ---------------------------------------------------------
TRACK_STEPS = ["Submitted", "Dispatched", "Resolved"]
TRACK_LABEL = {"New": "Submitted", "Dispatched": "Dispatched", "Resolved": "Resolved"}   # citizen-friendly names
TRACK_MAX_SESSION = 8     # wrong codes allowed per visitor ...
TRACK_MAX_ALL = 30        # ... and in total, within 5 minutes
TRACK_WINDOW_S = 300

def lookup_report(code: str):
    """Returns ONLY: incident type, submission time, a short area name and the status. Nothing else is ever read from the database."""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT hazard_type, created_at, location, status, status_updated_at FROM reports WHERE tracking_code = ?",
                          (code,)).fetchone()
    if not row:
        return None
    hazard, created, location, status, updated = row
    status = status if status in STATUS_ORDER else "New"
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    return {"incident": hazard or "Incident", "submitted_at": created, "area": ", ".join(parts[-2:])[:80] or "Not specified",
            "status": TRACK_LABEL[status], "updated_at": updated if status != "New" else None}

def track_report(raw_code: str):
    """Returns (result or None, message). Wrong codes are rate-limited so codes cannot be guessed."""
    now, tracker = time.time(), _login_tracker()
    mine = [t for t in st.session_state.get("track_fails", []) if now - t < TRACK_WINDOW_S]
    with tracker["lock"]:
        everyone = [t for t in tracker["fails"].get("track:*", []) if now - t < TRACK_WINDOW_S]
        tracker["fails"]["track:*"] = everyone
    st.session_state["track_fails"] = mine
    if len(mine) >= TRACK_MAX_SESSION or len(everyone) >= TRACK_MAX_ALL:
        return None, "Too many attempts. Please wait a few minutes and try again."
    code = normalize_code(raw_code)
    result = lookup_report(code) if code else None
    if result is None:
        st.session_state["track_fails"] = mine + [now]
        with tracker["lock"]:
            tracker["fails"].setdefault("track:*", []).append(now)
        return None, "We could not find a report with that code. Please check the code and try again."
    return result, ""

SUBMIT_MAX_SESSION = 5    # reports per visitor ...
SUBMIT_MAX_ALL = 60       # ... and in total, within the window below (protects the free AI quota)
SUBMIT_WINDOW_S = 600

def submit_wait() -> int:
    """Seconds a visitor must wait before sending another report (0 = allowed)."""
    now, tracker = time.time(), _login_tracker()
    mine = [t for t in st.session_state.get("submit_times", []) if now - t < SUBMIT_WINDOW_S]
    st.session_state["submit_times"] = mine
    with tracker["lock"]:
        everyone = [t for t in tracker["fails"].get("submit:*", []) if now - t < SUBMIT_WINDOW_S]
        tracker["fails"]["submit:*"] = everyone
    waits = []
    if len(mine) >= SUBMIT_MAX_SESSION:
        waits.append(SUBMIT_WINDOW_S - (now - mine[0]))
    if len(everyone) >= SUBMIT_MAX_ALL:
        waits.append(SUBMIT_WINDOW_S - (now - everyone[0]))
    return int(max(waits)) + 1 if waits else 0

def record_submit():
    now, tracker = time.time(), _login_tracker()
    st.session_state["submit_times"] = st.session_state.get("submit_times", []) + [now]
    with tracker["lock"]:
        tracker["fails"].setdefault("submit:*", []).append(now)

DUPLICATE_SUBMIT_WINDOW_S = 120   # an identical report from the same visitor within this time is treated as an accidental repeat

def submission_fingerprint(location, report_text, photo, audio) -> str:
    h = hashlib.sha256()
    for part in ((location or "").strip().lower().encode(), (report_text or "").strip().encode(),
                 photo.getvalue() if photo is not None else b"", audio[0] if audio is not None else b""):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()

def recent_submission(fp: str):
    rec = st.session_state.get("last_submission")
    return rec if rec and rec["fp"] == fp and time.time() - rec["at"] < DUPLICATE_SUBMIT_WINDOW_S else None

def fmt_time(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")

def ago_text(ts: str) -> str:
    m = int(minutes_since(ts))
    return "just now" if m < 1 else f"{m} min ago" if m < 60 else f"{m // 60} h {m % 60} min ago" if m < 1440 else f"{m // 1440} day(s) ago"

# ---------------------------------------------------------
# 7. Demo data (clearly labelled, for presenting a full map)
# ---------------------------------------------------------
DEMO = [
    ("Chamoli, Uttarakhand", 30.40, 79.32, "Landslide", 4, 28, "Rockfall blocked NH-58; passenger bus stranded behind debris.", ["JCB / bulldozers", "SDRF team", "Water and food"], 40),
    ("Guwahati, Assam", 26.14, 91.74, "Flash Flood", 5, 60, "Water entering homes in low-lying colony; families on rooftops.", ["NDRF boats", "Medical team", "Relief camp"], 12),
    ("Wayanad, Kerala", 11.68, 76.13, "Landslide", 4, 15, "Hillside slip near homes; several families cut off.", ["Rescue team", "Earthmover", "Ambulance"], 25),
    ("Dharavi, Mumbai", 19.04, 72.85, "Building Fire", 3, 8, "Fire in a shop row; smoke spreading, no injuries reported yet.", ["Fire tenders", "Police"], 5),
    ("Yamuna Bazaar, Delhi", 28.66, 77.25, "River Flooding", 2, 5, "River level rising near the embankment; road partially waterlogged.", ["Drainage pump", "Traffic police"], 55),
]

def load_demo_data():
    require_authority()
    for loc, lat, lon, hz, sev, ppl, summ, res, mins_ago in DEMO:
        save_report(dict(
            created_at=(datetime.now() - timedelta(minutes=mins_ago)).isoformat(), location=loc, lat=lat, lon=lon,
            hazard_type=hz, severity=sev, severity_reason="Demo data", people_count=ppl, people_text=f"About {ppl} people",
            resources=json.dumps(res), summary="[DEMO] " + summ, language="English", image_path="", is_demo=1))

# ---------------------------------------------------------
# 8. UI  (look and feel only. Everything above this line is unchanged.)
# ---------------------------------------------------------
esc = html.escape
HEX = {"High": "#FF4D5E", "Medium": "#FFA63D", "Low": "#34D399"}
CRED_HEX = {"Consistent": "#34D399", "Partly consistent": "#FFA63D", "Inconsistent": "#FF4D5E"}
MUTED = "#8FA3BF"
BLUE = "#4DA3FF"
STATUS_HEX = {"New": "#4DA3FF", "Dispatched": "#A78BFA", "Resolved": "#6B7C96"}
STATUS_MARKER = {"Dispatched": "purple", "Resolved": "gray"}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0B1220;--panel:#121B2E;--panel2:#16233B;--line:#22304A;--text:#E6EDF7;--muted:#8FA3BF;--brand:#FF5A4E;--blue:#4DA3FF;}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;}
.stApp{background:radial-gradient(1100px 480px at 88% -8%,rgba(77,163,255,.12),transparent 60%),radial-gradient(900px 420px at -5% 0%,rgba(255,90,78,.07),transparent 55%),var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stAppDeployButton"],.stDeployButton,.stAppDeployButton{display:none !important;}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1760px;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:6px 2px 14px;}
.brand{display:flex;align-items:center;gap:14px;}
.logo{width:46px;height:46px;border-radius:13px;background:linear-gradient(135deg,#FF5A4E,#E8384F);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:24px;color:#fff;box-shadow:0 6px 22px rgba(255,90,78,.35);}
.bt{font-size:26px;font-weight:800;letter-spacing:.04em;line-height:1.1;}
.bs{font-size:13px;color:var(--muted);margin-top:2px;}
.live{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:700;letter-spacing:.12em;color:#7BE5BC;background:#34D39914;border:1px solid #34D39944;padding:6px 14px;border-radius:999px;}
.pulse{width:8px;height:8px;border-radius:50%;background:#34D399;box-shadow:0 0 0 0 rgba(52,211,153,.7);animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.6);}70%{box-shadow:0 0 0 9px rgba(52,211,153,0);}100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:48px;padding:0 22px;border-radius:12px 12px 0 0;color:var(--muted);font-weight:600;font-size:15px;}
.stTabs [data-baseweb="tab"]:hover{color:var(--text);background:#ffffff08;}
.stTabs [aria-selected="true"]{color:var(--text);background:var(--panel);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--brand);height:3px;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
div[data-testid="stVerticalBlockBorderWrapper"]>div[data-testid="stVerticalBlock"]{gap:.6rem;}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:600;border:1px solid #2A3E63;background:var(--panel2);color:var(--text);transition:all .15s ease;}
.stButton>button:hover{border-color:var(--blue);color:#fff;transform:translateY(-1px);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#FF5A4E,#E8384F);border:none;color:#fff;font-weight:700;font-size:16px;height:52px;box-shadow:0 8px 24px rgba(232,56,79,.30);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);color:#fff;}
.stTextInput input,.stTextArea textarea,.stNumberInput input{border-radius:12px;background:#0F1A2E;border:1px solid #2A3E63;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(77,163,255,.18);}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #34476A;border-radius:14px;background:#0F1A2E;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--blue);}
[data-testid="stImage"] img{border-radius:12px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:var(--panel);overflow:hidden;}
[data-testid="stExpander"] summary{font-weight:600;padding:14px 16px;}
iframe{border-radius:14px;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
.sec-title{font-size:23px;font-weight:800;margin-top:4px;}
.sec-sub{color:var(--muted);font-size:13.5px;margin-bottom:10px;}
.sec-h{font-size:18px;font-weight:700;margin:26px 0 10px;display:flex;align-items:center;justify-content:space-between;}
.sec-h small{color:var(--muted);font-weight:500;font-size:13px;}
.legend{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12.5px;font-weight:500;}
.legend i{width:10px;height:10px;border-radius:50%;display:inline-block;margin-left:10px;}
.step{display:flex;align-items:center;gap:12px;}
.step b{width:28px;height:28px;border-radius:50%;background:var(--brand);color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;flex:none;}
.stp-t{font-weight:700;font-size:16px;line-height:1.2;}
.stp-h{color:var(--muted);font-size:12.5px;}
.or{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin:6px 0 2px;}
.or:before,.or:after{content:'';flex:1;height:1px;background:var(--line);}
.pill{display:inline-flex;align-items:center;padding:4px 12px;border-radius:999px;font-size:12.5px;font-weight:700;border:1px solid;white-space:nowrap;margin:0 6px 4px 0;}
.chip{display:inline-block;margin:0 6px 6px 0;padding:5px 11px;border-radius:9px;background:#1B2A45;border:1px solid #2A3E63;font-size:13px;}
.op{color:var(--muted);margin:0 4px;font-weight:700;}
.sc{font-size:20px;font-weight:800;margin-left:4px;}
.field{margin-bottom:16px;}
.fl{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:5px;}
.fv{font-size:15px;line-height:1.6;}
.fn{font-size:12.5px;color:var(--muted);margin-top:4px;}
.quote{display:block;border-left:3px solid var(--blue);padding:9px 13px;background:#0F1A2E;border-radius:6px;font-style:italic;}
.banner{padding:12px 15px;border-radius:12px;font-size:14px;margin:10px 0;border:1px solid;line-height:1.5;}
.banner.ok{background:#34D39914;border-color:#34D39955;color:#8FEBC6;}
.banner.warn{background:#FFA63D14;border-color:#FFA63D55;color:#FFC983;}
.banner.info{background:#4DA3FF14;border-color:#4DA3FF55;color:#A9D1FF;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:14px;}
.result{background:linear-gradient(180deg,#16233B,#121B2E);border:1px solid var(--line);border-radius:18px;padding:22px;margin-bottom:14px;}
.r-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.r-title{font-size:32px;font-weight:800;margin-top:2px;line-height:1.15;}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px;}
.stat{background:#0F1A2E;border:1px solid var(--line);border-radius:13px;padding:13px 15px;}
.sl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:700;}
.sv{font-size:32px;font-weight:800;line-height:1.25;}
.sv span{font-size:15px;color:var(--muted);font-weight:600;}
.ss{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.45;}
.sev{display:flex;gap:4px;margin-top:6px;}
.sev i{flex:1;height:6px;border-radius:3px;display:block;}
.empty{border:1.5px dashed #2A3E63;border-radius:18px;padding:34px 26px;text-align:center;background:#0F1A2E88;}
.e-ic{font-size:34px;}
.e-t{font-size:20px;font-weight:700;margin-top:6px;}
.e-s{color:var(--muted);font-size:14px;margin-top:4px;}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:22px;text-align:left;}
.how div{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:13.5px;font-weight:600;}
.how b{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--brand);color:#fff;align-items:center;justify-content:center;font-size:12px;margin-right:8px;}
.how small{display:block;color:var(--muted);font-weight:500;margin-top:4px;font-size:12px;}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;}
.kl{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:700;}
.kl i{width:9px;height:9px;border-radius:50%;display:inline-block;}
.kn{font-size:42px;font-weight:800;line-height:1.15;margin-top:4px;}
.ks{font-size:12.5px;color:var(--muted);}
.inc-head{margin-bottom:12px;}
.inc-score{color:var(--muted);font-size:13px;margin-left:6px;}
.inc-score b{color:var(--text);font-size:15px;}
.nophoto{height:170px;border:1.5px dashed #2A3E63;border-radius:12px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:13px;background:#0F1A2E;text-align:center;padding:10px;}
.merged{margin:4px 0 0 0;padding-left:18px;font-size:13.5px;line-height:1.55;color:#C9D6EA;}
.trk{display:flex;align-items:center;margin:18px 0 4px;}
.trk-s{display:flex;flex-direction:column;align-items:center;gap:6px;flex:none;width:96px;text-align:center;font-size:12.5px;font-weight:600;color:var(--muted);}
.trk-s b{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#1B2A45;border:2px solid #2A3E63;font-size:14px;}
.trk-s.done{color:var(--text);}
.trk-s.done b{background:#34D399;border-color:#34D399;color:#05281D;}
.trk-s.now{color:var(--text);}
.trk-s.now b{background:var(--brand);border-color:var(--brand);color:#fff;box-shadow:0 0 0 5px rgba(255,90,78,.20);}
.trk-l{flex:1;height:3px;background:#2A3E63;margin:0 2px 24px;border-radius:2px;}
.trk-l.done{background:#34D399;}
.trk-code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:30px;font-weight:800;letter-spacing:.08em;margin:6px 0 6px;color:#fff;}
@media (max-width:900px){.stats,.how{grid-template-columns:1fr;}.r-title{font-size:26px;}}
</style>"""

def H(markup, target=st):
    target.markdown("".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)

def pill(text, color):
    return f"<span class='pill' style='color:{color};background:{color}22;border-color:{color}66'>{esc(str(text))}</span>"

def chip(text):
    return f"<span class='chip'>{esc(str(text))}</span>"

def sev_bar(sev, color):
    return "<div class='sev'>" + "".join(f"<i style='background:{color if n < sev else '#22304A'}'></i>" for n in range(5)) + "</div>"

def field(label, body, note=""):
    extra = f"<div class='fn'>{note}</div>" if note else ""
    return f"<div class='field'><div class='fl'>{label}</div><div class='fv'>{body}</div>{extra}</div>"

def banner(kind, text):
    return f"<div class='banner {kind}'>{esc(text)}</div>"

def step_head(n, title, hint):
    return f"<div class='step'><b>{n}</b><div><div class='stp-t'>{title}</div><div class='stp-h'>{hint}</div></div></div>"

def kpi(label, value, color, sub, value_color="#E6EDF7"):
    return (f"<div class='kpi'><div class='kl'><i style='background:{color}'></i>{label}</div>"
            f"<div class='kn' style='color:{value_color}'>{value}</div><div class='ks'>{sub}</div></div>")

def formula(sev, pf, wf, cf, score, people_unknown=False):
    op = "<span class='op'>×</span>"
    parts = [chip(f"{sev} severity"), chip(f"{pf:.1f} people" + (" (unknown, assumed)" if people_unknown else "")), chip(f"{wf:.1f} waiting")]
    if cf > 1.0:
        parts.append(chip(f"{cf:.2f} confirmations"))
    raw = sev * pf * wf * cf
    out = op.join(parts) + f"<span class='op'>=</span><b class='sc'>{raw:.1f}</b>"
    if score > raw + 1e-9:   # severe emergencies never fall below their minimum level
        out += "<span class='op'>→</span>" + chip("minimum for severity " + str(sev)) + f"<b class='sc'>{score:.1f}</b>"
    return out

def render_login():
    H("<div class='sec-title'>Authority sign-in</div>"
      "<div class='sec-sub'>The control room is restricted to authorised personnel. "
      "Citizens can submit reports from the first tab without signing in.</div>")
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        H(banner("warn", notice))
    if not load_authorities():
        H(banner("warn", "Authority sign-in is not configured on this server, so the control room stays locked. "
                         "Add AUTHORITY_USERNAME and AUTHORITY_PASSWORD_HASH to the .env file (see make_password_hash.py)."))
        return
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        with st.container(border=True):
            H(step_head("🔒", "Control room access", "Authorised personnel only"))
            with st.form("authority_login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                ok, msg = attempt_login(username, password)
                if ok:
                    st.session_state["auth"] = {"user": username.strip().lower()[:64], "last": time.time()}
                    st.rerun()
                H(banner("warn", msg))

H(CSS)
H("<div class='topbar'><div class='brand'><div class='logo'>R</div><div><div class='bt'>RAHAT AI</div>"
  "<div class='bs'>Rapid Assessment &amp; Help Allocation Tool</div></div></div>"
  "<div class='live'><span class='pulse'></span>LIVE</div></div>")

def saved_confirmation(tracking_code, coords, approx):
    """Banner that the report is saved + the citizen's tracking code (shown for every saved report)."""
    if coords:
        H(banner("ok", "✅ Report saved and sent to the control room." + (" The pin uses the approximate centre of the named place." if approx else "")))
    else:
        H(banner("warn", "Report saved, but the location could not be found on the map. "
                         "Enter latitude/longitude under the optional coordinates box and submit again."))

    # tracking code for the citizen (shown now; also pre-filled in the Track report tab for this visit)
    st.session_state["track_code_input"] = tracking_code
    H("<div class='card'><div class='eyebrow'>Your tracking code</div>"
      f"<div class='trk-code'>{esc(tracking_code)}</div>"
      "<div class='fn'>Save this code now. Use it in the <b>Track report</b> tab to check the status of your report. "
      "It is shown only once.</div></div>")
    st.code(tracking_code, language=None)
    receipt = (f"RAHAT AI tracking code\n\nCode: {tracking_code}\nSubmitted: {datetime.now(IST).strftime('%d %b %Y, %I:%M %p IST')}\n\n"
               "To check your report, open RAHAT AI and use the 'Track report' tab.\nIf someone is in immediate danger, call 112.\n")
    try:
        st.download_button("Save code as a text file", receipt, file_name=f"{tracking_code}.txt", on_click="ignore", icon=":material/download:")
    except TypeError:   # older Streamlit versions
        st.download_button("Save code as a text file", receipt, file_name=f"{tracking_code}.txt")

tab_submit, tab_track, tab_dash = st.tabs(["Submit report", "Track report", "Control room"])

# ---------------- Tab 1: submit a report ----------------
with tab_submit:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        H("<div class='sec-title'>New incident report</div>"
          "<div class='sec-sub'>Add the location and a description or voice note. A photo is optional but helps.</div>")

        with st.container(border=True):
            H(step_head(1, "Photo (optional)", "Helps us verify the report. You can skip it and use text or voice"))
            uploaded_file = st.file_uploader("Upload incident photo", type=["jpg", "jpeg", "png", "webp"],
                                             label_visibility="collapsed",
                                             help="Upload an image showing the ground-level situation")
            image = None
            if uploaded_file:
                try:
                    if uploaded_file.size > MAX_PHOTO_MB * 1024 * 1024:
                        raise ValueError("too large")
                    image = Image.open(uploaded_file)
                    if image.width * image.height > MAX_PHOTO_PIXELS:
                        raise ValueError("too many pixels")
                    image.load()
                    st.image(image, caption="Uploaded ground photo", use_container_width=True)
                except Exception:
                    image = None
                    st.error(f"This photo could not be used. Please upload a normal JPG, PNG or WebP photo under {MAX_PHOTO_MB} MB.")

        with st.container(border=True):
            H(step_head(2, "Location", "Area, landmark, town and state"))
            location = st.text_input("Incident location", label_visibility="collapsed", max_chars=200,
                                     placeholder="e.g., Ward 12, Near Bus Stand, Joshimath, Uttarakhand")
            with st.expander("Exact coordinates (optional, use if the map cannot find your location)"):
                c1, c2 = st.columns(2)
                manual_lat = c1.number_input("Latitude", value=0.0, format="%.5f")
                manual_lon = c2.number_input("Longitude", value=0.0, format="%.5f")

        with st.container(border=True):
            H(step_head(3, "What happened", "Type it, speak it, or both. Hindi, English or Hinglish"))
            report_text = st.text_area("Incident details", label_visibility="collapsed", height=120, max_chars=1500,
                                       placeholder="e.g., यहाँ भारी बारिश से पुल टूट गया है और 15 लोग दूसरी तरफ फंसे हैं...")
            H("<div class='or'>or send a voice note</div>")
            recorded = st.audio_input("Record a voice note", label_visibility="collapsed") if hasattr(st, "audio_input") else None
            uploaded_audio = st.file_uploader("...or upload an audio file", type=list(AUDIO_MIME), key="voice_upload")
            audio = None   # (bytes, mime_type)
            if recorded is not None:
                audio = (recorded.getvalue(), "audio/wav")
            elif uploaded_audio is not None:
                audio = (uploaded_audio.getvalue(), AUDIO_MIME[uploaded_audio.name.rsplit(".", 1)[-1].lower()])

        submit_btn = st.button("Analyze & triage report", type="primary", use_container_width=True, icon=":material/bolt:")

    with col_output:
        H("<div class='sec-title'>Triage assessment</div>"
          "<div class='sec-sub'>AI analysis of the photo, text and voice note appears here.</div>")
        if submit_btn:
            fp = submission_fingerprint(location, report_text, uploaded_file, audio)
            if uploaded_file is not None and image is None:
                st.warning("The photo could not be used. Please remove it or upload a different one, or send the report without a photo.")
            elif not location.strip():
                st.warning("Please enter the incident location.")
            elif not report_text.strip() and audio is None:
                st.warning("Please type a short incident description or record a voice note.")
            elif audio is not None and len(audio[0]) > 15_000_000:
                st.warning("The voice note is too large. Please keep it under about 2 minutes.")
            elif (repeat := recent_submission(fp)):   # same visitor, same report, moments ago: do not save or assess it again
                H(banner("info", "This report was already submitted a moment ago, so it was not sent again. Your tracking code is below."))
                saved_confirmation(repeat["code"], repeat["coords"], repeat["approx"])
            elif submit_wait() > 0:
                st.warning(f"You have sent several reports in a short time. Please wait about {submit_wait() // 60 + 1} minute(s) "
                           "before sending another. If someone is in immediate danger, call 112.")
            else:
                record_submit()
                location, report_text = location.strip()[:200], report_text.strip()[:1500]
                # location -> coordinates (no Gemini involved; failures are handled inside)
                approx = False
                if manual_lat != 0.0 or manual_lon != 0.0:
                    coords = (manual_lat, manual_lon)
                else:
                    coords = geocode(location)
                    if coords is None:   # online geocoder failed: use the approximate centre of a known place
                        coords = fallback_place(location)
                        approx = coords is not None

                # SAVE FIRST (text, location, photo, voice note, tracking code), THEN ask Gemini
                tracking_code, data, ai_error = None, None, None
                with st.spinner("Analyzing visual hazards and cross-referencing text..."):
                    try:
                        tracking_code, data, ai_error = intake_report(image, audio, location, report_text, coords)
                        st.session_state["last_submission"] = dict(fp=fp, code=tracking_code, coords=coords, approx=approx, at=time.time())
                    except Exception as e:
                        st.error(str(e))

                if tracking_code and data is None:   # saved, but the AI assessment could not be completed
                    H(banner("warn", f"Your report was submitted successfully, but its AI assessment could not be completed ({AI_REASON[ai_error]}). "
                                     "The control room has your report and will review it. If someone is in immediate danger, call 112."))
                    saved_confirmation(tracking_code, coords, approx)

                if data:
                    sev = int(data.get("severity", 1))
                    people = int(data.get("people_at_risk_count", UNKNOWN_PEOPLE))   # -1 = unknown
                    resources = data.get("resources_needed", []) or []

                    score, pf, wf, cf, level = priority(sev, people, 0)
                    lc = HEX[level]
                    cred = data.get("photo_report_match", "Not checked")
                    cc = CRED_HEX.get(cred, MUTED)
                    hazard = esc(str(data.get("hazard_type", "Unknown")))
                    crit = "<div class='ss' style='color:#FF8A96;font-weight:700'>CRITICAL</div>" if sev >= 4 else ""
                    unknown_note = f" Count unknown, so it is scored as if about {UNKNOWN_PEOPLE_ASSUMED} people were at risk." if people < 0 else ""

                    H(f"<div class='result'><div class='r-top'><div><div class='eyebrow'>AI triage result</div>"
                      f"<div class='r-title'>{hazard}</div></div>{pill(level + ' priority', lc)}</div>"
                      f"<div class='stats'>"
                      f"<div class='stat'><div class='sl'>Severity</div><div class='sv'>{sev}<span> / 5</span></div>{sev_bar(sev, lc)}{crit}</div>"
                      f"<div class='stat'><div class='sl'>People at risk</div><div class='sv'>{people_label(people)}</div>"
                      f"<div class='ss'>{esc(str(data.get('people_at_risk', 'N/A')))}{unknown_note}</div></div>"
                      f"<div class='stat'><div class='sl'>Photo vs report</div><div style='margin-top:8px'>{pill(cred, cc)}</div>"
                      f"<div class='ss'>{esc(str(data.get('credibility_note', '')))}</div></div>"
                      f"</div></div>")

                    body = field("Dispatch summary", esc(str(data.get("english_summary", "N/A"))))
                    if data.get("voice_transcript"):
                        body += field("Voice note heard as", f"<span class='quote'>{esc(str(data['voice_transcript']))}</span>")
                    body += field("Why this severity", esc(str(data.get("severity_reason", "N/A"))))
                    body += field("Recommended resources", "".join(chip(r) for r in resources) or "N/A")
                    body += field("Priority score", formula(sev, pf, wf, cf, score, people < 0),
                                  "The waiting factor grows the longer a report stays unattended.")
                    body += field("Location and language",
                                  chip("📍 " + location) + chip("Language: " + str(data.get("language_detected", "Unknown"))))
                    H(f"<div class='card'>{body}</div>")

                    saved_confirmation(tracking_code, coords, approx)
        else:
            H("<div class='empty'><div class='e-ic'>⚡</div><div class='e-t'>Awaiting an incident report</div>"
              "<div class='e-s'>Submit an incident report on the left to see the AI triage analysis here.</div>"
              "<div class='how'><div><b>1</b>Report<small>Photo, location, text or voice</small></div>"
              "<div><b>2</b>AI triage<small>Hazard, severity, people at risk</small></div>"
              "<div><b>3</b>Dispatch<small>Ranked on the control room map</small></div></div></div>")

# ---------------- Tab 2: track a report (public, no sign-in) ----------------
with tab_track:
    col_form, col_status = st.columns([1, 1], gap="large")
    with col_form:
        H("<div class='sec-title'>Track your report</div>"
          "<div class='sec-sub'>Enter the tracking code you received after submitting your report.</div>")
        with st.container(border=True):
            H(step_head("🔎", "Your tracking code", "It looks like RAH-7K2M9-QXD4F"))
            with st.form("track_form"):
                code_in = st.text_input("Tracking code", key="track_code_input", placeholder="RAH-XXXXX-XXXXX",
                                        label_visibility="collapsed", max_chars=24)
                track_btn = st.form_submit_button("Check status", type="primary", use_container_width=True)
        H(banner("info", "If someone is in immediate danger, call 112 right away. Do not wait for an update here."))

    with col_status:
        H("<div class='sec-title'>Report status</div><div class='sec-sub'>Only the status of your own report is shown.</div>")
        if track_btn:
            found, message = track_report(code_in)
            if found:
                status = found["status"]
                idx = TRACK_STEPS.index(status)
                flow = ""
                for i, name in enumerate(TRACK_STEPS):
                    cls = "done" if (i < idx or idx == 2) else "now" if i == idx else ""
                    flow += f"<div class='trk-s {cls}'><b>{'✓' if cls == 'done' else i + 1}</b>{name}</div>"
                    if i < 2:
                        flow += f"<div class='trk-l {'done' if i < idx else ''}'></div>"
                colour = {"Submitted": BLUE, "Dispatched": "#A78BFA", "Resolved": "#34D399"}[status]
                H(f"<div class='result'><div class='r-top'><div><div class='eyebrow'>Your report</div>"
                  f"<div class='r-title'>{esc(found['incident'])}</div></div>{pill(status, colour)}</div>"
                  f"<div class='trk'>{flow}</div></div>")
                message_for = {"Submitted": "Your report has been received and is waiting for the control room to review it.",
                               "Dispatched": "Help has been dispatched for this incident.",
                               "Resolved": "This incident has been marked as resolved."}[status]
                body = field("What this means", esc(message_for))
                body += field("Incident type", esc(found["incident"]))
                body += field("Submitted", f"{esc(fmt_time(found['submitted_at']))} <span class='fn'>({ago_text(found['submitted_at'])})</span>")
                body += field("Area", esc(found["area"]))
                body += field("Last update", f"{status}, {ago_text(found['updated_at'])}" if found["updated_at"] else "Not updated yet")
                H(f"<div class='card'>{body}</div>")
            else:
                H(banner("warn", message))
        else:
            H("<div class='empty'><div class='e-ic'>🔎</div><div class='e-t'>Check on your report</div>"
              "<div class='e-s'>Enter your tracking code on the left to see whether your report has been "
              "submitted, dispatched or resolved.</div></div>")

# NOTE: the control room block below must stay LAST (it stops the script for visitors who are not signed in).
# ---------------- Tab 3: control room dashboard ----------------
with tab_dash:
    # Access control: nothing below runs for public visitors (no data loading, no actions, no AI calls).
    if not is_authenticated():
        render_login()
        st.stop()

    t1, t2, t3, t4, t5, t6 = st.columns([2.6, 1.1, 1.4, 1.2, 1.5, 1.2], vertical_alignment="center")
    with t1:
        H(f"<div class='sec-title'>Control room</div><div class='sec-sub'>Signed in as <b>{esc(current_user())}</b> · live incident picture, ranked by priority</div>")
    if t2.button("Refresh", icon=":material/refresh:", use_container_width=True):
        st.rerun()
    if t3.button("Load demo data", icon=":material/dataset:", use_container_width=True):
        load_demo_data()
        st.rerun()
    confirm = t4.checkbox("Confirm reset")
    if t5.button("Delete all reports", icon=":material/delete:", disabled=not confirm, use_container_width=True):
        clear_reports()
        st.rerun()
    if t6.button("Sign out", icon=":material/logout:", use_container_width=True):
        sign_out()
        st.rerun()

    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash, icon="✅")

    df = load_reports()
    if df.empty:
        H(banner("info", "No reports yet. Submit one in the first tab, or click 'Load demo data'."))
    else:
        # merge duplicates into incidents, then compute live priority for each incident
        decisions = load_decisions()
        rows = merge_reports(df.to_dict("records"), decisions)
        for x in rows:
            mins = minutes_since(x["first_reported"])
            if x["status"] != "New" and x["status_updated_at"]:   # waiting stops growing once help is dispatched
                mins = max(0.0, (datetime.fromisoformat(x["status_updated_at"]) - datetime.fromisoformat(x["first_reported"])).total_seconds() / 60)
            score, pf, wf, cf, level = priority(int(x["severity"]), int(x["people_count"]), mins, x["n_confirm"])
            x.update(minutes=mins, score=score, pf=pf, wf=wf, cf=cf, level=level)
        rows.sort(key=lambda x: (STATUS_ORDER[x["status"]], -x["score"]))   # incidents awaiting dispatch first
        reviews = find_review_pairs(rows, decisions)

        merged_away = len(df) - len(rows)
        n_new = sum(1 for x in rows if x["status"] == "New")
        n_disp = sum(1 for x in rows if x["status"] == "Dispatched")
        n_res = sum(1 for x in rows if x["status"] == "Resolved")
        counts = {lv: sum(1 for x in rows if x["level"] == lv and x["status"] == "New") for lv in ("High", "Medium", "Low")}
        k1, k2, k3, k4 = st.columns(4)
        H(kpi("Active incidents", n_new + n_disp, BLUE, f"{n_new} awaiting dispatch · {n_disp} dispatched"), k1)
        H(kpi("High priority", counts["High"], HEX["High"], "Awaiting dispatch", HEX["High"]), k2)
        H(kpi("Medium priority", counts["Medium"], HEX["Medium"], "Awaiting dispatch", HEX["Medium"]), k3)
        H(kpi("Low priority", counts["Low"], HEX["Low"], "Awaiting dispatch", HEX["Low"]), k4)
        if merged_away:
            H(banner("ok", f"🔗 {len(df)} reports received → {len(rows)} incidents ({merged_away} duplicate report(s) merged)"))

        # possible duplicates: NOT merged until an authority confirms
        if reviews:
            H(banner("warn", f"{len(reviews)} possible duplicate(s) need your review. They stay separate until you confirm."))
            with st.expander(f"Possible duplicates ({len(reviews)})", expanded=True):
                for rv in reviews[:10]:
                    A, B, info = rv["a_inc"], rv["b_inc"], rv["info"]
                    key = f"{rv['a']['id']}_{rv['b']['id']}"
                    c1, c2, c3 = st.columns([2, 2, 1.6], gap="medium")
                    for col, inc in ((c1, A), (c2, B)):
                        H(f"<div class='card'><div class='eyebrow'>{esc(inc['hazard_type'])} · {esc(inc['status'])}</div>"
                          f"<div class='fv'><b>{esc(inc['location'])}</b></div>"
                          f"<div class='fn'>{ago_text(inc['first_reported'])} · {inc['n_reports']} report(s) · People: {people_label(inc['people_count'])} · Severity {inc['severity']}/5</div>"
                          f"<div class='fv' style='margin-top:6px'>{esc(str(inc['summary'])[:220])}</div></div>", col)
                        if inc["image_path"] and os.path.exists(inc["image_path"]):
                            col.image(inc["image_path"], width=200)
                    with c3:
                        colour = HEX["High"] if info["level"] == "auto" else HEX["Medium"]
                        H(pill(f"Match {info['score']}%", colour) + "".join(chip(t) for t in info["reasons"]))
                        if st.button("Same incident: merge", key=f"rv_m_{key}", type="primary", use_container_width=True):
                            save_decision([(x["id"], y["id"]) for x in A["members"] for y in B["members"]], "merge", current_user())
                            st.session_state["flash"] = "Merged into one incident. Its status follows the less advanced of the two, so nothing is skipped."
                            st.rerun()
                        if st.button("Different incidents: keep separate", key=f"rv_s_{key}", use_container_width=True):
                            save_decision([(x["id"], y["id"]) for x in A["members"] for y in B["members"]], "separate", current_user())
                            st.session_state["flash"] = "Kept as separate incidents."
                            st.rerun()

        bar1, bar2 = st.columns([1.3, 3], vertical_alignment="center")
        show_resolved = bar2.toggle(f"Show resolved incidents ({n_res})", value=False)
        visible = [x for x in rows if show_resolved or x["status"] != "Resolved"]

        # situation report (AI summary of all incidents)
        if bar1.button("Generate situation report", icon=":material/summarize:"):
            with st.spinner("Summarising all incidents..."):
                try:
                    st.session_state["sitrep"] = (datetime.now(IST).strftime("%d %b %Y, %H:%M IST"), draft_sitrep(visible or rows))
                except Exception as e:
                    st.error(f"Could not generate report: {e}")
        if "sitrep" in st.session_state:
            ts, txt = st.session_state["sitrep"]
            with st.container(border=True):
                H(f"<div class='eyebrow'>Situation report</div><div class='fn' style='margin:2px 0 8px'>Generated {esc(ts)}</div>")
                st.markdown(txt)

        if not visible:
            H(banner("ok", "All incidents are resolved. Nothing is awaiting dispatch. Reopen one from 'Resolved incidents' below if needed."))
        else:
            # map
            H("<div class='sec-h'>Live incident map<span class='legend'><i style='background:#FF4D5E'></i>High"
              "<i style='background:#FFA63D'></i>Medium<i style='background:#34D399'></i>Low"
              "<i style='background:#A78BFA'></i>Dispatched<i style='background:#6B7C96'></i>Resolved</span></div>")
            mapped = [x for x in visible if has_coords(x)]
            if mapped:
                lats = [x["lat"] for x in mapped]; lons = [x["lon"] for x in mapped]
                center = [sum(lats) / len(lats), sum(lons) / len(lons)]
                def fit(span, pixels):   # highest zoom at which this span still fits on screen
                    return 11 if span < 0.01 else math.log2(0.85 * pixels * 360 / (256 * 1.15 * span))
                zoom = int(max(3, min(11, fit(max(lats) - min(lats), 620), fit(max(lons) - min(lons), 1400))))
                fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
                for rank, x in enumerate(visible, start=1):
                    if not has_coords(x):
                        continue
                    popup = (f"<b>#{rank} {html.escape(x['hazard_type'])}</b><br>"
                             f"{html.escape(x['location'])}<br>"
                             f"Status: <b>{x['status']}</b><br>"
                             f"Severity {x['severity']}/5 | People: {people_label(x['people_count'])} | Reports: {x['n_reports']}<br>"
                             f"Priority: <b>{x['level']}</b> (score {x['score']:.1f})<br>"
                             f"{html.escape(x['summary'])}")
                    pin = LEVEL_COLOR[x["level"]] if x["status"] == "New" else STATUS_MARKER[x["status"]]
                    folium.Marker([x["lat"], x["lon"]], popup=folium.Popup(popup, max_width=300),
                                  tooltip=f"#{rank} {x['hazard_type']} - {x['level']} - {x['status']}",
                                  icon=folium.Icon(color=pin, icon="exclamation-sign")).add_to(fmap)
                sig = "".join(x["status"][0] + x["level"][0] for x in mapped)
                map_key = f"map_{zoom}_{center[0]:.2f}_{center[1]:.2f}_{sig}"   # new key = map re-fits when incidents change
                try:   # pass centre and zoom explicitly so the view is always fitted to the pins
                    st_folium(fmap, center=center, zoom=zoom, key=map_key, height=620, use_container_width=True, returned_objects=[])
                except TypeError:
                    st_folium(fmap, key=map_key, height=620, use_container_width=True, returned_objects=[])
            if len(mapped) < len(visible):
                st.warning(f"{len(visible) - len(mapped)} incident(s) have no coordinates and are not shown on the map.")

            # ranked table
            H("<div class='sec-h'>Ranked incident list<small>awaiting dispatch first, then by priority</small></div>")
            table = pd.DataFrame([{
                "Rank": i, "Priority": f"{LEVEL_ICON[x['level']]} {x['level']}", "Status": x["status"], "Score": round(x["score"], 1),
                "Hazard": x["hazard_type"], "Severity": x["severity"], "People": people_label(x["people_count"]),
                "Reports": x["n_reports"], "Photo check": f"{CRED_ICON.get(cred_of(x), 'ℹ️')} {cred_of(x)}", "Waiting (min)": int(x["minutes"]), "Location": x["location"], "Summary": x["summary"],
            } for i, x in enumerate(visible, start=1)])
            st.dataframe(table, use_container_width=True, hide_index=True, column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=float(max(table["Score"].max(), 1)), format="%.1f"),
                "Severity": st.column_config.NumberColumn("Severity", format="%d / 5", width="small"),
                "People": st.column_config.TextColumn("People", width="small"),
                "Reports": st.column_config.NumberColumn("Reports", width="small"),
                "Summary": st.column_config.TextColumn("Summary", width="large"),
            })

            # details, status actions and alert for top 5
            H("<div class='sec-h'>Top incidents<small>details, dispatch status and responder alerts</small></div>")
            for i, x in enumerate(visible[:5], start=1):
                with st.expander(f"#{i}   {x['hazard_type']}  ·  {x['location']}  ·  {x['level']} priority  ·  {x['status']}", expanded=(i == 1)):
                    cr = cred_of(x)
                    H(f"<div class='inc-head'>{pill(x['level'] + ' priority', HEX[x['level']])}"
                      f"{pill(x['status'], STATUS_HEX[x['status']])}"
                      f"{pill(str(x['n_reports']) + ' report(s)', MUTED)}"
                      f"{pill('Photo: ' + cr, CRED_HEX.get(cr, MUTED))}"
                      f"<span class='inc-score'>Score <b>{x['score']:.1f}</b></span></div>")
                    d1, d2 = st.columns([1, 2], gap="large")
                    if x["image_path"] and os.path.exists(x["image_path"]):
                        d1.image(x["image_path"], use_container_width=True)
                    else:
                        H("<div class='nophoto'>No photo provided</div>", d1)

                    body = field("Summary", esc(str(x["summary"])))
                    body += field("People at risk", esc(str(x["people_text"])))
                    body += field("Why this severity", esc(str(x["severity_reason"])))
                    if isinstance(x.get("transcript"), str) and x["transcript"].strip():
                        body += field("Voice note transcript", f"<span class='quote'>{esc(x['transcript'])}</span>")
                    if cr in ("Partly consistent", "Inconsistent"):
                        body += banner("warn", f"Verify before dispatch: {x.get('credibility_note', '')}")
                    excluded = sum(1 for m in x["members"] if cred_of(m) == "Inconsistent")
                    if excluded and cr != "Inconsistent":
                        body += banner("warn", f"{excluded} report(s) failed the photo check. They are shown below but excluded from severity, people count and confirmations.")
                    body += field("Resources needed", "".join(chip(r) for r in json.loads(x["resources"] or "[]")) or "N/A")
                    body += field("Priority score", formula(x["severity"], x["pf"], x["wf"], x["cf"], x["score"], x["people_count"] < 0))
                    if x["status"] != "New" and x["status_updated_at"]:
                        body += field("Response", f"{esc(x['status'])} {int(minutes_since(x['status_updated_at']))} min ago, "
                                                  f"after {int(x['minutes'])} min waiting")
                    if x["n_reports"] > 1:
                        items = "".join(
                            f"<li>{int(minutes_since(m['created_at']))} min ago ({esc(str(m['language'] or 'n/a'))}) "
                            f"{CRED_ICON.get(cred_of(m), '')}{' <i>(excluded from score)</i>' if cred_of(m) == 'Inconsistent' else ''}{' <i>(confirmed by authority)</i>' if m['id'] in x['forced_ids'] else ''}: {esc(m['summary'][:140])}{'…' if len(m['summary']) > 140 else ''}</li>"
                            for m in x["members"])
                        body += field(f"🔗 Merged from {x['n_reports']} reports", f"<ul class='merged'>{items}</ul>")
                    H(body, d2)
                    for m in x["members"]:   # voice notes whose AI transcription is still pending can be played here
                        ap = m.get("audio_path")
                        if is_pending(m) and isinstance(ap, str) and ap and os.path.exists(ap):
                            d2.caption("Voice note (AI transcription pending)")
                            d2.audio(ap)

                    restore_to = min((prev_of(m) for m in x["members"] if status_of(m) == x["status"]),
                                     key=lambda name: STATUS_ORDER[name], default="New")
                    a1, a2, a3 = d2.columns(3)
                    if a1.button("Mark dispatched", key=f"disp_{x['id']}", disabled=x["status"] != "New",
                                 icon=":material/local_shipping:", use_container_width=True):
                        set_status(x["member_ids"], "Dispatched")
                        st.session_state["flash"] = "Incident marked as dispatched."
                        st.rerun()
                    if a2.button("Mark resolved", key=f"res_{x['id']}", disabled=x["status"] == "Resolved",
                                 icon=":material/check_circle:", use_container_width=True):
                        set_status(x["member_ids"], "Resolved")
                        st.session_state["flash"] = "Incident resolved. To reopen it, use 'Resolved incidents' at the bottom of this page."
                        st.rerun()
                    if a3.button("Undo dispatch" if x["status"] == "Dispatched" else "Reopen", key=f"reopen_{x['id']}",
                                 disabled=x["status"] == "New", help=f"Goes back to: {restore_to}",
                                 icon=":material/undo:", use_container_width=True):
                        restore_status(x["member_ids"])
                        st.session_state["flash"] = f"Incident restored to {restore_to}."
                        st.rerun()

                    akey = f"alert_{x['id']}"
                    if d2.button("Generate responder alert", key=f"btn_{x['id']}", icon=":material/campaign:"):
                        with st.spinner("Drafting alert..."):
                            try:
                                st.session_state[akey] = draft_alert(x)
                            except Exception as e:
                                st.error(f"Could not draft alert: {e}")
                    if akey in st.session_state:
                        d2.text_area("Responder alert (select and copy)", value=st.session_state[akey], height=190, key=f"txt_{x['id']}_{abs(hash(st.session_state[akey]))}")

        # resolved incidents stay reachable here, so they can always be reopened
        if n_res:
            with st.expander(f"Resolved incidents ({n_res})", expanded=not visible):
                for x in [r for r in rows if r["status"] == "Resolved"]:
                    back_to = min((prev_of(m) for m in x["members"] if status_of(m) == "Resolved"),
                                  key=lambda name: STATUS_ORDER[name], default="New")
                    when = f"resolved {int(minutes_since(x['status_updated_at']))} min ago" if x["status_updated_at"] else "resolved"
                    r1, r2 = st.columns([5, 1.4], vertical_alignment="center")
                    H(f"<div class='fv'><b>{esc(x['hazard_type'])}</b> · {esc(x['location'])} "
                      f"<span class='fn'>· {when} · {x['n_reports']} report(s)</span></div>", r1)
                    if r2.button("Reopen", key=f"rres_{x['id']}", help=f"Goes back to: {back_to}",
                                 icon=":material/undo:", use_container_width=True):
                        restore_status(x["member_ids"])
                        st.session_state["flash"] = f"Incident restored to {back_to}."
                        st.rerun()

        # the authority's duplicate decisions can be undone
        batches = list_decision_batches()
        if batches:
            with st.expander(f"Duplicate review decisions ({len(batches)})"):
                for bt in batches:
                    d1, d2 = st.columns([4, 1])
                    d1.markdown(f"**{'Merged' if bt['decision'] == 'merge' else 'Kept separate'}** · {bt['pairs']} report pair(s) · "
                                f"by {esc(str(bt['by']))} · {ago_text(bt['at'])}")
                    if d2.button("Undo", key=f"undo_{bt['batch']}"):
                        undo_decision(bt["batch"])
                        st.session_state["flash"] = "Decision undone. The reports are evaluated automatically again."
                        st.rerun()

        # reports whose AI assessment is still pending: an authority can retry them one at a time
        pending = [(m, x) for x in rows for m in x["members"] if is_pending(m)]
        if pending:
            H("<div class='sec-h'>Pending AI assessments<small>retry once the AI service is available again</small></div>")
            with st.expander(f"Reports awaiting AI assessment ({len(pending)})", expanded=True):
                for m, x in pending:
                    why = AI_REASON.get(m.get("ai_error") if isinstance(m.get("ai_error"), str) else "", "the assessment has not been completed yet")
                    tried = f", last tried {ago_text(m['ai_attempted_at'])}" if isinstance(m.get("ai_attempted_at"), str) else ""
                    evidence = "".join(chip(t) for t, p in (("photo", m.get("image_path")), ("voice note", m.get("audio_path"))) if isinstance(p, str) and p)
                    flag = chip("flagged by keyword check: urgent human review") if str(m.get("severity_reason") or "").startswith("URGENT HUMAN REVIEW") else ""
                    c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
                    H(f"<div class='fv'><b>{esc(m['location'])}</b> · reported {ago_text(m['created_at'])} · incident {esc(x['status'])}</div>"
                      f"<div class='fn'>{esc(why)}{tried}</div><div>{evidence}{flag}</div>", c1)
                    if c2.button("Retry AI assessment", key=f"retry_{m['id']}", icon=":material/refresh:", use_container_width=True):
                        with st.spinner("Asking the AI to assess this report..."):
                            try:
                                ok, msg = retry_assessment(m["id"])
                            except PermissionError:
                                ok, msg = False, "Your session has ended. Please sign in again."
                        if ok:
                            st.session_state["flash"] = msg
                            st.rerun()
                        else:
                            st.warning(msg)
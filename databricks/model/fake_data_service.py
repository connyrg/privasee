"""
Fake Data Service

Generates realistic fake replacement values for sensitive entities using the
Faker library.  Called by DocumentIntelligenceModel after entity extraction
so that replacement_text is pre-populated before the result is returned to
the backend.

Design notes
------------
- Stateless: callers own the consistency map (same original → same fake value).
- Names: structure-aware — mirrors token count, hyphenation, prefix/suffix, gender,
  and case style.  E.g. "Anne-Marie Doe" (female) → "Jane-Martha Smith".
- Dates: preserve the original year and exact format — separator, component order
  (D/M/Y vs M/D/Y), month representation (numeric / abbreviated / full),
  zero-padding.  E.g. "21/03/2004" → "22/01/2004",  "5 Jan 2025" → "19 Sep 2025".
- Gender-aware names when gender_guesser is installed (optional, graceful fallback).
"""

import logging
import re
from datetime import date
from typing import Optional

from faker import Faker
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

try:
    import gender_guesser.detector as _gg
    _gender_detector = _gg.Detector()
    _HAS_GENDER = True
except Exception:
    _gender_detector = None
    _HAS_GENDER = False

# Tokens that should be kept verbatim and not treated as name components
_PREFIXES = {"mr", "mrs", "ms", "miss", "dr", "prof", "rev", "sir", "lord", "lady", "mx", "master"}
_SUFFIXES = {"jr", "sr", "i", "ii", "iii", "iv", "v", "esq", "phd", "md", "dds", "dvm", "jd"}

# Address parsing helpers
_AU_STATE_RE = re.compile(r'\b(ACT|NSW|NT|QLD|SA|TAS|VIC|WA)\b')
_AU_POSTCODE_RE = re.compile(r'\b(\d{4})\b')
_STREET_TYPE_RE = re.compile(
    r'(?i)\b(street|st|road|rd|avenue|ave|drive|dr|lane|ln|court|ct|'
    r'place|pl|way|crescent|cres|terrace|tce|parade|pde|highway|hwy|'
    r'close|cl|circuit|cct|grove|gr|boulevard|blvd)\b'
)


class FakeDataService:
    """Generate a single realistic fake replacement for a sensitive entity."""

    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
        self.faker_au = Faker('en_AU')
        if seed is not None:
            Faker.seed(seed)

    def generate(self, entity_type: str, original_text: str) -> str:
        """
        Return a contextually appropriate fake value for *entity_type*.

        Parameters
        ----------
        entity_type:
            Human-readable label, e.g. ``"Full Name"``, ``"Date of Birth"``.
        original_text:
            The original text from the document — used for gender detection,
            date format / year preservation, and numeric-length matching.

        Returns
        -------
        str
            A realistic fake value.
        """
        t = entity_type.lower()

        # ---- Names -------------------------------------------------------
        if "name" in t:
            gender = self._detect_gender(original_text)
            if "first" in t or "given" in t:
                return self._fake_name_token(original_text.strip(), gender, "first")
            if "last" in t or "surname" in t or "family" in t:
                return self._fake_name_token(original_text.strip(), gender, "last")
            if "middle" in t:
                return self._fake_name_token(original_text.strip(), gender, "first")
            # Full name — mirror original structure
            return self._generate_structured_name(original_text, gender)

        # ---- Contact -----------------------------------------------------
        if "email" in t or "e-mail" in t:
            return self.faker.email()
        if "phone" in t or "telephone" in t or "mobile" in t:
            return self.faker.phone_number()

        # ---- Address -----------------------------------------------------
        if "address" in t:
            if "street" in t:
                return self.faker.street_address()
            if "city" in t or "suburb" in t:
                return self.faker.city()
            if "state" in t:
                return self.faker.state()
            if "zip" in t or "postal" in t or "postcode" in t:
                return self.faker.zipcode()
            return self._generate_address(original_text)

        # ---- Government / financial IDs ----------------------------------
        if "ssn" in t or "social security" in t:
            return self.faker.ssn()
        if "credit" in t or "card" in t:
            return self.faker.credit_card_number()
        if "account" in t or "bank" in t:
            return self.faker.bban()
        if "license" in t or "licence" in t:
            return self.faker.license_plate()

        # ---- Dates -------------------------------------------------------
        if "date" in t or "dob" in t or "birth" in t or "birthday" in t:
            return self._same_year_date(original_text)

        # ---- Professional ------------------------------------------------
        if "company" in t or "organization" in t or "employer" in t:
            return self.faker.company()
        if "job" in t or "title" in t or "position" in t:
            return self.faker.job()

        # ---- Technical ---------------------------------------------------
        if "url" in t or "website" in t:
            return self.faker.url()
        if "ip" in t:
            return self.faker.ipv4()
        if "username" in t or "user" in t:
            return self.faker.user_name()

        # ---- Fallback: preserve character structure ----------------------
        # Handles any unrecognised type (IDs, reference numbers, etc.) by
        # replacing each digit with a random digit and each letter with a
        # random letter, keeping separators (spaces, hyphens, slashes) intact.
        # E.g. "2123 45678 A 1" → "7654 12983 C 6" (Medicare)
        #      "123 456 789"   → "987 321 654"    (TFN)
        #      "N1234567"      → "K7865432"       (Passport)
        if "@" in original_text:
            return self.faker.email()
        return self._preserve_structure(original_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_gender(self, name: str) -> str:
        """Return 'male', 'female', or 'unknown' from a name string."""
        if not _HAS_GENDER:
            return "unknown"
        try:
            tokens = name.strip().split()
            # Skip leading prefix tokens (Dr., Mr., …) to reach the actual first name
            first = next(
                (t for t in tokens if t.strip(".,").lower() not in _PREFIXES),
                tokens[0] if tokens else "",
            )
            # Use first part of a hyphenated name (e.g. "Anne" from "Anne-Marie")
            first_part = first.split("-")[0].capitalize()
            result = _gender_detector.get_gender(first_part)
            if result in ("female", "mostly_female"):
                return "female"
            if result in ("male", "mostly_male"):
                return "male"
        except Exception:
            pass
        return "unknown"

    def _same_year_date(self, date_str: str) -> str:
        """Generate a fake date in the same year as *date_str*, same format."""
        s = date_str.strip()

        year = None
        try:
            year = dateutil_parser.parse(s, dayfirst=True, fuzzy=True).year
        except Exception:
            pass
        if year is None:
            m = re.search(r"\b(19|20)\d{2}\b", s)
            if m:
                year = int(m.group())

        if year is not None:
            try:
                fake_date = self.faker.date_between(
                    start_date=date(year, 1, 1),
                    end_date=date(year, 12, 31),
                )
                return self._reconstruct_date(fake_date, s)
            except Exception:
                pass

        return self._reconstruct_date(self.faker.date_of_birth(), s)

    def _reconstruct_date(self, fake: date, original: str) -> str:
        """
        Format *fake* date to mirror the structure of *original*.

        Preserves separator, component order (D/M/Y vs M/D/Y), month
        representation (numeric / abbreviated / full), zero-padding, and
        2- vs 4-digit year.

        D/M/Y vs M/D/Y detection: if the first numeric component is > 12
        it must be a day.  If the second is > 12, the first must be a month.
        Otherwise defaults to D/M/Y (AU/EU convention).
        """
        # ISO YYYY-MM-DD
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", original):
            return fake.strftime("%Y-%m-%d")

        # YYYY/MM/DD
        if re.fullmatch(r"\d{4}/\d{2}/\d{2}", original):
            return fake.strftime("%Y/%m/%d")

        # "D Mon YYYY" / "DD Mon YYYY" / "D Month YYYY"
        # e.g. "5 Jan 2025", "05 January 2025"
        m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})", original)
        if m:
            day_orig, month_orig, year_orig = m.group(1), m.group(2), m.group(3)
            day = str(fake.day).zfill(len(day_orig)) if day_orig.startswith("0") else str(fake.day)
            month_name = fake.strftime("%b") if len(month_orig) <= 3 else fake.strftime("%B")
            year = str(fake.year) if len(year_orig) == 4 else str(fake.year)[-2:]
            return f"{day} {month_name} {year}"

        # "Mon D, YYYY" / "Month D YYYY" / "Month D, YYYY"
        # e.g. "Jan 5, 2025", "January 5 2025"
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})(,?)\s*(\d{2,4})", original)
        if m:
            month_orig, day_orig, comma, year_orig = m.group(1), m.group(2), m.group(3), m.group(4)
            month_name = fake.strftime("%b") if len(month_orig) <= 3 else fake.strftime("%B")
            day = str(fake.day).zfill(len(day_orig)) if day_orig.startswith("0") else str(fake.day)
            year = str(fake.year) if len(year_orig) == 4 else str(fake.year)[-2:]
            sep = ", " if comma else " "
            return f"{month_name} {day}{sep}{year}"

        # Numeric ?sep?sep? where sep is / or -
        for sep in ("/", "-"):
            m = re.fullmatch(
                r"(\d{1,2})" + re.escape(sep) + r"(\d{1,2})" + re.escape(sep) + r"(\d{2,4})",
                original,
            )
            if m:
                p0, p1, p2 = m.group(1), m.group(2), m.group(3)
                year_str = str(fake.year) if len(p2) == 4 else str(fake.year)[-2:]
                p0_val, p1_val = int(p0), int(p1)
                if p0_val > 12:
                    # First component can only be a day → D/M/Y
                    day = str(fake.day).zfill(len(p0))
                    mon = str(fake.month).zfill(len(p1))
                    return sep.join([day, mon, year_str])
                elif p1_val > 12:
                    # Second component can only be a day → M/D/Y
                    mon = str(fake.month).zfill(len(p0))
                    day = str(fake.day).zfill(len(p1))
                    return sep.join([mon, day, year_str])
                else:
                    # Ambiguous — default to D/M/Y
                    day = str(fake.day).zfill(len(p0))
                    mon = str(fake.month).zfill(len(p1))
                    return sep.join([day, mon, year_str])

        # Fallback — no recognised format
        return fake.strftime("%d/%m/%Y")

    def _generate_address(self, original: str) -> str:
        """
        Generate a fake address that mirrors the structure and locale of *original*.

        Locale: uses ``en_AU`` Faker when the original contains an AU state
        abbreviation (NSW, VIC, …) or a 4-digit postcode; falls back to the
        default locale otherwise.

        Component detection (regex-based):
            has_unit    — "Unit N/", "Apt N", or bare "N/" prefix
            has_street  — street number followed by a word
            has_suburb  — text between the street line and the state/postcode
            has_state   — AU state abbreviation found
            has_postcode — 4-digit token found

        The fake components are assembled in the same order as the original
        and joined with the same separator (newline or ", ").
        """
        s = original.strip()

        # --- Locale detection ---
        # Default to AU locale — this is an Australian application.
        # Only override if we can positively identify a non-AU address
        # (currently no such heuristic), so always use faker_au.
        au_state_m = _AU_STATE_RE.search(s)
        au_pc_m = _AU_POSTCODE_RE.search(s)
        loc = self.faker_au

        # --- Separator ---
        sep = "\n" if "\n" in s else ", "

        # --- Component detection ---
        has_unit = bool(
            re.search(r'(?i)\b(unit|apt|apartment|flat|suite|level)\s+\d+', s)
            or re.match(r'^\d+/', s)
        )
        has_street = bool(re.search(r'\b\d{1,4}[A-Za-z]?\s+\w', s))
        has_state = bool(au_state_m)
        has_postcode = bool(au_pc_m)

        # Suburb: text that sits between the street line and the state/postcode.
        # Strategy: find where the first marker (state or postcode) begins, look
        # at everything before it, then take the segment after the last street-
        # type keyword (e.g. "Street", "Rd") or after the last comma.
        has_suburb = False
        if has_state or has_postcode:
            marker_pos = min(
                au_state_m.start() if au_state_m else len(s),
                au_pc_m.start() if au_pc_m else len(s),
            )
            before = s[:marker_pos].strip().rstrip(',').strip()
            # Prefer the segment after the last comma (most reliable split)
            last_comma = before.rfind(',')
            if last_comma >= 0:
                candidate = before[last_comma + 1:].strip()
            elif has_street:
                # No comma — try to find suburb after the street type keyword
                st_m = _STREET_TYPE_RE.search(before)
                candidate = before[st_m.end():].strip() if st_m else ""
            else:
                candidate = before
            has_suburb = len(candidate) > 1

        # --- Build fake components ---
        parts: list = []

        if has_street:
            if has_unit:
                unit_n = self.faker.random_int(1, 20)
                st_n = self.faker.random_int(1, 200)
                parts.append(f"Unit {unit_n}/{st_n} {loc.street_name()}")
            else:
                parts.append(loc.street_address())

        # Suburb, state, and postcode form a single space-joined "city line"
        # (e.g. "Blacktown VIC 3012") so they are never split by the address
        # separator.  Only add as one part to preserve the original structure.
        city_line: list = []
        if has_suburb:
            city_line.append(loc.city())
        if has_state and has_postcode:
            city_line.append(f"{loc.state_abbr()} {loc.postcode()}")
        elif has_state:
            city_line.append(loc.state_abbr())
        elif has_postcode:
            city_line.append(loc.postcode())
        if city_line:
            parts.append(" ".join(city_line))

        if not parts:
            return loc.address().replace("\n", sep)

        return sep.join(parts)

    # ------------------------------------------------------------------
    # Name structure helpers
    # ------------------------------------------------------------------

    def _generate_structured_name(self, original_text: str, gender: str) -> str:
        """
        Generate a fake full name that mirrors the token structure of *original_text*.

        Preserves: number of tokens (first / middle(s) / last), hyphenation,
        prefix (Dr., Mr., …) and suffix (Jr., III, …) kept verbatim, case
        style, and gender of first/middle name tokens.
        """
        tokens = original_text.strip().split()
        if not tokens:
            return self._simple_name(gender)

        prefix_tokens: list = []
        suffix_tokens: list = []
        name_tokens: list = []

        for i, tok in enumerate(tokens):
            clean = tok.strip(".,").lower()
            if i == 0 and clean in _PREFIXES:
                prefix_tokens.append(tok)
            elif i == len(tokens) - 1 and clean in _SUFFIXES:
                suffix_tokens.append(tok)
            else:
                name_tokens.append(tok)

        if not name_tokens:
            return self._simple_name(gender)

        fake_parts: list = []
        if len(name_tokens) == 1:
            fake_parts.append(self._fake_name_token(name_tokens[0], gender, "first"))
        elif len(name_tokens) == 2:
            fake_parts.append(self._fake_name_token(name_tokens[0], gender, "first"))
            fake_parts.append(self._fake_name_token(name_tokens[1], gender, "last"))
        else:
            # first + one or more middles + last
            fake_parts.append(self._fake_name_token(name_tokens[0], gender, "first"))
            for mid in name_tokens[1:-1]:
                fake_parts.append(self._fake_name_token(mid, gender, "first"))
            fake_parts.append(self._fake_name_token(name_tokens[-1], gender, "last"))

        return " ".join(prefix_tokens + fake_parts + suffix_tokens)

    def _fake_name_token(self, token: str, gender: str, role: str) -> str:
        """
        Generate a fake replacement for a single name token.

        Hyphenated tokens (e.g. "Anne-Marie") produce a hyphenated result
        ("Jane-Martha") with the same number of parts and same case style.
        """
        if "-" in token:
            parts = token.split("-")
            fake_parts = []
            for part in parts:
                raw = self.faker.last_name() if role == "last" else self._first_name_for_gender(gender)
                fake_parts.append(self._match_case(part, raw))
            return "-".join(fake_parts)

        raw = self.faker.last_name() if role == "last" else self._first_name_for_gender(gender)
        return self._match_case(token, raw)

    def _first_name_for_gender(self, gender: str) -> str:
        if gender == "female":
            return self.faker.first_name_female()
        if gender == "male":
            return self.faker.first_name_male()
        return self.faker.first_name()

    def _match_case(self, original: str, generated: str) -> str:
        """Mirror the case style of *original* onto *generated*."""
        if original.isupper():
            return generated.upper()
        if original.islower():
            return generated.lower()
        return generated.capitalize()

    def _simple_name(self, gender: str) -> str:
        """Fallback when no tokens can be parsed."""
        if gender == "female":
            return self.faker.name_female()
        if gender == "male":
            return self.faker.name_male()
        return self.faker.name()

    def _preserve_structure(self, original: str) -> str:
        """
        Generate a replacement that mirrors the character structure of *original*.

        Each digit is replaced with a random digit (0–9), each letter with a
        random letter (matching case), and all other characters (spaces, hyphens,
        slashes, etc.) are kept verbatim.

        Used as the generic fallback for unrecognised entity types such as
        government IDs, reference numbers, and mixed alphanumeric codes.
        """
        result = []
        for ch in original:
            if ch.isdigit():
                result.append(str(self.faker.random_int(0, 9)))
            elif ch.isupper():
                result.append(self.faker.random_letter().upper())
            elif ch.islower():
                result.append(self.faker.random_letter().lower())
            else:
                result.append(ch)
        return "".join(result)

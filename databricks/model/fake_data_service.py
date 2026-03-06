"""
Fake Data Service

Generates realistic fake replacement values for sensitive entities using the
Faker library.  Called by DocumentIntelligenceModel after entity extraction
so that replacement_text is pre-populated before the result is returned to
the backend.

Design notes
------------
- Stateless: callers own the consistency map (same original → same fake value).
- Dates: preserve the original year, randomise month/day, mirror the input format.
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


class FakeDataService:
    """Generate a single realistic fake replacement for a sensitive entity."""

    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
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
                return (self.faker.first_name_female() if gender == "female"
                        else self.faker.first_name_male() if gender == "male"
                        else self.faker.first_name())
            if "last" in t or "surname" in t or "family" in t:
                return self.faker.last_name()
            if "middle" in t:
                return (self.faker.first_name_female() if gender == "female"
                        else self.faker.first_name_male() if gender == "male"
                        else self.faker.first_name())
            # Full name
            return (self.faker.name_female() if gender == "female"
                    else self.faker.name_male() if gender == "male"
                    else self.faker.name())

        # ---- Contact -----------------------------------------------------
        if "email" in t or "e-mail" in t:
            return self.faker.email()
        if "phone" in t or "telephone" in t or "mobile" in t:
            return self.faker.phone_number()

        # ---- Address -----------------------------------------------------
        if "address" in t:
            if "street" in t:
                return self.faker.street_address()
            if "city" in t:
                return self.faker.city()
            if "state" in t:
                return self.faker.state()
            if "zip" in t or "postal" in t:
                return self.faker.zipcode()
            return self.faker.address().replace("\n", ", ")

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

        # ---- Fallback based on original text pattern ---------------------
        if original_text.strip().isdigit():
            return self.faker.numerify("#" * len(original_text.strip()))
        if "@" in original_text:
            return self.faker.email()
        return self.faker.name()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_gender(self, name: str) -> str:
        """Return 'male', 'female', or 'unknown' from a name string."""
        if not _HAS_GENDER:
            return "unknown"
        try:
            first = name.strip().split()[0].capitalize()
            result = _gender_detector.get_gender(first)
            if result in ("female", "mostly_female"):
                return "female"
            if result in ("male", "mostly_male"):
                return "male"
        except Exception:
            pass
        return "unknown"

    def _same_year_date(self, date_str: str) -> str:
        """Generate a fake date in the same year as *date_str*, same format."""
        format_patterns = [
            (r"\d{4}-\d{2}-\d{2}",         "%Y-%m-%d"),
            (r"\d{2}/\d{2}/\d{4}",         "%m/%d/%Y"),
            (r"\d{2}-\d{2}-\d{4}",         "%m-%d-%Y"),
            (r"\d{4}/\d{2}/\d{2}",         "%Y/%m/%d"),
            (r"\d{1,2}/\d{1,2}/\d{4}",     "%m/%d/%Y"),
            (r"\d{1,2}-\d{1,2}-\d{4}",     "%m-%d-%Y"),
            (r"[A-Za-z]+ \d{1,2},? \d{4}", "%B %d, %Y"),
            (r"\d{1,2} [A-Za-z]+ \d{4}",   "%d %B %Y"),
            (r"\d{2}/\d{2}/\d{2}",         "%m/%d/%y"),
        ]

        detected_fmt = None
        for pattern, fmt in format_patterns:
            if re.fullmatch(pattern, date_str.strip()):
                detected_fmt = fmt
                break

        year = None
        try:
            year = dateutil_parser.parse(date_str, fuzzy=True).year
        except Exception:
            pass
        if year is None:
            m = re.search(r"\b(19|20)\d{2}\b", date_str)
            if m:
                year = int(m.group())

        if year is not None:
            try:
                fake_date = self.faker.date_between(
                    start_date=date(year, 1, 1),
                    end_date=date(year, 12, 31),
                )
                return fake_date.strftime(detected_fmt or "%m/%d/%Y")
            except Exception:
                pass

        return self.faker.date_of_birth().strftime("%m/%d/%Y")

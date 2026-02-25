"""
Mapping Manager Service
Ensures consistent entity replacement across documents.
"""

from faker import Faker
import re
import logging
from datetime import date
from typing import Dict, Optional

from dateutil import parser as dateutil_parser

try:
    import gender_guesser.detector as gender_guesser_detector
    _gender_detector = gender_guesser_detector.Detector()
    HAS_GENDER_GUESSER = True
except ImportError:
    HAS_GENDER_GUESSER = False

logger = logging.getLogger(__name__)


class MappingManager:
    """
    Manages consistent entity replacement across document.
    Ensures that 'John Doe' always maps to the same replacement throughout.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mapping manager.

        Args:
            seed: Random seed for Faker (for reproducible fake data)
        """
        self.mappings: Dict[str, str] = {}  # normalized_original → replacement
        self.counter: Dict[str, int] = {}   # entity_type → count
        self.faker = Faker()

        if seed:
            Faker.seed(seed)

        logger.info("Mapping Manager initialized")

    def get_replacement(
        self,
        entity_type: str,
        original_text: str,
        strategy: str
    ) -> str:
        """
        Get consistent replacement for entity.

        Args:
            entity_type: Type of entity (e.g., "Full Name", "Email", "SSN")
            original_text: Original text to replace
            strategy: Replacement strategy ("fake_name", "redact", "entity_label")

        Returns:
            Replacement text (consistent for same original_text)
        """
        # Normalize text for consistency check
        normalized = self.normalize_text(original_text)

        # Check if we already have a mapping
        if normalized in self.mappings:
            logger.debug(f"Using existing mapping: '{original_text}' -> '{self.mappings[normalized]}'")
            return self.mappings[normalized]

        # Generate new replacement based on strategy
        if strategy == "fake_name" or strategy == "Fake Data":
            replacement = self._generate_fake_data(entity_type, original_text)
        elif strategy == "redact" or strategy == "Black Out":
            replacement = "[REDACTED]"
        elif strategy == "entity_label" or strategy == "Entity Label":
            replacement = self._generate_entity_label(entity_type)
        else:
            # Default to entity label
            logger.warning(f"Unknown strategy '{strategy}', using entity_label")
            replacement = self._generate_entity_label(entity_type)

        # Store mapping
        self.mappings[normalized] = replacement
        logger.info(f"New mapping created: '{original_text}' -> '{replacement}'")

        return replacement

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized text (lowercase, no extra spaces/punctuation)
        """
        # Convert to lowercase
        normalized = text.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Remove common punctuation (but keep hyphens in names)
        normalized = re.sub(r'[.,!?;:\'"()]', '', normalized)

        return normalized.strip()

    def _generate_fake_data(self, entity_type: str, original_text: str) -> str:
        """
        Generate realistic fake data using Faker.

        Args:
            entity_type: Type of entity
            original_text: Original text (used for context)

        Returns:
            Generated fake data
        """
        entity_lower = entity_type.lower()

        # Name-related entities
        if 'name' in entity_lower:
            gender = self._detect_gender(original_text)
            if 'first' in entity_lower or 'given' in entity_lower:
                if gender == 'female':
                    return self.faker.first_name_female()
                elif gender == 'male':
                    return self.faker.first_name_male()
                else:
                    return self.faker.first_name()
            elif 'last' in entity_lower or 'surname' in entity_lower or 'family' in entity_lower:
                return self.faker.last_name()
            elif 'middle' in entity_lower:
                if gender == 'female':
                    return self.faker.first_name_female()
                elif gender == 'male':
                    return self.faker.first_name_male()
                else:
                    return self.faker.first_name()
            else:
                # Full name — preserve gender
                if gender == 'female':
                    return self.faker.name_female()
                elif gender == 'male':
                    return self.faker.name_male()
                else:
                    return self.faker.name()

        # Email
        elif 'email' in entity_lower or 'e-mail' in entity_lower:
            return self.faker.email()

        # Phone
        elif 'phone' in entity_lower or 'telephone' in entity_lower or 'mobile' in entity_lower:
            return self.faker.phone_number()

        # Address-related
        elif 'address' in entity_lower:
            if 'street' in entity_lower:
                return self.faker.street_address()
            elif 'city' in entity_lower:
                return self.faker.city()
            elif 'state' in entity_lower:
                return self.faker.state()
            elif 'zip' in entity_lower or 'postal' in entity_lower:
                return self.faker.zipcode()
            else:
                # Full address
                return self.faker.address().replace('\n', ', ')

        # SSN / Government IDs
        elif 'ssn' in entity_lower or 'social security' in entity_lower:
            return self.faker.ssn()

        # Dates — preserve the original year
        elif 'date' in entity_lower or 'dob' in entity_lower or 'birth' in entity_lower or 'birthday' in entity_lower:
            return self._generate_same_year_date(original_text)

        # Company / Organization
        elif 'company' in entity_lower or 'organization' in entity_lower or 'employer' in entity_lower:
            return self.faker.company()

        # Job title
        elif 'job' in entity_lower or 'title' in entity_lower or 'position' in entity_lower:
            return self.faker.job()

        # Credit card
        elif 'credit' in entity_lower or 'card' in entity_lower:
            return self.faker.credit_card_number()

        # Bank account
        elif 'account' in entity_lower or 'bank' in entity_lower:
            return self.faker.bban()

        # License number
        elif 'license' in entity_lower or 'licence' in entity_lower:
            return self.faker.license_plate()

        # URL / Website
        elif 'url' in entity_lower or 'website' in entity_lower:
            return self.faker.url()

        # IP Address
        elif 'ip' in entity_lower:
            return self.faker.ipv4()

        # Username
        elif 'username' in entity_lower or 'user' in entity_lower:
            return self.faker.user_name()

        # Default: generate text based on length
        else:
            # Try to match the length/pattern of original text
            if original_text.isdigit():
                # Numeric ID
                return self.faker.numerify('#' * len(original_text))
            elif '@' in original_text:
                # Email-like
                return self.faker.email()
            else:
                # Generic text - use name as fallback
                return self.faker.name()

    def _detect_gender(self, name: str) -> str:
        """
        Detect gender from a name string.

        Args:
            name: Name to analyse (full or first name)

        Returns:
            'male', 'female', or 'unknown'
        """
        if not HAS_GENDER_GUESSER:
            return 'unknown'
        try:
            first_name = name.strip().split()[0].capitalize()
            result = _gender_detector.get_gender(first_name)
            if result in ('female', 'mostly_female'):
                return 'female'
            elif result in ('male', 'mostly_male'):
                return 'male'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'

    def _generate_same_year_date(self, date_str: str) -> str:
        """
        Generate a fake date that falls in the same year as the original.
        The output format mirrors the input format where possible.

        Args:
            date_str: Original date string

        Returns:
            Fake date string in the same format and year
        """
        # Common formats ordered from most specific to least
        format_patterns = [
            (r'\d{4}-\d{2}-\d{2}',        '%Y-%m-%d'),
            (r'\d{2}/\d{2}/\d{4}',        '%m/%d/%Y'),
            (r'\d{2}-\d{2}-\d{4}',        '%m-%d-%Y'),
            (r'\d{4}/\d{2}/\d{2}',        '%Y/%m/%d'),
            (r'\d{1,2}/\d{1,2}/\d{4}',   '%m/%d/%Y'),
            (r'\d{1,2}-\d{1,2}-\d{4}',   '%m-%d-%Y'),
            (r'[A-Za-z]+ \d{1,2},? \d{4}', '%B %d, %Y'),
            (r'\d{1,2} [A-Za-z]+ \d{4}', '%d %B %Y'),
            (r'\d{2}/\d{2}/\d{2}',        '%m/%d/%y'),
        ]

        detected_format = None
        for pattern, fmt in format_patterns:
            if re.fullmatch(pattern, date_str.strip()):
                detected_format = fmt
                break

        year = None
        try:
            parsed = dateutil_parser.parse(date_str, fuzzy=True)
            year = parsed.year
        except Exception as e:
            logger.warning(f"dateutil could not parse date '{date_str}': {e}")

        # Regex fallback: look for a 4-digit year in the string
        if year is None:
            match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if match:
                year = int(match.group())
                logger.info(f"Extracted year {year} from '{date_str}' via regex")

        if year is not None:
            try:
                fake_date = self.faker.date_between(
                    start_date=date(year, 1, 1),
                    end_date=date(year, 12, 31)
                )
                if detected_format:
                    return fake_date.strftime(detected_format)
                else:
                    return fake_date.strftime('%m/%d/%Y')
            except Exception as e:
                logger.warning(f"Could not generate date for year {year}: {e}")

        logger.warning(f"Could not extract year from '{date_str}', using random date_of_birth")
        return self.faker.date_of_birth().strftime('%m/%d/%Y')

    def _generate_entity_label(self, entity_type: str) -> str:
        """
        Generate entity label (e.g., Person_A, Email_1).

        Args:
            entity_type: Type of entity

        Returns:
            Generated label
        """
        # Get or initialize counter for this entity type
        if entity_type not in self.counter:
            self.counter[entity_type] = 0

        self.counter[entity_type] += 1
        count = self.counter[entity_type]

        # Clean entity type for label
        label_prefix = entity_type.replace(' ', '_').replace('-', '_')

        # Use letters for small counts, numbers for larger
        if count <= 26:
            suffix = chr(64 + count)  # A, B, C, ...
        else:
            suffix = str(count)

        return f"{label_prefix}_{suffix}"

    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get all current mappings.

        Returns:
            Dictionary of all mappings
        """
        return self.mappings.copy()

    def clear_mappings(self):
        """Clear all mappings and counters."""
        self.mappings.clear()
        self.counter.clear()
        logger.info("All mappings cleared")

    def add_manual_mapping(self, original_text: str, replacement: str):
        """
        Manually add or override a mapping.

        Args:
            original_text: Original text
            replacement: Replacement text
        """
        normalized = self.normalize_text(original_text)
        self.mappings[normalized] = replacement
        logger.info(f"Manual mapping added: '{original_text}' -> '{replacement}'")

    def check_consistency(self, entities: list) -> Dict:
        """
        Check consistency of entity replacements.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary with consistency statistics
        """
        original_counts = {}
        replacement_counts = {}

        for entity in entities:
            original = self.normalize_text(entity.get('original_text', ''))
            replacement = entity.get('replacement_text', '')

            original_counts[original] = original_counts.get(original, 0) + 1
            replacement_counts[replacement] = replacement_counts.get(replacement, 0) + 1

        return {
            'unique_originals': len(original_counts),
            'unique_replacements': len(replacement_counts),
            'total_entities': len(entities),
            'is_consistent': len(original_counts) == len(replacement_counts)
        }

"""Dataset validation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .types import TestCase, Message

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning
    record_index: Optional[int] = None
    value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [
                {
                    "field": e.field,
                    "message": e.message,
                    "severity": e.severity,
                    "record_index": e.record_index,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "field": w.field,
                    "message": w.message,
                    "severity": w.severity,
                    "record_index": w.record_index,
                }
                for w in self.warnings
            ],
            "statistics": self.statistics,
        }


class DatasetValidator:
    """Validates dataset structure and content.

    Provides comprehensive validation including:
    - Required field checks
    - Data type validation
    - Content length limits
    - Role validation
    - Duplicate detection
    """

    # Valid message roles
    VALID_ROLES: Set[str] = {"system", "user", "assistant", "tool", "function"}

    # Default validation rules
    DEFAULT_RULES = {
        "require_id": False,
        "require_system_message": False,
        "min_messages": 1,
        "max_messages": 100,
        "min_content_length": 1,
        "max_content_length": 100000,
        "check_duplicates": True,
        "valid_roles": None,  # Uses VALID_ROLES if None
    }

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        """Initialize validator with custom rules.

        Args:
            rules: Custom validation rules to override defaults.
        """
        self.rules = {**self.DEFAULT_RULES, **(rules or {})}

    def validate_test_case(
        self,
        test_case: TestCase,
        index: Optional[int] = None,
    ) -> List[ValidationError]:
        """Validate a single test case.

        Args:
            test_case: The test case to validate.
            index: Optional index for error reporting.

        Returns:
            List of validation errors.
        """
        errors: List[ValidationError] = []

        # Check ID requirement
        if self.rules["require_id"] and not test_case.id:
            errors.append(ValidationError(
                field="id",
                message="Test case ID is required",
                severity="error",
                record_index=index,
            ))

        # Check messages
        if not test_case.messages:
            errors.append(ValidationError(
                field="messages",
                message="Messages list is empty",
                severity="error",
                record_index=index,
            ))
            return errors

        # Check message count
        if len(test_case.messages) < self.rules["min_messages"]:
            errors.append(ValidationError(
                field="messages",
                message=f"Too few messages: {len(test_case.messages)} (minimum: {self.rules['min_messages']})",
                severity="error",
                record_index=index,
            ))

        if len(test_case.messages) > self.rules["max_messages"]:
            errors.append(ValidationError(
                field="messages",
                message=f"Too many messages: {len(test_case.messages)} (maximum: {self.rules['max_messages']})",
                severity="warning",
                record_index=index,
            ))

        # Check for system message requirement
        if self.rules["require_system_message"]:
            has_system = any(m.role == "system" for m in test_case.messages)
            if not has_system:
                errors.append(ValidationError(
                    field="messages",
                    message="No system message found",
                    severity="warning",
                    record_index=index,
                ))

        # Validate each message
        valid_roles = self.rules["valid_roles"] or self.VALID_ROLES

        for i, message in enumerate(test_case.messages):
            msg_errors = self._validate_message(message, i, valid_roles)
            for err in msg_errors:
                err.record_index = index
            errors.extend(msg_errors)

        return errors

    def _validate_message(
        self,
        message: Message,
        index: int,
        valid_roles: Set[str],
    ) -> List[ValidationError]:
        """Validate a single message.

        Args:
            message: The message to validate.
            index: Message index.
            valid_roles: Set of valid roles.

        Returns:
            List of validation errors.
        """
        errors: List[ValidationError] = []
        field_prefix = f"messages[{index}]"

        # Validate role
        if not message.role:
            errors.append(ValidationError(
                field=f"{field_prefix}.role",
                message="Role is missing",
                severity="error",
                value=message.role,
            ))
        elif message.role not in valid_roles:
            errors.append(ValidationError(
                field=f"{field_prefix}.role",
                message=f"Invalid role: '{message.role}'",
                severity="warning",
                value=message.role,
            ))

        # Validate content
        if message.content is None:
            errors.append(ValidationError(
                field=f"{field_prefix}.content",
                message="Content is missing",
                severity="error",
            ))
        else:
            content_len = len(message.content)

            if content_len < self.rules["min_content_length"]:
                errors.append(ValidationError(
                    field=f"{field_prefix}.content",
                    message=f"Content too short: {content_len} chars (minimum: {self.rules['min_content_length']})",
                    severity="warning",
                    value=f"<{content_len} chars>",
                ))

            if content_len > self.rules["max_content_length"]:
                errors.append(ValidationError(
                    field=f"{field_prefix}.content",
                    message=f"Content too long: {content_len} chars (maximum: {self.rules['max_content_length']})",
                    severity="warning",
                    value=f"<{content_len} chars>",
                ))

        return errors

    def validate_dataset(
        self,
        test_cases: List[TestCase],
    ) -> ValidationResult:
        """Validate a complete dataset.

        Args:
            test_cases: List of test cases to validate.

        Returns:
            ValidationResult with errors, warnings, and statistics.
        """
        all_errors: List[ValidationError] = []
        all_warnings: List[ValidationError] = []

        # Validate each test case
        for i, test_case in enumerate(test_cases):
            errors = self.validate_test_case(test_case, i)
            for err in errors:
                if err.severity == "error":
                    all_errors.append(err)
                else:
                    all_warnings.append(err)

        # Check for duplicates
        if self.rules["check_duplicates"]:
            seen_ids: Set[str] = set()
            for i, test_case in enumerate(test_cases):
                if test_case.id:
                    if test_case.id in seen_ids:
                        all_warnings.append(ValidationError(
                            field="id",
                            message=f"Duplicate ID: '{test_case.id}'",
                            severity="warning",
                            record_index=i,
                            value=test_case.id,
                        ))
                    seen_ids.add(test_case.id)

        # Calculate statistics
        statistics = self._calculate_statistics(test_cases)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            statistics=statistics,
        )

    def _calculate_statistics(
        self,
        test_cases: List[TestCase],
    ) -> Dict[str, Any]:
        """Calculate dataset statistics.

        Args:
            test_cases: List of test cases.

        Returns:
            Statistics dictionary.
        """
        if not test_cases:
            return {"total_records": 0}

        total_messages = sum(len(tc.messages) for tc in test_cases)
        content_lengths = []
        role_counts: Dict[str, int] = {}

        for tc in test_cases:
            for msg in tc.messages:
                content_lengths.append(len(msg.content))
                role_counts[msg.role] = role_counts.get(msg.role, 0) + 1

        return {
            "total_records": len(test_cases),
            "total_messages": total_messages,
            "avg_messages_per_record": total_messages / len(test_cases),
            "content_length": {
                "min": min(content_lengths) if content_lengths else 0,
                "max": max(content_lengths) if content_lengths else 0,
                "avg": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            },
            "role_distribution": role_counts,
            "unique_ids": len(set(tc.id for tc in test_cases if tc.id)),
            "records_without_id": sum(1 for tc in test_cases if not tc.id),
            "records_with_metadata": sum(1 for tc in test_cases if tc.metadata),
        }


def validate_jsonl_content(content: str) -> ValidationResult:
    """Validate JSONL content string.

    Args:
        content: JSONL content string.

    Returns:
        ValidationResult.
    """
    from .sources.jsonl import parse_jsonl_test_case

    test_cases: List[TestCase] = []
    parse_errors: List[ValidationError] = []

    for i, line in enumerate(content.strip().split("\n")):
        line = line.strip()
        if not line:
            continue

        try:
            test_case = parse_jsonl_test_case(line, index=i)
            test_cases.append(test_case)
        except Exception as e:
            parse_errors.append(ValidationError(
                field="validation",
                message=f"Validation error: {e}",
                severity="error",
                record_index=i,
            ))

    if parse_errors:
        return ValidationResult(
            valid=False,
            errors=parse_errors,
            statistics={"total_records": len(test_cases), "parse_errors": len(parse_errors)},
        )

    validator = DatasetValidator()
    return validator.validate_dataset(test_cases)


def validate_file(file_path: str) -> ValidationResult:
    """Validate a dataset file.

    Args:
        file_path: Path to dataset file.

    Returns:
        ValidationResult.
    """
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        return ValidationResult(
            valid=False,
            errors=[ValidationError(
                field="file",
                message=f"File not found: {file_path}",
                severity="error",
            )],
        )

    if path.suffix == ".jsonl":
        content = path.read_text(encoding="utf-8")
        return validate_jsonl_content(content)
    elif path.suffix == ".csv":
        # Use CSV source to load and validate
        from .sources.csv import CSVDatasetSource
        try:
            source = CSVDatasetSource(name="validation", config={"path": str(path)})
            test_cases = source.load()
            validator = DatasetValidator()
            return validator.validate_dataset(test_cases)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="csv",
                    message=f"CSV parse error: {e}",
                    severity="error",
                )],
            )
    else:
        return ValidationResult(
            valid=False,
            errors=[ValidationError(
                field="file",
                message=f"Unsupported file format: {path.suffix}",
                severity="error",
            )],
        )

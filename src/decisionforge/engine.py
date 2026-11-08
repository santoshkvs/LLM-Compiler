"""Runtime engine (table + optional extractor) and JSONL audit logging."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from typing import Any, Dict, Mapping, Optional

from .extract import FeatureExtractor
from .table import Decision, DecisionTable


class AuditLogger:
    """Append one JSON line per decision.

    Input text is hashed by default (it may contain personal data). Pass
    ``log_input=True`` to also store it.
    """

    def __init__(self, path: str, *, log_input: bool = False):
        self.path = path
        self.log_input = log_input
        self._lock = threading.Lock()

    def log(self, decision: Decision, *, request_id: Optional[str] = None, input_text: Optional[str] = None) -> Dict[str, Any]:
        rec = decision.to_record()
        rec["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")
        rec["request_id"] = request_id
        if input_text is not None:
            rec["input_sha256"] = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            if self.log_input:
                rec["input"] = input_text
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return rec


class DecisionEngine:
    """Evaluate a frozen table, optionally extracting features from text first.

    Any extraction failure (bad JSON, out-of-domain value, low confidence,
    provider error) returns the table's ``safe_default`` with ``fallback=True``
    instead of guessing.
    """

    def __init__(
        self,
        table: DecisionTable,
        extractor: Optional[FeatureExtractor] = None,
        audit: Optional[AuditLogger] = None,
    ):
        self.table = table
        self.extractor = extractor
        self.audit = audit

    def decide(self, features: Mapping[str, str], *, request_id: Optional[str] = None) -> Decision:
        """Deterministic: the same features always give the same decision."""
        d = self.table.evaluate(features)
        if self.audit:
            self.audit.log(d, request_id=request_id)
        return d

    def decide_text(
        self,
        text: str,
        *,
        known_features: Optional[Mapping[str, str]] = None,
        request_id: Optional[str] = None,
    ) -> Decision:
        if self.extractor is None:
            raise RuntimeError("decide_text needs an extractor; use decide(features) or pass one")
        ex = self.extractor.extract(text, known=known_features)
        if ex.ok:
            d = self.table.evaluate(ex.features)
            d.confidence = ex.confidence
        else:
            d = Decision(
                output=self.table.safe_default,
                rule_id="fallback:extraction",
                rule_index=None,
                table_name=self.table.name,
                table_version=self.table.version,
                table_hash=self.table.content_hash(),
                features=dict(ex.features),
                fallback=True,
                reason=ex.reason,
                confidence=ex.confidence,
            )
        if self.audit:
            self.audit.log(d, request_id=request_id, input_text=text)
        return d

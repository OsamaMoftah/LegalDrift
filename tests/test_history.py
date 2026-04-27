"""Tests for legaldrift.core.history module."""

import json
import os
from pathlib import Path

import pytest

from legaldrift.core.history import DriftHistory, DriftRecord
from legaldrift.core.detector import DriftResult


class TestDriftRecord:
    def test_to_dict(self):
        record = DriftRecord(
            timestamp="2024-01-01T00:00:00Z",
            baseline_id="doc1",
            current_id="doc2",
            result={"drift_detected": True, "p_value": 0.01},
            notes="test",
            tags=["tag1"],
        )
        d = record.to_dict()
        assert d["baseline_id"] == "doc1"
        assert d["current_id"] == "doc2"
        assert d["tags"] == ["tag1"]

    def test_from_dict(self):
        data = {
            "timestamp": "2024-01-01T00:00:00Z",
            "baseline_id": "doc1",
            "current_id": "doc2",
            "result": {"drift_detected": False},
            "notes": "",
            "tags": [],
        }
        record = DriftRecord.from_dict(data)
        assert record.baseline_id == "doc1"
        assert record.result["drift_detected"] is False


class TestDriftHistoryJSON:
    def setup_method(self):
        self.path = Path("test_history.json")
        if self.path.exists():
            self.path.unlink()

    def teardown_method(self):
        if self.path.exists():
            self.path.unlink()

    def test_save_and_query(self):
        history = DriftHistory(path=self.path, backend="json")
        result = DriftResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.99,
            severity=0.5,
            effect_size=0.3,
            threshold=0.05,
            tests={},
        )
        history.save("doc1", "doc2", result, notes="test", tags=["a", "b"])

        records = history.query()
        assert len(records) == 1
        assert records[0].baseline_id == "doc1"
        assert records[0].current_id == "doc2"
        assert records[0].notes == "test"
        assert records[0].tags == ["a", "b"]

    def test_query_by_baseline(self):
        history = DriftHistory(path=self.path, backend="json")
        result = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("docA", "docB", result)
        history.save("docC", "docD", result)

        records = history.query(baseline_id="docA")
        assert len(records) == 1
        assert records[0].baseline_id == "docA"

    def test_query_by_drift_status(self):
        history = DriftHistory(path=self.path, backend="json")
        result_drift = DriftResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.99,
            severity=0.5,
            effect_size=0.3,
            threshold=0.05,
            tests={},
        )
        result_no_drift = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("d1", "d2", result_drift)
        history.save("d3", "d4", result_no_drift)

        drift_records = history.query(drift_detected=True)
        assert len(drift_records) == 1
        assert drift_records[0].result["drift_detected"] is True

    def test_get_latest(self):
        history = DriftHistory(path=self.path, backend="json")
        result = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("doc1", "doc2", result)
        history.save("doc1", "doc3", result)

        latest = history.get_latest(baseline_id="doc1")
        assert latest is not None
        assert latest.current_id == "doc3"

    def test_get_latest_none(self):
        history = DriftHistory(path=self.path, backend="json")
        latest = history.get_latest()
        assert latest is None

    def test_clear(self):
        history = DriftHistory(path=self.path, backend="json")
        result = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("a", "b", result)
        history.clear()
        records = history.query()
        assert len(records) == 0
        assert not self.path.exists()

    def test_load_existing(self):
        # Pre-populate file
        with open(self.path, "w") as f:
            json.dump(
                [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "baseline_id": "old",
                        "current_id": "new",
                        "result": {"drift_detected": True},
                        "notes": "",
                        "tags": [],
                    }
                ],
                f,
            )
        history = DriftHistory(path=self.path, backend="json")
        records = history.query()
        assert len(records) == 1
        assert records[0].baseline_id == "old"


class TestDriftHistorySQLite:
    def setup_method(self):
        self.path = Path("test_history.db")
        if self.path.exists():
            self.path.unlink()

    def teardown_method(self):
        if self.path.exists():
            self.path.unlink()

    def test_save_and_query(self):
        history = DriftHistory(path=self.path, backend="sqlite")
        result = DriftResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.99,
            severity=0.5,
            effect_size=0.3,
            threshold=0.05,
            tests={},
        )
        history.save("doc1", "doc2", result, notes="sqlite test", tags=["sql"])

        records = history.query()
        assert len(records) == 1
        assert records[0].baseline_id == "doc1"
        assert records[0].notes == "sqlite test"
        assert records[0].tags == ["sql"]

    def test_query_by_baseline_sqlite(self):
        history = DriftHistory(path=self.path, backend="sqlite")
        result = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("base", "curr1", result)
        history.save("other", "curr2", result)

        records = history.query(baseline_id="base")
        assert len(records) == 1
        assert records[0].current_id == "curr1"

    def test_query_by_drift_status_sqlite(self):
        history = DriftHistory(path=self.path, backend="sqlite")
        result_drift = DriftResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.99,
            severity=0.5,
            effect_size=0.3,
            threshold=0.05,
            tests={},
        )
        result_no_drift = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("d1", "d2", result_drift)
        history.save("d3", "d4", result_no_drift)

        drift_records = history.query(drift_detected=True)
        assert len(drift_records) == 1
        assert drift_records[0].result["drift_detected"] is True

    def test_clear_sqlite(self):
        history = DriftHistory(path=self.path, backend="sqlite")
        result = DriftResult(
            drift_detected=False,
            p_value=0.5,
            confidence=0.5,
            severity=0.0,
            effect_size=0.0,
            threshold=0.05,
            tests={},
        )
        history.save("a", "b", result)
        history.clear()
        records = history.query()
        assert len(records) == 0
        # SQLite file may persist (empty table), so we don't assert path absence

    def test_default_path(self):
        history = DriftHistory(backend="json")
        assert history.path.name == "drift_history.json"
        if history.path.exists():
            history.path.unlink()

        history2 = DriftHistory(backend="sqlite")
        assert history2.path.name == "drift_history.db"
        if history2.path.exists():
            history2.path.unlink()

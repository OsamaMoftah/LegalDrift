"""Tests for legaldrift CLI commands."""

import subprocess
import sys
from pathlib import Path

import pytest

PYTHON = sys.executable
CLI = "-m legaldrift.cli"


@pytest.fixture
def sample_contracts(tmp_path):
    v1 = tmp_path / "contract_v1.txt"
    v2 = tmp_path / "contract_v2.txt"
    v1.write_text(
        "This contract is between Party A and Party B.\n\n"
        "1. SERVICES\n"
        "Party A shall provide consulting services.\n\n"
        "2. PAYMENT\n"
        "Party B shall pay within 30 days.\n"
    )
    v2.write_text(
        "This contract is between Party A and Party B.\n\n"
        "1. SERVICES\n"
        "Party A shall provide consulting and AI training services.\n\n"
        "2. PAYMENT\n"
        "Party B shall pay within 15 days.\n"
    )
    return str(v1), str(v2)


class TestCLIDetect:
    def test_detect_text_output(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "--no-legal-bert", "detect", v1, v2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Drift Detection Results" in result.stdout

    def test_detect_json_output(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "detect",
                v1,
                v2,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert '"drift_detected"' in result.stdout
        assert '"p_value"' in result.stdout

    def test_detect_with_history(self, sample_contracts, tmp_path):
        v1, v2 = sample_contracts
        history_path = tmp_path / "history.json"
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "detect",
                v1,
                v2,
                "--history",
                str(history_path),
                "--notes",
                "test run",
                "--tags",
                "gdpr",
                "contract",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert history_path.exists()
        content = history_path.read_text()
        assert "test run" in content
        assert "contract" in content


class TestCLIAnalyze:
    def test_analyze_text_output(self, sample_contracts):
        v1, _ = sample_contracts
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "--no-legal-bert", "analyze", v1],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Document Analysis" in result.stdout
        assert "obligation" in result.stdout

    def test_analyze_json_output(self, sample_contracts):
        v1, _ = sample_contracts
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "--no-legal-bert", "analyze", v1, "--output", "json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert '"concepts"' in result.stdout

    def test_analyze_jurisdiction(self, sample_contracts):
        v1, _ = sample_contracts
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "-j",
                "EU",
                "analyze",
                v1,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert '"jurisdiction": "EU"' in result.stdout


class TestCLIChunks:
    def test_chunks_text_output(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "--no-legal-bert", "chunks", v1, v2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Chunked Drift Detection Results" in result.stdout

    def test_chunks_json_output(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "chunks",
                v1,
                v2,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert '"status"' in result.stdout


class TestCLIHistory:
    def test_history_empty(self, tmp_path):
        hist = tmp_path / "empty.json"
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "history", "--path", str(hist)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0 records" in result.stdout or "records" in result.stdout

    def test_history_with_data(self, sample_contracts, tmp_path):
        v1, v2 = sample_contracts
        hist = tmp_path / "hist.json"
        # First, save a detection
        subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "detect",
                v1,
                v2,
                "--history",
                str(hist),
            ],
            capture_output=True,
            text=True,
        )
        # Then query
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "history", "--path", str(hist)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "1 records" in result.stdout or "records" in result.stdout


class TestCLICompare:
    def test_compare_output(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli", "--no-legal-bert", "compare", v1, v2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "LegalDrift" in result.stdout
        assert "ADWIN" in result.stdout

    def test_compare_json(self, sample_contracts):
        v1, v2 = sample_contracts
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "legaldrift.cli",
                "--no-legal-bert",
                "compare",
                v1,
                v2,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert '"legal_drift"' in result.stdout
        assert '"baselines"' in result.stdout


class TestCLIGeneral:
    def test_no_command_prints_help(self):
        result = subprocess.run(
            [PYTHON, "-m", "legaldrift.cli"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode != 0
            or "usage" in result.stdout.lower()
            or "usage" in result.stderr.lower()
        )

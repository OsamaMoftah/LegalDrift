#!/usr/bin/env python3
"""Command Line Interface for LegalDrift."""

import argparse
import json
import logging
import sys
from pathlib import Path

from legaldrift.core.document import LegalDocument
from legaldrift.core.embedding import EmbeddingEngine
from legaldrift.core.detector import DriftDetector
from legaldrift.core.concepts import LegalConceptExtractor
from legaldrift.core.baselines import ADWIN, DDM, HDP
from legaldrift.core.chunker import chunk_by_sections, align_chunks
from legaldrift.core.history import DriftHistory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_document(path: Path, jurisdiction: str = "US") -> LegalDocument:
    """Load a document from file."""
    text = path.read_text(encoding="utf-8")
    return LegalDocument(
        text=text, document_id=path.stem, jurisdiction=jurisdiction, metadata={"source": str(path)}
    )


def detect_command(args):
    """Detect drift between two documents."""
    logger.info("Detecting drift between %s and %s", args.file1, args.file2)

    doc1 = load_document(Path(args.file1), jurisdiction=args.jurisdiction)
    doc2 = load_document(Path(args.file2), jurisdiction=args.jurisdiction)

    engine = EmbeddingEngine(use_legal_bert=not args.no_legal_bert)
    detector = DriftDetector(threshold=args.threshold)

    emb1 = engine.encode([doc1.text])
    emb2 = engine.encode([doc2.text])

    result = detector.detect(emb1, emb2)

    if args.history:
        history = DriftHistory(path=Path(args.history))
        history.save(
            baseline_id=doc1.document_id,
            current_id=doc2.document_id,
            result=result,
            notes=args.notes or "",
            tags=args.tags or [],
        )
        logger.info("Saved drift record to %s", args.history)

    if args.output == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\nDrift Detection Results")
        print("=" * 40)
        print(f"Drift Detected: {result.drift_detected}")
        print(f"P-value: {result.p_value:.4f}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Severity: {result.severity:.4f}")
        print(f"Effect Size: {result.effect_size:.4f}")
        print(f"\nIndividual Tests:")
        for test_name, test_data in result.tests.items():
            print(f"  {test_name}: p={test_data['p_value']:.4f}")

    return 0


def analyze_command(args):
    """Analyze a single document."""
    logger.info("Analyzing document: %s", args.file)

    doc = load_document(Path(args.file), jurisdiction=args.jurisdiction)

    engine = EmbeddingEngine(use_legal_bert=not args.no_legal_bert)
    extractor = LegalConceptExtractor()

    concepts = extractor.extract_from_text(doc.text)

    if args.output == "json":
        result = {
            "document_id": doc.document_id,
            "jurisdiction": doc.jurisdiction,
            "word_count": doc.word_count,
            "char_count": doc.char_count,
            "concepts": list(concepts),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\nDocument Analysis")
        print("=" * 40)
        print(f"Document ID: {doc.document_id}")
        print(f"Jurisdiction: {doc.jurisdiction}")
        print(f"Word Count: {doc.word_count}")
        print(f"Character Count: {doc.char_count}")
        print(f"\nLegal Concepts Detected:")
        if concepts:
            for concept in sorted(concepts):
                print(f"  - {concept}")
        else:
            print("  (none)")

    return 0


def compare_baselines_command(args):
    """Compare our detector against baselines."""
    logger.info("Running baseline comparison")

    doc1 = load_document(Path(args.file1), jurisdiction=args.jurisdiction)
    doc2 = load_document(Path(args.file2), jurisdiction=args.jurisdiction)

    engine = EmbeddingEngine(use_legal_bert=not args.no_legal_bert)
    detector = DriftDetector(threshold=args.threshold)

    emb1 = engine.encode([doc1.text])
    emb2 = engine.encode([doc2.text])

    our_result = detector.detect(emb1, emb2)

    adwin = ADWIN()
    ddm = DDM()
    hdp = HDP()

    adwin_result = adwin.detect(emb1, emb2)
    ddm_result = ddm.detect(emb1, emb2)
    hdp_result = hdp.detect(emb1, emb2)

    if args.output == "json":
        result = {
            "legal_drift": our_result.to_dict(),
            "baselines": {
                "ADWIN": {
                    "drift_detected": adwin_result.drift_detected,
                    "p_value": adwin_result.p_value,
                    "severity": adwin_result.severity,
                },
                "DDM": {
                    "drift_detected": ddm_result.drift_detected,
                    "p_value": ddm_result.p_value,
                    "severity": ddm_result.severity,
                },
                "HDP": {
                    "drift_detected": hdp_result.drift_detected,
                    "p_value": hdp_result.p_value,
                    "severity": hdp_result.severity,
                },
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\nComparison: LegalDrift vs Baselines")
        print("=" * 50)
        print(f"\nLegalDrift:")
        print(f"  Drift: {our_result.drift_detected}, p={our_result.p_value:.4f}")

        print(f"\nADWIN:")
        print(f"  Drift: {adwin_result.drift_detected}, p={adwin_result.p_value:.4f}")

        print(f"\nDDM:")
        print(f"  Drift: {ddm_result.drift_detected}, p={ddm_result.p_value:.4f}")

        print(f"\nHDP:")
        print(f"  Drift: {hdp_result.drift_detected}, p={hdp_result.p_value:.4f}")

    return 0


def chunks_command(args):
    """Detect drift section-by-section between two documents."""
    logger.info("Chunked drift detection: %s vs %s", args.file1, args.file2)

    doc1 = load_document(Path(args.file1), jurisdiction=args.jurisdiction)
    doc2 = load_document(Path(args.file2), jurisdiction=args.jurisdiction)

    chunks1 = chunk_by_sections(doc1)
    chunks2 = chunk_by_sections(doc2)

    aligned = align_chunks(chunks1, chunks2)

    engine = EmbeddingEngine(use_legal_bert=not args.no_legal_bert)
    detector = DriftDetector(threshold=args.threshold)

    results = []

    for c1, c2 in aligned:
        if c1 is None:
            results.append(
                {
                    "status": "NEW",
                    "section": c2.metadata.get("header", f"Section {c2.chunk_index}"),
                    "drift_detected": True,
                    "p_value": 0.0,
                    "severity": 1.0,
                    "message": "Added in current document",
                }
            )
            continue
        if c2 is None:
            results.append(
                {
                    "status": "DELETED",
                    "section": c1.metadata.get("header", f"Section {c1.chunk_index}"),
                    "drift_detected": True,
                    "p_value": 0.0,
                    "severity": 1.0,
                    "message": "Removed in current document",
                }
            )
            continue

        e1 = engine.encode([c1.text])
        e2 = engine.encode([c2.text])
        result = detector.detect(e1, e2)

        header = c1.metadata.get("header", f"Section {c1.chunk_index}")
        results.append(
            {
                "status": "DRIFT" if result.drift_detected else "OK",
                "section": header,
                "drift_detected": result.drift_detected,
                "p_value": result.p_value,
                "severity": result.severity,
                "effect_size": result.effect_size,
            }
        )

    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        print(f"\nChunked Drift Detection Results")
        print("=" * 60)
        for r in results:
            status_icon = "🔴" if r["drift_detected"] else "🟢"
            if r["status"] in ("NEW", "DELETED"):
                status_icon = "🟡"
            print(f"{status_icon} [{r['status']}] {r['section'][:50]}")
            if r["status"] not in ("NEW", "DELETED"):
                print(f"    p={r['p_value']:.4f}, severity={r['severity']:.4f}")

    return 0


def history_command(args):
    """Query drift history."""
    history = DriftHistory(path=Path(args.path))
    records = history.query(
        baseline_id=args.baseline,
        drift_detected=args.drift_only if args.drift_only else None,
        limit=args.limit,
    )

    if args.output == "json":
        data = [r.to_dict() for r in records]
        print(json.dumps(data, indent=2))
    else:
        print(f"\nDrift History ({len(records)} records)")
        print("=" * 60)
        for r in records:
            status = "DRIFT" if r.result.get("drift_detected") else "OK"
            print(
                f"[{status}] {r.timestamp} | {r.baseline_id} -> {r.current_id} | "
                f"p={r.result.get('p_value', 'N/A'):.4f}"
            )
            if r.notes:
                print(f"    Note: {r.notes}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LegalDrift: Detect semantic drift in legal documents"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-legal-bert", action="store_true", help="Disable Legal-BERT (use hash embeddings)"
    )
    parser.add_argument(
        "--jurisdiction",
        "-j",
        default="US",
        help="Default legal jurisdiction (e.g., US, EU, DE, UK)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # detect
    detect_parser = subparsers.add_parser("detect", help="Detect drift between documents")
    detect_parser.add_argument("file1", help="Baseline document")
    detect_parser.add_argument("file2", help="Current document")
    detect_parser.add_argument("--threshold", type=float, default=0.05, help="Drift threshold")
    detect_parser.add_argument("--output", choices=["text", "json"], default="text")
    detect_parser.add_argument("--history", help="Path to save drift history (JSON or SQLite)")
    detect_parser.add_argument("--notes", help="Optional notes for history entry")
    detect_parser.add_argument("--tags", nargs="+", help="Optional tags for history entry")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a document")
    analyze_parser.add_argument("file", help="Document to analyze")
    analyze_parser.add_argument("--output", choices=["text", "json"], default="text")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare against baselines")
    compare_parser.add_argument("file1", help="First document")
    compare_parser.add_argument("file2", help="Second document")
    compare_parser.add_argument("--threshold", type=float, default=0.05)
    compare_parser.add_argument("--output", choices=["text", "json"], default="text")

    # chunks
    chunks_parser = subparsers.add_parser("chunks", help="Section-level drift detection")
    chunks_parser.add_argument("file1", help="Baseline document")
    chunks_parser.add_argument("file2", help="Current document")
    chunks_parser.add_argument("--threshold", type=float, default=0.05)
    chunks_parser.add_argument("--output", choices=["text", "json"], default="text")

    # history
    history_parser = subparsers.add_parser("history", help="Query drift history")
    history_parser.add_argument("--path", default="drift_history.json", help="History file path")
    history_parser.add_argument("--baseline", help="Filter by baseline document ID")
    history_parser.add_argument(
        "--drift-only", action="store_true", help="Show only drift detections"
    )
    history_parser.add_argument("--limit", type=int, default=50, help="Max records to show")
    history_parser.add_argument("--output", choices=["text", "json"], default="text")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    commands = {
        "detect": detect_command,
        "analyze": analyze_command,
        "compare": compare_baselines_command,
        "chunks": chunks_command,
        "history": history_command,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

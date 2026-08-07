"""Run inference or deterministic prediction-cache replay and write reports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.benchmark.adapter import (
    load_prediction_document,
    write_prediction_document,
)
from app.benchmark.annotations import (
    load_manifest,
    validate_manifest_references,
)
from app.benchmark.evaluator import evaluate_benchmark
from app.benchmark.reports import (
    attach_failure_artifacts,
    compare_with_baseline,
    write_reports,
)
from app.benchmark.runner import (
    build_reproducibility_snapshot,
    file_sha256,
    git_commit_hash,
    git_worktree_dirty,
    run_video_inference,
    select_videos,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark traffic review candidates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        choices=["development", "validation", "test", "all"],
        default="validation",
        help="Dataset partition to evaluate; test data should remain reserved.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Evaluate prediction JSON caches without running YOLO.",
    )
    parser.add_argument(
        "--predictions-dir",
        help="Cache directory; defaults to <output>/predictions.",
    )
    parser.add_argument("--baseline", help="Previous benchmark_report.json")
    parser.add_argument(
        "--allow-incomparable-baseline",
        action="store_true",
        help=(
            "Developer-only override: show deltas for fingerprint-mismatched runs "
            "with a prominent non-comparable warning."
        ),
    )
    parser.add_argument("--no-failure-artifacts", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    videos = select_videos(manifest, args.split)
    reference_errors = validate_manifest_references(
        manifest_path,
        manifest.model_copy(update={"videos": videos}),
        require_videos=not args.skip_inference,
        require_configs=not args.skip_inference,
    )
    if reference_errors:
        raise ValueError(
            "invalid benchmark references:\n- " + "\n- ".join(reference_errors)
        )

    repository = Path(__file__).resolve().parents[2]
    commit = git_commit_hash(repository)
    dirty = git_worktree_dirty(repository)
    cache_directory = (
        Path(args.predictions_dir).resolve()
        if args.predictions_dir
        else output / "predictions"
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for video in videos:
        cache_path = cache_directory / f"{video.id}.json"
        if args.skip_inference:
            document = load_prediction_document(cache_path)
        else:
            document = run_video_inference(
                manifest_path, video, output, git_commit=commit
            )
            write_prediction_document(document, cache_path)
        predictions[video.id] = document

    reproducibility = build_reproducibility_snapshot(
        manifest_path,
        manifest,
        videos,
        output,
        git_commit=commit,
        git_dirty=dirty,
    )
    reproducibility["prediction_cache_hashes_sha256"] = {
        video.id: file_sha256(cache_directory / f"{video.id}.json") for video in videos
    }
    reproducibility["prediction_versions"] = {
        video.id: predictions[video.id].versions.model_dump(mode="json")
        for video in videos
    }
    report = evaluate_benchmark(
        manifest_path,
        manifest,
        predictions,
        videos=videos,
        reproducibility=reproducibility,
    )
    artifacts_enabled = (
        manifest.benchmark.failure_artifacts and not args.no_failure_artifacts
    )
    if artifacts_enabled:
        attach_failure_artifacts(
            report,
            output,
            manifest_path,
            videos,
            clip_padding_seconds=manifest.benchmark.failure_clip_padding_seconds,
        )
    if args.baseline:
        with Path(args.baseline).open("r", encoding="utf-8") as stream:
            baseline = json.load(stream)
        report["baseline_comparison"] = compare_with_baseline(
            report,
            baseline,
            manifest.benchmark.baseline_tolerances,
            allow_incomparable=args.allow_incomparable_baseline,
        )
    json_path, markdown_path = write_reports(report, output)
    metrics = report["overall_metrics"]
    LOGGER.info("JSON report: %s", json_path)
    LOGGER.info("Markdown report: %s", markdown_path)
    print(
        "TP={true_positives} FP={false_positives} FN={false_negatives} "
        "precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}".format(**metrics)
    )
    acceptance = report["acceptance"]
    return 2 if acceptance["configured"] and not acceptance["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

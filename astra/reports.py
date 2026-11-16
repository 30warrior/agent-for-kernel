"""Atomic per-version reports; missing and failed measurements stay empty."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile


def atomic_write(path, content):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue())


def total_time(version, origin="dataset"):
    if not version.get("complete") or version.get("correctness") != "passed":
        return None
    cases = [case for case in version.get("cases", []) if case.get("origin", "dataset") == origin]
    return sum(case["benchmark"]["median_ms"] for case in cases) if cases else None


def summarize(payload):
    versions = payload["versions"]
    v1 = next((version for version in versions if version["version"] == "v1"), {})
    best = next((version for version in versions if version["version"] == payload["best_version"]), {})
    dataset_count = payload.get("test_suite", {}).get("dataset_case_count", len(payload["baseline"]))
    baseline_time = sum(case["median_ms"] for case in payload["baseline"][:dataset_count])
    best_time, first_time = total_time(best), total_time(v1)
    valid = best_time is not None and best_time > 0
    return dict(task=payload["task"], target=payload["target"], best_version=payload["best_version"],
                success=valid, test_suite=payload.get("test_suite"),
                generated_best_sum_median_ms=total_time(best, "generated"),
                compiled=any(version.get("compiled") for version in versions),
                correctness_passed=any(version.get("correctness") == "passed" for version in versions),
                baseline_sum_median_ms=baseline_time,
                best_sum_median_ms=best_time, v1_sum_median_ms=first_time,
                reference_speedup=baseline_time / best_time if valid else None,
                refinement_speedup=first_time / best_time if valid and first_time is not None else None)


def write_reports(payload, directory):
    directory = Path(directory)
    summary = summarize(payload)
    atomic_write(directory / "benchmark_results.json", json.dumps(dict(payload, summary=summary), indent=2))
    atomic_write(directory / "summary.json", json.dumps(summary, indent=2))
    baseline = payload["baseline"]
    dataset_count = payload.get("test_suite", {}).get("dataset_case_count", len(baseline))
    baseline_total = sum(case["median_ms"] for case in baseline[:dataset_count])
    first = next((version for version in payload["versions"] if version["version"] == "v1"), {})
    first_total = total_time(first)
    rows, details = [], []
    for version in payload["versions"]:
        value = total_time(version)
        valid = value is not None and value > 0
        rows.append(dict(version=version["version"], status=version["status"],
                         correctness=version["correctness"], complete=version["complete"],
                         sum_median_ms=value, generated_sum_median_ms=total_time(version, "generated"),
                         reference_speedup=baseline_total/value if valid else None,
                         refinement_speedup=first_total/value if valid and first_total is not None else None,
                         error=version.get("error")))
        for case in version["cases"]:
            timing = case.get("benchmark") or {}
            index = case["case_id"]
            elapsed = timing.get("median_ms")
            details.append(dict(version=version["version"], case_id=index,
                                name=case.get("name"), origin=case.get("origin", "dataset"),
                                correctness=case["correctness"], median_ms=elapsed,
                                mean_ms=timing.get("mean_ms"), std_ms=timing.get("std_ms"),
                                p95_ms=timing.get("p95_ms"), baseline_median_ms=baseline[index]["median_ms"],
                                speedup=baseline[index]["median_ms"]/elapsed if elapsed else None,
                                error=case.get("error")))
    write_csv(directory / "benchmark_summary.csv", rows,
              ["version","status","correctness","complete","sum_median_ms","generated_sum_median_ms","reference_speedup","refinement_speedup","error"])
    write_csv(directory / "benchmark_details.csv", details,
              ["version","case_id","name","origin","correctness","median_ms","mean_ms","std_ms","p95_ms","baseline_median_ms","speedup","error"])
    return summary

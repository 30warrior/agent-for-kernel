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
    if "selection" in payload:
        final = payload.get("final_evaluation", {})
        metric = payload["selection"]["metric"]
        speedups = final.get("speedups") or {}
        valid = final.get("status") == "passed" and bool(speedups)
        baseline_time = sum(case["median_ms"] for case in final.get("baseline", [])) if valid else None
        best_time = total_time(final) if valid else None
        first_score = (v1.get("selection_speedups") or {}).get(metric)
        best_score = (best.get("selection_speedups") or {}).get(metric)
        return dict(task=payload["task"], target=payload["target"], best_version=payload["best_version"],
            success=valid, test_suite=payload.get("test_suite"), selection=payload["selection"],
            final_evaluation_status=final.get("status", "pending"), reference_speedup_metric=metric,
            compiled=any(version.get("compiled") for version in versions),
            correctness_passed=any(version.get("correctness") == "passed" for version in versions),
            generated_best_sum_median_ms=total_time(best, "generated"),
            baseline_sum_median_ms=baseline_time, best_sum_median_ms=best_time,
            v1_sum_median_ms=total_time(v1),
            reference_speedup=speedups.get(metric) if valid else None,
            final_speedups=speedups if valid else None,
            final_total_time_speedup=baseline_time/best_time if valid and best_time else None,
            refinement_speedup=best_score/first_score if first_score and best_score else None,
            refinement_speedup_scope="generated_cases")
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
        refinement = first_total/value if valid and first_total is not None else None
        if "selection" in payload:
            metric = payload["selection"]["metric"]
            first_score = (first.get("selection_speedups") or {}).get(metric)
            current_score = (version.get("selection_speedups") or {}).get(metric)
            refinement = current_score/first_score if first_score and current_score else None
        rows.append(dict(version=version["version"], status=version["status"],
                         selection_arithmetic_speedup=(version.get("selection_speedups") or {}).get("arithmetic"),
                         selection_geometric_speedup=(version.get("selection_speedups") or {}).get("geometric"),
                         correctness=version["correctness"], complete=version["complete"],
                         sum_median_ms=value, generated_sum_median_ms=total_time(version, "generated"),
                         reference_speedup=baseline_total/value if valid else None,
                         refinement_speedup=refinement,
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
              ["version","status","correctness","complete","selection_arithmetic_speedup","selection_geometric_speedup","sum_median_ms","generated_sum_median_ms","reference_speedup","refinement_speedup","error"])
    write_csv(directory / "benchmark_details.csv", details,
              ["version","case_id","name","origin","correctness","median_ms","mean_ms","std_ms","p95_ms","baseline_median_ms","speedup","error"])
    final = payload.get("final_evaluation", {})
    final_rows = []
    for case in final.get("cases", []):
        timing = case.get("benchmark") or {}
        index = case["case_id"]
        baselines = final.get("baseline", [])
        final_rows.append(dict(version=final.get("version"), case_id=index, name=case.get("name"),
            correctness=case["correctness"], median_ms=timing.get("median_ms"),
            baseline_median_ms=baselines[index]["median_ms"] if index < len(baselines) else None,
            speedup=case.get("reference_speedup"), error=case.get("error")))
    if "selection" in payload:
        write_csv(directory / "final_benchmark_details.csv", final_rows,
                  ["version","case_id","name","correctness","median_ms","baseline_median_ms","speedup","error"])
    return summary

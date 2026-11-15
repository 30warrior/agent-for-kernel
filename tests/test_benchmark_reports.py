"""Exercise report persistence without calling models or running GPU kernels."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import cuda_kernel_optimizer_multi as app
from test import BenchmarkResult, CorrectnessResult


def result(time_ms, baseline_ms):
    return BenchmarkResult(
        status="success",
        generated_metrics={
            "mean_time_ms": time_ms,
            "std_time_ms": 0.01,
            "throughput_elements_per_second": 16000 / time_ms,
        },
        baseline_metrics={"mean_time_ms": baseline_ms},
        comparison_metrics={"speedup": baseline_ms / time_ms},
        total_elements=16,
        input_shape_str="B=1, D=16",
    )


class FakeCollection:
    def __init__(self, results):
        self.results = results

    def __len__(self):
        return len(self.results)

    def get(self, name):
        return SimpleNamespace(description="same shape, distinct test case")

    def run_all_benchmarks(self, **kwargs):
        return self.results

    def run_all_correctness(self, **kwargs):
        return {name: CorrectnessResult(status="failed") for name in self.results}


class BenchmarkReportsTest(unittest.TestCase):
    def setUp(self):
        self.previous_state = copy.deepcopy(app.optimization_state)
        self.folder = tempfile.TemporaryDirectory(prefix="astra-report-test-")
        self.run_dir = Path(self.folder.name)
        (self.run_dir / "kernels").mkdir()
        app.optimization_state.clear()
        app.optimization_state.update({
            "kernel_output_dir": str(self.run_dir / "kernels"),
            "kernel_versions": {},
            "benchmark_results": {},
            "detailed_correctness_results": {},
            "baseline": {"func": "silu_and_mul"},
            "generic_test_cases": {"small_fp16": {"B": np.int64(1), "value_dtype": app.torch.float16}},
        })
        self.callables = patch.multiple(app, generated_callable=object(), baseline_callable=object())
        self.callables.start()
        # The reporting integration does not require an installed NVML driver.
        self.nvml = patch.dict(sys.modules, {"pynvml": None})
        self.nvml.start()

    def tearDown(self):
        self.nvml.stop()
        self.callables.stop()
        app.optimization_state.clear()
        app.optimization_state.update(self.previous_state)
        self.folder.cleanup()

    def benchmark(self, version, results):
        app.optimization_state["generated_wrapper"] = {"module": f"gen_{version}", "func": "sgl_silu_mul"}
        app.optimization_state["test_collection"] = FakeCollection(results)
        app.optimization_state["kernel_versions"][version] = {
            "path": str(self.run_dir / "kernels" / f"kernel_{version}.cu"),
        }
        text = app.benchmark_kernel_detailed(version)
        self.assertNotIn("Error", text)

    def csv_rows(self, name):
        with (self.run_dir / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_each_version_is_saved_with_case_identity_and_failures(self):
        app.optimization_state["detailed_correctness_results"]["v1"] = {
            "overall_status": "passed", "test_results": {"small_fp16": {"status": "passed"}},
        }
        self.benchmark("v1", {"small_fp16": result(1, 2), "small_bf16": result(2, 4)})
        first_json = json.loads((self.run_dir / "benchmark_results.json").read_text())
        self.assertEqual(list(first_json["benchmark_results"]), ["v1"])
        self.assertEqual(first_json["test_cases"]["small_fp16"]["B"], 1)
        self.assertEqual(first_json["test_cases"]["small_fp16"]["value_dtype"], "torch.float16")

        self.benchmark("v2", {
            "small_fp16": result(0.5, 2),
            "small_bf16": result(1, 4),
            "failed_case": BenchmarkResult(status="error", error_message="launch failed, try again\nsecond line"),
        })
        payload = json.loads((self.run_dir / "benchmark_results.json").read_text())
        self.assertEqual(list(payload["benchmark_results"]), ["v1", "v2"])
        self.assertIn("measured_at", payload["benchmark_results"]["v2"])
        rows = self.csv_rows("benchmark_summary.csv")
        self.assertEqual([row["version"] for row in rows], ["v1", "v2"])
        self.assertEqual(rows[0]["correctness_status"], "passed")
        self.assertEqual(rows[1]["correctness_status"], "not_tested")
        self.assertEqual(rows[1]["failed_tests"], "1")
        self.assertAlmostEqual(float(rows[1]["mean_time_ms"]), 0.75)
        self.assertAlmostEqual(float(rows[1]["aggregate_speedup"]), 4)
        detail = self.csv_rows("benchmark_details.csv")
        self.assertEqual(len(detail), 5)
        failed = next(row for row in detail if row["test_name"] == "failed_case")
        self.assertEqual(failed["mean_time_ms"], "")
        self.assertIn("\nsecond line", failed["error_message"])
        text = (self.run_dir / "performance.log").read_text()
        self.assertIn("---- Benchmark Summary (v1) ----", text)
        self.assertIn("---- Benchmark Summary (v2) ----", text)
        self.assertIn("COMPREHENSIVE CUDA PERFORMANCE ANALYSIS:", text)
        self.assertIn("Execution Time:", text)
        self.assertFalse(list(self.run_dir.glob(".*.tmp")))

        # A later failed correctness recheck refreshes the already saved reports.
        app.verify_kernel_correctness_detailed("v2")
        self.assertEqual(self.csv_rows("benchmark_summary.csv")[1]["correctness_status"], "failed")

    def test_all_failed_measurements_are_blank_not_zero(self):
        self.benchmark("v1", {"broken": BenchmarkResult(status="error", error_message="no measurement")})
        row = self.csv_rows("benchmark_summary.csv")[0]
        self.assertEqual(row["successful_tests"], "0")
        self.assertEqual(row["failed_tests"], "1")
        for field in ["mean_time_ms", "baseline_mean_time_ms", "aggregate_speedup"]:
            self.assertEqual(row[field], "")

    def test_current_performance_is_logged_before_strategy_request_fails(self):
        self.benchmark("v1", {"small": result(1, 2)})
        optimizer = object.__new__(app.CUDAKernelOptimizer)
        with patch.object(optimizer, "_get_kernel_code", return_value="source"), \
             patch.object(app.Runner, "run_sync", side_effect=RuntimeError("simulated model failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated model failure"):
                optimizer._step_get_suggestions("v1", "v2")
        text = (self.run_dir / "performance.log").read_text()
        self.assertIn("---- Benchmark Summary (v1) ----", text)
        self.assertIn("---- CURRENT PERFORMANCE (v1 -> v2) ----", text)
        self.assertTrue((self.run_dir / "benchmark_results.json").exists())

    def test_interrupted_file_replacement_keeps_previous_report(self):
        path = self.run_dir / "benchmark_results.json"
        path.write_text('{"previous": true}', encoding="utf-8")
        with patch.object(app.os, "replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                app._write_report_atomic(path, lambda stream: stream.write("new contents"))
        self.assertEqual(json.loads(path.read_text()), {"previous": True})
        self.assertFalse(list(self.run_dir.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()

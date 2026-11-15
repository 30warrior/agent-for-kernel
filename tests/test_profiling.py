"""Check device matching, optional NVML failures and profiling display."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cuda_kernel_optimizer_multi as app
from format import format_benchmark_summary
import test as framework


class ProfilingTest(unittest.TestCase):
    def test_nvml_matches_active_cuda_uuid_and_balances_initialization(self):
        nvml = Mock()
        with patch.dict(sys.modules, {"pynvml": nvml}), \
             patch.object(app.torch.cuda, "current_device", return_value=0), \
             patch.object(app.torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid="physical-one")):
            with app._nvml_for_current_device() as (handle, note):
                self.assertIs(handle, nvml.nvmlDeviceGetHandleByUUID.return_value)
                self.assertIsNone(note)
            nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-physical-one")
            nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
            nvml.nvmlShutdown.assert_called_once()

    def test_nvml_failure_is_a_note_not_a_benchmark_failure(self):
        nvml = Mock()
        nvml.nvmlInit.side_effect = RuntimeError("driver unavailable")
        with patch.dict(sys.modules, {"pynvml": nvml}):
            with app._nvml_for_current_device() as (handle, note):
                self.assertIsNone(handle)
                self.assertIn("driver unavailable", note)
        nvml.nvmlShutdown.assert_not_called()

    def test_test_collection_forwards_the_nvml_handle(self):
        collection = framework.TestCollection("silu")
        case = Mock(name="case")
        case.name = "small"
        collection.add(case)
        handle = object()
        collection.run_all_benchmarks("generated", "baseline", nvml_handle=handle, nvml_note="note")
        case.benchmark.assert_called_once_with("generated", "baseline", 20, 100, nvml_handle=handle, nvml_note="note")

    def run_case(self, nvml, handle="handle", note=None):
        case = framework.SiluTest("small", 1, 16)
        case._tensors_prepared = True
        with patch.dict(sys.modules, {"pynvml": nvml}), \
             patch.object(case, "get_memory_bytes", return_value=128), \
             patch.object(case, "make_generated_caller", return_value=lambda: None), \
             patch.object(case, "make_baseline_caller", return_value=lambda: None), \
             patch.object(framework.torch.cuda, "empty_cache"), \
             patch.object(framework.torch.cuda, "synchronize"), \
             patch.object(framework, "time_cuda_kernel", side_effect=[framework.TimingStats(1, 0.1), framework.TimingStats(2, 0.1)]):
            return case.benchmark(None, None, nvml_handle=handle, nvml_note=note)

    def test_real_sample_fields_and_zero_utilization_are_preserved(self):
        nvml = Mock()
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=64 * 1024**2, total=256 * 1024**2)
        nvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=0, memory=13)
        nvml.nvmlDeviceGetUUID.return_value = "GPU-physical-one"
        result = self.run_case(nvml)
        self.assertEqual(result.status, "success", result.error_message)
        prof = result.profiling_data
        self.assertTrue(prof["nvml_available"])
        self.assertEqual(prof["real_gpu_utilization_percent"], 0)
        self.assertEqual(prof["real_memory_controller_utilization_percent"], 13)
        self.assertEqual(prof["memory_used_mb"], 64)
        self.assertEqual(prof["memory_utilization_percent"], 25)
        self.assertNotIn("arithmetic_intensity_ops_per_byte", prof)
        self.assertNotIn("memory_bandwidth_efficiency_percent", prof)
        self.assertNotIn("theoretical_memory_bandwidth_gb_s", prof)

    def test_failed_sampling_keeps_timing_and_reason(self):
        nvml = Mock()
        nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("sampling unavailable")
        result = self.run_case(nvml)
        self.assertEqual(result.status, "success", result.error_message)
        self.assertEqual(result.generated_metrics["mean_time_ms"], 1)
        self.assertFalse(result.profiling_data["nvml_available"])
        self.assertIn("sampling unavailable", result.profiling_data["nvml_note"])

    def test_summary_uses_test_names_and_does_not_request_reinstallation(self):
        prof = {
            "nvml_available": True, "real_gpu_utilization_percent": 0,
            "real_memory_controller_utilization_percent": 13,
            "memory_used_mb": 64, "memory_utilization_percent": 25,
            "actual_memory_bandwidth_gb_s": 1, "gpu_uuid": "GPU-physical-one",
        }
        cases = {
            name: {"status": "success", "total_elements": 16, "generated_metrics": {"mean_time_ms": value}}
            for name, value in [("fp16", 1), ("bf16", 2)]
        }
        text = format_benchmark_summary({
            "success": True, "results": {"16": {"mean_time_ms": 2}},
            "detailed_results": cases, "profiling": {"fp16": prof, "bf16": prof},
        })
        self.assertIn("Size 16 (fp16)", text)
        self.assertIn("Size 16 (bf16)", text)
        self.assertIn("device sample): 0.0%", text)
        self.assertIn("64 MiB (25.0% of capacity)", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("CLASSIFICATION:", text)
        legacy = format_benchmark_summary({"success": True, "results": {"16": {"mean_time_ms": 1}}})
        self.assertIn("NVML samples unavailable", legacy)
        self.assertNotIn("pip install", legacy)


if __name__ == "__main__":
    unittest.main()

"""
Formatting utilities for CUDA kernel optimization output.
"""

import numpy as np
from typing import Dict, Any, List


def format_benchmark_summary(benchmark_data: Dict[str, Any]) -> str:
    """Show measured test cases without inventing hardware profiler metrics."""
    if not benchmark_data or not benchmark_data.get("success"):
        return "No benchmark data available"

    details = benchmark_data.get("detailed_results", {})
    profiling = benchmark_data.get("profiling", {})
    records = []
    # Use the same test-name keys as the stored profiling data. Distinct cases
    # with identical sizes must retain their own metrics.
    for name, case in details.items():
        if case.get("status") == "success":
            label = f"Size {case.get('total_elements', '?')} ({name})"
            records.append((label, case.get("generated_metrics", {}),
                            profiling.get(name, case.get("profiling_data", {}))))
    if not details:
        for size, metrics in benchmark_data.get("results", {}).items():
            records.append((f"Size {size}", metrics, profiling.get(size, {})))
    if not records:
        lines = ["No successful benchmark measurements available"]
        for name, case in details.items():
            lines.append(f"  - {name}: {case.get('status', 'unknown')} | {case.get('error_message') or ''}")
        return "\n".join(lines)

    lines = ["COMPREHENSIVE CUDA PERFORMANCE ANALYSIS:"]
    for label, metrics, prof in records:
        lines.extend([
            f"  {label}:",
            f"    - Execution Time: {metrics.get('mean_time_ms', 0):.4f}ms",
            f"    - Throughput: {metrics.get('throughput_elements_per_second', 0):,.0f} elem/s",
        ])
        if prof.get("actual_memory_bandwidth_gb_s", 0) > 0:
            lines.append(f"    - Effective Memory Bandwidth (estimated bytes/time): {prof['actual_memory_bandwidth_gb_s']:.1f} GiB/s")
        if prof.get("nvml_available"):
            lines.extend([
                f"    - NVML GPU Utilization (device sample): {prof['real_gpu_utilization_percent']:.1f}%",
                f"    - NVML Memory Controller Activity: {prof['real_memory_controller_utilization_percent']:.1f}%",
                f"    - GPU Memory Used: {prof['memory_used_mb']:.0f} MiB ({prof['memory_utilization_percent']:.1f}% of capacity)",
                f"    - GPU UUID: {prof.get('gpu_uuid', 'unavailable')}",
            ])
        if prof.get("nvml_note"):
            lines.append(f"    - NVML Note: {prof['nvml_note']}")
        lines.append("")

    lines.append("OPTIMIZATION ANALYSIS BASED ON REAL MEASUREMENTS:")
    bandwidths = [prof["actual_memory_bandwidth_gb_s"] for _, _, prof in records
                  if prof.get("actual_memory_bandwidth_gb_s", 0) > 0]
    samples = [prof for _, _, prof in records if prof.get("nvml_available")]
    if bandwidths:
        lines.append(f"  - Average Effective Memory Bandwidth: {float(np.mean(bandwidths)):.1f} GiB/s")
        lines.append("  - Effective bandwidth is calculated from estimated bytes per call and CUDA event timing.")
    if samples:
        utilization = float(np.mean([prof["real_gpu_utilization_percent"] for prof in samples]))
        lines.append(f"  - Average NVML GPU Utilization (device samples): {utilization:.1f}%")
        lines.append("  - NVML uses device-wide sampling windows; these samples are not kernel-specific occupancy or utilization.")
    else:
        lines.append("  - NVML samples unavailable; CUDA event timing and throughput are still recorded.")
        if not any(prof.get("nvml_note") for _, _, prof in records):
            lines.append("  - No NVML samples were collected for these results; this does not establish a missing dependency.")
    lines.append("  - Occupancy, arithmetic intensity, theoretical peak bandwidth and bandwidth efficiency were not measured.")
    return "\n".join(lines) + "\n"

def format_test_results_for_suggestions(correctness_results: Dict[str, Any]) -> str:
    """Format detailed test results for optimization suggestions."""
    if not correctness_results or not correctness_results.get("test_results"):
        return "No detailed test results available."
    
    test_results = correctness_results.get("test_results", {})
    summary_stats = correctness_results.get("summary_stats", {})
    
    detail = f"CORRECTNESS ANALYSIS ({summary_stats.get('passed', 0)}/{summary_stats.get('total_tests', 0)} passed):\n"
    
    # List passed tests
    passed_tests = [name for name, result in test_results.items() if result.get("status") == "passed"]
    if passed_tests:
        detail += f"\n✅ PASSED TESTS ({len(passed_tests)}):\n"
        for test_name in passed_tests:
            result = test_results[test_name]
            max_abs_diff = result.get("max_abs_diff")
            rel_diff = result.get("relative_diff")
            # Safe formatting
            try:
                abs_str = f"{(max_abs_diff if max_abs_diff is not None else 0.0):.2e}"
            except Exception:
                abs_str = str(max_abs_diff)
            try:
                rel_str = f"{(rel_diff if rel_diff is not None else 0.0):.2e}"
            except Exception:
                rel_str = str(rel_diff)
            detail += f"  - {test_name}: abs_diff={abs_str} rel_diff={rel_str}\n"
    
    # List failed tests
    failed_tests = [name for name, result in test_results.items() if result.get("status") == "failed"]
    if failed_tests:
        detail += f"\n❌ FAILED TESTS ({len(failed_tests)}):\n"
        for test_name in failed_tests:
            result = test_results[test_name]
            max_abs_diff = result.get("max_abs_diff", "N/A")
            rel_diff = result.get("relative_diff", "N/A") 
            error_msg = result.get("error_message", "Unknown error")
            detail += f"  - {test_name}: abs_diff={max_abs_diff} rel_diff={rel_diff} - {error_msg}\n"
    
    # List error tests  
    error_tests = [name for name, result in test_results.items() if result.get("status") == "error"]
    if error_tests:
        detail += f"\n⚠️ ERROR TESTS ({len(error_tests)}):\n"
        for test_name in error_tests:
            result = test_results[test_name]
            error_msg = result.get("error_message", "Unknown error")
            detail += f"  - {test_name}: {error_msg}\n"
    
    return detail


def format_comparison_summary(
    prev_version: str,
    curr_version: str,
    benchmark_results: Dict[str, Any]
) -> str:
    """
    Return a concise comparison summary between two versions.
    If previous version data is unavailable, fall back to current version summary.
    
    Args:
        prev_version: Previous version string (e.g., 'v1')
        curr_version: Current version string (e.g., 'v2')
        benchmark_results: Dictionary containing benchmark results for all versions
        
    Returns:
        Formatted comparison summary string
    """
    if prev_version not in benchmark_results or curr_version not in benchmark_results:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    prev_data = benchmark_results[prev_version]
    curr_data = benchmark_results[curr_version]
    prev_details = prev_data.get("detailed_results", {}) or {}
    curr_details = curr_data.get("detailed_results", {}) or {}
    
    lines = [f"PERFORMANCE COMPARISON: {prev_version} → {curr_version}"]
    compared_any = False
    
    if prev_details and curr_details:
        common_tests = sorted(set(prev_details.keys()) & set(curr_details.keys()))
        for test_name in common_tests:
            prev_res = prev_details.get(test_name, {})
            curr_res = curr_details.get(test_name, {})
            if prev_res.get("status") != "success" or curr_res.get("status") != "success":
                continue
            p_gen = prev_res.get("generated_metrics", {})
            c_gen = curr_res.get("generated_metrics", {})
            p_time = p_gen.get("mean_time_ms", 0.0) or 0.0
            c_time = c_gen.get("mean_time_ms", 0.0) or 0.0
            p_tp = p_gen.get("throughput_elements_per_second", 0.0) or 0.0
            c_tp = c_gen.get("throughput_elements_per_second", 0.0) or 0.0
            time_improv = ((p_time - c_time) / p_time * 100.0) if p_time > 0 else 0.0
            tp_improv = ((c_tp - p_tp) / p_tp * 100.0) if p_tp > 0 else 0.0
            lines.append(
                f"  - {test_name}: time {p_time:.4f}ms → {c_time:.4f}ms ({time_improv:+.1f}%), "
                f"throughput {p_tp/1e6:.1f}M/s → {c_tp/1e6:.1f}M/s ({tp_improv:+.1f}%)"
            )
            compared_any = True
    
    if compared_any:
        return "\n".join(lines)
    
    # Fallback to legacy summary if detailed data unavailable
    prev_results = prev_data.get("results", {})
    curr_results = curr_data.get("results", {})
    if not prev_results or not curr_results:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return x
    
    common_sizes = sorted(set(prev_results.keys()) & set(curr_results.keys()), key=_safe_int)
    if not common_sizes:
        return format_benchmark_summary(benchmark_results.get(curr_version, {}))

    for size in common_sizes:
        p = prev_results[size]
        c = curr_results[size]
        p_time = p.get("mean_time_ms", 0.0) or 0.0
        c_time = c.get("mean_time_ms", 0.0) or 0.0
        p_tp = p.get("throughput_elements_per_second", 0.0) or 0.0
        c_tp = c.get("throughput_elements_per_second", 0.0) or 0.0
        time_improv = ((p_time - c_time) / p_time * 100.0) if p_time > 0 else 0.0
        tp_improv = ((c_tp - p_tp) / p_tp * 100.0) if p_tp > 0 else 0.0
        lines.append(
            f"  Size {size}: time {p_time:.4f}ms → {c_time:.4f}ms ({time_improv:+.1f}%), "
            f"throughput {p_tp/1e6:.1f}M/s → {c_tp/1e6:.1f}M/s ({tp_improv:+.1f}%)"
        )

    return "\n".join(lines)


def format_results(benchmark_results: Dict[str, Any]) -> str:
    """Format benchmark results for the report."""
    if not benchmark_results:
        return "No benchmark results available."
    
    formatted = ""
    for version, results in benchmark_results.items():
        if results.get("success") and "results" in results:
            formatted += f"\n### Version {version}\n"
            for size, result in results["results"].items():
                formatted += f"- Size {size}: {result['mean_time_ms']:.4f}ms\n"
    
    return formatted


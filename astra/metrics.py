"""Equal-case speedup aggregates; reject missing or invalid measurements."""
import math


def speedup_metrics(result, baseline, origin):
    if result.get("correctness") != "passed" or not result.get("complete"):
        return None
    ratios = []
    for case in result.get("cases", []):
        if case.get("origin", "dataset") != origin:
            continue
        try:
            actual = float(case["benchmark"]["median_ms"])
            reference = float(baseline[case["case_id"]]["median_ms"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) and value > 0 for value in (actual, reference)):
            return None
        ratio = reference / actual
        if not math.isfinite(ratio) or ratio <= 0:
            return None
        ratios.append(ratio)
    if not ratios:
        return None
    return dict(case_count=len(ratios), arithmetic=sum(ratios) / len(ratios),
                geometric=math.exp(sum(math.log(value) for value in ratios) / len(ratios)))

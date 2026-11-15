"""Shared dataset-driven multi-agent workflow for all native targets."""
import json
from pathlib import Path
import traceback
from .backends import Backend
from .execution import execute
from .generation import generate_candidate


def score(result):
    if not result.get("complete") or result.get("correctness") != "passed":
        return None
    values = [case["benchmark"]["median_ms"] for case in result["cases"]]
    return sum(values) / len(values) if values else None


def sources(folder, target):
    folder = Path(folder)
    kernel = folder / ("kernel.cu" if target == "cuda" else "kernel.cpp")
    wrapper = folder / "model.py"
    return dict(kernel=kernel.read_text() if kernel.exists() else "",
                wrapper=wrapper.read_text() if wrapper.exists() else "")


def run(task, reference, baseline, folder, settings, llm, on_update=None):
    from agents import Agent, Runner, function_tool
    folder = Path(folder)
    backend = Backend(settings.target, settings.compile_timeout)
    records, best, best_score, first_valid_score = [], None, None, None
    context = dict(task=task.metadata(), execution=reference.metadata)

    @function_tool
    def record_plan(plan: str) -> str:
        """Record the optimization plan for this fixed task and target."""
        (folder / "plan.txt").write_text(plan)
        return "Plan recorded"

    orchestrator = Agent(name="Orchestrator", model=llm.model, tools=[record_plan],
                         instructions="Plan this task's native kernel optimization. Use record_plan; prioritize correctness, target ISA and fixed-case performance. Do not change task semantics or cases.")
    planner = Runner.run_sync(orchestrator, json.dumps(context, default=str), max_turns=llm.max_turns)
    (folder / "orchestrator.txt").write_text(str(planner.final_output))
    strategist = Agent(name="OptimizationStrategist", model=llm.model,
                       instructions="Give concrete native-kernel optimization or repair suggestions, no code. Respect target ISA, dtype, thread count and task semantics. Analyze actual errors/timings, and distinguish measured facts from hypotheses. Never replace measured timings with estimates.")
    previous, feedback = None, None
    for index in range(1, settings.max_iterations + 2):
        version = f"v{index}"
        candidate_dir = folder / version
        candidate_dir.mkdir(exist_ok=True)
        record = dict(version=version, status="generating", correctness="not_checked", complete=False, cases=[])
        try:
            generate_candidate(task, reference.metadata, settings.target, candidate_dir, llm,
                               previous=previous, feedback=feedback, compiler=backend.compile_feedback)
            source = candidate_dir / ("kernel.cu" if settings.target == "cuda" else "kernel.cpp")
            backend.compile(source, candidate_dir)  # Compilation is required even if the LLM skipped its tool.
            verification = {}

            @function_tool
            def verify_candidate() -> str:
                """Run actual correctness checks on every fixed task case."""
                verification.update(execute(folder, candidate_dir, "verify", settings))
                return json.dumps(verification)

            tester = Agent(name="CorrectnessTester", model=llm.model, tools=[verify_candidate],
                           instructions="Call verify_candidate to check this fixed candidate. Do not generate new cases or claim results without the tool.")
            response = Runner.run_sync(tester, "Verify " + version, max_turns=llm.max_turns)
            (candidate_dir / "correctness_agent.txt").write_text(str(response.final_output))
            if not verification:
                verification.update(execute(folder, candidate_dir, "verify", settings))
            record.update(verification)
            if verification["correctness"] == "passed":
                measurement = {}

                @function_tool
                def benchmark_candidate() -> str:
                    """Measure this candidate on all fixed cases, rechecking correctness first."""
                    measurement.update(execute(folder, candidate_dir, "benchmark", settings))
                    return json.dumps(measurement)

                benchmarker = Agent(name="Benchmarker", model=llm.model, tools=[benchmark_candidate],
                                    instructions="Call benchmark_candidate to collect actual timings. Use the fixed task cases, never synthesize measurements.")
                response = Runner.run_sync(benchmarker, "Benchmark " + version, max_turns=llm.max_turns)
                (candidate_dir / "benchmark_agent.txt").write_text(str(response.final_output))
                if not measurement:
                    measurement.update(execute(folder, candidate_dir, "benchmark", settings))
                record.update(measurement)
            record["status"] = "measured" if record["complete"] else "invalid"
        except Exception:
            record["status"] = "error"
            record["error"] = traceback.format_exc()[-24000:]
        current_score = score(record)
        if index == 1:
            first_valid_score = current_score
        if current_score is not None and (best_score is None or current_score < best_score):
            best, best_score = version, current_score
        records.append(record)
        payload = dict(task=task.metadata(), target=settings.target, baseline=baseline,
                       versions=records, best_version=best, first_candidate_score=first_valid_score)
        temporary = folder / ".progress.tmp"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(folder / "progress.json")
        if on_update:
            on_update(payload)
        if index == settings.max_iterations + 1:
            break
        # Repair an invalid candidate; otherwise refine the fastest valid source.
        base_version = version if current_score is None else best
        previous = sources(folder / base_version, settings.target)
        strategy_input = dict(context, base_version=base_version, previous=previous,
                              latest_result=record, best_version=best, history=records,
                              baseline=baseline)
        (candidate_dir / "strategy_prompt.json").write_text(json.dumps(strategy_input, indent=2))
        try:
            response = Runner.run_sync(strategist, json.dumps(strategy_input, default=str), max_turns=llm.max_turns)
            feedback = str(response.final_output)
            (candidate_dir / "suggestions.txt").write_text(feedback)
        except Exception as error:
            feedback = dict(error=str(error), latest_result=record,
                            instruction="Repair the recorded failure or improve the best valid candidate")
            (candidate_dir / "strategy_error.txt").write_text(str(error))
    return payload

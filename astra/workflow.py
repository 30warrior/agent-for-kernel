"""Shared dataset-driven multi-agent workflow for all native targets."""
import json
from pathlib import Path
import traceback
from .backends import Backend
from .execution import execute
from .generation import generate_candidate
from .prompts import ORCHESTRATOR, TESTER, PROFILER, execution_context, public_metadata, strategy_instructions
from .test_cases import generate_suite


def score(result):
    if not result.get("complete") or result.get("correctness") != "passed":
        return None
    values = [case["benchmark"]["median_ms"] for case in result["cases"]
              if case.get("origin", "dataset") == "dataset"]
    return sum(values) / len(values) if values else None


def sources(folder, target):
    folder = Path(folder)
    kernel = folder / ("kernel.cu" if target == "cuda" else "kernel.cpp")
    wrapper = folder / "model.py"
    return dict(kernel=kernel.read_text(encoding="utf-8") if kernel.exists() else "",
                wrapper=wrapper.read_text(encoding="utf-8") if wrapper.exists() else "")


def run(task, reference, baseline, folder, settings, llm, on_update=None, backend=None, executor=None,
        suite_builder=None):
    from agents import Agent, Runner, function_tool
    folder = Path(folder)
    backend = backend or Backend(settings.target, settings.compile_timeout)
    executor = executor or execute
    records, best, best_score, first_valid_score = [], None, None, None

    def checked_execute(candidate_dir, mode):
        result = executor(folder, candidate_dir, mode, settings)
        labels = reference.metadata["case_labels"]
        if result.get("suite_id") != reference.metadata["suite_id"] or len(result.get("cases", [])) != len(labels):
            raise RuntimeError("Evaluation did not use the frozen test suite")
        for index, case in enumerate(result["cases"]):
            if case.get("case_id") != index or any(case.get(key) != labels[index][key] for key in ("name", "origin")):
                raise RuntimeError("Evaluation changed the fixed test case order or identity")
            elapsed = (case.get("benchmark") or {}).get("median_ms")
            if elapsed:
                case["reference_speedup"] = baseline[index]["median_ms"] / elapsed
        return result
    orchestrator = Agent(name="Orchestrator", model=llm.model, tools=[], instructions=ORCHESTRATOR)
    notifications = []

    def notify(event, **details):
        message = dict(event=event, **details)
        try:
            response = Runner.run_sync(orchestrator, json.dumps(message), max_turns=llm.max_turns)
            message["response"] = str(response.final_output)
        except Exception as error:
            message["error"] = str(error)
        notifications.append(message)
        (folder / "orchestrator_events.json").write_text(json.dumps(notifications, indent=2), encoding="utf-8")

    notify("workflow_initialized", target=settings.target, task_id=task.task_id)
    if not reference.metadata.get("suite_id"):
        if suite_builder is None:
            raise ValueError("A test suite materializer is required before generating candidates")
        suite = generate_suite(task, reference, folder, settings, llm, suite_builder)
        reference.metadata, baseline = suite["reference"], suite["baseline"]
    notify("test_suite_frozen", case_count=len(reference.metadata["cases"]),
           dataset_case_count=reference.metadata["dataset_case_count"])
    context = dict(task=public_metadata(task.metadata()), reference_source=task.source,
                   execution=execution_context(reference.metadata, settings))
    context["execution"]["baseline"] = baseline
    (folder / "role_instructions.json").write_text(json.dumps(dict(orchestrator=ORCHESTRATOR,
        testing=TESTER, profiling=PROFILER, planning=strategy_instructions(settings.target)), indent=2), encoding="utf-8")
    strategist = Agent(name="PlanningAgent", model=llm.model, instructions=strategy_instructions(settings.target))
    previous, feedback = None, None
    for index in range(1, settings.max_iterations + 2):
        version = f"v{index}"
        print("Generating and evaluating", version, flush=True)
        candidate_dir = folder / version
        candidate_dir.mkdir(exist_ok=True)
        record = dict(version=version, status="generating", correctness="not_checked", complete=False, cases=[])
        try:
            generate_candidate(task, context["execution"], settings.target, candidate_dir, llm,
                               previous=previous, feedback=feedback, compiler=backend.compile_feedback)
            source = candidate_dir / ("kernel.cu" if settings.target == "cuda" else "kernel.cpp")
            backend.compile(source, candidate_dir)  # Compilation is required even if the LLM skipped its tool.
            record["compiled"] = True
            verification = {}

            @function_tool
            def verify_candidate() -> str:
                """Run actual correctness checks on every fixed task case."""
                if not verification:
                    verification.update(checked_execute(candidate_dir, "verify"))
                return json.dumps(public_metadata(verification))

            tester = Agent(name="TestingAgent", model=llm.model, tools=[verify_candidate], instructions=TESTER)
            testing_input = dict(context, action="verify_current_version", version=version,
                                 candidate_sources=sources(candidate_dir, settings.target),
                                 next_step="Send pass/fail to PlanningAgent; send the fixed inputs to ProfilingAgent on pass")
            (candidate_dir / "testing_prompt.json").write_text(json.dumps(testing_input, indent=2), encoding="utf-8")
            try:
                response = Runner.run_sync(tester, json.dumps(testing_input), max_turns=llm.max_turns)
                (candidate_dir / "correctness_agent.txt").write_text(str(response.final_output), encoding="utf-8")
            except Exception as error:
                (candidate_dir / "correctness_agent_error.txt").write_text(str(error), encoding="utf-8")
            if not verification:
                verification.update(checked_execute(candidate_dir, "verify"))
            record.update(verification)
            notify("correctness_checked", version=version, correctness=verification["correctness"])
            if verification["correctness"] == "passed":
                measurement = {}

                @function_tool
                def benchmark_candidate() -> str:
                    """Measure this candidate on all fixed cases, rechecking correctness first."""
                    if not measurement:
                        measurement.update(checked_execute(candidate_dir, "benchmark"))
                    return json.dumps(public_metadata(measurement))

                benchmarker = Agent(name="ProfilingAgent", model=llm.model, tools=[benchmark_candidate], instructions=PROFILER)
                profiling_input = dict(context, action="profile_current_version", version=version,
                                       verification=public_metadata(verification), next_step="Send measured speedups to PlanningAgent")
                (candidate_dir / "profiling_prompt.json").write_text(json.dumps(profiling_input, indent=2), encoding="utf-8")
                try:
                    response = Runner.run_sync(benchmarker, json.dumps(profiling_input), max_turns=llm.max_turns)
                    (candidate_dir / "benchmark_agent.txt").write_text(str(response.final_output), encoding="utf-8")
                except Exception as error:
                    (candidate_dir / "benchmark_agent_error.txt").write_text(str(error), encoding="utf-8")
                if not measurement:
                    measurement.update(checked_execute(candidate_dir, "benchmark"))
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
                       test_suite={key: reference.metadata[key] for key in ("suite_id", "case_labels", "dataset_case_count")},
                       versions=records, best_version=best, first_candidate_score=first_valid_score,
                       configuration={key:value for key,value in vars(settings).items() if key not in ("api_key", "result_file")})
        temporary = folder / ".progress.tmp"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(folder / "progress.json")
        if on_update:
            on_update(payload)
        notify("version_evaluated", version=version, status=record["status"], best_version=best)
        if index == settings.max_iterations + 1:
            break
        # Repair an invalid candidate; otherwise refine the fastest valid source.
        base_version = version if current_score is None else best
        previous = sources(folder / base_version, settings.target)
        strategy_input = public_metadata(dict(context, base_version=base_version, previous=previous,
                              latest_result=record, best_version=best, history=records,
                              next_version=f"v{index + 1}"))
        (candidate_dir / "strategy_prompt.json").write_text(json.dumps(strategy_input, indent=2))
        try:
            response = Runner.run_sync(strategist, json.dumps(strategy_input, default=str), max_turns=llm.max_turns)
            suggestions = str(response.final_output)
            (candidate_dir / "suggestions.txt").write_text(suggestions, encoding="utf-8")
        except Exception as error:
            suggestions = "Repair the recorded failure or improve the best valid candidate"
            (candidate_dir / "strategy_error.txt").write_text(str(error), encoding="utf-8")
        feedback = public_metadata(dict(suggestions=suggestions, latest_result=record, base_version=base_version))
        notify("suggestions_ready", from_version=version, next_version=f"v{index + 1}")
    notify("workflow_completed", best_version=best)
    return payload

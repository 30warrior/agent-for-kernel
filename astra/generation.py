"""Task-to-native-source generation; the SDK is imported only on request."""
import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = None
    model: str = "o4-mini"
    model_api: str = "responses"
    tracing: bool = False
    max_turns: int = 12


def configure_llm(config):
    from openai import AsyncOpenAI
    from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled
    if not config.api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key before generation")
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url or None)
    set_default_openai_client(client, use_for_tracing=config.tracing)
    set_default_openai_api(config.model_api)
    set_tracing_disabled(not config.tracing)


TARGET_GUIDANCE = {
    "cuda": "Generate CUDA C++ (.cu), launch on PyTorch's current CUDA stream, check shapes/dtypes and handle tails.",
    "neon": "Generate AArch64 C++ with <arm_neon.h> NEON intrinsics. Handle fixed-width vector tails and unaligned inputs safely.",
    "sve": "Generate AArch64 C++ with <arm_sve.h> SVE intrinsics. Use predicates and runtime lane counts; do not assume a fixed vector length."
}

CONTRACT = """
Provide TWO complete sources through save_candidate(kernel_source, wrapper_source).
1. kernel_source: native C++/CUDA implementation, including
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m). No main() or build script.
2. wrapper_source: Python defining class ModelNew(torch.nn.Module) with
   __init__(self, reference, extension) and forward with the task's exact inputs
   and output structure. extension is the compiled PyBind module. Copy needed
   parameters/buffers from reference into independent storage; preserve values.
   Do not compile/import a generated extension yourself. The framework builds it.
All numerical computation must use your native extension; the Python wrapper may
manage arguments, allocate outputs and expose state, but must not call the reference
forward, PyTorch numerical operators or hidden baseline libraries to compute results.
Implement the complete task, including broadcasting, strides and required input
mutations. Never special-case test values or hard-code outputs. Use supported ISA
only. Single-threaded means no OpenMP, worker pools or external threaded libraries.
Do not relax math accuracy or dtype without satisfying the task tolerances.
"""


def generate_candidate(task, reference_metadata, target, folder, config, previous=None, feedback=None, compiler=None):
    from agents import Agent, Runner, function_tool
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    extension = ".cu" if target == "cuda" else ".cpp"
    kernel_path, wrapper_path = folder / ("kernel" + extension), folder / "model.py"

    @function_tool
    def save_candidate(kernel_source: str, wrapper_source: str) -> str:
        """Save complete native kernel and ModelNew sources for this fixed candidate."""
        if "PYBIND11_MODULE" not in kernel_source:
            return "ERROR: native source must export a PYBIND11_MODULE"
        try:
            tree = ast.parse(wrapper_source)
        except SyntaxError as error:
            return "ERROR: wrapper syntax: " + str(error)
        if not any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body):
            return "ERROR: wrapper must define ModelNew"
        kernel_path.write_text(kernel_source)
        wrapper_path.write_text(wrapper_source)
        return "Saved candidate sources"

    tools = [save_candidate]
    if compiler is not None:
        @function_tool
        def compile_candidate() -> str:
            """Compile the saved candidate and return actual compiler diagnostics."""
            if not kernel_path.exists() or not wrapper_path.exists():
                return "ERROR: save both sources first"
            return compiler(kernel_path, folder)
        tools.append(compile_candidate)
    agent = Agent(name="CodeGenerator", model=config.model,
                  instructions=TARGET_GUIDANCE[target] + CONTRACT +
                  " After saving, compile with compile_candidate if available and repair compiler errors.", tools=tools)
    prompt = dict(task=task.metadata(), reference_source=task.source,
                  execution=reference_metadata, previous_sources=previous, feedback=feedback)
    prompt_text = json.dumps(prompt, indent=2, default=str)
    (folder / "generation_prompt.json").write_text(prompt_text)
    result = Runner.run_sync(agent, prompt_text, max_turns=config.max_turns)
    (folder / "generation_response.txt").write_text(str(result.final_output))
    if not kernel_path.exists() or not wrapper_path.exists():
        raise RuntimeError("CodeGenerator did not save both required sources")
    return kernel_path, wrapper_path

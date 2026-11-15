"""Load the small, explicit local API configuration without shell evaluation."""
import os
from pathlib import Path


def load_environment(path):
    path = Path(path)
    if not path.is_file():
        return
    allowed = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if key not in allowed:
            continue
        if not separator:
            raise ValueError(f"Invalid API configuration at line {number}")
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed API configuration quote at line {number}")
            value = value[1:-1]
        os.environ.setdefault(key, value)

# Repository Guidelines

## Project Structure & Module Organization
`token_prepending/` is the main project directory. Key files include `evaluate.py` for model loading and evaluation, `run.sh` as the entry script, `config.yaml` for model presets, and `requirements.txt` for Python dependencies. The `senllm/` package contains model implementations, while `SentEval/` is vendored evaluation code and should be treated as third-party source. Research artifacts and notes live under `docs/`, and notebooks such as `kaggle.ipynb` are kept alongside the experiment code.

## Build, Test, and Development Commands
- Do not run `pip install -r requirements.txt` on the local machine.
- Do not run `bash run.sh ...` on the local machine.
- Use those commands only on a GPU-enabled server or other approved environment.
- On a server, `bash run.sh llama-2-7b` runs the default preset, and `python evaluate.py --config llama-2-7b --config_file config.yaml` is useful for direct debugging.

## Coding Style & Naming Conventions
Use standard Python style with 4-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for constants. Keep YAML config keys lowercase and use hyphenated model preset names such as `llama-2-7b-tp`. Prefer small, explicit helper functions over large inline blocks. No formatter or linter is enforced in the repo, so match the surrounding style.

## Testing Guidelines
There is no dedicated automated test suite in the repository. Validate changes only on a GPU-capable server by running the relevant evaluation command and confirming that the selected model config loads, GPU settings are applied, and the expected score table is printed. Do not use the local machine for installs, smoke tests, or evaluation runs.

## Commit & Pull Request Guidelines
Recent commits use short, imperative prefixes such as `feat:` and `docs:` followed by a concise summary. Keep commit messages focused on one change. Pull requests should describe the model or configuration impact, list the exact command used to verify the change, and include screenshots or logs only when they clarify a result.

## Configuration & Data Notes
Keep local model paths and GPU IDs in `config.yaml` or override them at runtime; do not hard-code machine-specific paths in source files. Large checkpoints, downloaded datasets, and generated results should stay out of version control.

# Contributing

Thanks for looking. The most useful contributions right now are, in order:

1. **Running a real model adapter and reporting what happened.** Only the ONNX Kokoro voice has been run for real; SDXL, FLUX, Wan and the Whisper adapters have not. Run `python scripts/e2e.py` and attach `e2e-runs/report.md`. A bug report with `shortgen doctor --json` output, or a pull request that replaces an estimate in `docs/hardware.md` with a measurement, is worth more than a new feature.
2. Bug fixes with a test.
3. New adapters (see [docs/adding-a-model.md](docs/adding-a-model.md)).

## Setup

```bash
git clone https://github.com/OWNER/shortgen && cd shortgen
python3.12 -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev,mcp]"
make check            # ruff, mypy --strict, pytest
make smoke            # the whole pipeline with fake models and real ffmpeg
```

The test suite needs no GPU and downloads nothing. Tests marked `ffmpeg` are skipped if ffmpeg is not installed.

## What a pull request needs

- `make check` passes: zero ruff warnings, mypy strict clean, all tests green.
- New pure logic has unit tests. New pipeline behaviour has a test in `tests/integration/` using the fake providers.
- No new hard-coded model names, paths or tuning numbers outside `configs/`.
- No heavy import (`torch`, `diffusers`, ...) at module top level. They belong inside adapter methods.
- Public classes and functions have a docstring that says why they exist. Methods that implement an interface (`load`, `generate`, `units`, `execute`, ...) inherit the interface's docstring and only need their own when they do something a reader would not expect.
- If you could not run something against a real model, say so in the docstring and the pull request. Unverified is fine; unverified and unlabelled is not.

## Architecture rules

These are what keep the project small. Please do not bend them without discussing it first.

- `domain/` imports nothing from the rest of the package except `errors`.
- Stages depend on `providers/interfaces.py`, never on anything in `adapters/`.
- Subprocesses are started only in `process.py`, from argument lists.
- A stage writes only to the temp paths it is given.
- No global mutable state. Pass dependencies through `PipelineContext` or `ProviderContext`.

[docs/architecture.md](docs/architecture.md) has the full picture.

## Adding a model

Follow [docs/adding-a-model.md](docs/adding-a-model.md). Every registry entry must record the weights' license and whether commercial use is allowed, and the model must be added to [docs/licenses.md](docs/licenses.md). Models with non-commercial licenses are welcome as opt-in entries with `commercial_use: false`, never as defaults.

## Commits

Small, focused commits with [Conventional Commits](https://www.conventionalcommits.org/) prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`. A scope is welcome, for example `feat(adapters): ...`.

## Scope

shortgen does one thing: script in, narrated captioned short video out, locally and for free. Features that need a paid or hosted service can be added as adapters that are off by default. Ideas outside that scope go on the roadmap in the README first.

## License

By contributing you agree that your contribution is licensed under Apache-2.0, the license of this project.

# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `generate`, `plan`, `render` and `status` work on a folder that only has a `script.md`: it is set up in place, as `run` already did. Before, they stopped with "no project".
- `generate`, `plan`, `render` and `status` now see an edited `script.md` in the project folder; before, only `run` did, so `generate --stage image` after changing a picture description drew the old description. A plan that is out of date is brought up to date first even when the chosen stages leave it out.
- The final render no longer needs an ffmpeg built with libass. Captions are rasterised with Pillow and composited with the `overlay` filter, so Homebrew's default ffmpeg works. Previously the pipeline failed at the last stage on a stock Mac.

### Added

- `kokoro_onnx` speech provider (now the default): Kokoro-82M on ONNX Runtime, no PyTorch, with measured word timings. Weights download from the kokoro-onnx GitHub release and are verified against pinned SHA-256 hashes.
- `folder` image provider: bring your own pictures, one per scene.
- Scripts can describe the picture for each scene with `<!-- image: ... -->` (also `motion:` and `camera:`). Without it the image model was prompted with the narration sentence, which describes a story beat, not a picture. All examples now carry a description for every scene.
- Five example projects in different short-form formats, with original art for the cartoon one.
- `stories/`: nine scripts written to be posted as Shorts, three of them for young children, with upload text, fact sources and notes on YouTube's rules.
- A test that every shipped script plans, describes every picture and keeps its prompts within the image model's limit.
- `scripts/e2e.py`: renders every example, times each stage and verifies the output.
- A folder containing `script.md` becomes a project in place; editing a project's `script.md` is picked up on the next run.
- Optional adapters for Higgsfield, a hosted, paid service: `higgsfield_image` and `higgsfield_video`, selected with `configs/higgsfield.yaml` and installed with the `hosted` extra. Off by default, flagged by `doctor`, and never resubmit a job that was already accepted. **Not run against the live service**; see `docs/hosted.md`.
- Registry entries can declare `hosted: true`, and `commercial_use: null` when a service's terms decide.
- Captions break at clauses and avoid dangling function words. Caption font lookup with fallbacks and a `font_path` override; `doctor` reports the font in use.

### Verified

- Four complete videos have now been made on the 16 GB Mac (`octopus-facts`, `one-light-day`, `pip-the-owl`, `moon-night-off`), about 50 pictures in all. At 24 steps a picture takes 3.8 minutes on average.
- First complete real run of the default pipeline on a 16 GB Apple Silicon Mac: Kokoro voice, SDXL pictures, Ken Burns animation, burned-in captions and the final render. SDXL took about 4.5 minutes per 768x1344 picture, slower than the earlier estimate; `docs/hardware.md` has the numbers.

### Changed

- The README is rebuilt around real output: a banner, a demo GIF and a gallery, all drawn by the default models on the 16 GB Mac, plus a section on what the image model gets wrong.
- The `octopus-facts` example's picture descriptions and style were rewritten after the first real run: the key detail first, one subject, no words that invite fake text.

- Default transcriber is `proportional`; it is only used when the speech model returns no timings.
- Pillow is a core dependency. The `speech` extra installs `kokoro-onnx`; the PyTorch build moved to `speech-torch`.
- Placeholder images are labelled with their prompt.

## [0.1.0] - 2026-11-11

First public version. Alpha: the pipeline, renderer, CLI and MCP server are tested; the real-model adapters are not yet verified against real weights.

### Added

- Pure-Python domain: data contracts, deterministic scene splitting, timing, word alignment onto script words, subtitle layout (SRT and ASS).
- Layered configuration (`configs/default.yaml`, project file, `--config`, `SHORTGEN__*` environment variables) validated with Pydantic.
- Provider interfaces for planner, speech, transcriber, image and video, and a YAML model registry that records each model's ID and license.
- Resumable pipeline: typed stage inputs hashed for caching, a single atomic JSON manifest, lazy model loading with one model in memory at a time, bounded retry for transient errors, per-scene failure isolation, safe interruption.
- Stages: plan, speech, align, image, video, subtitles, compose, render.
- ffmpeg rendering through argument-list command builders: Ken Burns animation, frame-exact scene segments, concatenation, burned-in captions, 9:16 and 16:9.
- Adapters: Kokoro-82M speech, faster-whisper and mlx-whisper transcribers, Stable Diffusion XL (with IP-Adapter reference images) and FLUX.1-schnell images, Wan 2.2 TI2V-5B image-to-video, Ollama planner, rule-based planner, proportional aligner, Ken Burns animator.
- Fake providers and `configs/smoke.yaml` for running the whole pipeline without models.
- CLI: `init`, `plan`, `generate`, `render`, `run`, `status`, `doctor`.
- Optional MCP server (`shortgen-mcp`) exposing the same operations as tools.
- Documentation: architecture, adding a model, licenses, hardware expectations.
- CI: ruff, mypy strict, pytest and a smoke run on Linux; tests on macOS.

### Known limitations

- Kokoro, Whisper, SDXL, FLUX and Wan adapters have not been run against real weights or on real hardware.
- Nothing has been run on macOS or on a GPU; all render times in `docs/hardware.md` are estimates except the ffmpeg-only measurements.
- The Ollama planner is tested against a stub of Ollama's API, not a real server.
- One shot per scene; no background music; no word-by-word caption highlighting.

[Unreleased]: https://github.com/OWNER/shortgen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/shortgen/releases/tag/v0.1.0

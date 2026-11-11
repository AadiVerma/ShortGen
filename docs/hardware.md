# Hardware and expected render times

**Read this first.** Most numbers on this page are **estimates**. The default pipeline has been measured once on a 16 GB Apple Silicon Mac and on a small CPU-only Linux container; those measurements are in the last two sections. Nothing has been run on an NVIDIA GPU, and FLUX and Wan have not been run anywhere. Estimates are reasoned from model size, the publishers' own statements and typical community reports, and they may be wrong by a factor of two or more in either direction. The first estimate that was checked against reality was wrong: SDXL on the Mac was guessed at 1 to 3 minutes per image and took about 4.5.

When you run it on real hardware, please open a pull request replacing an estimate with your measurement and the machine it came from.

## What fits where

| Model (registry name) | Role | Apple Silicon, 16 GB | NVIDIA, 12 GB | NVIDIA, 24 GB | CPU only |
|---|---|---|---|---|---|
| Kokoro-82M on ONNX Runtime (`kokoro_onnx`, default) | speech | Yes, on CPU by design | Yes | Yes | Yes (measured below) |
| Kokoro-82M on PyTorch (`kokoro`) | speech | Expected yes, on CPU | Expected yes | Expected yes | Expected yes |
| faster-whisper small (`faster_whisper`) | align | Yes, on CPU (CTranslate2 has no MPS backend) | Yes | Yes | Yes |
| Whisper large-v3-turbo on MLX (`mlx_whisper`) | align | Expected yes | n/a | n/a | n/a |
| SDXL base 1.0, half precision (`sdxl`) | image | **Yes (measured below)**, tight; close other apps | Expected yes | Expected yes | Works, very slow |
| FLUX.1-schnell (`flux_schnell`) | image | **Not expected to fit** at full precision | Not expected to fit | Expected yes with CPU offload | No |
| Ken Burns via ffmpeg (`kenburns`) | video | Yes (no model) | Yes | Yes | Yes |
| Wan 2.2 TI2V-5B (`wan`) | video | **Not expected to fit** | Not expected to fit | The model card states 24 GB is enough | No |
| qwen2.5:7b via Ollama (`ollama`) | plan | Expected yes (about 5 GB while planning) | Yes | Yes | Yes, slowly |

"Expected" means reasoned, not observed. `shortgen doctor` compares each selected model's `min_memory_gb` from the registry against your machine and warns when it looks too large.

Peak memory is one model at a time: the runner loads a model for its stage and unloads it before the next stage starts.

## Estimated time per stage

For a typical 60-second short: about 150 words, 8 to 12 scenes, 1080x1920.

| Stage | MacBook Air M4, 16 GB (estimate) | NVIDIA 24 GB, e.g. RTX 4090 (estimate) | Basis for the estimate |
|---|---|---|---|
| `plan` (rule-based) | under 1 second | under 1 second | Pure Python; measured below |
| `plan` (Ollama, qwen2.5:7b) | 20 to 90 seconds | 5 to 20 seconds | One or two short LLM calls; first call includes loading the model |
| `speech` (Kokoro, ONNX) | 1 to 3 seconds per scene | about 1 second per scene | Measured at 0.7x real time on 2 slow CPU cores (below); an M4 should beat that |
| `align` | under 1 second per scene with Kokoro's own timestamps; 2 to 10 seconds per scene if Whisper is needed | same or faster | No model is loaded in the first case |
| `image` (SDXL, 768x1344) | **about 4.4 minutes per image at 30 steps, 3.8 at 24 (measured, see below)** | 5 to 15 seconds per image | Mac: one real run. NVIDIA: typical reports for high-end cards |
| `video` (Ken Burns) | a few seconds per scene | a few seconds per scene | CPU-bound ffmpeg; see measured numbers below |
| `video` (Wan 2.2 TI2V-5B, about 5 s clip at 720p) | not expected to run | **several minutes per clip**; the publisher states a 5-second 720p clip takes under 9 minutes on a consumer GPU | Publisher's statement, not ours |
| `subtitles` | under 1 second | under 1 second | Pure Python; measured below |
| `compose` | a few seconds per scene | a few seconds per scene | CPU-bound ffmpeg |
| `render` | 10 to 40 seconds | 10 to 40 seconds | CPU-bound libx264 encode of about 60 s of 1080x1920 video |

Estimated total for the default configuration (Kokoro, SDXL, Ken Burns):

- **MacBook Air M4, 16 GB: roughly 30 to 55 minutes** (3.8 to 4.4 minutes per scene, measured), almost all of it image generation.
- **NVIDIA 24 GB: roughly 3 to 6 minutes.**
- **With Wan on an NVIDIA 24 GB card: add about 1 to 1.5 hours** for ten clips, on the publisher's figure.

The first run on any machine is longer because the weights (several GB) have to download.

## Clip length versus narration length

Image-to-video models produce a fixed, short clip (Wan is capped here at 121 frames, about 5 seconds at 24 fps). A scene with more narration than that gets the longest clip the model can make; the `compose` stage then slows it down by at most `render.max_slowdown` (default 2x) and holds the last frame for whatever time remains. Keep `planning.max_scene_words` low enough that scenes stay near 5 seconds if you use Wan. The Ken Burns animator has no such limit and always renders exactly the scene's length.

## Making it fit, or go faster

- Fewer image steps: `provider_options: {sdxl: {steps: 20}}`.
- A smaller image budget: `provider_options: {sdxl: {max_area: 589824}}` (768x768 worth of pixels).
- On a low-VRAM NVIDIA card: `provider_options: {sdxl: {cpu_offload: true}}`.
- On a Mac, close browsers and other memory-heavy apps before the image stage. Unified memory is shared with the GPU.
- Render at 720x1280 by overriding `render.resolutions`.
- Nothing is lost by stopping: Ctrl+C, then re-run later. Finished scenes are not redone.

## CPU fallback

Without CUDA or MPS, shortgen prints a warning and runs on the CPU. Speech, alignment, the Ken Burns animator and all rendering are fine there. SDXL on a CPU should be expected to take many minutes per image.

## Measured on a 16 GB Apple Silicon Mac

One real run of the default pipeline, `shortgen run examples/octopus-facts`: 7 scenes, a 27.0-second 1080x1920 video at 30 fps. The machine is the project's target, reported by its owner as a MacBook Air M4 with 16 GB, with Homebrew ffmpeg; about 4.5 GB of memory was free before the run because other apps were open. Times are from the project manifest. The run's log was not kept, so the device (MPS) is inferred from the speed and peak memory was not recorded.

| Stage | Time | Notes |
|---|---|---|
| `speech` (Kokoro ONNX) | about 1 second per scene | The first scene took 150 s, which was the one-time 350 MB weight download and model load |
| `image` (SDXL, 30 steps, 768x1344) | **3.6 to 5.3 minutes per image, 4.4 on average** | About 7.8 seconds per step. The first image took 42 minutes because the weights (about 7 GB) downloaded first |
| `video` (Ken Burns) | 0.6 seconds per scene | |
| `compose` | 0.7 seconds per scene | |
| `render` | 11 seconds | For the whole 27-second video |

What that says:

- Image generation is **over 99% of the time**: about 31 minutes for 7 pictures, against 20 seconds for everything after it.
- A fanless laptop gets hot and stays hot for the whole image stage. It slows itself down as it warms, which is probably why the per-image time varied.
- With little free memory the model competes with other apps. Closing them should help; that has not been measured.
- Picture quality from that run, judged by eye: consistent style and attractive single subjects, but the model often ignored the specific detail a scene asked for, and a style containing the words "infographic poster" produced fake text. The example's prompts were rewritten afterwards.

### Later runs at 24 steps

Three more stories were drawn on the same Mac with `provider_options: {sdxl: {steps: 24}}`. Across 20 pictures from two of them (`pip-the-owl` and `moon-night-off`), an image took **2.6 to 7.7 minutes, 3.8 on average and 3.3 at the median**. The slow ones were all in one run and are unexplained; the laptop may have been in use, or warm. A ten-scene short therefore takes about 40 minutes of drawing, and each picture you reword costs another three or four.

Counting every redraw, about 50 pictures have been drawn on that machine without a crash or an out-of-memory error.

## Measured on a CPU-only Linux container

These come from `python scripts/e2e.py --images demo`, which renders the five example projects with the real voice model and the real renderer, using bundled art or labelled placeholders instead of an image model. **No image or video model time is included.**

**Machine:** Linux x86-64 container, 2 virtual CPUs, 8 GB RAM, no GPU, ffmpeg 6.1, Python 3.11. Output at full resolution (1080x1920 or 1920x1080), 30 fps.

| Video | Shape | Scenes | Video length | Total | speech (Kokoro ONNX) | video (Ken Burns) | compose | render |
|---|---|---|---|---|---|---|---|---|
| midnight-knock | 9:16 | 7 | 33.9 s | 87 s | 20.5 s | 17.0 s | 15.4 s | 33.8 s |
| mona-lisa-heist | 16:9 | 10 | 50.7 s | 159 s | 35.3 s | 29.5 s | 27.5 s | 66.3 s |
| octopus-facts | 9:16 | 7 | 27.0 s | 71 s | 19.3 s | 13.1 s | 11.9 s | 26.6 s |
| odyssey-cartoon | 16:9 | 11 | 54.4 s | 161 s | 37.0 s | 33.0 s | 29.2 s | 60.9 s |
| stoic-morning | 9:16 | 7 | 29.2 s | 78 s | 18.9 s | 14.2 s | 12.8 s | 32.0 s |

What that says:

- Everything except image generation takes roughly **3x the length of the video** on two slow CPU cores.
- Kokoro on ONNX Runtime synthesises speech at about **0.7x real time** there (a 54 s narration in 37 s, including loading the model), and returned measured word timings for all 45 scenes.
- Peak memory was **under 0.7 GB** for the Python process and under 0.7 GB for ffmpeg.
- `plan`, `align`, `subtitles` and a full no-op re-run each take well under a second.

The same test suite, including the real-ffmpeg tests, also passes on Linux arm64 with ffmpeg 4.4.

Run `python scripts/e2e.py` on your machine to get your own table, image generation included.

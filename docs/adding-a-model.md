# Adding a model

Adding a model takes two things: one adapter module and one registry entry. The pipeline, the runner and the CLI do not change.

## 1. Pick the interface

All five live in `src/shortgen/providers/interfaces.py`.

| Interface | Method to implement | Must do |
|---|---|---|
| `ScriptPlanner` | `write_script(topic, target_words, language)`, `plan(request)` | Return scenes whose narration is a verbatim slice of the script |
| `SpeechSynthesizer` | `synthesize(request) -> SpeechResult` | Write 16-bit PCM WAV to `request.output_path`; return word timings or an empty list |
| `Transcriber` | `transcribe(audio_path, language)` | Return the words heard, with start and end times in seconds |
| `ImageGenerator` | `generate(request)` | Write a PNG to `request.output_path` |
| `VideoGenerator` | `generate(request)` | Write an MP4 to `request.output_path` |

## 2. Write the adapter

Create `src/shortgen/adapters/my_image.py` (or put it in your own package; any importable path works).

```python
"""MyModel image generator (license: ...)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from shortgen.adapters._torch import place_pipeline, release_memory, require, seeded_generator
from shortgen.domain.models import FrozenModel
from shortgen.providers.interfaces import ImageGenerator, ImageRequest
from shortgen.render.layout import fit_area


class MyImageOptions(FrozenModel):
    steps: int = Field(ge=1)
    max_area: int = Field(gt=0)
    size_multiple: int = Field(gt=0)
    cpu_offload: bool


class MyImage(ImageGenerator):
    _pipeline: Any = None

    def load(self) -> None:
        options = self.parse_options(MyImageOptions)
        torch = require("torch", self.spec.install_hint)
        diffusers = require("diffusers", self.spec.install_hint)
        pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
            self.spec.model_id, torch_dtype=torch.float16
        )
        place_pipeline(pipeline, self.context.device.kind, options.cpu_offload)
        self._pipeline = pipeline

    def unload(self) -> None:
        self._pipeline = None
        release_memory()

    def generate(self, request: ImageRequest) -> None:
        options = self.parse_options(MyImageOptions)
        torch = require("torch", self.spec.install_hint)
        size = fit_area(request.target, options.max_area, options.size_multiple)
        image = self._pipeline(
            prompt=request.prompt,
            width=size.width,
            height=size.height,
            num_inference_steps=options.steps,
            generator=seeded_generator(torch, request.seed),
        ).images[0]
        image.save(request.output_path, format="PNG")
```

The rules an adapter has to follow:

- **Import heavy libraries inside methods**, through `require()`, never at module top level. Importing your module must work on a machine without torch, so that `shortgen doctor` can report what is missing instead of crashing.
- **`__init__` must stay cheap.** Load weights in `load()`. The runner calls it lazily, once, before the first request of a stage.
- **`unload()` must drop every reference to the model** and call `release_memory()`. The next stage's model needs that memory.
- **Write only to `request.output_path`.** It is a temp path; the runner renames it after your method returns. Do not write anything else into the project directory.
- **Use `self.context.device.kind`** (`cuda`, `mps` or `cpu`) for device selection. Do not detect devices yourself.
- **Use the request's `seed`** so reruns are reproducible.
- **Take every number from options.** Declare a Pydantic options model with required fields and call `self.parse_options(...)`. The values live in the registry entry.
- **Raise the right error.** `ProviderUnavailableError` if a dependency or service is missing (stops the stage at once), `TransientError` for failures worth retrying (timeouts), `ProviderError` for anything else.
- **Do not guess API signatures.** If you could not run something, say so in the module docstring, the way the shipped adapters do.

## 3. Register it

Add an entry to `configs/models.yaml`, or to your own file referenced by `runtime.models_file` (entries there are merged over the shipped registry, so you can also override a single option of an existing model).

```yaml
models:
  my_image:
    kind: image
    adapter: shortgen.adapters.my_image:MyImage
    model_id: some-org/some-model
    license: Apache-2.0            # the license of the WEIGHTS, read from the model card
    commercial_use: true           # be conservative: false if the license restricts it
    source: https://huggingface.co/some-org/some-model
    weights_host: huggingface
    requires: [torch, diffusers]   # module names `doctor` checks for
    install_hint: "pip install 'shortgen[image]'"
    devices: [cuda, mps]           # where you have reason to believe it works
    min_memory_gb: 12
    notes: Anything a user should know before choosing it.
    options:
      steps: 20
      max_area: 1048576
      size_multiple: 64
      cpu_offload: false
```

The license fields are required. A model with a non-commercial license must set `commercial_use: false`; `shortgen doctor` then warns whenever it is selected. Add the model to the table in `docs/licenses.md` as well: a test fails if a registered model is missing from it.

An adapter for a hosted service must set `hosted: true` (so `doctor` warns that requests leave the machine) and may set `commercial_use: null` when the service's terms, rather than a model license, decide. It must never be made a default; a test enforces that. Read credentials from `self.context.env`, not from a config file, and see `adapters/_higgsfield.py` for how to avoid paying twice for a job when a run is retried or interrupted.

## 4. Select it

```yaml
# <project>/config.yaml
providers:
  image: my_image
```

or `SHORTGEN__PROVIDERS__IMAGE=my_image shortgen run story.md`.

## 5. Test it

- `tests/unit/test_registry.py` already checks that every shipped entry has a license and that its adapter imports and subclasses the right interface.
- Add a test in `tests/unit/test_adapters.py` that installs a stand-in for the library and asserts on how your adapter calls it. Look at the SDXL tests for the pattern. Such a test checks your glue code, not the model.
- Run the real model once and record what you measured in `docs/hardware.md`, with the machine it was measured on.

```bash
make check                     # ruff, mypy --strict, pytest
shortgen doctor                # should list your model and its dependencies
shortgen generate projects/demo --stage image --scene s001 --force
```

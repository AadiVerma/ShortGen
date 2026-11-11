# Licenses

shortgen's **code** is licensed under Apache-2.0 (see [LICENSE](../LICENSE)).

The **models** it can drive are separate works. Their weights are not in this repository; they are downloaded from their publishers, under their publishers' terms, the first time you use them. The Apache-2.0 license of this code does not apply to them, and their licenses govern what you may do with the weights and, in some cases, with the output.

The source of truth for the table below is [`configs/models.yaml`](../configs/models.yaml); a test keeps the two in sync. The license of each model was taken from its model card when the entry was written. **Licenses change and this is not legal advice. Follow the link and read the license yourself before any commercial use.**

## Models

| Registry name | Role | Model | License of the weights | Commercial use | Default |
|---|---|---|---|---|---|
| `kokoro_onnx` | speech | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), ONNX export from [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (MIT code) | Apache-2.0 | Yes | Yes |
| `kokoro` | speech | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), PyTorch | Apache-2.0 | Yes | No |
| `faster_whisper` | transcriber | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | MIT | Yes | No |
| `mlx_whisper` | transcriber | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT (OpenAI Whisper's license; the MLX conversion's card declares none) | Yes | No |
| `sdxl` | image | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | CreativeML Open RAIL++-M | Yes, with use restrictions | Yes |
| `flux_schnell` | image | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Apache-2.0 | Yes | No |
| `wan` | video | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Apache-2.0 | Yes | No |
| `ollama` | planner | [qwen2.5:7b](https://ollama.com/library/qwen2.5) (via Ollama) | Apache-2.0 | Yes | No |

### Hosted services (optional, off by default)

These two entries are not models you download. They send each scene to a hosted, paid service, and that service's terms and pricing apply instead of a model license. shortgen has not reviewed those terms and makes no claim about commercial use. See [hosted.md](hosted.md).

| Registry name | Role | Service and model | Terms | Commercial use | Default |
|---|---|---|---|---|---|
| `higgsfield_image` | image | [Higgsfield](https://cloud.higgsfield.ai), hosted and paid: `bytedance/seedream/v4/text-to-image` | Higgsfield terms of service (hosted, paid) | Not determined: read Higgsfield's terms for your plan | No |
| `higgsfield_video` | video | [Higgsfield](https://cloud.higgsfield.ai), hosted and paid: `higgsfield-ai/dop/standard` | Higgsfield terms of service (hosted, paid) | Not determined: read Higgsfield's terms for your plan | No |

The models behind a hosted service have their own owners and terms as well; the model ids above are examples from Higgsfield's documentation and were not checked against a live account.

### Auxiliary weights

| Used by | Weights | Purpose | License | Commercial use |
|---|---|---|---|---|
| `sdxl` | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | VAE that stays stable in half precision | MIT | Yes |
| `sdxl` | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) | Reference-image conditioning; downloaded only when a scene has a reference image | Apache-2.0 | Yes |

### Entries with no model weights

These adapters contain no learned weights, so no model license applies. They are covered by the code's Apache-2.0 license.

| Registry name | Role | What it is |
|---|---|---|
| `heuristic` | planner | Rule-based scene splitting (default) |
| `kenburns` | video | Zoom and pan over the still image, rendered by ffmpeg (default) |
| `proportional` | transcriber | Estimates word timings from word length, no recognition (default; only used when the speech model returns no timings) |
| `folder` | image | Uses pictures you supply, one per scene. You are responsible for the rights to them |
| `fake_planner`, `fake_speech`, `fake_image`, `fake_video` | testing | Placeholder generators for tests and the smoke run |

## Non-commercial models: flagged, and not registered

No model with a non-commercial license is enabled by default, and none is currently registered at all. The ones you are most likely to reach for:

| Model | License | Why it matters |
|---|---|---|
| FLUX.1-dev | FLUX.1 [dev] Non-Commercial License | Works with the same adapter as `flux_schnell`, but **commercial use is not permitted** without a separate license from Black Forest Labs. If you register it yourself, set `commercial_use: false`. |
| SDXL-Turbo, SD-Turbo | Stability AI non-commercial / community license terms | **Check the current terms**; they have changed over time and may restrict commercial use. |
| Stable Video Diffusion | Stability AI community license terms | **Check the current terms** before commercial use. |

If you add such a model to your own registry file, set `commercial_use: false`. `shortgen doctor` then shows a warning whenever it is selected.

## A note on the SDXL license

CreativeML Open RAIL++-M permits commercial use of the model and its outputs but attaches use-based restrictions (for example on generating content that is illegal or harmful). It is not an OSI-approved open-source license. Read it before publishing.

## Other software

- **ffmpeg** is called as an external program and is not distributed with shortgen. Its license (LGPL or GPL) depends on how your copy was built.
- **Ollama** is a separate program you install yourself (MIT). Each model you pull through it has its own license; `qwen2.5:7b` is the only one this registry records.
- **higgsfield-client** (Apache-2.0) is only installed with the optional `hosted` extra.
- **Python dependencies** (PyTorch, diffusers, kokoro, faster-whisper, mlx-whisper, Typer, Pydantic and others) are installed from PyPI under their own licenses.

## Voices and likeness

Kokoro's bundled voices are synthetic. shortgen does not do voice cloning. If you add a model that can clone a voice, only use voices you have the right to use.

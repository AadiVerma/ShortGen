# Launch thread (X)

Draft, 8 posts. Every claim below matches what the code does today. Before posting: attach a real demo clip to post 1, and re-read post 7 against the current state of the adapters. If you have run them on real hardware by then, update it with what you measured.

---

**1/**

I built shortgen: an open-source pipeline that turns a script into a narrated, captioned vertical video with one command.

`shortgen run story.md`

It runs locally and for free. No paid API anywhere.

It's alpha. Here's how it's built, and what is and isn't tested yet. 🧵

---

**2/**

The models are the least interesting part.

Generating video on a laptop is slow. If scene 9 of 12 crashes, I don't want to redo scenes 1 to 8. If I fix one sentence, I want one scene regenerated, not the whole video.

So the design started from resumability, not from a model.

---

**3/**

How resume works:

Every unit of work (one stage, one scene) has typed inputs. They get hashed. Upstream files are part of the inputs, as content hashes.

Same hash and the output file still exists: skip.

Redraw one image and only its clip, its segment and the final render rerun.

---

**4/**

One JSON manifest per project is the source of truth. No database.

It's written atomically before and after every unit. Stages write to temp files that are renamed only on success.

Ctrl+C at any point leaves a valid project. Run the same command again and it continues.

---

**5/**

The pipeline never imports a model.

Stages depend on 5 small interfaces: planner, speech, transcriber, image, video. A YAML registry maps a name to an adapter class and records the model's license.

Adding a model = one adapter file + one registry entry. The pipeline doesn't change.

---

**6/**

Built for a 16 GB MacBook, so memory is a rule, not a hope:

One model in memory at a time. Each loads lazily on first use and is unloaded when its stage ends. A stage with nothing to do never loads its model.

There's a test that fails if two models are ever loaded together.

---

**7/**

What it doesn't do, honestly:

On 16 GB the default "animation" is a Ken Burns zoom/pan rendered by ffmpeg. Real image-to-video (Wan 2.2) is one config line, but its model card asks for a 24 GB GPU.

And it is slow on a laptop: about four and a half minutes per picture on my 16 GB Mac, so half an hour for a short. The FLUX and Wan adapters are written but not yet run on real hardware.

---

**8/**

Defaults: Kokoro (on ONNX, no PyTorch) for speech, SDXL for images, ffmpeg for everything else. All allow commercial use; every model's license is listed in the repo.

Apache-2.0 code. Python, Pydantic, Typer, mypy strict. There's an MCP server so an AI assistant can drive it.

If you try it on your machine, tell me what breaks.

github.com/AadiVerma/ShortGen

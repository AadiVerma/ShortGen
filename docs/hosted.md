# Optional: pictures and animation from Higgsfield (hosted, paid)

shortgen is local and free by default, and that does not change. This page describes the one exception you can opt into: two adapters that have Higgsfield draw the picture for each scene and animate it. They are **off by default**, they are **not free**, and they **have not been verified** against the live service (see [What is not verified](#what-is-not-verified)).

Reach for them only if the local options are not good enough for what you want. On a 16 GB laptop the local pipeline gives you Stable Diffusion XL stills with a zoom or pan. It does not give you consistent characters that actually move. Hosted models do, and you pay for each one.

## What leaves your machine, and what it costs

With these adapters selected, each scene's picture description (including your style and character descriptions) is sent to `api.higgsfield.ai`. For animation, the generated picture is uploaded as well. Reference images are uploaded only if you turn that on. Your script's narration text is not sent as narration, but picture descriptions you did not write yourself default to the narration sentence, so assume the text of your script is sent. The voice, the captions and the final render still run locally.

Every image and every clip spends credits on your Higgsfield account. shortgen does not know the prices and cannot show you a total. Count the jobs before you start: a script with nine scenes is nine image jobs and nine video jobs. Video costs far more than images. Run `shortgen plan <project>` and then `shortgen status <project>` to see how many scenes a script has before anything is sent, and start with one short script.

Their terms, not shortgen's Apache-2.0 license, decide what you may do with the results, including commercial use. Read them.

## Setup

```bash
pip install -e ".[hosted]"
export HF_KEY='your-key:your-secret'        # create one at https://cloud.higgsfield.ai
shortgen --config configs/higgsfield.yaml doctor
shortgen --config configs/higgsfield.yaml run examples/odyssey-cartoon
```

`HF_KEY` is Higgsfield's variable name. It has nothing to do with Hugging Face, whose variables also start with `HF_`. You can set `HF_API_KEY` and `HF_API_SECRET` instead. Either form can go in a `.env` file in the directory you run shortgen from; `.env` is git-ignored. Never put the key in a YAML file or commit it.

`shortgen doctor` shows a warning for each hosted provider and fails if the key is missing. It does not contact Higgsfield, so it cannot tell you whether the key is valid or the account has credits.

To pay for pictures only and animate them locally with zoom and pan:

```bash
SHORTGEN__PROVIDERS__VIDEO=kenburns shortgen --config configs/higgsfield.yaml run my-project
```

## Choosing models

The registry entries `higgsfield_image` and `higgsfield_video` name one model each (`bytedance/seedream/v4/text-to-image` and `higgsfield-ai/dop/standard`). Which models your account offers, and what they are called, is for Higgsfield's documentation to say. To use another one, put an override in your own registry file and point `runtime.models_file` at it:

```yaml
# my-models.yaml
models:
  higgsfield_video:
    model_id: kling-video/v2.1/pro/image-to-video
    options:
      durations_s: [5, 10]        # the clip lengths that model offers
      image_argument: image_url   # the name that model uses for the input picture
```

```bash
SHORTGEN__RUNTIME__MODELS_FILE=my-models.yaml shortgen --config configs/higgsfield.yaml run my-project
```

Anything under `options.arguments` is sent with every job, which is how you pass a model-specific setting that shortgen does not know about.

For recurring characters, set `reference_argument` on `higgsfield_image` to the argument name your chosen model documents for input images. The reference images of the characters in a scene are then uploaded and passed under that name. It is unset by default because the right name depends on the model.

## You do not pay twice for the same job

A hosted job that was accepted is never submitted again by a retry, a crash or Ctrl+C:

- When a job is accepted, its request id is saved under `~/.cache/shortgen/higgsfield/`. If the run stops before the output is saved, the next run asks for that same job's result instead of starting a new one.
- Only a failure to connect is retried automatically, because nothing was accepted yet. A connection that breaks mid-submit is reported as an error, since the job may or may not exist; check your account before running again.
- A download that fails is retried in place. If it keeps failing, run the same command again: the job is remembered.
- Scenes that are already done are skipped, as everywhere in shortgen.

What does start a new, paid job: changing a scene's picture description, the style, a character description or the model. Changing `project.style` regenerates every image. Use `shortgen generate --scene s003` to limit a run to the scenes you mean.

To deliberately start over (for example after a job got stuck), delete the files in `~/.cache/shortgen/higgsfield/`.

## What is not verified

No request has ever been sent to Higgsfield from this project; there was no account to test with. The adapters were written from the source code of the official `higgsfield-client` package (0.2.0) and from published examples, and they are tested against that real client talking to a pretend server. That proves the adapters use the client correctly. It does not prove the service answers the way the examples say. In order of confidence:

1. How to authenticate, submit a job, poll it, upload a file and read errors: taken from the client's source code.
2. The text-to-image model id, its arguments (`prompt`, `aspect_ratio`, `resolution`) and where the image URL is in the result: taken from the one example in the client's README.
3. The image-to-video model ids, their argument names (`image_url`, `prompt`, `duration`) and where the clip URL is in the result: taken from examples in the hosted API's documentation. This is the least certain part. The argument names are options so they can be corrected without a code change; the result fields are not.

Also unknown: whether a seed or a negative prompt can be sent (neither is, by default), and what a job costs.

If something does not match, the error message says what the service returned. Please open an issue with it.

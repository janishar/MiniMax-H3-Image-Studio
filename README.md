# minimax h3 image studio

**A command line and a local web studio for [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)** —
text-to-image and multi-image editing on Apple Silicon, with MiniMax's PE prompt
enhancer.

`minimax-h3-image` is a small PyTorch CLI around diffusers' modular
`MiniMaxH3ImageBlocks`. The 33B transformer and the 33B text encoder are 66 GB
each in bf16, so on a 64 GB Mac both run **int8 weight-only**, quantized on the
GPU as the bf16 files stream in — nothing extra is written to disk — and one at a
time: the text encoder encodes the prompt and is freed before the transformer
loads. The studio in `web/` is a standard-library Python server that drives that
CLI — it builds the arguments, queues renders and streams their progress, in a
browser tab, nothing sent off your machine. Like
[qwen image studio](https://github.com/janishar/qwen-image-2.1-studio), it runs
as a [helmstudio][helmstudio] studio and only that way; the CLI needs none of it.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%28Apple%20Silicon%29-lightgrey?logo=apple)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

## Table of contents

- [Requirements](#requirements) · [Weights](#weights)
- [The studio](#the-studio) — [Path A: the launcher](#path-a-install-from-the-launcher-recommended) ·
  [Path B: a checkout](#path-b-run-from-a-checkout) · [Using it](#using-the-studio) ·
  [Sessions and state](#sessions-and-state)
- [Command line](#command-line) — [Sizes](#sizes) · [Prompt enhancer](#prompt-enhancer) ·
  [Editing](#editing-with-reference-images) · [Flags](#cli-reference) ·
  [Environment variables](#environment-variables)
- [How it fits in 64 GB](#how-it-fits-in-64-gb) · [Security](#security) · [Limits](#limits) · [License](#license)

## Requirements

macOS on Apple Silicon with **64 GB** of unified memory is what this is built and
tested on (an M5 Pro). See [How it fits in 64 GB](#how-it-fits-in-64-gb).
`--device cuda` and `--device cpu` exist, untested.

| | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | The environment. `uv sync` installs exactly `uv.lock` (torch 2.14, diffusers 0.43). |
| [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) | Downloading the weights, if helmstudio is not doing it. |
| **helmstudio's `helm`** | The studio only. |

```bash
uv sync
```

## Weights

| Weight | Hugging Face | Disk | Used for |
| --- | --- | --- | --- |
| Image model | [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | 68.6 GB + the 66.7 GB text encoder | every render |
| Prompt enhancer | [`MiniMaxAI/MiniMax-H3-Image-PE`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) | 17.9 GB | `--enhance` |

Only the diffusers layout is needed: the repository also carries the transformer
and VAE in their original format (`transformer/`, `vae/`, another 68 GB),
`examples/` and `assets/`.

```bash
hf download MiniMaxAI/MiniMax-H3-Image --local-dir ~/models/MiniMax-H3-Image \
  --include "model_index.json" --include "modular_model_index.json" --include "scheduler/*" \
  --include "tokenizer/*" --include "processor/*" --include "vae_diffusers/*" --include "transformer_diffusers/*"
hf download MiniMaxAI/MiniMax-H3-Image-PE --local-dir ~/models/MiniMax-H3-Image-PE \
  --exclude "README.md" --exclude "assets/*" --exclude "scripts/*"
```

**The text encoder** is MiniMax-H3's, byte for byte (`text_encoder/`, as are
`tokenizer/` and `processor/`). With [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
already on disk, link its files instead of downloading 66.7 GB again:

```bash
mkdir -p ~/models/MiniMax-H3-Image/text_encoder
ln -s ~/models/MiniMax-H3/Ref2VA/text_encoder/* ~/models/MiniMax-H3-Image/text_encoder/
```

Without it: `hf download MiniMaxAI/MiniMax-H3-Image --local-dir ~/models/MiniMax-H3-Image --include "text_encoder/*"`.
Link the files, not the directory: helmstudio follows a file link anywhere, but
refuses a directory link that leaves the weight's own folder.

Both weights are under the
[MiniMax-H3-Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE),
not this repository's.

## The studio

Two ways in, as for qwen image studio — **the studio never starts on its own**,
since whatever launches it also gives it the platform that keeps its sessions and
takes.

| | **A — Install it** | **B — Run from a checkout** |
| --- | --- | --- |
| For | using the studio | changing the studio |
| Needs | the **launcher**, helmstudio's daemon and web UI | the **`helm` CLI**; `helm dev` runs one studio, no daemon |
| Install | one click in its library | `git clone`, `uv sync`, `bash web/run.sh` |
| Weights | it downloads or links them | you point `MINIMAX_MODELS` at them |

### Path A: Install from the launcher (recommended)

Install helmstudio (the Mac app from [its releases][helm-releases], or
`git clone https://github.com/janishar/helmstudio && cd helmstudio && make build && ./bin/helmstudio`),
open **http://127.0.0.1:8700**, and install minimax h3 image studio from its
library. Install shows every command it will run — `uv sync --locked` — and the
weights it will fetch. Start it, and the launcher gives the studio a port and the
weights' paths, then opens its page.

### Path B: Run from a checkout

```bash
# 1. helm, the CLI that runs this checkout
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/janishar/helmstudio/main/installer/install.sh)"

# 2. the checkout and its environment
git clone https://github.com/janishar/MiniMax-H3-Image-Studio && cd MiniMax-H3-Image-Studio && uv sync

# 3. start, or restart; then `stop` to end it
MINIMAX_MODELS=~/models bash web/run.sh
bash web/run.sh stop
```

`web/run.sh` links `MiniMax-H3-Image` and `MiniMax-H3-Image-PE` from
`MINIMAX_MODELS` (default `~/models`) and starts the studio under `helm dev` on
**http://127.0.0.1:8740**, keeping what it stores in `./.helm`. `helm dev` checks
that every weight file is there, so it refuses to start mid-download.

### Using the studio

Write a prompt, press **Generate**. The left pane is every CLI flag:

- **Model** — the image model's path (change it, and the enhancer's, under **⚙ Paths**) — and **Device**.
- **Mode** — *Text → image*, or *Edit* as soon as there is a reference image.
- **Prompt** — type `@` to pick a reference image, or click one on the right; it
  is written as `@name` and sent as `<Picture N>` by the image's place in the
  list, so removing another image never breaks it.
- **Prompt enhancer** — **Enhance prompt** and **Let it think**.
- **Aspect ratio** — *Auto*, a preset, or *Free*; **Size** — 4, 8, 12 or 16 MP; and
  the size it will render. Type a **Width** or a **Height** and the other side
  follows the ratio (shown in italics, still editable — typing there makes it the
  side that leads); with *Free*, both are yours.
- **Seed** (⚄ for a random one). There are no steps to set: the checkpoint is
  distilled to six.
- **Warm re-roll** — keep the image model loaded between renders; see
  [Warm re-roll](#warm-re-roll).
- Under **Generate**, the command it will run.

The centre shows the take, the six stages of the render as they happen
(Enhance → Encode → Load → Denoise → Decode → Save), the enhanced prompt beside the
original, and the session's takes. The right pane holds the reference images (up
to 16), the queue, and the terminal.

### Sessions and state

The studio keeps nothing of its own; helmstudio keeps it all — a session's
settings and reference images (session state), references and takes (assets), a
take with every setting that made it and its enhanced prompt (a gallery item), a
render and its log (a job), the paths chosen under **⚙ Paths** (kv). Its gallery,
references from other studios' takes, the render log and launcher cancellation
work as in qwen image studio.

## Command line

```bash
uv run minimax-h3-image "A neon shop sign that reads H3" -m ~/models/MiniMax-H3-Image
```

Or `export MINIMAX_H3_IMAGE_PATH=~/models/MiniMax-H3-Image` once and drop `-m`.
The take is written to `output/output.png`; `--output` puts it elsewhere.

MiniMax-H3-Image expects a long, complete English description of the whole image
— a short prompt works, but `--enhance` is how MiniMax's own API gets its quality.

### Sizes

`--ratio W:H` takes any whole-number ratio and `--mp` the megapixels (1–20,
default **4**); the size is the ratio at that area, in multiples of 32:

| `--ratio` | `--mp 4` (default) | `--mp 16` (MiniMax's recommendation) |
| --- | --- | --- |
| 1:1 | 2048×2048 | 4096×4096 |
| 4:3 / 3:4 | 2368×1792 / 1792×2368 | 4736×3552 / 3552×4736 |
| 3:2 / 2:3 | 2496×1664 / 1664×2496 | 5024×3360 / 3360×5024 |
| 16:9 / 9:16 | 2720×1536 / 1536×2720 | 5472×3072 / 3072×5472 |
| 21:9 | 3136×1344 | 6272×2688 |

At 16 MP that is MiniMax's own table. Without `--ratio` it is 1:1, `<Picture 1>`'s
aspect in an edit, or the enhancer's pick. Give one of `--width` or `--height`
and the other follows the ratio; give both for an exact, free size (multiples of
32, 256–8192). Below 4 MP the pipeline generates on a 4 MP canvas and resizes
down, so it is no faster.

### Prompt enhancer

`--enhance` first rewrites the prompt with MiniMax-H3-Image-PE (a Qwen3.8-27B
fine-tune) and lets it choose the aspect ratio; `--ratio`, `--width` and
`--height` still win. It thinks before answering — briefly by default, at length
with `--think` (MiniMax's own setting, often minutes slower). The PE ships only as
NVFP4, which nothing on a Mac runs, so it too is loaded as int8; it is freed
before the image model loads.

```bash
export MINIMAX_H3_IMAGE_PE_PATH=~/models/MiniMax-H3-Image-PE
uv run minimax-h3-image "a corgi playing guitar in the rain" --enhance
```

With reference images it uses MiniMax's image-editing PE prompt on the same
model, as diffusers' PE blocks do; MiniMax calls that workflow still in
development.

### Editing with reference images

`--input` takes up to 16 images, comma-separated, and makes the render an edit.
The prompt names them `<Picture 1>`, `<Picture 2>`, … in that order. Each keeps
its own size, resized only when its short edge is outside 1024–2048 px.

```bash
uv run minimax-h3-image "Keep everything in <Picture 1> as it is, but put the mug from <Picture 2> on the table" \
  --input room.jpg,mug.png --enhance
```

### CLI reference

| Flag | Default | Description |
| --- | --- | --- |
| `prompt` | *(required)* | The text prompt. |
| `--input` | *(none)* | Reference images for editing, comma-separated, at most 16. |
| `--ratio` | `1:1`, `<Picture 1>`'s, or the PE's | Aspect ratio `W:H`. |
| `--mp` | `4` | Megapixels, 1–20. |
| `--width`, `--height` | from `--ratio` and `--mp` | One: the other side follows the ratio. Both: exactly that size. Multiples of 32, 256–8192. |
| `--seed` | `42` | Seed for the CPU generator, so a seed repeats across devices. For an edit it is mixed with the inputs' pixels, so editing a take with the seed it was made with does not start from the noise that made it. |
| `-m`, `--model` | `$MINIMAX_H3_IMAGE_PATH`, else `MiniMaxAI/MiniMax-H3-Image` | Local directory or Hub id. |
| `--enhance` | off | Rewrite the prompt, and choose the ratio, with MiniMax-H3-Image-PE first. |
| `--think` | off | With `--enhance`: reason at length (`xhigh`) rather than briefly (`low`). |
| `--pe-model` | `$MINIMAX_H3_IMAGE_PE_PATH`, else `MiniMaxAI/MiniMax-H3-Image-PE` | Enhancer directory or Hub id. |
| `--device` | `$MINIMAX_H3_IMAGE_DEVICE`, else `auto` | `auto`, `mps`, `cuda` or `cpu`. |
| `--output` | `output/output.png` | Where the take is written. |

Besides its human-readable output the CLI prints one line per event for the
studio to read: `@stage enhance|encode|load|denoise|decode|save`, `@step 3/6`, and
`@enhanced {"prompt": …, "ratio": …}`.

### Environment variables

| Variable | Used as |
| --- | --- |
| `MINIMAX_H3_IMAGE_PATH` | `--model` |
| `MINIMAX_H3_IMAGE_PE_PATH` | `--pe-model` |
| `MINIMAX_H3_IMAGE_DEVICE` | `--device` |
| `MINIMAX_MODELS` | `web/run.sh`: the directory holding both weights |

## How it fits in 64 GB

`src/minimax_h3_image/int8.py` builds each model with its weights on the meta
device, swaps every large `nn.Linear` for an `Int8Linear`, then streams the
safetensors in one tensor at a time: a big weight is quantized on the GPU as it
arrives — int8, one bf16 scale per output row — and everything else keeps its
stored dtype (the transformer's float32 embeddings, timestep and output heads
stay float32). A forward dequantizes each weight to bf16 just before its matmul;
a one-token forward (the enhancer generating) uses torch's int8 kernel instead.

Peak GPU memory by phase, each model freed before the next loads (M5 Pro, 64 GB):

| Phase | 4 MP text → image | 4 MP edit, one 2048 px reference | 16 MP text → image |
| --- | --- | --- | --- |
| Enhance — the PE, `--enhance` only | 34.1 GB | | |
| Encode — text encoder, 51 of its 64 layers (the transformer reads layer 50) | 29.6 GB | 29.6 GB | 29.6 GB |
| Load, denoise, decode — transformer + VAE | 37.9 GB | 39.0 GB | 41.9 GB |
| **Time, end to end** | **115 s** | **218 s** | **403 s** |

The enhancer adds about 110 s to a text → image render with brief reasoning,
and about 4 minutes to an edit, whose instructions alone are 8.5k tokens (38 GB
on the GPU); it writes about 8 tokens/s once loaded. `--think` is often minutes
more. Attention at 16 MP (16.7k tokens) runs in MPS's
memory-efficient kernel, about 1 GB. The VAE works in 256 px tiles, as MiniMax
released it; two things keep that small on MPS (`load_vae` in `generate.py`): the
encoder's causal 3-D convolutions, which see a single frame, run as the exact 2-D
convolutions they reduce to, and each tile waits for the GPU before the next is
queued. Without them a 2048 px reference took 38 GB, and conv3d alone 33 GB for
one untiled 1024 px encode.

### Warm re-roll

Without it, every render starts from nothing: its own process loads the text
encoder and the transformer from disk and frees them at the end. With **Warm
re-roll** ticked, the studio sends renders to one long-lived worker
(`python -m minimax_h3_image.worker`, the CLI fed one argv per line) that keeps
the transformer and VAE loaded — about 38 GB — with the last prompt encoding and
the last enhanced prompt:

| Next render changes | What happens | 4 MP, measured |
| --- | --- | --- |
| Only the seed, size or ratio | Encoding and enhanced prompt reused, straight to denoising | **76 s** (vs 115 s cold, 254 s cold with the enhancer) |
| The prompt or a reference | The image model makes room, the text encoder loads and goes, the image model reloads | 113 s, as cold |
| Enhance on, prompt/references/Think unchanged | The enhanced prompt is reused, not asked again | as the first row |

64 GB cannot hold the image model beside the text encoder or the 27B prompt
enhancer, so whichever has to load, the image model unloads first; the enhancer
is never kept. Untick the toggle (or start a render without it, or stop the
studio) and the worker ends, giving the memory back.

## Security

The studio has **no authentication**: anyone who can reach its port can run
renders. It binds to `127.0.0.1`, refuses a `Host` it does not recognise (which
blocks DNS rebinding), refuses a cross-origin write, and accepts only JSON for
anything that changes state. `--allow-host` adds a name to accept; binding to
anything but loopback is on you. The terminal is output only, never a shell.

## Limits

- int8 is not bf16: a take is close to, not bit-identical with, what the same
  seed gives on a GPU that holds the bf16 weights.
- One render at a time; the rest wait in the queue. Each render loads the models
  afresh, streaming 50 GB (text encoder) and 66 GB (transformer) of bf16 from disk,
  unless [Warm re-roll](#warm-re-roll) keeps the transformer.
- A warm re-roll keeps the last enhanced prompt rather than asking the enhancer
  again with the new seed; change the prompt, or untick Enhance and paste a
  variant, for a different rewrite.
- Load has no progress of its own: its bar jumps from nothing to done.
- CUDA and CPU are untested; `helmstudio.yaml` still requires an Apple Silicon Mac.

## License

MIT — see [LICENSE](LICENSE). The model weights have their own license, above.

[helmstudio]: https://github.com/janishar/helmstudio
[helm-releases]: https://github.com/janishar/helmstudio/releases

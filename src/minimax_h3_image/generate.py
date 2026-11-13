"""Generate / edit images with MiniMax-H3-Image on Apple Silicon (MPS), CUDA or CPU.

The 33B transformer and the 33B text encoder are each 66 GB in bf16, so both run int8 weight-only
(int8.py), one at a time: the text encoder encodes the prompt and is freed before the transformer loads.

Usage:
    uv run minimax-h3-image "A neon shop sign that reads H3" -m ~/models/MiniMax-H3-Image
    uv run minimax-h3-image "Keep <Picture 1>, make it night" --input photo.png
    uv run minimax-h3-image "a corgi playing guitar in the rain" --enhance
    uv run minimax-h3-image "Prompt" --ratio 16:9 --mp 16
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps

from minimax_h3_image.int8 import load_int8, shards

DEVICES = ("auto", "mps", "cuda", "cpu")
#: The checkpoint is distilled to six steps, one output head each.
STEPS = 6
MAX_INPUTS = 16
#: hidden_states[50] conditions the transformer; a 51-layer stack still leaves it un-normed.
TEXT_LAYERS = 51
RATIO_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")
#: Widest ratio rendered; MiniMax's own sizes stop at 21:9.
MAX_RATIO = 4.0


def patch_mps_empty_cache() -> None:
    # torch 2.14: mps.empty_cache() waits on the GPU while holding the GIL, but finishing
    # non_blocking weight copies needs the GIL -> deadlock. synchronize() releases it.
    # ponytail: global monkeypatch, drop once torch's empty_cache releases the GIL itself
    empty_cache = torch.mps.empty_cache
    if empty_cache.__name__ == "safe_empty_cache":  # already patched (a warm worker renders again)
        return

    def safe_empty_cache() -> None:
        torch.mps.synchronize()
        empty_cache()

    torch.mps.empty_cache = safe_empty_cache


def patch_single_frame_conv() -> None:
    """Run the VAE encoder's causal 3-D convolutions on a single frame as the 2-D convolutions they are.

    Each pads ``kernel_t - 1`` zero frames before the frame, so only the last temporal tap sees it: exactly a
    conv2d with that tap. On MPS, conv3d keeps driver memory it never frees (a 2048 px reference took 85 GB).
    """
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import MiniMaxH3ImageCausalConv3d as Conv

    conv3d = Conv.forward
    if getattr(conv3d, "single_frame", False):  # already patched (a warm worker loads the VAE again)
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[2] != 1 or self.temporal_padding != self.kernel_size[0] - 1:
            return conv3d(self, hidden_states)
        # A tile arrives as a view into the whole image; MPS would keep one set of buffers per view offset.
        hidden_states = hidden_states.contiguous()
        if self.spatial_padding > 0:
            p = self.spatial_padding
            hidden_states = torch.nn.functional.pad(hidden_states, (p, p, p, p, 0, 0), mode=self.spatial_padding_mode)
        out = torch.nn.functional.conv2d(hidden_states[:, :, 0], self.weight[:, :, -1], self.bias,
                                         stride=self.stride[1:], dilation=self.dilation[1:])  # fmt: skip
        return out[:, :, None]

    forward.single_frame = True
    Conv.forward = forward


def pick_device(requested: str) -> str:
    """The torch device to run on: the one asked for, or with "auto" the first of MPS, CUDA, CPU present."""
    available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
    if requested == "auto":
        return next(d for d, ok in available.items() if ok)
    if not available[requested]:
        raise ValueError(f"{requested.upper()} is not available in this torch build")
    return requested


def empty_cache(device: str) -> None:
    """Hand freed memory back to the device, so the next model has room."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def parse_ratio(text: str) -> float:
    m = RATIO_RE.match(text or "")
    if not m or not int(m[1]) or not int(m[2]):
        raise ValueError(f"a ratio is W:H in whole numbers, got {text!r}")
    return int(m[1]) / int(m[2])


def size_for(ratio: float, mp: float) -> tuple[int, int]:
    """(width, height) at ``mp`` megapixels, multiples of 32. At 16 it is MiniMax's own table (4:3 is 4736x3552).

    Worked out on the long side and swapped for a portrait, so 3:4 is exactly 4:3 turned.
    """
    if ratio < 1:
        return size_for(1 / ratio, mp)[::-1]
    width = round(math.sqrt(mp * 1024 * 1024 * ratio) / 32) * 32
    return width, round(width / ratio / 32) * 32


def side(pixels: float) -> int:
    """A side in whole multiples of 32, within 256-8192."""
    return min(max(round(pixels / 32) * 32, 256), 8192)


def plan_size(ratio: float, mp: float, width: int | None, height: int | None) -> tuple[int, int]:
    """The size to render: both sides as given (a free size), one side with the other from ``ratio``, else ``mp``."""
    if width and height:
        return width, height
    if width:
        return width, side(width / ratio)
    if height:
        return side(height * ratio), height
    return size_for(ratio, mp)


def noise_seed(seed: int, images: list[Image.Image] | None) -> int:
    """The seed for the starting noise: `seed` itself, or for an edit, `seed` mixed with the inputs' pixels.

    Editing a take with the seed it was made with would otherwise start from the noise that made it.
    """
    if not images:
        return seed
    digest = hashlib.sha256(str(seed).encode())
    for img in images:
        digest.update(f"{img.mode}{img.size}".encode())
        digest.update(img.tobytes())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


def marker(line: str) -> None:
    # a line the studio UI reads. tqdm leaves its bar unterminated on the same pipe, so end it first:
    # \r keeps it a bar the log rewrites, \n gives this line its own start
    print(f"\r\n{line}", flush=True)


def stage(name: str) -> None:
    # machine-readable marker for the studio UI: enhance, encode, load, denoise, decode, save
    marker(f"@stage {name}")


def resolve_model(model: str) -> Path:
    """A local MiniMax-H3-Image directory, downloading the files the CLI reads when given a Hub id."""
    if os.path.isdir(model):
        return Path(model)
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "scheduler/*", "tokenizer/*", "processor/*", "text_encoder/*",
                "transformer_diffusers/*", "vae_diffusers/*"]  # fmt: skip
    return Path(snapshot_download(model, allow_patterns=patterns))


def load_text_encoder(model_dir: Path, device: str):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(model_dir / "text_encoder")
    text = config.text_config
    text.num_hidden_layers = TEXT_LAYERS  # the 13 layers after hidden_states[50] are never read
    if getattr(text, "layer_types", None):
        text.layer_types = text.layer_types[:TEXT_LAYERS]
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLForConditionalGeneration(config)
    model.lm_head = torch.nn.Identity()  # get_qwen3vl_prompt_embeds reads text_encoder.model's hidden states

    layer = re.compile(r"\.layers\.(\d+)\.")

    def rename(key: str) -> str | None:
        if key.startswith("lm_head.") or ((m := layer.search(key)) and "language_model" in key and int(m[1]) >= TEXT_LAYERS):
            return None
        return key

    return load_int8(model, shards(model_dir / "text_encoder"), device,
                     quantize=lambda name, _: ".language_model.layers." in name, rename=rename)  # fmt: skip


def load_vae(model_dir: Path, device: str):
    """The VAE, which encodes references and decodes in 256 px tiles (as released), one tile at a time.

    Python queues every tile before the GPU runs the first, and MPS keeps each queued tile's buffers:
    121 tiles of a 2048 px reference held 38 GB. Waiting after each tile keeps it to one.
    """
    from diffusers.models import AutoencoderKLMiniMaxH3Image

    patch_single_frame_conv()
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(model_dir / "vae_diffusers").to(device).eval()
    if device == "mps":
        for part in (vae.encoder, vae.decoder):
            part.register_forward_hook(lambda *_: torch.mps.synchronize())
    return vae


def load_transformer(model_dir: Path, device: str):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel

    sub = model_dir / "transformer_diffusers"
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3ImageTransformer2DModel.from_config(MiniMaxH3ImageTransformer2DModel.load_config(sub))
    # The 52 blocks are int8; the float32 embeddings, timestep and output heads and the small projections stay as stored.
    return load_int8(model, shards(sub), device, quantize=lambda name, _: "blocks." in name)


def digests(images: list[Image.Image] | None) -> tuple[str, ...]:
    """Each reference's identity by its pixels, for the warm caches."""
    return tuple(f"{img.size}{hashlib.sha256(img.tobytes()).hexdigest()}" for img in images or ())


class Warm:
    """What the studio's warm worker (worker.py) keeps between renders.

    The transformer and VAE stay loaded, with the last prompt encoding and the last enhanced prompt, so
    a render that only changes the seed or the size goes straight to denoising. 64 GB cannot hold the
    image model beside the text encoder (or the prompt enhancer), so a new prompt unloads it first.
    """

    def __init__(self) -> None:
        self.render = None  # the vae_encoder/denoise/decode pipeline, with its models
        self.render_key = None  # (model dir, device) it was loaded for
        self.encoding = None  # (key, the text encoder's PipelineState) of the last prompt
        self.enhanced = None  # (key, (prompt, ratio)) of the last enhancement

    def unload(self, device: str) -> None:
        """Free the image model, to make room for a model that has to load."""
        if self.render is not None:
            self.render = self.render_key = None
            empty_cache(device)

    def forget(self) -> None:
        """Free everything, after a render failed partway."""
        self.render = self.render_key = self.encoding = self.enhanced = None
        empty_cache(pick_device("auto"))


def main(argv: list[str] | None = None, warm: Warm | None = None) -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3-Image generation")
    parser.add_argument("prompt", help="Text prompt. In an edit, the inputs are <Picture 1>, <Picture 2>, …")
    parser.add_argument("--input", help=f"Reference image(s) for editing, comma-separated, at most {MAX_INPUTS}")
    parser.add_argument("--ratio", help="Aspect ratio W:H (default: 1:1, <Picture 1>'s with --input, the PE's with --enhance)")
    parser.add_argument("--mp", type=float, default=4.0,
                        help="Megapixels, 1-20 (default: %(default)s; MiniMax recommends 16)")  # fmt: skip
    parser.add_argument("--width", type=int, help="Width (multiple of 32); alone, the height follows the ratio")
    parser.add_argument("--height", type=int, help="Height (multiple of 32); alone, the width follows the ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-m", "--model", type=os.path.expanduser,
                        default=os.environ.get("MINIMAX_H3_IMAGE_PATH") or "MiniMaxAI/MiniMax-H3-Image",
                        help="Local path or HF repo ID (default: %(default)s, set via MINIMAX_H3_IMAGE_PATH)")  # fmt: skip
    parser.add_argument("--enhance", action="store_true",
                        help="Rewrite the prompt (and pick the aspect ratio) with MiniMax-H3-Image-PE first")  # fmt: skip
    parser.add_argument("--think", action="store_true", help="With --enhance: reason harder first (slower)")
    parser.add_argument("--pe-model", type=os.path.expanduser,
                        default=os.environ.get("MINIMAX_H3_IMAGE_PE_PATH") or "MiniMaxAI/MiniMax-H3-Image-PE",
                        help="Prompt enhancer path (default: %(default)s, set via MINIMAX_H3_IMAGE_PE_PATH)")  # fmt: skip
    parser.add_argument("--device", choices=DEVICES, default=os.environ.get("MINIMAX_H3_IMAGE_DEVICE") or "auto",
                        help="Where to run (default: %(default)s = MPS, else CUDA, else CPU; MINIMAX_H3_IMAGE_DEVICE)")  # fmt: skip
    parser.add_argument("--output", default="output/output.png")
    args = parser.parse_args(argv)

    if not 1 <= args.mp <= 20:
        parser.error("--mp must be between 1 and 20")
    for side in ("width", "height"):
        value = getattr(args, side)
        if value is not None and (value % 32 or not 256 <= value <= 8192):
            parser.error(f"--{side} must be a multiple of 32 between 256 and 8192")
    try:
        ratio = parse_ratio(args.ratio) if args.ratio else None
        device = pick_device(args.device)
    except ValueError as e:
        parser.error(str(e))
    # ponytail: a relative "dir/name" path looks like a Hub ID, so only absolute, dotted or multi-segment paths are checked
    for model in [args.model, args.pe_model] if args.enhance else [args.model]:
        if not os.path.isdir(model) and (model.startswith((".", "/")) or model.count("/") != 1):
            parser.error(f"model directory not found: {model}")
    print(f"Device: {device}", flush=True)

    images = None
    if args.input:
        images = [ImageOps.exif_transpose(Image.open(p.strip())).convert("RGB") for p in args.input.split(",")]
        if len(images) > MAX_INPUTS:
            parser.error(f"MiniMax-H3-Image takes at most {MAX_INPUTS} reference images")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "mps":
        patch_mps_empty_cache()

    prompt = args.prompt
    if args.enhance:
        from minimax_h3_image.enhance import enhance

        stage("enhance")
        pe_dir = args.pe_model
        if not os.path.isdir(pe_dir):
            from huggingface_hub import snapshot_download

            pe_dir = snapshot_download(pe_dir)
        # An edit's PE writes its caption for the canvas, so it is told one the user fixed.
        fixed = f"{args.width}:{args.height}" if args.width and args.height else args.ratio
        if ratio and not 1 / MAX_RATIO <= ratio <= MAX_RATIO:
            fixed = f"{MAX_RATIO:g}:1" if ratio > 1 else f"1:{MAX_RATIO:g}"
        # Warm, a re-roll keeps its enhanced prompt; any change to what the PE reads asks it again.
        pe_key = (args.prompt, digests(images), args.think, fixed if images else None, os.path.abspath(pe_dir))  # t2i's PE ignores it
        if warm and warm.enhanced and warm.enhanced[0] == pe_key:
            prompt, pe_ratio = warm.enhanced[1]
            print("Reusing the enhanced prompt: same prompt, references and Think as the last render")
        else:
            if warm:
                warm.unload(device)  # the 27B PE needs the room
            prompt, pe_ratio = enhance(prompt, images, pe_dir, args.seed, args.think, device, fixed)
            if warm:
                warm.enhanced = (pe_key, (prompt, pe_ratio))
        print(f"Enhanced prompt ({pe_ratio or 'no ratio'}):\n{prompt}\n")
        marker(f"@enhanced {json.dumps({'prompt': prompt, 'ratio': pe_ratio})}")
        if ratio is None and pe_ratio:
            ratio = parse_ratio(pe_ratio)
    if ratio is None:
        ratio = images[0].width / images[0].height if images else 1.0
    if not (args.width and args.height) and not 1 / MAX_RATIO <= ratio <= MAX_RATIO:
        ratio = min(max(ratio, 1 / MAX_RATIO), MAX_RATIO)
        print(f"Aspect ratio clamped to {ratio:.3g}:1: this studio renders no wider than {MAX_RATIO:g}:1")
    width, height = plan_size(ratio, args.mp, args.width, args.height)

    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines import MiniMaxH3ImageBlocks, MiniMaxH3ImageModularPipeline, SequentialPipelineBlocks
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    model_dir = resolve_model(args.model)
    blocks = MiniMaxH3ImageBlocks()

    def part(*names: str) -> MiniMaxH3ImageModularPipeline:
        sub = SequentialPipelineBlocks.from_blocks_dict({n: blocks.sub_blocks[n] for n in names})
        return MiniMaxH3ImageModularPipeline(blocks=sub, pretrained_model_name_or_path=str(model_dir))

    # 1. The prompt (and the references' vision blocks) through the text encoder, which then goes.
    stage("encode")
    encoding_key = (str(model_dir), device, prompt, digests(images))
    if warm and warm.encoding and warm.encoding[0] == encoding_key:
        state = warm.encoding[1]
        print("Reusing the prompt encoding: same prompt and references as the last render")
    else:
        if warm:
            warm.unload(device)  # the text encoder needs the room
        encoder = part("before_encode", "text_encoder")
        encoder.update_components(
            text_encoder=load_text_encoder(model_dir, device),
            tokenizer=Qwen2TokenizerFast.from_pretrained(model_dir / "tokenizer"),
            processor=Qwen3VLProcessor.from_pretrained(model_dir / "processor"),
        )
        state = encoder(prompt=prompt, image=images)
        del encoder
        empty_cache(device)
        if warm:
            warm.encoding = (encoding_key, state)

    # 2. References through the VAE, the six steps, the decode.
    stage("load")
    if warm and warm.render is not None and warm.render_key == (str(model_dir), device):
        render = warm.render
        print("The image model is still loaded")
    else:
        render = part("vae_encoder", "denoise", "decode")
        render.update_components(
            transformer=load_transformer(model_dir, device),
            vae=load_vae(model_dir, device),
            scheduler=MiniMaxH3Scheduler.from_pretrained(model_dir / "scheduler"),
        )
        if warm:
            warm.render, warm.render_key = render, (str(model_dir), device)
    steps = iter(range(1, STEPS + 1))

    def on_step(*_) -> None:
        # The GPU runs behind Python: without waiting, the last five steps would report at once.
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        step = next(steps)
        marker(f"@step {step}/{STEPS}")
        if step == STEPS:
            stage("decode")

    def on_first_step(*_) -> None:
        # The references' VAE encode runs before it, so denoising starts here, not at render().
        stage("denoise")
        first.remove()

    first = render.transformer.register_forward_pre_hook(on_first_step)
    progress = render.transformer.register_forward_hook(on_step)
    try:
        out = render(state=state, height=height, width=width, output="images",
                     generator=torch.Generator().manual_seed(noise_seed(args.seed, images)))[0]  # fmt: skip
    finally:  # a warm transformer outlives this render's hooks
        first.remove()
        progress.remove()

    stage("save")
    try:
        out.save(out_path)
    except (OSError, ValueError) as e:  # e.g. an unknown extension: keep the render as PNG
        out_path = out_path.with_suffix(".png")
        print(f"Could not save {args.output} ({e}); saving PNG instead")
        out.save(out_path)
    marker(f"Saved {out_path.resolve()} ({out.size[0]}x{out.size[1]}, mode={out.mode})")


if __name__ == "__main__":
    main()

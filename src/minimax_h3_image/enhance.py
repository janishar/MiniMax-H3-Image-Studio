"""Rewrite prompts with MiniMax-H3-Image-PE before generation.

The PE (a Qwen3.8-27B fine-tune) ships only as NVFP4, which nothing on a Mac runs, and its bf16
form (54 GB) doesn't fit next to the system. So each NVFP4 Linear is dequantized on the device as
it streams in and requantized to int8 by the shared loader: about 27 GB on the GPU.
"""

from __future__ import annotations

import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from minimax_h3_image.int8 import load_int8, shards

T2I_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")
#: An edit's references reach the PE at about 1 MP: ~1k vision tokens each, where the processor would
#: take up to 16k of a phone photo. ponytail: fixed cap, raise it if captions miss fine detail.
PE_IMAGE_PIXELS = 1 << 20

# E2M1 values by 4-bit code (bit 3 is the sign), and by packed byte: low nibble is the first element.
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
FP4_PAIRS = torch.stack((FP4.repeat(16), FP4.repeat_interleave(16)), dim=1)  # [byte] -> (low, high)


def nvfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """ModelOpt NVFP4 -> bf16: uint8 [out, in/2] codes, float8_e4m3fn [out, in/16] block scales, fp32 global scale."""
    out = packed.shape[0]
    # code * block scale is exact in bf16 (2 + 4 significant bits), so only the global scale rounds.
    w = FP4_PAIRS.to(packed.device, torch.bfloat16)[packed.int()].view(out, -1, 16)
    w = w * scale.to(torch.bfloat16)[..., None]
    return (w.float() * scale_2.float()).to(torch.bfloat16).view(out, -1)


def reduced_ratio(w: int, h: int) -> str:
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def parse_answer(gen: str, ratio_key: str) -> tuple[str, str | None]:
    """The (caption, W:H or None) in the JSON after </think>; SystemExit with the raw output if there is none."""
    head, sep, answer = gen.rpartition("</think>")
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    try:
        result = json.loads(answer) if sep else None
        caption = result["expanded_caption"].strip()
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        # The streamer already printed the output; the studio shows this last line as the failure.
        raise SystemExit("Prompt enhancer did not return the JSON answer (its output is above)") from None
    ratio = result.get(ratio_key)
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", ratio) if isinstance(ratio, str) else None
    return caption, (reduced_ratio(int(m[1]), int(m[2])) if m and int(m[1]) and int(m[2]) else None)


def edit_content(task_prompt: str, prompt: str, images: list[Image.Image], ratio: str | None = None) -> list[dict]:
    """The one user message of the README's "How to use Image editing PE". ``ratio``: W:H the user fixed."""
    n = len(images)
    content = [
        {"type": "text", "text": task_prompt},
        {"type": "text", "text": f"Reference Image Count:\n{n} input image(s), labeled <Picture 1> through <Picture {n}>."},
    ]
    for i, img in enumerate(images, start=1):
        content += [
            {"type": "text", "text": f"Reference Media:\n- <Picture {i}> [aspect_ratio {reduced_ratio(*img.size)}]"},
            {"type": "image", "image": shrink(img)},
        ]
    content.append({"type": "text", "text": (
        "\nUser Editing Instruction (JSON string):\n"
        f"{json.dumps(prompt, ensure_ascii=False)}\n\n"
        f"User Aspect Ratio Selection:\n{ratio or 'adaptive'}\n\n"
        "Apply the instructions above to the supplied inputs and reference images. "
        "Output only the required two-field JSON object."
    )})
    return content


def shrink(img: Image.Image) -> Image.Image:
    """``img`` at no more than PE_IMAGE_PIXELS, its aspect kept."""
    scale = math.sqrt(PE_IMAGE_PIXELS / (img.width * img.height))
    return img if scale >= 1 else img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def load_pe(model_path: str, device: str) -> torch.nn.Module:
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

    path = Path(model_path)
    config = AutoConfig.from_pretrained(path)
    del config.quantization_config  # or transformers sets up the (CUDA-only) modelopt quantizer
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained(path)

    # Checkpoint keys are already transformers' module names; the NVFP4 scales are folded into the weight.
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def get(key: str) -> torch.Tensor:
        file = weight_map[key]
        if file not in files:
            files[file] = safe_open(path / file, "pt", device="cpu")
        tensor = files[file].get_tensor(key)
        # MPS has no float8: the e4m3 block scales become bf16 (exactly) before they move.
        return (tensor.to(torch.bfloat16) if tensor.dtype == torch.float8_e4m3fn else tensor).to(device)

    def nvfp4(name: str) -> bool:
        return f"{name}.weight_scale" in weight_map

    def hook(name: str, tensor: torch.Tensor) -> torch.Tensor:
        module, _, attr = name.rpartition(".")
        if attr != "weight" or not nvfp4(module):
            return tensor
        return nvfp4_dequant(tensor, get(f"{module}.weight_scale"), get(f"{module}.weight_scale_2"))

    def rename(key: str) -> str | None:
        return None if key.endswith(("input_scale", "weight_scale", "weight_scale_2")) else key

    return load_int8(model, shards(path), device, lambda name, _: nvfp4(name), rename, hook)


def empty_cache(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def enhance(
    prompt: str,
    images: list[Image.Image] | None,
    model_path: str,
    seed: int,
    think: bool = False,
    device: str = "mps",
    ratio: str | None = None,
) -> tuple[str, str | None]:
    """(expanded caption, 'W:H' ratio or None). Loads the PE on `device` and frees it (gc + empty cache) before returning.

    ``ratio`` is a W:H the user fixed, which an edit's PE writes its caption for; None lets it choose.
    """
    from transformers import AutoProcessor, TextStreamer

    prompts = Path(model_path) / "prompt"
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    if images:
        images = [ImageOps.exif_transpose(img).convert("RGB") for img in images]
        task_prompt = (prompts / "edit" / "task_prompt.txt").read_text(encoding="utf-8")
        if ratio and (m := re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", ratio)) and int(m[1]) and int(m[2]):
            ratio = reduced_ratio(int(m[1]), int(m[2]))
        messages = [{"role": "user", "content": edit_content(task_prompt, prompt, images, ratio)}]
        ratio_key = "target_aspect_ratio"
    else:
        system_prompt = (prompts / "t2i" / "system_prompt.txt").read_text(encoding="utf-8").strip()
        task_prompt = (prompts / "t2i" / "task_prompt.txt").read_text(encoding="utf-8")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt.format(prompt)},
        ]
        ratio_key = "aspect_ratio"
    # The system prompt asks for a thinking section, so thinking stays on; `think` sets its length.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
        reasoning_effort="xhigh" if think else "low",
    )

    t0 = time.perf_counter()
    model = load_pe(model_path, device)
    load_s = time.perf_counter() - t0
    inputs = inputs.to(device)

    # print the thinking + answer live, the rewrite takes minutes
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stamps = []  # one per put(): the prompt, then each new token
    put = streamer.put
    streamer.put = lambda value: (stamps.append(time.perf_counter()), put(value))[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4096,  # thinking included, as released
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            streamer=streamer,
        )
    n = out.shape[1] - inputs["input_ids"].shape[1]
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    peak = torch.mps.driver_allocated_memory() / 2**30 if device.startswith("mps") else float("nan")
    decode = (len(stamps) - 2) / (stamps[-1] - stamps[1]) if len(stamps) > 2 else float("nan")
    print(
        f"\nPE: loaded in {load_s:.0f}s, prompt {inputs['input_ids'].shape[1]} tokens in {stamps[1] - t0 - load_s:.0f}s, "
        f"{n} new tokens at {decode:.2f} tok/s, {peak:.1f} GB on the GPU",
        file=sys.stderr,
    )

    # the 27B PE and the image pipeline don't fit in memory together
    del model, inputs, out
    gc.collect()
    empty_cache(device)
    return parse_answer(gen, ratio_key)


if __name__ == "__main__":
    assert reduced_ratio(1920, 1080) == "16:9" and reduced_ratio(1000, 1000) == "1:1"
    assert parse_answer('x\n</think>\n\n{"expanded_caption": " A cat. ", "aspect_ratio": "3:2"}', "aspect_ratio") == ("A cat.", "3:2")
    assert parse_answer('</think>```json\n{"expanded_caption": "A", "target_aspect_ratio": "1024:768"}\n```', "target_aspect_ratio") == ("A", "4:3")
    assert parse_answer('</think>{"expanded_caption": "A", "aspect_ratio": "wide"}', "aspect_ratio") == ("A", None)
    for bad in ("no think tags", "</think> not json", '</think>{"caption": "A"}', "</think>[1]"):
        try:
            parse_answer(bad, "aspect_ratio")
            raise AssertionError(bad)
        except SystemExit:
            pass
    # packed byte 0x2F: first element code 0xF (-6), second 0x2 (1); block scales 2 and 0.5, global 0.25
    packed = torch.tensor([[0x2F] + [0] * 7 + [0x2F] + [0] * 7], dtype=torch.uint8)
    w = nvfp4_dequant(packed, torch.tensor([[2.0, 0.5]]).to(torch.float8_e4m3fn), torch.tensor(0.25))
    assert w.shape == (1, 32) and w[0, :2].tolist() == [-3.0, 0.5] and w[0, 16:18].tolist() == [-0.75, 0.125], w
    content = edit_content("T", 'make "it" night', [Image.new("RGB", (4032, 3024))], "16:9")
    assert len(content) == 5 and "aspect_ratio 4:3]" in content[2]["text"] and "Selection:\n16:9" in content[4]["text"]
    assert content[3]["image"].size == (1182, 887), content[3]["image"].size  # ~1 MP, still 4:3
    assert "Selection:\nadaptive" in edit_content("T", "x", [Image.new("RGB", (64, 64))])[-1]["text"]
    print("ok")

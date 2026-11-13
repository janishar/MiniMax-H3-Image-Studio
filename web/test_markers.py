"""The CLI's markers reach the studio whole, though tqdm leaves its bar unterminated on the same pipe,
and a warm worker's render ends at its @done line while the worker stays up for the next one.

Run: uv run python web/test_markers.py
"""

import os
import subprocess
import sys
from types import SimpleNamespace

from server import STEP_RE, Runner

# The real marker() beside a real bar that refreshes every step, as a render's slow steps make it.
CLI = """
from tqdm import tqdm
from minimax_h3_image.generate import marker, stage

stage("denoise")
with tqdm(total=6, mininterval=0, miniters=1) as bar:
    for i in range(6):
        marker(f"@step {i + 1}/6")
        bar.update()
stage("decode")
marker("Saved /tmp/take.png (1024x1024, mode=RGB)")
"""


# A warm worker as minimax_h3_image.worker speaks: per stdin line, a render's output, then @done.
WORKER = """
import sys
from minimax_h3_image.generate import marker, stage
for n, line in enumerate(sys.stdin):
    stage("denoise")
    marker("@step 6/6")
    marker(f"Saved /tmp/take{n}.png (512x512, mode=RGB)")
    marker("@done 0" if n == 0 else "@done 1")
"""


def warm() -> None:
    runner = Runner.__new__(Runner)
    runner.state = SimpleNamespace(job_progress=lambda job: None)
    runner.emit = lambda kind, data: None
    runner.log = lambda job, line, rewrites=False: None
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKER], env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    for n, code in enumerate((0, 1)):
        job = {"id": str(n), "progress": {"stage": None, "step": 0, "total": 0}, "result": {}}
        proc.stdin.write(b'["a prompt"]\n')
        proc.stdin.flush()
        assert runner._pump(job, proc.stdout, []) == code  # returns at @done, not at the end of the stream
        assert proc.poll() is None and job["result"]["width"] == 512  # the worker waits for the next render
    proc.stdin.close()
    assert proc.wait() == 0


def main() -> None:
    assert not STEP_RE.match(" 17%|█| 1/6 [00:08<00:40]@step 1/6")  # glued, it is lost

    logged: list[tuple[str, bool]] = []
    steps: list[int] = []
    runner = Runner.__new__(Runner)  # no helmstudio, no queue thread
    runner.state = SimpleNamespace(job_progress=lambda job: None)
    runner.emit = lambda kind, data: steps.append(data["progress"]["step"])
    runner.log = lambda job, line, rewrites=False: logged.append((line, rewrites))
    job = {"id": "x", "progress": {"stage": None, "step": 0, "total": 0}, "result": {}}

    proc = subprocess.Popen(
        [sys.executable, "-c", CLI],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    runner._pump(job, proc.stdout, [])
    assert proc.wait() == 0, logged

    assert job["result"] == {"steps": 6, "width": 1024, "height": 1024, "mode": "RGB"}
    assert [s for s in steps if s] == list(range(1, 7)), steps  # the page's bar moves
    assert job["progress"]["stage"] == "decode"

    bars = [(line, rewrites) for line, rewrites in logged if "/6 [" in line]
    # The bar stays in the log, rewriting itself, and no marker shows there.
    assert any("1/6 [" in line for line, _ in bars), logged
    assert all(rewrites for _, rewrites in bars[:-1]), bars
    assert not any("@" in line for line, _ in logged), logged
    warm()
    print("ok")


if __name__ == "__main__":
    main()

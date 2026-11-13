"""The studio's warm renderer: the CLI kept alive, so the image model stays loaded between renders.

Each stdin line is one render's argv as a JSON list (the CLI's own flags). Its output is the CLI's,
markers included, then ``@done <exit code>``. Everything a render loads, it frees again — the
prompt enhancer, the text encoder — except the transformer and VAE, which wait for the next render
with the last prompt encoding (generate.Warm). The worker ends when stdin closes.

    python -m minimax_h3_image.worker   # the studio starts it; one render per line
"""

from __future__ import annotations

import json
import sys
import traceback

from minimax_h3_image.generate import Warm, main, marker


def serve() -> None:
    warm = Warm()
    for line in sys.stdin:
        if not line.strip():
            continue
        argv = json.loads(line)
        try:
            main(argv, warm)
            code = 0
        except SystemExit as exit:  # argparse's errors, and a prompt enhancer that gave no JSON
            if isinstance(exit.code, str):
                print(exit.code, flush=True)
            code = exit.code if isinstance(exit.code, int) else 1
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            code = 1
        if code:
            warm.forget()  # a render that failed partway may have left half a model behind
        marker(f"@done {code}")


if __name__ == "__main__":
    serve()

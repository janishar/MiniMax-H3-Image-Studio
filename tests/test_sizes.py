"""The CLI's render size: MiniMax's own 16 MP table, and one side following the ratio.

Run: uv run python tests/test_sizes.py  (web/static/app.js mirrors these rules for the page's readout)
"""

from minimax_h3_image.generate import plan_size, size_for

# MiniMax's recommended sizes at 16 MP, portrait as the landscape turned
TABLE = {(1, 1): (4096, 4096), (4, 3): (4736, 3552), (3, 2): (5024, 3360), (16, 9): (5472, 3072), (21, 9): (6272, 2688)}
for (w, h), size in TABLE.items():
    assert size_for(w / h, 16) == size, (w, h, size_for(w / h, 16))
    assert size_for(h / w, 16) == size[::-1], (h, w)
assert size_for(1, 4) == (2048, 2048) and size_for(3 / 4, 4) == (1792, 2368)

# one side given: the other follows the ratio, in multiples of 32, within 256-8192
assert plan_size(16 / 9, 4, 2048, None) == (2048, 1152)
assert plan_size(16 / 9, 4, None, 1152) == (2048, 1152)
assert plan_size(3 / 4, 16, 3072, None) == (3072, 4096)
assert plan_size(1 / 4, 4, 4096, None) == (4096, 8192)  # 16384 clamped
assert plan_size(4, 4, 512, None) == (512, 256)  # 128 clamped
# both given: a free size, exactly as given; neither: the ratio at mp
assert plan_size(16 / 9, 4, 1024, 1024) == (1024, 1024)
assert plan_size(16 / 9, 16, None, None) == (5472, 3072)
print("ok")

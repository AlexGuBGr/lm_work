"""
Mål trænings-throughput (tokens/sek) for forskellige batch-størrelser, med og uden torch.compile.
Kør:  .venv/bin/python bench.py [--compile] [batch sizes ...]
"""
import os
import sys
import time

TOOLCHAIN_CXX = os.path.expanduser("~/.local/share/toolchain/bin/x86_64-conda-linux-gnu-g++")
if os.path.exists(TOOLCHAIN_CXX):
    os.environ.setdefault("CXX", TOOLCHAIN_CXX)

import torch  # noqa: E402

from model import GPT, ModelConfig  # noqa: E402

use_compile = "--compile" in sys.argv
sizes = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [8, 16, 32, 64]
torch.set_num_threads(os.cpu_count())
torch.manual_seed(0)

model = GPT(ModelConfig())
opt = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95))
fwd = torch.compile(model) if use_compile else model
T = model.cfg.block_size
print(f"compile={use_compile} threads={torch.get_num_threads()} params={model.num_params():,}")

for B in sizes:
    x = torch.randint(0, model.cfg.vocab_size, (B, T))
    y = torch.randint(0, model.cfg.vocab_size, (B, T))

    def step(x=x, y=y):
        _, loss = fwd(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    t0 = time.time()
    step()  # opvarmning (inkl. kompilering)
    warm = time.time() - t0
    step()
    n = 5
    t0 = time.time()
    for _ in range(n):
        step()
    dt = (time.time() - t0) / n
    print(f"batch {B:>3}: {dt * 1000:7.0f} ms/iter | {B * T / dt:8,.0f} tok/s | første iter {warm:.1f} s",
          flush=True)

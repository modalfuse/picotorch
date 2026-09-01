#!/usr/bin/env python3
"""Compare C reference MHA against NumPy. Run from the CMake build directory or repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_mha import linear, mha  # noqa: E402


def main() -> int:
    L, D = 8, 16
    x = (0.01 * np.arange(1, L * D + 1, dtype=np.float32)).reshape(L, D)
    I = np.eye(D, dtype=np.float32)
    z = np.zeros(D, dtype=np.float32)
    y = mha(x, x, 2, I, z, I, z, I, z, I, z, True)

    W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    b = np.array([0.10, 0.20], dtype=np.float32)
    ylin = linear(np.array([[1, 2, 3]], dtype=np.float32), W, b).reshape(-1)
    lin_err = float(np.max(np.abs(ylin - np.array([1.10, 2.20], dtype=np.float32))))

    build = ROOT / "build"
    exe = build / "test_mha_maxabs"
    if not exe.exists():
        exe = build / "Release" / "test_mha_maxabs"
    if not exe.exists():
        print("test_mha_maxabs binary missing; configure CMake first", file=sys.stderr)
        return 1
    out = subprocess.check_output([str(exe)], text=True)
    print(out, end="")
    # Parse C y00 and compare to NumPy row 0
    line = [ln for ln in out.splitlines() if ln.startswith("linear_maxabs")][0]
    y00 = float(line.split("mha_y00=")[1].split()[0])
    mha_err = abs(y00 - float(y[0, 0]))
    print(f"numpy_linear_maxabs={lin_err:.6e} numpy_mha_y00={y[0, 0]:.6f} c_vs_numpy={mha_err:.6e}")
    if lin_err > 1e-5 or mha_err > 1e-5:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

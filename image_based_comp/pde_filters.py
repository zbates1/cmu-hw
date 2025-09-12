#!/usr/bin/env python3
"""
PDE-based image smoothing (finite differences) with PGM I/O.

Implements:
  1) Linear filtering (heat equation):    ∂u/∂t = Δu
  2) Nonlinear filtering (Perona–Malik):  ∂u/∂t = div( g(|∇u|) ∇u )

- Explicit time stepping with 4-neighbor stencil (Neumann "no-flux" boundaries).
- Reads and writes ASCII PGM (P2) so you can compare in IrfanView, etc.

Usage examples
--------------
Linear (two iteration counts):
    python pde_filters.py foot.pgm --linear --iters 10 50 --dt 0.2

Nonlinear (two iteration counts and two lambdas):
    python pde_filters.py foot.pgm --nonlinear --iters 10 50 --lam 1 10 --g exp --dt 0.2

Both (one run each):
    python pde_filters.py foot.pgm --linear --iters 25 --dt 0.2 \
                                   --nonlinear --iters 25 --lam 10 --g lorentz

Notes
-----
- Stability (explicit scheme, h=1): choose dt ≤ 0.25 for the linear heat equation.
- For Perona–Malik, similar dt (e.g., 0.1–0.25) is typically safe.
- Conductance g options:
    "exp":     g(s) = exp( - (s/lam)^2 )
    "lorentz": g(s) = 1 / (1 + (s/lam)^2 )

Outputs
-------
Saves files next to the input with names like:
    <stem>_linear_it{N}.pgm
    <stem>_pm_g{exp|lorentz}_lam{L}_it{N}.pgm
"""

import argparse
import os
from typing import Tuple, List
import numpy as np


# ----------------------- PGM (P2) I/O -----------------------

def read_pgm_p2(path: str) -> Tuple[np.ndarray, int]:
    """
    Read an ASCII PGM (P2) file. Returns (image as float64 array, maxval).
    Preserves shape and converts to float in [0, maxval].
    """
    with open(path, 'r') as f:
        # Read header
        magic = f.readline().strip()
        if magic != 'P2':
            raise ValueError(f"Expected P2 magic number, got: {magic}")

        # Skip comments and find width/height
        def next_token():
            # Generator that yields non-comment tokens
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                for tok in line.split():
                    yield tok

        tok = next_token()
        try:
            w = int(next(tok))
            h = int(next(tok))
            maxval = int(next(tok))
        except StopIteration:
            raise ValueError("Invalid PGM header (width/height/maxval missing).")

        # Read pixels
        vals: List[int] = []
        for t in tok:
            vals.append(int(t))
        if len(vals) != w * h:
            raise ValueError(f"Expected {w*h} pixels, got {len(vals)}")

        img = np.array(vals, dtype=np.float64).reshape((h, w))  # PGM is row-major: height x width
        return img, maxval


def write_pgm_p2(path: str, img: np.ndarray, maxval: int = 255, comment: str = "Created by pde_filters.py") -> None:
    """
    Write an ASCII PGM (P2) file from a 2D array. Values are clipped to [0, maxval] and rounded to ints.
    """
    img_clipped = np.clip(img, 0, maxval)
    img_int = np.rint(img_clipped).astype(int)

    h, w = img_int.shape
    with open(path, 'w') as f:
        f.write("P2\n")
        f.write(f"# {comment}\n")
        f.write(f"{w} {h}\n")
        f.write(f"{maxval}\n")
        # Write pixel values, up to ~17 numbers per line for readability
        count = 0
        for val in img_int.flatten():
            f.write(f"{val} ")
            count += 1
            if count % 17 == 0:
                f.write("\n")
        f.write("\n")


# ------------------ Finite-difference operators ------------------

def laplacian_neumann(u: np.ndarray) -> np.ndarray:
    """
    4-neighbor Laplacian with Neumann (reflecting) boundaries via edge padding.
    Δu ≈ u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4 u_{i,j}
    """
    pad = np.pad(u, 1, mode='edge')
    center = pad[1:-1, 1:-1]
    lap = (pad[2:, 1:-1] + pad[:-2, 1:-1] + pad[1:-1, 2:] + pad[1:-1, :-2] - 4.0 * center)
    return lap


def perona_malik_step(u: np.ndarray, dt: float, lam: float, gtype: str = "exp") -> np.ndarray:
    """
    One explicit PM step with 4-neighbor fluxes and Neumann boundaries.

    Discretization (4-neighbor, forward/backward differences):
        u^{t+dt} = u^t + dt * [ div( g(|∇u|) ∇u ) ]

    We compute directional gradients (N,S,E,W), conductances on those edges,
    then the divergence of fluxes.
    """
    # Pad for boundaries (Neumann)
    pad = np.pad(u, 1, mode='edge')
    c = pad[1:-1, 1:-1]
    n = pad[:-2, 1:-1]
    s = pad[2:,  1:-1]
    w = pad[1:-1, :-2]
    e = pad[1:-1, 2:]

    # Directional differences
    dn = n - c
    ds = s - c
    dw = w - c
    de = e - c

    # Magnitudes per edge
    if lam <= 0:
        raise ValueError("lam must be > 0 for Perona–Malik.")
    def gfunc(m):
        if gtype == "exp":
            return np.exp(-(m/lam)**2)
        elif gtype == "lorentz":
            return 1.0 / (1.0 + (m/lam)**2)
        else:
            raise ValueError("gtype must be 'exp' or 'lorentz'")

    gn = gfunc(np.abs(dn))
    gs = gfunc(np.abs(ds))
    gw = gfunc(np.abs(dw))
    ge = gfunc(np.abs(de))

    # Fluxes
    fn = gn * dn
    fs = gs * ds
    fw = gw * dw
    fe = ge * de

    # Divergence (sum of incoming minus outgoing; here symmetric so sum of four)
    divF = (fn + fs + fw + fe)

    return u + dt * divF


def heat_equation(u0: np.ndarray, n_iter: int, dt: float) -> np.ndarray:
    """
    Linear diffusion (heat equation) explicit scheme.
    """
    u = u0.copy()
    for _ in range(n_iter):
        u += dt * laplacian_neumann(u)
    return u


def perona_malik(u0: np.ndarray, n_iter: int, dt: float, lam: float, gtype: str = "exp") -> np.ndarray:
    """
    Nonlinear anisotropic diffusion (Perona–Malik) explicit scheme.
    """
    u = u0.copy()
    for _ in range(n_iter):
        u = perona_malik_step(u, dt=dt, lam=lam, gtype=gtype)
    return u


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser(description="Finite-difference linear and nonlinear (Perona–Malik) filters with PGM I/O.")
    p.add_argument("input", help="Input PGM (P2) image, e.g., foot.pgm")
    p.add_argument("--dt", type=float, default=0.2, help="Time step (suggested ≤ 0.25). Default: 0.2")
    p.add_argument("--linear", action="store_true", help="Run linear (heat equation) filtering")
    p.add_argument("--nonlinear", action="store_true", help="Run nonlinear (Perona–Malik) filtering")
    p.add_argument("--iters", type=int, nargs="+", default=[25], help="Iterations to run (you can pass multiple). Default: 25")
    p.add_argument("--lam", type=float, nargs="+", default=[10.0], help="Lambda(s) for PM. Default: 10.0")
    p.add_argument("--g", choices=["exp", "lorentz"], default="exp", help="Conductance function for PM. Default: exp")
    args = p.parse_args()

    # Load
    img_in, maxval = read_pgm_p2(args.input)
    # Keep original for writing later if needed
    base = os.path.splitext(args.input)[0]

    # Linear runs
    if args.linear:
        for it in args.iters:
            out = heat_equation(img_in, n_iter=it, dt=args.dt)
            out_path = f"{base}_linear_it{it}.pgm"
            write_pgm_p2(out_path, out, maxval=maxval, comment=f"Linear diffusion, iters={it}, dt={args.dt}")
            print(f"Wrote {out_path}")

    # Nonlinear runs
    if args.nonlinear:
        for it in args.iters:
            for lam in args.lam:
                out = perona_malik(img_in, n_iter=it, dt=args.dt, lam=lam, gtype=args.g)
                out_path = f"{base}_pm_g{args.g}_lam{lam}_it{it}.pgm"
                write_pgm_p2(out_path, out, maxval=maxval, comment=f"PM {args.g}, lam={lam}, iters={it}, dt={args.dt}")
                print(f"Wrote {out_path}")

    # Always save the input back out (for side-by-side comparison/reproducibility), if desired
    # out_in = f"{base}_input_copy.pgm"
    # write_pgm_p2(out_in, img_in, maxval=maxval, comment="Unmodified input copy")
    # print(f"Wrote {out_in}")


if __name__ == "__main__":
    main()

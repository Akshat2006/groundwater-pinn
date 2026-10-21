"""Finite-difference solver for 2D depth-averaged transient groundwater flow:

    S ∂h/∂t = ∇·(T ∇h) + R − Q

Units:
- x, y [m], t [days], h [m]
- T [m²/day], S [–]
- R, Q [m/day] (volume per unit area per day)

Discretisation:
- **Cell-centred grid.** Arrays are indexed [j, i], with j = row (south → north) and i = column (west → east).
- **Backward Euler in time.** It is unconditionally stable. The system matrix is constant, so it is
  LU-factorised once and each time step is just a forward/back substitution.
- **Face transmissivity** is the harmonic mean of the two neighbouring cells. This conserves flux across
  jumps in T, e.g. between zones.
- **Boundaries:** each edge is either no-flow or fixed head. A fixed head is imposed on the edge *face*,
  half a cell from the boundary cell's centre.

Forcing convention:
- R and Q may be scalars, (ny, nx) arrays, or callables f(t_mid) returning either.
- t_mid is the midpoint of the step in days, measured from the start of the run. It is evaluated as
  t0_days + (n + 0.5)·dt.

Water budget, recorded at every step (m³):
- storage change = sources (R − Q) + inflow across fixed-head edges
- `rel_error` is the leftover divided by the total gross flow in that step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

Forcing = Union[float, np.ndarray, Callable[[float], Union[float, np.ndarray]]]
EdgeValue = Union[None, float, np.ndarray]


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0          # west edge [m]
    y0: float = 0.0          # south edge [m]

    @property
    def xc(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self) -> np.ndarray:
        return self.y0 + (np.arange(self.ny) + 0.5) * self.dy

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.nx * self.dx, self.y0, self.y0 + self.ny * self.dy)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates X, Y with shape (ny, nx)."""
        return np.meshgrid(self.xc, self.yc)


@dataclass(frozen=True)
class BoundarySpec:
    """None = no flow. A float or 1D array = fixed head on that edge face.
    West/east arrays have length ny (south → north); south/north arrays have length nx (west → east)."""
    west: EdgeValue = None
    east: EdgeValue = None
    south: EdgeValue = None
    north: EdgeValue = None


@dataclass
class FDResult:
    times: np.ndarray                 # days, saved states (index 0 = initial state)
    heads: np.ndarray                 # (n_saved, ny, nx)
    budget: dict[str, np.ndarray] = field(default_factory=dict)   # per step, m³ (see module docstring)

    @property
    def h_final(self) -> np.ndarray:
        return self.heads[-1]


def _as_field(value, shape) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.broadcast_to(arr, shape) if arr.shape != shape else arr


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


class FDModel:
    """Assembles and factorises the backward-Euler system once; `run` then steps it forward."""

    def __init__(self, grid: Grid, S, T, bc: BoundarySpec, dt_days: float):
        self.grid, self.bc, self.dt = grid, bc, float(dt_days)
        ny, nx = grid.shape
        self.S = _as_field(S, grid.shape)
        self.T = _as_field(T, grid.shape)
        if np.any(self.S <= 0) or np.any(self.T <= 0):
            raise ValueError("S and T must be positive")
        n = nx * ny
        idx = np.arange(n).reshape(ny, nx)
        rows, cols, vals = [], [], []
        diag = np.zeros(n)

        def connect(k1, k2, c):          # symmetric conductance between cells k1 and k2
            rows.extend([k1, k2]); cols.extend([k2, k1]); vals.extend([-c, -c])
            np.add.at(diag, k1, c); np.add.at(diag, k2, c)

        cx = _harmonic(self.T[:, :-1], self.T[:, 1:]) * grid.dy / grid.dx       # x-faces
        connect(idx[:, :-1].ravel(), idx[:, 1:].ravel(), cx.ravel())
        cy = _harmonic(self.T[:-1, :], self.T[1:, :]) * grid.dx / grid.dy       # y-faces
        connect(idx[:-1, :].ravel(), idx[1:, :].ravel(), cy.ravel())
        rows = np.concatenate([np.atleast_1d(r) for r in rows])
        cols = np.concatenate([np.atleast_1d(c) for c in cols])
        vals = np.concatenate([np.atleast_1d(v) for v in vals])

        # fixed-head edges: conductance from the edge face to the boundary cell centre (half a cell)
        self.bc_cells, self.bc_cond, self.bc_head = [], [], []
        edges = {
            "west": (idx[:, 0], self.T[:, 0] * grid.dy / (grid.dx / 2), ny),
            "east": (idx[:, -1], self.T[:, -1] * grid.dy / (grid.dx / 2), ny),
            "south": (idx[0, :], self.T[0, :] * grid.dx / (grid.dy / 2), nx),
            "north": (idx[-1, :], self.T[-1, :] * grid.dx / (grid.dy / 2), nx),
        }
        for name, (cells, cond, length) in edges.items():
            value = getattr(bc, name)
            if value is None:
                continue
            head = np.broadcast_to(np.asarray(value, dtype=float), (length,))
            self.bc_cells.append(cells); self.bc_cond.append(cond); self.bc_head.append(head)
        if self.bc_cells:
            self.bc_cells = np.concatenate(self.bc_cells)
            self.bc_cond = np.concatenate(self.bc_cond)
            self.bc_head = np.concatenate(self.bc_head)
            np.add.at(diag, self.bc_cells, self.bc_cond)
        else:
            self.bc_cells = np.zeros(0, int); self.bc_cond = np.zeros(0); self.bc_head = np.zeros(0)

        self.storage_coef = (self.S * grid.cell_area / self.dt).ravel()      # m²/day
        K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
        self.matrix = (K + sp.diags(self.storage_coef)).tocsc()
        self._lu = splu(self.matrix)
        self.bc_rhs = np.zeros(n)
        np.add.at(self.bc_rhs, self.bc_cells, self.bc_cond * self.bc_head)

    def run(self, h0, R: Forcing, Q: Forcing, n_steps: int, t0_days: float = 0.0,
            save_every: int = 1, return_budget: bool = True, dtype=np.float64) -> FDResult:
        g, dt = self.grid, self.dt
        h = np.array(_as_field(h0, g.shape), dtype=float).ravel()
        n_saved = n_steps // save_every + 1
        heads = np.empty((n_saved,) + g.shape, dtype=dtype)
        times = t0_days + dt * save_every * np.arange(n_saved)
        heads[0] = h.reshape(g.shape)
        budget = {k: np.empty(n_steps) for k in
                  ("storage", "recharge", "pumping", "boundary", "residual", "rel_error")} if return_budget else {}
        area = g.cell_area

        def evaluate(f, t):
            v = f(t) if callable(f) else f
            return _as_field(v, g.shape).ravel()

        for n in range(n_steps):
            t_mid = t0_days + (n + 0.5) * dt
            r, q = evaluate(R, t_mid), evaluate(Q, t_mid)
            rhs = self.storage_coef * h + self.bc_rhs + (r - q) * area
            h_new = self._lu.solve(rhs)
            if return_budget:
                d_storage = float(np.sum(self.storage_coef * (h_new - h)) * dt)
                rech = float(np.sum(r) * area * dt)
                pump = float(np.sum(q) * area * dt)
                bflux = self.bc_cond * (self.bc_head - h_new[self.bc_cells]) * dt
                bnd = float(np.sum(bflux))
                resid = d_storage - (rech - pump) - bnd
                gross = abs(d_storage) + abs(rech) + abs(pump) + float(np.sum(np.abs(bflux)))
                budget["storage"][n], budget["recharge"][n], budget["pumping"][n] = d_storage, rech, pump
                budget["boundary"][n], budget["residual"][n] = bnd, resid
                budget["rel_error"][n] = abs(resid) / gross if gross > 0 else 0.0
            h = h_new
            if (n + 1) % save_every == 0:
                heads[(n + 1) // save_every] = h.reshape(g.shape)
        return FDResult(times=times, heads=heads, budget=budget)


def solve_fd(grid: Grid, S, T, R: Forcing, Q: Forcing, h0, bc: BoundarySpec, dt_days: float,
             n_steps: int, t0_days: float = 0.0, save_every: int = 1, return_budget: bool = True,
             dtype=np.float64) -> FDResult:
    """One-call interface (PLAN.md §4.2). Use FDModel directly to reuse a factorisation across runs."""
    model = FDModel(grid, S, T, bc, dt_days)
    return model.run(h0, R, Q, n_steps, t0_days=t0_days, save_every=save_every,
                     return_budget=return_budget, dtype=dtype)

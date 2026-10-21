"""Exact solutions of S ∂h/∂t = T ∇²h + R − Q, used to test the finite-difference solver (PLAN.md §6.3).
Units: m, days, m²/day, m/day."""
from __future__ import annotations

import numpy as np
from scipy.special import exp1


def sine_mode(x, y, t, L: float, S: float, T: float, h0: float, amp: float):
    """Square domain [0, L]², head fixed at h0 on all four edges, R = Q = 0.
    A single sine bump decays exponentially at rate 2π²(T/S)/L²."""
    decay = np.exp(-2.0 * np.pi**2 * (T / S) * t / L**2)
    return h0 + amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L) * decay


def closed_box(t, h0: float, S: float, net_source: float):
    """No-flow edges and uniform constant R − Q = net_source [m/day]: the head rises (or falls) uniformly."""
    return h0 + net_source * t / S


def linear_steady(x, h_west: float, h_east: float, L: float):
    """Fixed heads on the west/east edges, no flow north/south, no forcing: a straight-line head profile."""
    return h_west + (h_east - h_west) * x / L


def theis_drawdown(r, t, rate_m3_day: float, S: float, T: float):
    """Theis (1935) drawdown [m] at distance r [m] and time t [days] from a well pumping at a constant rate
    in an infinite, uniform aquifer: s = Q/(4πT)·W(u), with u = r²S/(4Tt) and W the exponential integral."""
    u = np.asarray(r, dtype=float) ** 2 * S / (4.0 * T * t)
    return rate_m3_day / (4.0 * np.pi * T) * exp1(u)

"""
RietveldApp – Standalone Python Rietveld Refinement Suite (single file)
=======================================================================
Run with:
    pip install streamlit numpy scipy plotly pandas
    streamlit run rietveld_single.py
"""
# ============================================================
#  Standard imports
# ============================================================
import re
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import plotly.graph_objects as go
import streamlit as st

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ============================================================
#  SECTION 1 – Crystal Structure
# ============================================================

@dataclass
class UnitCell:
    a: float = 5.0;  b: float = 5.0;  c: float = 5.0
    alpha: float = 90.0;  beta: float = 90.0;  gamma: float = 90.0

    def metric_tensor(self) -> np.ndarray:
        a, b, c = self.a, self.b, self.c
        ca = np.cos(np.radians(self.alpha))
        cb = np.cos(np.radians(self.beta))
        cg = np.cos(np.radians(self.gamma))
        return np.array([[a*a, a*b*cg, a*c*cb],
                         [a*b*cg, b*b, b*c*ca],
                         [a*c*cb, b*c*ca, c*c]])

    def volume(self) -> float:
        return np.sqrt(max(np.linalg.det(self.metric_tensor()), 0.0))

    def d_spacing(self, h, k, l) -> float:
        Ginv = np.linalg.inv(self.metric_tensor())
        hkl = np.array([h, k, l], dtype=float)
        q2 = hkl @ Ginv @ hkl
        return 1.0 / np.sqrt(q2) if q2 > 0 else np.inf

    def two_theta(self, h, k, l, wavelength) -> Optional[float]:
        d = self.d_spacing(h, k, l)
        val = wavelength / (2.0 * d)
        if abs(val) > 1.0: return None
        return 2.0 * np.degrees(np.arcsin(val))


@dataclass
class AtomSite:
    label: str = "Fe1";  element: str = "Fe"
    x: float = 0.0;  y: float = 0.0;  z: float = 0.0
    occ: float = 1.0;  Biso: float = 0.5


@dataclass
class Phase:
    name: str = "Phase_1"
    cell: UnitCell = field(default_factory=UnitCell)
    atoms: List[AtomSite] = field(default_factory=list)
    spacegroup: str = "P 1"
    scale: float = 1.0;  eta: float = 0.5
    U: float = 0.01;  V: float = -0.005;  W: float = 0.002
    X: float = 0.0;  Y: float = 0.0
    pref_orient: bool = False;  pref_r: float = 1.0
    pref_hkl: Tuple[int, int, int] = (0, 0, 1)
    refine_cell: bool = True;  refine_scale: bool = True

    def fwhm_tch(self, tt_deg: float) -> float:
        theta = np.radians(tt_deg / 2.0)
        t, c = np.tan(theta), np.cos(theta)
        fg = np.sqrt(max(self.U*t**2 + self.V*t + self.W, 1e-8))
        fl = self.X / c + self.Y * t
        f5 = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
              + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5)
        return max(f5**0.2, 1e-4)

    def eta_tch(self, tt_deg: float) -> float:
        fwhm = self.fwhm_tch(tt_deg)
        theta = np.radians(tt_deg / 2.0)
        fl = self.X / np.cos(theta) + self.Y * np.tan(theta)
        r = fl / fwhm
        return float(np.clip(1.36603*r - 0.47719*r**2 + 0.11116*r**3, 0.0, 1.0))


# Cromer-Mann coefficients
_CM: Dict[str, tuple] = {
    "H":  (0.493,10.511,0.323,26.126,0.140,3.142,0.041,57.800,0.003),
    "C":  (2.310,20.844,1.020,10.208,1.589,0.569,0.865,51.651,0.216),
    "N":  (12.213,0.006,3.132,9.893,2.013,28.997,1.166,0.583,-11.529),
    "O":  (3.049,13.277,2.287,5.701,1.546,0.324,0.867,32.909,0.251),
    "Al": (6.420,3.039,1.900,0.743,1.594,31.547,1.964,85.089,1.115),
    "Si": (6.292,2.439,3.035,32.334,1.989,0.678,1.541,81.694,1.141),
    "Fe": (11.770,4.761,7.069,0.307,3.565,15.352,2.322,76.880,1.036),
    "Cu": (13.338,3.583,7.168,0.247,5.616,11.396,1.674,64.812,1.191),
    "Zn": (14.074,3.266,7.032,0.233,5.165,10.316,2.410,58.710,1.304),
    "La": (20.578,2.948,19.599,0.244,11.373,18.773,3.288,133.124,2.146),
    "Ti": (9.759,7.851,7.356,0.500,3.586,35.634,1.491,116.105,1.170),
    "Ba": (19.919,0.021,19.013,5.985,11.858,20.866,4.865,134.685,2.101),
}
_Z_FALLBACK = {"Li":3,"Na":11,"Mg":12,"Mn":25,"Ni":28,"Co":27,"Ca":20,"K":19,"Cl":17,"S":16,"P":15}


def atomic_f0(element: str, s: float) -> float:
    s2 = s * s
    if element in _CM:
        a1,b1,a2,b2,a3,b3,a4,b4,c = _CM[element]
        return a1*np.exp(-b1*s2)+a2*np.exp(-b2*s2)+a3*np.exp(-b3*s2)+a4*np.exp(-b4*s2)+c
    return float(_Z_FALLBACK.get(element, 20)) * np.exp(-2.0*s2)


def structure_factor(phase: Phase, h, k, l, wavelength) -> complex:
    d = phase.cell.d_spacing(h, k, l)
    s = 1.0 / (2.0 * d) if d > 0 else 0.0
    F = 0j
    for atom in phase.atoms:
        f0 = atomic_f0(atom.element, s)
        DW = np.exp(-atom.Biso * s * s)
        F += atom.occ * f0 * DW * np.exp(2j * np.pi * (h*atom.x + k*atom.y + l*atom.z))
    return F


def generate_reflections(phase: Phase, wavelength: float,
                         tt_min=5.0, tt_max=80.0) -> List[dict]:
    cell = phase.cell
    d_min = wavelength / (2.0 * np.sin(np.radians(tt_max / 2.0)))
    d_max = wavelength / (2.0 * np.sin(np.radians(tt_min / 2.0)))
    hmax = int(cell.a / d_min) + 2
    kmax = int(cell.b / d_min) + 2
    lmax = int(cell.c / d_min) + 2
    refs, seen = [], set()
    for h in range(-hmax, hmax+1):
        for k in range(-kmax, kmax+1):
            for l in range(-lmax, lmax+1):
                if h == k == l == 0: continue
                d = cell.d_spacing(h, k, l)
                if not (d_min <= d <= d_max): continue
                tt = cell.two_theta(h, k, l, wavelength)
                if tt is None or not (tt_min <= tt <= tt_max): continue
                F = structure_factor(phase, h, k, l, wavelength)
                I = abs(F)**2
                if I < 1e-6: continue
                key = tuple(sorted([abs(h), abs(k), abs(l)]))
                if key in seen: continue
                seen.add(key)
                refs.append({"h": h, "k": k, "l": l, "d": d,
                             "two_theta": tt, "F2": I, "mult": 8, "I_calc": I*8})
    refs.sort(key=lambda r: r["two_theta"])
    return refs


# ============================================================
#  SECTION 2 – Peak Profile Functions
# ============================================================

def _gauss(x, x0, fwhm):
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5*((x-x0)/sig)**2) / (sig * np.sqrt(2.0*np.pi))

def _lorentz(x, x0, fwhm):
    g = fwhm / 2.0
    return (1.0/np.pi) * g / ((x-x0)**2 + g**2)

def _mod_lorentz(x, x0, fwhm):
    g = fwhm / 2.0
    return (2.0/(np.pi*g)) / (1.0+((x-x0)/g)**2)**2

def _pv(x, x0, fwhm, eta):
    eta = float(np.clip(eta, 0, 1))
    return eta*_lorentz(x, x0, fwhm) + (1-eta)*_gauss(x, x0, fwhm)

def _p7(x, x0, fwhm, m):
    m = max(m, 0.5)
    g = fwhm / (2.0 * np.sqrt(2.0**(1.0/m)-1.0))
    norm = _gamma(m) / (_gamma(m-0.5) * g * np.sqrt(np.pi))
    return norm * (1.0+((x-x0)/g)**2)**(-m)

def _tch(x, x0, fg, fl):
    f5 = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
          + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5)
    fwhm = f5**0.2
    if fwhm < 1e-10: return np.zeros_like(x)
    r = fl / fwhm
    eta = float(np.clip(1.36603*r - 0.47719*r**2 + 0.11116*r**3, 0, 1))
    return _pv(x, x0, fwhm, eta)

def _split_pv(x, x0, fwhm_l, fwhm_r, eta_l, eta_r):
    result = np.where(x <= x0, _pv(x, x0, fwhm_l, eta_l), _pv(x, x0, fwhm_r, eta_r))
    area = _trapz(result, x) if len(x) > 1 else 1.0
    return result / max(area, 1e-20)

PROFILE_TYPES = ["TCH Pseudo-Voigt", "Pseudo-Voigt", "Gaussian",
                 "Lorentzian", "Modified Lorentzian", "Pearson-VII", "Split Pseudo-Voigt"]

def profile_peak(x, x0, fwhm, profile_type="TCH Pseudo-Voigt",
                 eta=0.5, m=2.0, fwhm_g=None, fwhm_l=None,
                 fwhm_right=None, eta_right=None):
    if profile_type == "Gaussian":         return _gauss(x, x0, fwhm)
    elif profile_type == "Lorentzian":     return _lorentz(x, x0, fwhm)
    elif profile_type == "Modified Lorentzian": return _mod_lorentz(x, x0, fwhm)
    elif profile_type == "Pseudo-Voigt":   return _pv(x, x0, fwhm, eta)
    elif profile_type == "Pearson-VII":    return _p7(x, x0, fwhm, m)
    elif profile_type == "TCH Pseudo-Voigt":
        fg = fwhm_g if fwhm_g is not None else fwhm*(1-eta)
        fl = fwhm_l if fwhm_l is not None else fwhm*eta
        return _tch(x, x0, max(fg, 1e-8), max(fl, 1e-8))
    elif profile_type == "Split Pseudo-Voigt":
        fr = fwhm_right if fwhm_right is not None else fwhm
        er = eta_right if eta_right is not None else eta
        return _split_pv(x, x0, fwhm, fr, eta, er)
    return _gauss(x, x0, fwhm)


# ============================================================
#  SECTION 3 – Background Models
# ============================================================

BG_MODES = ["Chebyshev", "Polynomial", "Interpolated points", "Fourier filter", "Fixed (zero)"]


def _cheb_bg(x, coeffs):
    if len(x) == 0: return np.zeros_like(x)
    xn = 2.0*(x - x.min()) / max(x.max()-x.min(), 1e-10) - 1.0
    bg = np.zeros_like(x, dtype=float)
    Tp, Tc = np.ones_like(x), xn.copy()
    for i, c in enumerate(coeffs):
        if i == 0:   bg += c * Tp
        elif i == 1: bg += c * Tc
        else:
            Tn = 2.0*xn*Tc - Tp;  bg += c*Tn;  Tp, Tc = Tc, Tn
    return bg

def _poly_bg(x, coeffs):
    bg = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs): bg += c * x**i
    return bg

def _interp_bg(x, xp, yp):
    if len(xp) < 2: return np.full_like(x, yp[0] if len(yp) else 0.0)
    idx = np.argsort(xp)
    return CubicSpline(np.array(xp)[idx], np.array(yp)[idx], extrapolate=True)(x)

def _fourier_bg(y, cutoff=0.08):
    Y = np.fft.rfft(y)
    cut = max(1, int(cutoff * len(Y)))
    Yf = np.zeros_like(Y);  Yf[:cut] = Y[:cut]
    return np.fft.irfft(Yf, n=len(y))


class Background:
    def __init__(self, mode="Chebyshev", n_coeffs=6):
        self.mode = mode
        self.coeffs = [0.0] * n_coeffs
        self.x_points: List[float] = []
        self.y_points: List[float] = []
        self.fourier_cutoff = 0.08
        self.refine = True

    def evaluate(self, x, y_obs=None):
        if self.mode == "Fixed (zero)":        return np.zeros_like(x)
        elif self.mode == "Chebyshev":         return _cheb_bg(x, self.coeffs)
        elif self.mode == "Polynomial":        return _poly_bg(x, self.coeffs)
        elif self.mode == "Interpolated points":
            if len(self.x_points) >= 2:
                return _interp_bg(x, self.x_points, self.y_points)
            return np.zeros_like(x)
        elif self.mode == "Fourier filter":
            return _fourier_bg(y_obs, self.fourier_cutoff) if y_obs is not None else np.zeros_like(x)
        return np.zeros_like(x)


# ============================================================
#  SECTION 4 – Pattern Calculator
# ============================================================

def _lp(tt_deg, mono=False):
    tt = np.radians(tt_deg);  th = tt/2.0
    sin_t, cos_t, cos2t = np.sin(th), np.cos(th), np.cos(tt)
    if abs(sin_t) < 1e-10: return 1.0
    LP = (1.0 + cos2t**2) / (2.0 * sin_t**2 * cos_t)
    if mono: LP *= (1.0 + cos2t**2) / 2.0
    return LP

def _absorption(tt_deg, muR):
    th = np.radians(tt_deg/2.0)
    return np.exp(-muR / np.sin(th)) if np.sin(th) > 1e-10 else 1.0


def calc_pattern(x_grid, phases, wavelength, background,
                 profile_type="TCH Pseudo-Voigt", monochromator=False,
                 mu_R=0.0, peak_cutoff_fwhm=10.0, y_obs=None):
    y_calc = background.evaluate(x_grid, y_obs).copy()
    all_refs = []
    for phase in phases:
        refs = generate_reflections(phase, wavelength,
                                    tt_min=max(float(x_grid.min()), 1.0),
                                    tt_max=min(float(x_grid.max()), 170.0))
        for ref in refs:
            tt = ref["two_theta"]
            fwhm = phase.fwhm_tch(tt)
            eta  = phase.eta_tch(tt)
            intensity = phase.scale * ref["I_calc"] * _lp(tt, monochromator)
            if mu_R > 0: intensity *= _absorption(tt, mu_R)
            half = peak_cutoff_fwhm * fwhm
            mask = np.abs(x_grid - tt) < half
            if not np.any(mask): continue
            pk = profile_peak(x_grid[mask], tt, fwhm, profile_type=profile_type,
                              eta=eta, fwhm_g=fwhm*(1-eta), fwhm_l=fwhm*eta)
            if np.sum(mask) > 1:
                pk = pk * np.mean(np.diff(x_grid[mask]))
            y_calc[mask] += intensity * pk
            ref["phase_name"] = phase.name
            all_refs.append(ref)
    return y_calc, all_refs


# ============================================================
#  SECTION 5 – Refinement Engine
# ============================================================

def _r_wp(yo, yc, w): return 100.0*np.sqrt(np.sum(w*(yo-yc)**2)/max(np.sum(w*yo**2), 1e-20))
def _r_p(yo, yc):    return 100.0*np.sum(np.abs(yo-yc))/max(np.sum(np.abs(yo)), 1e-20)
def _r_exp(yo, w, p):
    return 100.0*np.sqrt((len(yo)-p)/max(np.sum(w*yo**2), 1e-20))


@dataclass
class RefinementResult:
    success: bool;  message: str;  n_iter: int;  cost: float
    r_wp: float;  r_p: float;  r_exp: float;  gof: float
    parameters: Dict[str, float]
    y_calc: np.ndarray;  residual: np.ndarray;  reflections: List[dict]


def _build_ps(phases, bg, ref_bg, ref_prof, ref_cell, ref_scale):
    ps: Dict[str, dict] = {}

    def add(name, val, lo=-np.inf, hi=np.inf, refine=True):
        ps[name] = {"val": float(val), "lo": lo, "hi": hi, "ref": refine}

    if bg.mode in ("Chebyshev", "Polynomial"):
        for i, c in enumerate(bg.coeffs):
            add(f"bg_{i}", c, -1e6, 1e6, ref_bg and bg.refine)

    for pi, ph in enumerate(phases):
        p = f"ph{pi}"
        add(f"{p}_scale", ph.scale, 1e-12, 1e12, ref_scale and ph.refine_scale)
        add(f"{p}_a", ph.cell.a, 0.5, 50.0, ref_cell and ph.refine_cell)
        add(f"{p}_b", ph.cell.b, 0.5, 50.0, ref_cell and ph.refine_cell)
        add(f"{p}_c", ph.cell.c, 0.5, 50.0, ref_cell and ph.refine_cell)
        add(f"{p}_alpha", ph.cell.alpha, 60., 120., False)
        add(f"{p}_beta",  ph.cell.beta,  60., 120., False)
        add(f"{p}_gamma", ph.cell.gamma, 60., 120., False)
        add(f"{p}_U", ph.U, 0.0, 1.0, ref_prof)
        add(f"{p}_V", ph.V, -0.5, 0.5, ref_prof)
        add(f"{p}_W", ph.W, 1e-6, 0.5, ref_prof)
        add(f"{p}_X", ph.X, 0.0, 1.0, ref_prof)
        add(f"{p}_Y", ph.Y, 0.0, 1.0, ref_prof)
    return ps


def _apply_ps(ps, phases, bg):
    if bg.mode in ("Chebyshev", "Polynomial"):
        for i in range(len(bg.coeffs)):
            k = f"bg_{i}"
            if k in ps: bg.coeffs[i] = ps[k]["val"]
    for pi, ph in enumerate(phases):
        p = f"ph{pi}"
        ph.scale = ps[f"{p}_scale"]["val"]
        ph.cell.a = ps[f"{p}_a"]["val"];  ph.cell.b = ps[f"{p}_b"]["val"]
        ph.cell.c = ps[f"{p}_c"]["val"];  ph.cell.alpha = ps[f"{p}_alpha"]["val"]
        ph.cell.beta = ps[f"{p}_beta"]["val"];  ph.cell.gamma = ps[f"{p}_gamma"]["val"]
        ph.U = ps[f"{p}_U"]["val"];  ph.V = ps[f"{p}_V"]["val"]
        ph.W = ps[f"{p}_W"]["val"];  ph.X = ps[f"{p}_X"]["val"]
        ph.Y = ps[f"{p}_Y"]["val"]


def run_refinement(x_obs, y_obs, phases, background,
                   wavelength=1.5406, profile_type="TCH Pseudo-Voigt",
                   weighting="standard", refine_bg=True, refine_profile=True,
                   refine_cell=True, refine_scale=True, le_bail=False,
                   max_iter=50, damping=1.0,
                   progress_callback: Optional[Callable] = None) -> RefinementResult:
    if weighting == "standard":
        w = 1.0 / np.maximum(y_obs, 1.0)
    elif weighting == "ml":
        w = 1.0 / (np.maximum(y_obs, 1.0) + 1.0)
    else:
        w = np.ones_like(y_obs)

    ps = _build_ps(phases, background, refine_bg, refine_profile, refine_cell, refine_scale)
    active = [k for k, v in ps.items() if v["ref"]]
    n_params = len(active)
    iteration = [0]

    def residuals(p_vec):
        for k, v in zip(active, p_vec): ps[k]["val"] = float(v)
        _apply_ps(ps, phases, background)
        yc, _ = calc_pattern(x_obs, phases, wavelength, background,
                              profile_type=profile_type, y_obs=y_obs)
        res = np.sqrt(w) * (y_obs - yc)
        iteration[0] += 1
        if progress_callback:
            progress_callback(iteration[0], float(np.sum(res**2)))
        return res

    p0 = np.array([ps[k]["val"] for k in active])
    lo = np.array([ps[k]["lo"] for k in active])
    hi = np.array([ps[k]["hi"] for k in active])

    try:
        res = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                            ftol=1e-8, xtol=1e-8, gtol=1e-8,
                            max_nfev=max_iter*max(len(p0),1)*2, verbose=0)
        for k, v in zip(active, res.x): ps[k]["val"] = float(v)
        _apply_ps(ps, phases, background)
        success, message = res.success or res.cost < 1e-6, res.message
    except Exception as e:
        success, message = False, str(e)
        res = type("R", (), {"nfev": 0, "cost": np.inf})()

    yc_final, refs_final = calc_pattern(x_obs, phases, wavelength, background,
                                         profile_type=profile_type, y_obs=y_obs)
    rwp  = _r_wp(y_obs, yc_final, w)
    rp   = _r_p(y_obs, yc_final)
    rexp = _r_exp(y_obs, w, n_params)
    s    = rwp / max(rexp, 1e-20)
    params_out = {k: v["val"] for k, v in ps.items()}

    return RefinementResult(success=success, message=message, n_iter=iteration[0],
                            cost=float(getattr(res, "cost", np.inf)),
                            r_wp=rwp, r_p=rp, r_exp=rexp, gof=s,
                            parameters=params_out, y_calc=yc_final,
                            residual=y_obs - yc_final, reflections=refs_final)


# ============================================================
#  SECTION 6 – I/O (data files + CIF)
# ============================================================

def _strip_comments(lines):
    return [l.strip() for l in lines if l.strip() and l.strip()[0] not in "#!;"]

def read_powder_data(filename, text):
    ext = filename.rsplit(".", 1)[-1].lower()
    lines = text.splitlines()
    data = []
    for line in (lines if ext != "fxye" else lines):
        if ext == "fxye" and (line.strip().upper().startswith("BANK") or line.strip().startswith("#")):
            continue
        parts = line.split()
        try:
            row = [float(p) for p in parts[:3]]
            if len(row) >= 2: data.append(row)
        except ValueError:
            continue
    if not data: raise ValueError("No numeric data found.")
    arr = np.array(data)
    x = arr[:, 0] / (100.0 if ext == "fxye" else 1.0)
    y = arr[:, 1]
    sig = arr[:, 2] if arr.shape[1] >= 3 else None
    return x, y, sig


def _cif_val(v):
    return re.sub(r"\(\d+\)$", "", v.strip().strip("'\""))

def _try_f(s):
    try: return float(_cif_val(s))
    except: return None

def parse_cif(text) -> List[Phase]:
    phases = []
    for block in re.split(r"(?m)^data_", text):
        if not block.strip(): continue
        lines = block.splitlines()
        kv, loop_data = {}, {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.lower() == "loop_":
                hdrs, rows, i = [], [], i+1
                while i < len(lines) and lines[i].strip().startswith("_"):
                    hdrs.append(lines[i].strip().lower()); i += 1
                while i < len(lines):
                    lx = lines[i].strip()
                    if lx.startswith("_") or lx.lower() == "loop_" or lx.startswith("data_"): break
                    if lx and not lx.startswith("#"): rows.append(lx.split())
                    i += 1
                for hi, h in enumerate(hdrs):
                    loop_data[h] = [r[hi] if hi < len(r) else "." for r in rows]
                continue
            if line.startswith("_"):
                parts = line.split(None, 1); key = parts[0].lower()
                kv[key] = parts[1].strip() if len(parts) == 2 else (lines[i+1].strip() if i+1 < len(lines) else "")
                if len(parts) < 2: i += 1
            i += 1

        cell = UnitCell(
            a=_try_f(kv.get("_cell_length_a","5")) or 5.0,
            b=_try_f(kv.get("_cell_length_b","5")) or 5.0,
            c=_try_f(kv.get("_cell_length_c","5")) or 5.0,
            alpha=_try_f(kv.get("_cell_angle_alpha","90")) or 90.0,
            beta= _try_f(kv.get("_cell_angle_beta","90")) or 90.0,
            gamma=_try_f(kv.get("_cell_angle_gamma","90")) or 90.0)
        sg = kv.get("_symmetry_space_group_name_h-m",
             kv.get("_space_group_name_h-m_alt",
             kv.get("_symmetry_int_tables_number", "P 1")))
        atoms = []
        lk = next((k for k in loop_data if "label" in k and "atom_site" in k), None)
        tk = next((k for k in loop_data if "type_symbol" in k), None)
        xk = next((k for k in loop_data if "fract_x" in k), None)
        yk = next((k for k in loop_data if "fract_y" in k), None)
        zk = next((k for k in loop_data if "fract_z" in k), None)
        ok = next((k for k in loop_data if "occupancy" in k), None)
        bk = next((k for k in loop_data if "b_iso" in k or "b_eq" in k or "u_iso" in k), None)
        if lk and xk and yk and zk:
            for j in range(len(loop_data[lk])):
                lbl = loop_data[lk][j]
                elem = loop_data[tk][j] if tk else re.sub(r"[^A-Za-z]","",lbl)[:2]
                xf = _try_f(loop_data[xk][j]) or 0.0
                yf = _try_f(loop_data[yk][j]) or 0.0
                zf = _try_f(loop_data[zk][j]) or 0.0
                occ  = _try_f(loop_data[ok][j]) if ok else 1.0; occ = occ or 1.0
                biso = _try_f(loop_data[bk][j]) if bk else 0.5; biso = biso or 0.5
                if bk and "u_iso" in bk: biso *= 8*np.pi**2
                atoms.append(AtomSite(label=lbl, element=elem.capitalize(),
                                       x=xf, y=yf, z=zf, occ=occ, Biso=biso))
        name = lines[0].strip().split("_",1)[-1][:30] if lines else "Phase"
        phases.append(Phase(name=name or "Phase", cell=cell, atoms=atoms,
                            spacegroup=_cif_val(sg)))
    return [p for p in phases if p.cell.a > 0]


# ============================================================
#  SECTION 7 – Plotting
# ============================================================

_COLORS = ["#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0",
           "#00BCD4","#795548","#607D8B","#E91E63","#CDDC39"]

def make_pattern_figure(x_obs=None, y_obs=None, y_calc=None, y_bg=None,
                        residual=None, reflections=None,
                        x_label="2θ (°)", title="Diffraction Pattern", log_scale=False):
    fig = go.Figure()
    if x_obs is not None and y_obs is not None:
        fig.add_trace(go.Scatter(x=x_obs, y=y_obs, mode="markers",
                                  marker=dict(size=2, color="#333"), name="Observed"))
    if x_obs is not None and y_calc is not None:
        fig.add_trace(go.Scatter(x=x_obs, y=y_calc, mode="lines",
                                  line=dict(color="#E53935", width=1.5), name="Calculated"))
    if x_obs is not None and y_bg is not None:
        fig.add_trace(go.Scatter(x=x_obs, y=y_bg, mode="lines",
                                  line=dict(color="#1E88E5", width=1, dash="dash"), name="Background"))
    if x_obs is not None and residual is not None:
        off = -(np.max(y_obs)*0.15 if y_obs is not None else 0.0)
        fig.add_trace(go.Scatter(x=x_obs, y=residual+off, mode="lines",
                                  line=dict(color="#43A047", width=1), name="Difference"))
    if reflections:
        seen_ph = {}
        for r in reflections:
            ph = r.get("phase_name","Phase")
            if ph not in seen_ph: seen_ph[ph] = len(seen_ph)
        for ph, idx in seen_ph.items():
            pr = [r for r in reflections if r.get("phase_name","") == ph]
            ty = -(np.max(y_obs)*0.05 if y_obs is not None else 100)
            fig.add_trace(go.Scatter(
                x=[r["two_theta"] for r in pr], y=[ty]*len(pr), mode="markers",
                marker=dict(symbol="line-ns", size=8, color=_COLORS[idx%len(_COLORS)],
                            line=dict(width=1, color=_COLORS[idx%len(_COLORS)])),
                name=f"Ticks: {ph}", text=[f"({r['h']}{r['k']}{r['l']})" for r in pr],
                hoverinfo="x+text"))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Intensity",
                      hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=60,r=20,t=50,b=50),
                      yaxis=dict(type="log" if log_scale else "linear", gridcolor="#EEE"),
                      xaxis=dict(gridcolor="#EEE"))
    return fig


def make_crystal_view(phase):
    cell = phase.cell
    a, b, c = cell.a, cell.b, cell.c
    ca = np.cos(np.radians(cell.alpha));  cb = np.cos(np.radians(cell.beta))
    cg = np.cos(np.radians(cell.gamma));  sg = np.sin(np.radians(cell.gamma))
    v = cell.volume()
    M = np.array([[a, b*cg, c*cb],
                  [0, b*sg, c*(ca-cb*cg)/sg],
                  [0, 0,    v/(a*b*sg)]])
    def f2c(fxyz): return M @ np.array(fxyz)
    corners = list(itertools.product([0,1],[0,1],[0,1]))
    ex, ey, ez = [], [], []
    for v1 in corners:
        for v2 in corners:
            if sum(abs(aa-bb) for aa,bb in zip(v1,v2)) == 1:
                p1, p2 = f2c(v1), f2c(v2)
                ex += [p1[0],p2[0],None];  ey += [p1[1],p2[1],None];  ez += [p1[2],p2[2],None]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="gray", width=2), name="Unit cell"))
    elem_col = {"Fe":"#B71C1C","O":"#1565C0","Si":"#FFA000","Al":"#6A1B9A",
                "C":"#212121","N":"#0D47A1","Cu":"#BF360C","Ti":"#004D40","H":"#F9A825"}
    for atom in phase.atoms:
        cart = f2c([atom.x, atom.y, atom.z])
        fig.add_trace(go.Scatter3d(x=[cart[0]], y=[cart[1]], z=[cart[2]],
                                   mode="markers+text", name=atom.label,
                                   marker=dict(size=8, color=elem_col.get(atom.element,"#607D8B"), opacity=0.85),
                                   text=[atom.label], textposition="top center"))
    fig.update_layout(scene=dict(xaxis_title="x(Å)", yaxis_title="y(Å)", zaxis_title="z(Å)"),
                      title=f"Structure: {phase.name}  [{phase.spacegroup}]",
                      margin=dict(l=0,r=0,b=0,t=40), showlegend=False)
    return fig


# ============================================================
#  SECTION 8 – Demo Data
# ============================================================

def make_demo(name):
    np.random.seed(42)
    x = np.linspace(10, 80, 3500)
    if "hematite" in name or "Fe" in name:
        peaks = [(24.14,800),(33.15,6000),(35.61,3000),(40.85,800),(49.48,2500),
                 (54.09,4000),(57.60,1800),(62.45,5500),(64.02,2000),(71.95,1200)]
        bg_level = 200
    elif "LaB" in name:
        d0, wl = 4.1569, 1.54056
        peaks = []
        for hkl in [(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,1,0),(2,1,1),(2,2,0),(3,0,0)]:
            h,k,l = hkl
            dh = d0/np.sqrt(h**2+k**2+l**2)
            v = wl/(2*dh)
            if abs(v)<=1: peaks.append((2*np.degrees(np.arcsin(v)), 10000))
        bg_level = 100
    elif "Quartz" in name:
        peaks = [(20.85,2500),(26.64,10000),(36.55,1800),(39.47,1200),(40.29,900),
                 (42.45,1500),(45.79,800),(50.14,1600),(54.88,900),(59.96,1200)]
        bg_level = 150
    else:
        y = np.random.poisson(500, len(x)).astype(float)
        return x, y, np.sqrt(np.maximum(y,1))
    y = np.full(len(x), float(bg_level)) + np.random.normal(0, 10, len(x))
    for tt, I in peaks:
        sig = 0.12/(2*np.sqrt(2*np.log(2)))
        y += I * np.exp(-0.5*((x-tt)/sig)**2)
    y = np.random.poisson(np.maximum(y,0)).astype(float)
    return x, y, np.sqrt(np.maximum(y,1))


def default_phase(name="Phase_1"):
    cell = UnitCell(a=2.8664, b=2.8664, c=2.8664)
    atoms = [AtomSite("Fe1","Fe",0,0,0), AtomSite("Fe2","Fe",0.5,0.5,0.5)]
    return Phase(name=name, cell=cell, atoms=atoms, spacegroup="Im-3m",
                 U=0.01, V=-0.004, W=0.002)


# ============================================================
#  SECTION 9 – Streamlit UI helpers
# ============================================================

def _rfactors_row(y_obs, y_calc, ncols=3):
    w = 1.0/np.maximum(y_obs,1.0)
    cols = st.columns(ncols)
    cols[0].metric("Rwp",  f"{_r_wp(y_obs,y_calc,w):.2f}%")
    cols[1].metric("Rp",   f"{_r_p(y_obs,y_calc):.2f}%")
    cols[2].metric("Δ max",f"{np.max(np.abs(y_obs-y_calc)):.0f}")


def _rfactors_result(r):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rwp", f"{r.r_wp:.2f}%");  c2.metric("Rp",   f"{r.r_p:.2f}%")
    c3.metric("Rexp",f"{r.r_exp:.2f}%"); c4.metric("GoF",  f"{r.gof:.3f}")


def _make_report(r):
    lines = ["="*60,"  RIETVELD REFINEMENT REPORT","="*60,
             f"  Rwp  = {r.r_wp:.4f} %", f"  Rp   = {r.r_p:.4f} %",
             f"  Rexp = {r.r_exp:.4f} %", f"  GoF  = {r.gof:.4f}",
             f"  Cycles = {r.n_iter}", f"  Converged: {r.success}",
             "", "  REFINED PARAMETERS", "-"*60]
    for k,v in r.parameters.items(): lines.append(f"  {k:<30s}  {v:.8g}")
    lines += ["","  REFLECTIONS","-"*60,
              f"  {'Phase':<15}{'h':>4}{'k':>4}{'l':>4}  {'d/A':>8}  {'2th':>8}  {'I_calc':>12}"]
    for ref in r.reflections:
        lines.append(f"  {ref.get('phase_name',''):<15}{ref['h']:>4}{ref['k']:>4}{ref['l']:>4}"
                     f"  {ref['d']:>8.4f}  {ref['two_theta']:>8.3f}  {ref['I_calc']:>12.1f}")
    return "\n".join(lines)


# ============================================================
#  SECTION 10 – Page: Data Import
# ============================================================

def page_data():
    st.header("📂 Data Import & Pattern Visualization")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Import Data")
        uploaded = st.file_uploader("Upload diffraction data",
                                     type=["xy","xye","dat","txt","csv","fxye","esg"])
        st.markdown("**— or —**")
        demo = st.selectbox("Load demo dataset",
                             ["(none)","α-Fe₂O₃ (hematite) – Cu Kα",
                              "LaB₆ standard – Cu Kα","Quartz – Cu Kα",
                              "Synthetic noisy flat"])

        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="replace")
            try:
                x, y, sig = read_powder_data(uploaded.name, text)
                st.session_state.update(x_obs=x, y_obs=y, sigma=sig,
                                         data_filename=uploaded.name)
                st.success(f"Loaded {len(x)} points")
            except Exception as e:
                st.error(f"Parse error: {e}")
        elif demo != "(none)":
            x, y, sig = make_demo(demo)
            st.session_state.update(x_obs=x, y_obs=y, sigma=sig, data_filename=demo)

        if "x_obs" in st.session_state:
            xd = st.session_state["x_obs"]
            x_range = st.slider("2θ display range (°)", float(xd.min()), float(xd.max()),
                                  (float(xd.min()), float(xd.max())), key="vis_range")
        scat_var = st.selectbox("Scattering variable",
                                 ["2θ (degrees)","TOF (µs)","Energy (keV)"])
        log_y    = st.checkbox("Log scale Y")
        show_bg  = st.checkbox("Estimate background", True)
        bg_mode  = st.selectbox("Background model", BG_MODES, key="bg_mode_vis")
        n_coeffs = st.slider("# BG coefficients", 2, 12, 6, key="bg_n_vis")

    with col2:
        if "x_obs" not in st.session_state:
            st.info("👈 Upload a file or pick a demo dataset.")
            with st.expander("Supported formats"):
                st.markdown("""| Ext | Format |
|-----|--------|
| `.xy` `.dat` `.txt` `.csv` | Plain `x  y` columns |
| `.xye` `.esg` | `x  y  sigma` |
| `.fxye` | GSAS (2θ×100, y, sigma) |""")
            return

        x, y = st.session_state["x_obs"], st.session_state["y_obs"]
        lo, hi = st.session_state.get("vis_range", (x.min(), x.max()))
        mask = (x >= lo) & (x <= hi);  xp, yp = x[mask], y[mask]

        y_bg = None
        if show_bg and len(xp) > 4:
            try:
                from numpy.polynomial.chebyshev import chebfit, chebval
                xn = 2*(xp-xp.min())/(xp.max()-xp.min()+1e-10)-1
                coeffs = chebfit(xn, yp, deg=min(n_coeffs-1, len(xp)-1))
                y_bg = np.minimum(chebval(xn, coeffs), yp)
            except: pass

        st.plotly_chart(make_pattern_figure(xp, yp, y_bg=y_bg,
                         x_label=scat_var,
                         title=f"Pattern: {st.session_state.get('data_filename','')}",
                         log_scale=log_y), use_container_width=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Points", len(xp))
        c2.metric("Range", f"{xp.min():.2f}°–{xp.max():.2f}°")
        c3.metric("Max I", f"{yp.max():.0f}")
        c4.metric("Mean I", f"{yp.mean():.0f}")
        c5.metric("Step", f"{np.mean(np.diff(xp)):.4f}°" if len(xp)>1 else "—")

        with st.expander("Raw data table (first 100 rows)"):
            df = pd.DataFrame({"2θ": xp[:100], "I_obs": yp[:100]})
            st.dataframe(df, use_container_width=True)


# ============================================================
#  SECTION 11 – Page: Crystal Structure Editor
# ============================================================

CRYSTAL_SYSTEMS = {
    "Triclinic":    dict(a=5,b=6,c=7,alpha=80,beta=85,gamma=70),
    "Monoclinic":   dict(a=5,b=6,c=7,alpha=90,beta=110,gamma=90),
    "Orthorhombic": dict(a=5,b=6,c=7,alpha=90,beta=90,gamma=90),
    "Tetragonal":   dict(a=5,b=5,c=7,alpha=90,beta=90,gamma=90),
    "Trigonal":     dict(a=5,b=5,c=7,alpha=90,beta=90,gamma=120),
    "Hexagonal":    dict(a=5,b=5,c=7,alpha=90,beta=90,gamma=120),
    "Cubic":        dict(a=5,b=5,c=5,alpha=90,beta=90,gamma=90),
}

def page_phases():
    st.header("🔬 Phase & Crystal Structure Editor")
    if "phases" not in st.session_state:
        st.session_state["phases"] = [default_phase()]
    phases = st.session_state["phases"]

    with st.sidebar:
        st.subheader("Phase Manager")
        sel = st.selectbox("Active phase", range(len(phases)),
                           format_func=lambda i: phases[i].name, key="active_phase_idx")
        ca, cb_ = st.columns(2)
        if ca.button("➕ Add") and len(phases) < 16:
            phases.append(default_phase(f"Phase_{len(phases)+1}"))
            st.session_state["active_phase_idx"] = len(phases)-1; st.rerun()
        if cb_.button("🗑 Remove", disabled=len(phases)<=1):
            phases.pop(sel); st.session_state["active_phase_idx"] = 0; st.rerun()
        st.markdown("---")
        cif_f = st.file_uploader("Import CIF", type=["cif"], key="cif_up")
        if cif_f:
            try:
                new_ph = parse_cif(cif_f.read().decode("utf-8", errors="replace"))
                for p in new_ph:
                    if len(phases) < 16: phases.append(p)
                st.success(f"Imported {len(new_ph)} phase(s)"); st.rerun()
            except Exception as e:
                st.error(str(e))

    ph = phases[sel]
    t_cell, t_atoms, t_prof, t_view = st.tabs(["🏛 Unit Cell","⚛ Atoms","📈 Profile","🌐 3D View"])

    with t_cell:
        c1, c2 = st.columns(2)
        with c1:
            ph.name = st.text_input("Name", ph.name)
            ph.spacegroup = st.text_input("Space group (H-M)", ph.spacegroup)
            csys = st.selectbox("Crystal system preset", list(CRYSTAL_SYSTEMS.keys()), key=f"cs_{sel}")
            if st.button("Apply preset", key=f"ap_{sel}"):
                d = CRYSTAL_SYSTEMS[csys]
                ph.cell.a,ph.cell.b,ph.cell.c = d["a"],d["b"],d["c"]
                ph.cell.alpha,ph.cell.beta,ph.cell.gamma = d["alpha"],d["beta"],d["gamma"]
                st.rerun()
        with c2:
            st.markdown("**Lattice parameters**")
            r1, r2 = st.columns(2)
            ph.cell.a     = r1.number_input("a (Å)", 0.5,50.0,ph.cell.a,  step=0.001,format="%.4f",key=f"a_{sel}")
            ph.cell.b     = r2.number_input("b (Å)", 0.5,50.0,ph.cell.b,  step=0.001,format="%.4f",key=f"b_{sel}")
            ph.cell.c     = r1.number_input("c (Å)", 0.5,50.0,ph.cell.c,  step=0.001,format="%.4f",key=f"c_{sel}")
            ph.cell.alpha = r2.number_input("α (°)", 30.,150.,ph.cell.alpha,step=0.01,format="%.3f",key=f"al_{sel}")
            ph.cell.beta  = r1.number_input("β (°)", 30.,150.,ph.cell.beta, step=0.01,format="%.3f",key=f"be_{sel}")
            ph.cell.gamma = r2.number_input("γ (°)", 30.,150.,ph.cell.gamma,step=0.01,format="%.3f",key=f"ga_{sel}")
            st.info(f"Volume: **{ph.cell.volume():.4f} Å³**")
            ph.refine_cell  = st.checkbox("Refine cell",  ph.refine_cell,  key=f"rc_{sel}")
            ph.refine_scale = st.checkbox("Refine scale", ph.refine_scale, key=f"rs_{sel}")
            ph.scale = st.number_input("Scale", 1e-10,1e10,float(ph.scale),step=0.01,format="%.4f",key=f"sc_{sel}")
        st.markdown("---")
        st.markdown("**March-Dollase preferred orientation**")
        pa, pb = st.columns(2)
        ph.pref_orient = pa.checkbox("Enable", ph.pref_orient, key=f"po_{sel}")
        if ph.pref_orient:
            ph.pref_r = pb.slider("r", 0.1, 3.0, float(ph.pref_r), step=0.01, key=f"pr_{sel}")
            try:
                h,k,l = [int(v) for v in pa.text_input("axis hkl","0 0 1",key=f"phkl_{sel}").split()]
                ph.pref_hkl = (h,k,l)
            except: pass

    with t_atoms:
        st.markdown("**Atom sites** — edit inline, add/delete rows")
        df_ed = st.data_editor(
            pd.DataFrame({"Label":[a.label for a in ph.atoms],"Element":[a.element for a in ph.atoms],
                          "x":[a.x for a in ph.atoms],"y":[a.y for a in ph.atoms],
                          "z":[a.z for a in ph.atoms],"Occ":[a.occ for a in ph.atoms],
                          "Biso":[a.Biso for a in ph.atoms]}),
            num_rows="dynamic", use_container_width=True, key=f"ae_{sel}")
        ph.atoms = []
        for _, row in df_ed.iterrows():
            try: ph.atoms.append(AtomSite(str(row["Label"]),str(row["Element"]),
                                           float(row["x"]),float(row["y"]),float(row["z"]),
                                           float(row["Occ"]),float(row["Biso"])))
            except: pass
        if len(ph.atoms) >= 2:
            with st.expander("📏 Distances"):
                rows = []
                for i,a1 in enumerate(ph.atoms):
                    for j,a2 in enumerate(ph.atoms):
                        if j<=i: continue
                        d = np.sqrt(((a1.x-a2.x)*ph.cell.a)**2+((a1.y-a2.y)*ph.cell.b)**2+((a1.z-a2.z)*ph.cell.c)**2)
                        rows.append({"Atom 1":a1.label,"Atom 2":a2.label,"d (Å)":f"{d:.4f}"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with t_prof:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Profile function**")
            pk = f"pt_{sel}"
            if pk not in st.session_state: st.session_state[pk] = "TCH Pseudo-Voigt"
            st.session_state[pk] = st.selectbox("Shape", PROFILE_TYPES,
                                                  index=PROFILE_TYPES.index(st.session_state[pk]),
                                                  key=f"pf_{sel}")
            st.markdown("**Caglioti (TCH)**")
            ph.U = st.number_input("U", -1.0,2.0,float(ph.U),step=0.0001,format="%.5f",key=f"U_{sel}")
            ph.V = st.number_input("V", -1.0,1.0,float(ph.V),step=0.0001,format="%.5f",key=f"V_{sel}")
            ph.W = st.number_input("W",  1e-6,1.0,float(ph.W),step=0.0001,format="%.5f",key=f"W_{sel}")
        with p2:
            st.markdown("**Lorentzian broadening**")
            ph.X = st.number_input("X (size)",   0.0,2.0,float(ph.X),step=0.001,format="%.4f",key=f"X_{sel}")
            ph.Y = st.number_input("Y (strain)",  0.0,2.0,float(ph.Y),step=0.001,format="%.4f",key=f"Y_{sel}")
            ph.eta = st.slider("η (pV mixing)", 0.0,1.0,float(ph.eta),step=0.01,key=f"eta_{sel}")
            tt_a = np.linspace(10,80,200)
            fw_a = np.array([ph.fwhm_tch(t) for t in tt_a])
            fwhm_fig = go.Figure(go.Scatter(x=tt_a,y=fw_a,line=dict(color="#1565C0")))
            fwhm_fig.update_layout(title="FWHM(2θ)",xaxis_title="2θ(°)",yaxis_title="FWHM(°)",
                                   height=200,margin=dict(l=40,r=10,t=30,b=40),plot_bgcolor="white")
            st.plotly_chart(fwhm_fig, use_container_width=True)

    with t_view:
        if not ph.atoms:
            st.info("Add atom sites to see structure.")
        else:
            st.plotly_chart(make_crystal_view(ph), use_container_width=True)
            wl = st.session_state.get("refinement_wavelength", 1.54056)
            refs = generate_reflections(ph, wl, 5.0, 80.0)
            if refs:
                with st.expander(f"hkl list ({len(refs)} reflections)"):
                    st.dataframe(pd.DataFrame([{"h":r["h"],"k":r["k"],"l":r["l"],
                                                "d(Å)":f"{r['d']:.4f}","2θ(°)":f"{r['two_theta']:.3f}",
                                                "I_calc":f"{r['I_calc']:.1f}"} for r in refs]),
                                 use_container_width=True)


# ============================================================
#  SECTION 12 – Page: Le Bail
# ============================================================

def page_lebail():
    st.header("⚡ Le Bail Profile Matching")
    st.markdown("Full-profile fit **without structural constraints** — only cell + profile params needed.")
    if "x_obs" not in st.session_state:
        st.warning("⚠️ Load data first (Data Import page)."); return
    if not st.session_state.get("phases"):
        st.warning("⚠️ Define phases first (Crystal Structure page)."); return

    x_obs = st.session_state["x_obs"]
    y_obs = st.session_state["y_obs"]
    phases = st.session_state["phases"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Settings")
        wl = st.number_input("Wavelength (Å)", 0.1,10.0,
                              st.session_state.get("refinement_wavelength",1.54056),
                              step=0.00001,format="%.5f",key="lb_wl")
        st.session_state["refinement_wavelength"] = wl
        profile_type = st.selectbox("Profile", PROFILE_TYPES, key="lb_pt")
        bg_mode = st.selectbox("Background", BG_MODES, key="lb_bg")
        n_bg    = st.slider("# BG params", 2,12,6, key="lb_nbg")
        refine_bg   = st.checkbox("Refine background", True,  key="lb_rbg")
        refine_cell = st.checkbox("Refine cell",        True,  key="lb_rc")
        refine_prof = st.checkbox("Refine profile",     True,  key="lb_rp")
        max_iter    = st.slider("Max cycles", 5,200,30, key="lb_iter")
        weighting   = st.selectbox("Weighting", ["standard","unit","ml"], key="lb_w")
        run_btn     = st.button("▶ Run Le Bail", type="primary")

    with col2:
        if "lb_bg" not in st.session_state or "lb_background" not in st.session_state:
            st.session_state["lb_background"] = Background(bg_mode, n_bg)
        bg = st.session_state["lb_background"]
        bg.mode = bg_mode
        if len(bg.coeffs) != n_bg: bg.coeffs = [0.0]*n_bg
        bg.refine = refine_bg

        yc, refs = calc_pattern(x_obs, phases, wl, bg,
                                 profile_type=profile_type, y_obs=y_obs)
        st.plotly_chart(make_pattern_figure(x_obs, y_obs, yc,
                                             y_bg=bg.evaluate(x_obs,y_obs),
                                             residual=y_obs-yc, reflections=refs,
                                             title="Le Bail – current params"),
                        use_container_width=True)
        _rfactors_row(y_obs, yc)

    if run_btn:
        pbar, stat, log = st.progress(0), st.empty(), []
        def upd(it, cost):
            pbar.progress(min(int(it/max_iter*100),99))
            stat.text(f"Cycle {it}  cost={cost:.4e}")
            log.append(f"{it:4d}  {cost:.6e}")
        with st.spinner("Le Bail fitting…"):
            res = run_refinement(x_obs, y_obs, phases, bg, wavelength=wl,
                                  profile_type=profile_type, weighting=weighting,
                                  refine_bg=refine_bg, refine_profile=refine_prof,
                                  refine_cell=refine_cell, refine_scale=True,
                                  le_bail=True, max_iter=max_iter, progress_callback=upd)
        pbar.progress(100);  stat.text("Done!")
        st.session_state["lb_background"] = bg
        st.plotly_chart(make_pattern_figure(x_obs, y_obs, res.y_calc,
                                             y_bg=bg.evaluate(x_obs,y_obs),
                                             residual=res.residual, reflections=res.reflections,
                                             title=f"Le Bail  Rwp={res.r_wp:.2f}%  Rp={res.r_p:.2f}%"),
                        use_container_width=True)
        _rfactors_result(res)
        with st.expander("Refined parameters"):
            st.dataframe(pd.DataFrame([{"Parameter":k,"Value":f"{v:.6g}"}
                                        for k,v in res.parameters.items()]),
                         use_container_width=True)
        with st.expander("Log"):
            st.code("\n".join(log[-50:]))
        if res.success: st.success(f"✅ Converged  Rwp={res.r_wp:.2f}%")
        else:           st.warning(f"⚠️ {res.message}")


# ============================================================
#  SECTION 13 – Page: Rietveld Refinement
# ============================================================

def page_rietveld():
    st.header("🔩 Rietveld Refinement")
    if "x_obs" not in st.session_state:
        st.warning("⚠️ Load data first."); return
    if not st.session_state.get("phases"):
        st.warning("⚠️ Define phases first."); return

    x_obs  = st.session_state["x_obs"]
    y_obs  = st.session_state["y_obs"]
    phases = st.session_state["phases"]

    with st.sidebar:
        st.subheader("⚙️ Rietveld Settings")
        wl = st.number_input("λ (Å)", 0.01,10.0,
                              st.session_state.get("refinement_wavelength",1.54056),
                              step=0.00001,format="%.5f",key="ri_wl")
        st.session_state["refinement_wavelength"] = wl
        if st.checkbox("Second wavelength (Kα₂)", key="ri_wl2"):
            wl2 = st.number_input("λ₂ (Å)", 0.01,10.0,wl*1.002,step=0.00001,format="%.5f",key="ri_wl2v")
            st.slider("I(Kα₂)/I(Kα₁)", 0.0,1.0,0.5,key="ri_wl2r")
        st.selectbox("Scattering variable",["2θ (degrees)","TOF (µs)","Energy (keV)"],key="ri_sv")
        profile_type = st.selectbox("Profile", PROFILE_TYPES, key="ri_pt")
        st.markdown("---")
        bg_mode = st.selectbox("Background", BG_MODES, key="ri_bg")
        n_bg    = st.slider("# BG params", 2,12,6, key="ri_nbg")
        st.markdown("---")
        refine_bg   = st.checkbox("Refine background",     True,  key="ri_rbg")
        refine_cell = st.checkbox("Refine cell parameters", True,  key="ri_rc")
        refine_prof = st.checkbox("Refine profile params",  True,  key="ri_rp")
        refine_scal = st.checkbox("Refine scale factors",   True,  key="ri_rsc")
        st.markdown("---")
        mono  = st.checkbox("Monochromator LP", False, key="ri_mono")
        mu_R  = st.number_input("Absorption µR", 0.0,50.0,0.0,step=0.1,key="ri_muR")
        st.markdown("---")
        weighting = st.selectbox("Weighting",["standard","unit","ml"],key="ri_w")
        max_iter  = st.slider("Max cycles", 5,500,50, key="ri_mi")
        damping   = st.slider("Damping", 0.1,2.0,1.0,step=0.05,key="ri_damp")

    if "riet_background" not in st.session_state:
        st.session_state["riet_background"] = Background(bg_mode, n_bg)
    bg = st.session_state["riet_background"]
    bg.mode = bg_mode
    if len(bg.coeffs) != n_bg: bg.coeffs = [0.0]*n_bg
    bg.refine = refine_bg

    yc, refs = calc_pattern(x_obs, phases, wl, bg,
                             profile_type=profile_type, monochromator=mono,
                             mu_R=mu_R, y_obs=y_obs)
    st.plotly_chart(make_pattern_figure(x_obs, y_obs, yc,
                                         y_bg=bg.evaluate(x_obs,y_obs),
                                         residual=y_obs-yc, reflections=refs,
                                         title="Rietveld – pre-refinement preview"),
                    use_container_width=True)
    _rfactors_row(y_obs, yc)

    col_run, col_rst = st.columns([2,1])
    run_btn   = col_run.button("▶▶ Run Rietveld Refinement", type="primary")
    reset_btn = col_rst.button("↩ Reset")
    if reset_btn and "riet_result" in st.session_state:
        del st.session_state["riet_result"];  st.rerun()

    if run_btn:
        pbar, stat, log = st.progress(0), st.empty(), []
        def upd(it, cost):
            pbar.progress(min(int(it/max_iter*100),99))
            stat.text(f"Cycle {it}  cost={cost:.4e}")
            log.append(f"{it:5d}  {cost:.8e}")
        with st.spinner("Running Rietveld refinement…"):
            res = run_refinement(x_obs, y_obs, phases, bg, wavelength=wl,
                                  profile_type=profile_type, weighting=weighting,
                                  refine_bg=refine_bg, refine_profile=refine_prof,
                                  refine_cell=refine_cell, refine_scale=refine_scal,
                                  max_iter=max_iter, damping=damping, progress_callback=upd)
        pbar.progress(100);  stat.empty()
        st.session_state.update(riet_result=res, riet_background=bg, riet_log=log)
        if res.success: st.success(f"✅ Rwp={res.r_wp:.2f}%  Rp={res.r_p:.2f}%  GoF={res.gof:.3f}")
        else:           st.warning(f"⚠️ {res.message}")

    if "riet_result" in st.session_state:
        res = st.session_state["riet_result"]
        bg  = st.session_state.get("riet_background", bg)
        st.markdown("---");  st.subheader("Results")
        st.plotly_chart(make_pattern_figure(x_obs, y_obs, res.y_calc,
                                             y_bg=bg.evaluate(x_obs,y_obs),
                                             residual=res.residual, reflections=res.reflections,
                                             title=f"Rietveld  Rwp={res.r_wp:.2f}%  Rp={res.r_p:.2f}%  GoF={res.gof:.3f}"),
                        use_container_width=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Rwp",  f"{res.r_wp:.3f}%");  c2.metric("Rp",   f"{res.r_p:.3f}%")
        c3.metric("Rexp", f"{res.r_exp:.3f}%"); c4.metric("GoF",  f"{res.gof:.4f}")
        c5.metric("Cycles", res.n_iter)

        tp, tr, tl, te = st.tabs(["📊 Parameters","🔷 Reflections","📋 Log","💾 Export"])
        with tp:
            st.dataframe(pd.DataFrame([{"Parameter":k,"Value":f"{v:.8g}"}
                                        for k,v in res.parameters.items()]),
                         use_container_width=True)
        with tr:
            if res.reflections:
                st.dataframe(pd.DataFrame([{"Phase":r.get("phase_name",""),
                    "h":r["h"],"k":r["k"],"l":r["l"],"d(Å)":f"{r['d']:.4f}",
                    "2θ(°)":f"{r['two_theta']:.3f}","I_calc":f"{r['I_calc']:.1f}"}
                    for r in res.reflections]), use_container_width=True)
        with tl:
            st.text_area("Log", "\n".join(st.session_state.get("riet_log",[])), height=300)
        with te:
            st.download_button("⬇ Report (.txt)", _make_report(res),
                               "rietveld_report.txt","text/plain")
            arr = np.column_stack([x_obs,y_obs,res.y_calc,res.residual])
            csv = "# 2theta  I_obs  I_calc  I_diff\n" + \
                  "\n".join(f"{r[0]:.4f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f}" for r in arr)
            st.download_button("⬇ Pattern (.xy)", csv, "rietveld_pattern.xy","text/plain")


# ============================================================
#  SECTION 14 – App Entry Point
# ============================================================

st.set_page_config(page_title="RietveldApp", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3{font-weight:700;letter-spacing:-0.02em;}
code,pre,.stCode{font-family:'IBM Plex Mono',monospace;}
section[data-testid="stSidebar"]>div:first-child{
  background:linear-gradient(180deg,#0D1B2A 0%,#1B2838 100%);color:#ECF0F1;}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label{color:#ECF0F1 !important;}
div[data-testid="metric-container"]{background:#F8F9FA;border:1px solid #DEE2E6;border-radius:8px;padding:12px 16px;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1565C0,#0D47A1);color:white;border:none;border-radius:6px;font-weight:600;}
</style>""", unsafe_allow_html=True)

PAGES = {
    "📂 Data Import":         page_data,
    "🔬 Crystal Structure":   page_phases,
    "⚡ Le Bail Fitting":      page_lebail,
    "🔩 Rietveld Refinement": page_rietveld,
}

with st.sidebar:
    st.markdown("## 🔬 RietveldApp")
    st.markdown("*Standalone Rietveld refinement suite*")
    st.markdown("---")
    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    n_ph = len(st.session_state.get("phases", []))
    st.markdown(f"**Phases loaded:** {n_ph}")
    if "x_obs" in st.session_state:
        x = st.session_state["x_obs"]
        st.markdown(f"**Data:** {len(x)} pts · {x.min():.1f}°–{x.max():.1f}°")
    else:
        st.markdown("**Data:** *(not loaded)*")
    st.markdown("---")
    st.caption("numpy · scipy · plotly · streamlit")

PAGES[selected]()
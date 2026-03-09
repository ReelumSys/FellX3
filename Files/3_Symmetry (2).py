"""
HKL Indexer & Structure Factor Calculator
==========================================
From an uploaded (or simulated) powder diffractogram:

  Step 1 — Peak detection
    scipy find_peaks + Gaussian/pseudo-Voigt fitting → 2θ_peak, d_hkl

  Step 2 — HKL Indexing
    • Bragg's law: d = λ / (2·sinθ)
    • For a chosen mineral / crystal system, generate all (hkl) and match
      observed d-spacings to calculated d-spacings within a tolerance window.
    • Reports: best-match (hkl), Δd, Δ2θ, figure-of-merit M20

  Step 3 — Structure Factor Calculation (for matched reflections)
    • F(hkl) = Σ fⱼ · DWⱼ · exp(2πi(hxⱼ + kyⱼ + lzⱼ))
    • 4-Gaussian atomic scattering factors (Int. Tables Vol. C)
    • Debye-Waller correction
    • Returns |F|, phase φ, F_real, F_imag, I∝|F|²

  Visualisations
    • Annotated diffractogram with HKL labels
    • d-spacing chart (measured vs calculated)
    • Argand diagram  (complex F plane)
    • Polar phase plot
    • Per-peak structure-factor bar chart

  Export
    • Full CSV with peak position, d, hkl, |F|, φ, I, Δd …
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import io, itertools

# ─────────────────────────────────────────────────────────────────────────────
# Atomic scattering factor database  (a1..a4, b1..b4, c)  Int. Tables Vol. C
# ─────────────────────────────────────────────────────────────────────────────
ASF = {
    "H":  ([0.4899,0.2620,0.1967,0.0490,0.0],[20.6593,7.7404,49.5519,2.2016,0.0],0.0010),
    "C":  ([2.3100,1.0200,1.5886,0.8650,0.0],[20.8439,10.2075,0.5687,51.6512,0.0],0.2156),
    "O":  ([3.0485,2.2868,1.0624,0.1156,0.0],[13.2771,5.7011,0.3239,32.9089,0.0],0.3006),
    "Na": ([4.7626,3.1736,1.2674,1.1128,0.0],[3.2850,8.8422,0.3136,129.424,0.0],0.6760),
    "Mg": ([5.4204,2.1735,1.2269,2.3073,0.0],[2.8275,79.2611,0.3808,7.1937,0.0],0.8584),
    "Al": ([6.4202,1.9002,1.5936,1.9646,0.0],[3.0387,0.7426,31.5472,85.0886,0.0],1.1151),
    "Si": ([6.2915,3.0353,1.9891,0.5399,1.1410],[2.4386,32.3337,0.6785,81.6937,0.0],1.1407),
    "S":  ([6.9053,5.2034,1.4379,1.5863,0.0],[1.4679,22.2151,0.2536,56.1720,0.0],0.8669),
    "Cl": ([11.4604,7.1964,6.2556,1.6455,0.0],[0.0104,1.1664,18.5194,47.7784,0.0],-9.5574),
    "K":  ([8.2186,7.4398,1.0519,0.8659,0.0],[12.7949,0.7748,213.187,41.6841,0.0],1.4228),
    "Ca": ([8.6266,7.3873,1.5899,1.0211,0.0],[10.4421,0.6599,85.7484,178.437,0.0],1.3751),
    "Ti": ([9.7595,7.3558,1.6991,1.9021,0.0],[7.8508,0.5,35.6338,116.105,0.0],1.2807),
    "Mn": ([11.2819,7.3573,3.5490,2.1645,0.0],[5.3409,0.3432,17.8674,83.7543,0.0],1.0896),
    "Fe": ([11.7695,7.3573,3.5222,2.3045,0.0],[4.7611,0.3072,15.3535,76.8805,0.0],1.0369),
}

# ─────────────────────────────────────────────────────────────────────────────
# Mineral structure database
# ─────────────────────────────────────────────────────────────────────────────
MINERALS = {
    "Quartz (SiO₂)": {
        "system":"Hexagonal","sg":"P3₂21 (154)","Z":3,
        "a":4.9133,"b":4.9133,"c":5.4053,
        "alpha":90.0,"beta":90.0,"gamma":120.0,
        "atoms":[
            {"element":"Si","x":0.4697,"y":0.0000,"z":0.0000,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.0000,"y":0.4697,"z":0.6667,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.5303,"y":0.5303,"z":0.3333,"occ":1.0,"Biso":0.50},
            {"element":"O", "x":0.4135,"y":0.2669,"z":0.1188,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.2669,"y":0.4135,"z":0.8812,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.7331,"y":0.1466,"z":0.4521,"occ":1.0,"Biso":0.80},
        ],
    },
    "Calcite (CaCO₃)": {
        "system":"Trigonal","sg":"R3̄c (167)","Z":6,
        "a":4.9896,"b":4.9896,"c":17.0610,
        "alpha":90.0,"beta":90.0,"gamma":120.0,
        "atoms":[
            {"element":"Ca","x":0.0000,"y":0.0000,"z":0.0000,"occ":1.0,"Biso":0.60},
            {"element":"C", "x":0.0000,"y":0.0000,"z":0.2500,"occ":1.0,"Biso":0.50},
            {"element":"O", "x":0.2573,"y":0.0000,"z":0.2500,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.0000,"y":0.2573,"z":0.2500,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.7427,"y":0.7427,"z":0.2500,"occ":1.0,"Biso":1.00},
        ],
    },
    "Forsterite (Mg₂SiO₄)": {
        "system":"Orthorhombic","sg":"Pbnm (62)","Z":4,
        "a":4.7540,"b":10.1971,"c":5.9806,
        "alpha":90.0,"beta":90.0,"gamma":90.0,
        "atoms":[
            {"element":"Mg","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.50},
            {"element":"Mg","x":0.5,"y":0.5,"z":0.0,"occ":1.0,"Biso":0.50},
            {"element":"Mg","x":0.0,"y":0.2211,"z":0.5,"occ":1.0,"Biso":0.50},
            {"element":"Mg","x":0.5,"y":0.7789,"z":0.5,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.0,"y":0.0940,"z":0.4232,"occ":1.0,"Biso":0.40},
            {"element":"Si","x":0.5,"y":0.4060,"z":0.4232,"occ":1.0,"Biso":0.40},
            {"element":"O", "x":0.0,"y":0.0926,"z":0.7656,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.5,"y":0.4074,"z":0.7656,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.0,"y":0.4512,"z":0.2199,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.5,"y":0.0488,"z":0.2199,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.2724,"y":0.1643,"z":0.2801,"occ":1.0,"Biso":0.80},
            {"element":"O", "x":0.7276,"y":0.8357,"z":0.2801,"occ":1.0,"Biso":0.80},
        ],
    },
    "Albite (NaAlSi₃O₈)": {
        "system":"Triclinic","sg":"P1̄ (2)","Z":4,
        "a":8.1360,"b":12.7870,"c":7.1582,
        "alpha":94.253,"beta":116.605,"gamma":87.756,
        "atoms":[
            {"element":"Na","x":0.2690,"y":0.9890,"z":0.1470,"occ":1.0,"Biso":1.50},
            {"element":"Al","x":0.0088,"y":0.1680,"z":0.2082,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.0036,"y":0.8200,"z":0.2390,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.6900,"y":0.1120,"z":0.3150,"occ":1.0,"Biso":0.50},
            {"element":"Si","x":0.6813,"y":0.8820,"z":0.3610,"occ":1.0,"Biso":0.50},
            {"element":"O", "x":0.0055,"y":0.1310,"z":0.9680,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.5934,"y":0.9970,"z":0.2800,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.8194,"y":0.1085,"z":0.1902,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.0203,"y":0.3027,"z":0.2700,"occ":1.0,"Biso":1.00},
        ],
    },
    "Halite (NaCl)": {
        "system":"Cubic","sg":"Fm3̄m (225)","Z":4,
        "a":5.6402,"b":5.6402,"c":5.6402,
        "alpha":90.0,"beta":90.0,"gamma":90.0,
        "atoms":[
            {"element":"Na","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":1.20},
            {"element":"Na","x":0.5,"y":0.5,"z":0.0,"occ":1.0,"Biso":1.20},
            {"element":"Na","x":0.5,"y":0.0,"z":0.5,"occ":1.0,"Biso":1.20},
            {"element":"Na","x":0.0,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.20},
            {"element":"Cl","x":0.5,"y":0.0,"z":0.0,"occ":1.0,"Biso":1.50},
            {"element":"Cl","x":0.0,"y":0.5,"z":0.0,"occ":1.0,"Biso":1.50},
            {"element":"Cl","x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":1.50},
            {"element":"Cl","x":0.5,"y":0.5,"z":0.5,"occ":1.0,"Biso":1.50},
        ],
    },
    "Pyrite (FeS₂)": {
        "system":"Cubic","sg":"Pa3̄ (205)","Z":4,
        "a":5.4166,"b":5.4166,"c":5.4166,
        "alpha":90.0,"beta":90.0,"gamma":90.0,
        "atoms":[
            {"element":"Fe","x":0.0,  "y":0.0,  "z":0.0,  "occ":1.0,"Biso":0.50},
            {"element":"Fe","x":0.5,  "y":0.0,  "z":0.5,  "occ":1.0,"Biso":0.50},
            {"element":"Fe","x":0.0,  "y":0.5,  "z":0.5,  "occ":1.0,"Biso":0.50},
            {"element":"Fe","x":0.5,  "y":0.5,  "z":0.0,  "occ":1.0,"Biso":0.50},
            {"element":"S", "x":0.385,"y":0.385,"z":0.385,"occ":1.0,"Biso":0.60},
            {"element":"S", "x":0.615,"y":0.615,"z":0.385,"occ":1.0,"Biso":0.60},
            {"element":"S", "x":0.615,"y":0.385,"z":0.615,"occ":1.0,"Biso":0.60},
            {"element":"S", "x":0.385,"y":0.615,"z":0.615,"occ":1.0,"Biso":0.60},
        ],
    },
    "Dolomite (CaMg(CO₃)₂)": {
        "system":"Trigonal","sg":"R3̄ (148)","Z":3,
        "a":4.8070,"b":4.8070,"c":16.0020,
        "alpha":90.0,"beta":90.0,"gamma":120.0,
        "atoms":[
            {"element":"Ca","x":0.0,"y":0.0,"z":0.0,"occ":1.0,"Biso":0.60},
            {"element":"Mg","x":0.0,"y":0.0,"z":0.5,"occ":1.0,"Biso":0.50},
            {"element":"C", "x":0.0,"y":0.0,"z":0.2436,"occ":1.0,"Biso":0.50},
            {"element":"O", "x":0.2498,"y":0.0,"z":0.2436,"occ":1.0,"Biso":1.00},
            {"element":"O", "x":0.0,"y":0.2498,"z":0.2436,"occ":1.0,"Biso":1.00},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Crystallography core
# ─────────────────────────────────────────────────────────────────────────────

def metric_tensor(a, b, c, al, be, ga):
    ca,cb,cg = np.cos(np.radians(al)),np.cos(np.radians(be)),np.cos(np.radians(ga))
    return np.array([[a*a,a*b*cg,a*c*cb],
                     [a*b*cg,b*b,b*c*ca],
                     [a*c*cb,b*c*ca,c*c]])

def d_calc(h, k, l, G):
    Gi = np.linalg.inv(G)
    v  = np.array([h,k,l], dtype=float)
    q2 = v @ Gi @ v
    return 1.0/np.sqrt(q2) if q2 > 1e-12 else np.inf

def f_atom(elem, s2):
    """Atomic scattering factor f(sinθ/λ)²"""
    if elem not in ASF:
        return 1.0
    a, b, c = ASF[elem]
    return c + sum(ai*np.exp(-bi*s2) for ai,bi in zip(a,b))

def structure_factor(h, k, l, atoms, d, lam):
    """Returns (|F|, phase_deg, F_real, F_imag)"""
    s  = lam / (2*d)       # sinθ/λ
    s2 = s*s
    F  = 0+0j
    for at in atoms:
        f  = f_atom(at["element"], s2)
        DW = np.exp(-at.get("Biso",0.5) * s2)
        ph = 2*np.pi*(h*at["x"] + k*at["y"] + l*at["z"])
        F += at.get("occ",1.0) * f * DW * np.exp(1j*ph)
    return abs(F), np.degrees(np.angle(F)), F.real, F.imag

def gen_hkl_table(mineral, lam, tt_min, tt_max, hkl_max=8):
    """
    Build a lookup table of all symmetry-distinct (hkl) reflections with
    calculated d, 2θ, F², phase, multiplicity.
    Returns list of dicts sorted by 2θ.
    """
    a,b,c   = mineral["a"],mineral["b"],mineral["c"]
    al,be,ga= mineral["alpha"],mineral["beta"],mineral["gamma"]
    G       = metric_tensor(a,b,c,al,be,ga)
    atoms   = mineral["atoms"]

    bucket = {}  # key = round(d,3) → best representative
    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h==0 and k==0 and l==0:
                    continue
                d = d_calc(h,k,l,G)
                if d<=0 or d>30:
                    continue
                st = lam/(2*d)
                if st > 1.0:
                    continue
                tt = np.degrees(2*np.arcsin(st))
                if not (tt_min <= tt <= tt_max):
                    continue
                amp, phi, Fr, Fi = structure_factor(h,k,l,atoms,d,lam)
                F2 = amp**2
                key = round(d, 3)
                if key not in bucket:
                    bucket[key] = {
                        "h":h,"k":k,"l":l,"d_calc":d,"tt_calc":tt,
                        "|F|":amp,"phase_deg":phi,"F_real":Fr,"F_imag":Fi,
                        "F2":F2,"mult":1
                    }
                else:
                    bucket[key]["mult"] += 1

    refs = sorted(bucket.values(), key=lambda r: r["tt_calc"])
    return refs

# ─────────────────────────────────────────────────────────────────────────────
# Peak detection & fitting
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_peak(x, A, mu, sigma, bg):
    return bg + A*np.exp(-((x-mu)**2)/(2*sigma**2))

def fit_one_peak(x_win, y_win, pos_est, amp_est):
    """Fit a Gaussian to a windowed segment; return (mu, amp, sigma, bg, fwhm)."""
    bg_est  = np.percentile(y_win, 10)
    sig_est = 0.15
    try:
        popt, _ = curve_fit(
            gaussian_peak, x_win, y_win,
            p0=[amp_est-bg_est, pos_est, sig_est, bg_est],
            bounds=([0, pos_est-2, 0.01, 0],
                    [amp_est*5, pos_est+2, 3.0, amp_est*2]),
            maxfev=4000,
        )
        mu, sigma = popt[1], abs(popt[2])
        fwhm = 2*np.sqrt(2*np.log(2))*sigma
        return mu, popt[0]+popt[3], sigma, popt[3], fwhm
    except Exception:
        return pos_est, amp_est, sig_est, bg_est, 2*np.sqrt(2*np.log(2))*sig_est

def detect_peaks(two_theta, intensity, min_prominence_pct, min_sep_deg, win_deg):
    """Detect and fit peaks; return list of dicts with tt, intensity, fwhm, d."""
    step = (two_theta[-1]-two_theta[0]) / max(len(two_theta)-1,1)
    min_dist_pts = max(3, int(min_sep_deg/step))
    smooth = gaussian_filter1d(intensity, sigma=max(2, int(0.05/step)))

    peak_idx, _ = find_peaks(
        smooth,
        prominence=min_prominence_pct/100 * smooth.max(),
        distance=min_dist_pts,
    )

    peaks = []
    for idx in peak_idx:
        pos = two_theta[idx]
        amp = intensity[idx]
        half = max(int(win_deg/step), 8)
        lo   = max(0, idx-half)
        hi   = min(len(two_theta)-1, idx+half)
        x_w  = two_theta[lo:hi+1]
        y_w  = intensity[lo:hi+1]
        if len(x_w) < 5:
            continue
        mu, peak_amp, sigma, bg, fwhm = fit_one_peak(x_w, y_w, pos, amp)
        peaks.append({"tt_obs":mu, "I_obs":peak_amp, "fwhm":fwhm, "bg":bg})
    return peaks

# ─────────────────────────────────────────────────────────────────────────────
# HKL Indexing: match observed d-spacings → calculated hkl
# ─────────────────────────────────────────────────────────────────────────────

def obs_d(tt_deg, lam):
    th = np.radians(tt_deg/2)
    return lam/(2*np.sin(th)) if np.sin(th) > 0 else np.inf

def index_peaks(obs_peaks, hkl_table, lam, tol_deg):
    """
    For each observed peak, find the best-matching hkl from the table
    within ±tol_deg in 2θ.  Returns enriched list.
    """
    indexed = []
    for pk in obs_peaks:
        d_obs = obs_d(pk["tt_obs"], lam)
        best  = None
        best_delta = np.inf
        for ref in hkl_table:
            delta = abs(ref["tt_calc"] - pk["tt_obs"])
            if delta < tol_deg and delta < best_delta:
                best_delta = delta
                best = ref

        row = dict(pk)
        row["d_obs"]    = round(d_obs, 5)
        if best is not None:
            row["h"]         = best["h"]
            row["k"]         = best["k"]
            row["l"]         = best["l"]
            row["d_calc"]    = round(best["d_calc"], 5)
            row["tt_calc"]   = round(best["tt_calc"], 4)
            row["delta_d"]   = round(d_obs - best["d_calc"], 5)
            row["delta_2t"]  = round(pk["tt_obs"] - best["tt_calc"], 4)
            row["|F|"]       = round(best["|F|"], 3)
            row["phase_deg"] = round(best["phase_deg"], 2)
            row["F_real"]    = round(best["F_real"], 3)
            row["F_imag"]    = round(best["F_imag"], 3)
            row["F2"]        = round(best["F2"], 2)
            row["mult"]      = best["mult"]
            row["indexed"]   = True
        else:
            row.update({"h":"?","k":"?","l":"?","d_calc":np.nan,
                        "tt_calc":np.nan,"delta_d":np.nan,"delta_2t":np.nan,
                        "|F|":np.nan,"phase_deg":np.nan,"F_real":np.nan,
                        "F_imag":np.nan,"F2":np.nan,"mult":np.nan,"indexed":False})
        indexed.append(row)
    return indexed

def figure_of_merit_M20(indexed):
    """
    de Wolff M20: M20 = Q20 / (2·ε·N20)
    Q = 1/d², ε = mean |ΔQ|, N = number of possible lines up to 20th line.
    Simplified version using first 20 indexed peaks.
    """
    ok = [r for r in indexed if r["indexed"] and isinstance(r["h"], int)][:20]
    if len(ok) < 3:
        return np.nan
    N  = len(ok)
    Q_obs  = np.array([(1/r["d_obs"])**2  for r in ok])
    Q_calc = np.array([(1/r["d_calc"])**2 for r in ok])
    eps    = np.mean(np.abs(Q_obs - Q_calc))
    Q20    = Q_obs[-1]
    M20    = Q20 / (2 * eps * N) if eps > 0 else np.inf
    return round(float(M20), 2)

# ─────────────────────────────────────────────────────────────────────────────
# Demo pattern generator
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def make_demo(mineral_name, lam, tt_min, tt_max, noise_pct, cryst_nm, n=5000):
    mineral = MINERALS[mineral_name]
    a,b,c   = mineral["a"],mineral["b"],mineral["c"]
    al,be,ga= mineral["alpha"],mineral["beta"],mineral["gamma"]
    G       = metric_tensor(a,b,c,al,be,ga)
    atoms   = mineral["atoms"]
    tt      = np.linspace(tt_min, tt_max, n)
    pat     = np.zeros(n)

    for h in range(-6,7):
        for k in range(-6,7):
            for l in range(-6,7):
                if h==k==l==0: continue
                d = d_calc(h,k,l,G)
                if d<=0 or d>30: continue
                st_ = lam/(2*d)
                if st_>1: continue
                tt0 = np.degrees(2*np.arcsin(st_))
                if not (tt_min < tt0 < tt_max): continue
                amp, *_ = structure_factor(h,k,l,atoms,d,lam)
                F2 = amp**2
                if F2 < 0.1: continue
                th_r = np.radians(tt0/2)
                H_inst = np.sqrt(max(0.01*np.tan(th_r)**2+0.005*np.tan(th_r)+0.002, 1e-6))
                H_size = np.degrees(0.9*lam/(cryst_nm*10*max(np.cos(th_r),1e-6)))
                H = np.sqrt(H_inst**2+H_size**2)
                sig = H/(2*np.sqrt(2*np.log(2)))
                pat += F2*np.exp(-((tt-tt0)**2)/(2*sig**2))

    # LP correction + background + noise
    th_arr = np.radians(tt/2)
    lp = (1+np.cos(2*np.radians(tt))**2)/(np.sin(th_arr)**2*np.cos(th_arr)+1e-9)
    lp /= lp.max()
    pat *= lp
    pat = pat/max(pat.max(),1)*9000
    bg  = 150 + 400*np.exp(-tt/25)
    pat += bg
    if noise_pct > 0:
        pat += np.random.normal(0, noise_pct/100*pat.max(), n)
    return tt, np.clip(pat, 0, None)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HKL Indexer & Structure Factors", page_icon="🔎", layout="wide")

st.title("🔎 HKL Indexer & Structure Factor Calculator")
st.markdown(
    "Upload a powder diffractogram → auto-detect peaks → index to **(hkl)** → "
    "compute full **structure factors F(hkl)** with amplitudes, phases, and Argand diagram."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader(
        "Upload diffractogram (CSV / XY / DAT)",
        type=["csv","txt","xy","dat","xye"],
        help="Two-column file: 2θ [°], Intensity. Delimiter: comma, tab or space."
    )
    use_demo = st.checkbox("Use simulated demo data", value=True)

    st.divider()
    st.header("🔬 Structure Model")
    mineral_name = st.selectbox("Mineral / Phase", list(MINERALS.keys()))
    mineral      = MINERALS[mineral_name]

    st.divider()
    st.header("⚙️ Instrument")
    lam      = st.number_input("λ (Å)", value=1.54056, format="%.5f", help="Cu Kα₁ = 1.54056 Å")
    tt_min   = st.number_input("2θ min (°)", value=5.0)
    tt_max   = st.number_input("2θ max (°)", value=80.0)

    st.divider()
    st.header("🔍 Peak Detection")
    min_prom = st.slider("Min prominence (% of max)", 1, 40, 4)
    min_sep  = st.slider("Min peak separation (°)", 0.1, 5.0, 0.4, 0.05)
    win_deg  = st.slider("Fit window ±(°)", 0.1, 3.0, 0.6, 0.05)

    st.divider()
    st.header("📐 Indexing")
    tol_deg  = st.slider("Match tolerance (°2θ)", 0.05, 2.0, 0.30, 0.05)
    hkl_max  = st.slider("HKL search limit", 3, 10, 6)

    if use_demo:
        st.divider()
        st.header("🎛️ Demo Settings")
        cryst_nm  = st.slider("Crystallite size (nm)", 10, 300, 80)
        noise_pct = st.slider("Noise (%)", 0, 15, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Load / generate data
# ─────────────────────────────────────────────────────────────────────────────
two_theta = intensity = None

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8", errors="replace")
        lines   = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        df_raw  = pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python",
                               header=None, on_bad_lines="skip")
        df_raw  = df_raw.apply(pd.to_numeric, errors="coerce").dropna(subset=[0,1])
        two_theta = df_raw.iloc[:,0].values.astype(float)
        intensity = df_raw.iloc[:,1].values.astype(float)
        st.sidebar.success(f"Loaded {len(two_theta)} points")
    except Exception as e:
        st.sidebar.error(f"Parse error: {e}")

if two_theta is None:
    if use_demo:
        with st.spinner("Generating demo pattern…"):
            two_theta, intensity = make_demo(
                mineral_name, lam, tt_min, tt_max, noise_pct, cryst_nm)
        st.info(f"ℹ️ Using **simulated** {mineral_name} pattern. Upload your own file to analyse real data.")
    else:
        st.warning("Please upload a diffractogram file, or enable the demo data option.")
        st.stop()

# Crop to range
mask      = (two_theta >= tt_min) & (two_theta <= tt_max)
two_theta = two_theta[mask]
intensity = intensity[mask]

if len(two_theta) < 10:
    st.error("Too few data points in selected 2θ range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Peak detection
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Detecting peaks…"):
    obs_peaks = detect_peaks(two_theta, intensity, min_prom, min_sep, win_deg)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate HKL table from structure model
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Computing HKL reflection table…"):
    hkl_table = gen_hkl_table(mineral, lam, tt_min, tt_max, hkl_max)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Index observed peaks
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Indexing peaks…"):
    indexed = index_peaks(obs_peaks, hkl_table, lam, tol_deg)

n_found   = len(indexed)
n_indexed = sum(1 for r in indexed if r["indexed"])
M20       = figure_of_merit_M20(indexed)

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Peaks detected",  n_found)
m2.metric("Peaks indexed",   n_indexed)
m3.metric("Unindexed",       n_found - n_indexed)
m4.metric("Index rate",      f"{n_indexed/max(n_found,1)*100:.0f}%")
m5.metric("Figure of merit M₂₀", f"{M20}" if not (isinstance(M20,float) and np.isnan(M20)) else "—",
          help="de Wolff M₂₀ — higher = better indexing (>10 = good)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Annotated Pattern",
    "📋 Index & Structure Factors",
    "🌀 Phase / Argand",
    "📊 d-spacing Match",
    "🔬 All HKL (calc)",
    "💾 Export CSV",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Annotated diffractogram
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Diffractogram with HKL Assignments")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=two_theta, y=intensity,
        mode="lines", name="Measured",
        line=dict(color="#90caf9", width=1.3),
    ))

    # Indexed peaks
    ok_idx = [r for r in indexed if r["indexed"]]
    if ok_idx:
        fig.add_trace(go.Scatter(
            x=[r["tt_obs"] for r in ok_idx],
            y=[r["I_obs"]  for r in ok_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#a5d6a7",
                        line=dict(color="white", width=0.8)),
            name="Indexed peaks",
            hovertemplate="<b>(%{customdata[0]} %{customdata[1]} %{customdata[2]})</b><br>"
                          "2θ=%{x:.3f}°  d=%{customdata[3]:.4f} Å<br>"
                          "|F|=%{customdata[4]:.2f}  φ=%{customdata[5]:.1f}°<extra></extra>",
            customdata=[[r["h"],r["k"],r["l"],r["d_obs"],r["|F|"],r["phase_deg"]] for r in ok_idx],
        ))
        # HKL text labels
        for r in ok_idx:
            fig.add_annotation(
                x=r["tt_obs"], y=r["I_obs"] * 1.04,
                text=f"<b>({r['h']}{r['k']}{r['l']})</b>",
                showarrow=False,
                font=dict(size=8, color="#a5d6a7"),
                textangle=-65,
            )

    # Unindexed peaks
    no_idx = [r for r in indexed if not r["indexed"]]
    if no_idx:
        fig.add_trace(go.Scatter(
            x=[r["tt_obs"] for r in no_idx],
            y=[r["I_obs"]  for r in no_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#ef9a9a",
                        line=dict(color="white", width=0.8)),
            name="Unindexed peaks",
        ))

    # Calculated Bragg ticks
    fig.add_trace(go.Scatter(
        x=[r["tt_calc"] for r in hkl_table],
        y=[-0.025*intensity.max()] * len(hkl_table),
        mode="markers",
        marker=dict(symbol="line-ns", size=7, color="#ffcc80",
                    line=dict(color="#ffcc80", width=1.2)),
        name="Calc. Bragg pos.",
        hovertemplate="(%{customdata[0]}%{customdata[1]}%{customdata[2]}) "
                      "2θ=%{x:.3f}°<extra></extra>",
        customdata=[[r["h"],r["k"],r["l"]] for r in hkl_table],
    ))

    fig.update_layout(
        height=520,
        xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"▽ green = indexed  |  ▽ red = unindexed  |  | orange ticks = all calculated Bragg positions for {mineral_name}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Full index + structure factor table
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Peak Index & Structure Factor Results")

    rows = []
    for i,r in enumerate(indexed):
        hkl_str = f"({r['h']} {r['k']} {r['l']})" if r["indexed"] else "unindexed"
        rows.append({
            "Peak #":      i+1,
            "2θ_obs (°)":  round(r["tt_obs"],4),
            "2θ_calc (°)": round(r["tt_calc"],4) if r["indexed"] else "—",
            "Δ2θ (°)":     round(r["delta_2t"],4) if r["indexed"] else "—",
            "d_obs (Å)":   round(r["d_obs"],5),
            "d_calc (Å)":  round(r["d_calc"],5) if r["indexed"] else "—",
            "Δd (Å)":      round(r["delta_d"],5) if r["indexed"] else "—",
            "(hkl)":       hkl_str,
            "I_obs":       round(r["I_obs"],1),
            "|F(hkl)|":    round(r["|F|"],3) if r["indexed"] else "—",
            "φ (°)":       round(r["phase_deg"],2) if r["indexed"] else "—",
            "F_real":      round(r["F_real"],3) if r["indexed"] else "—",
            "F_imag":      round(r["F_imag"],3) if r["indexed"] else "—",
            "I∝|F|²":      round(r["F2"],2) if r["indexed"] else "—",
            "Mult.":       r["mult"] if r["indexed"] else "—",
            "FWHM (°)":    round(r["fwhm"],4),
        })

    df_res = pd.DataFrame(rows)

    # Colour-code indexed vs not
    def style_row(row):
        color = "color: #a5d6a7" if row["(hkl)"] != "unindexed" else "color: #ef9a9a"
        return [color]*len(row)

    st.dataframe(
        df_res.style.apply(style_row, axis=1),
        use_container_width=True, height=520,
    )

    # Quick stats
    if n_indexed > 0:
        ok_rows = [r for r in indexed if r["indexed"]]
        st.markdown(
            f"**Mean |Δ2θ|** = {np.mean([abs(r['delta_2t']) for r in ok_rows]):.4f}°  |  "
            f"**Mean |Δd|** = {np.mean([abs(r['delta_d']) for r in ok_rows]):.5f} Å  |  "
            f"**Mean |F|** = {np.mean([r['|F|'] for r in ok_rows]):.2f}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Phase & Argand diagrams
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    ok_rows = [r for r in indexed if r["indexed"]]

    if not ok_rows:
        st.info("No indexed peaks to display.")
    else:
        fig3 = make_subplots(rows=1, cols=2,
            specs=[[{"type":"polar"},{"type":"xy"}]],
            subplot_titles=["Polar Phase Plot (|F| vs φ)", "Argand Diagram (Complex Plane)"])

        colors = [f"hsl({int(i*360/max(len(ok_rows),1))},80%,60%)" for i in range(len(ok_rows))]

        # Polar
        fig3.add_trace(go.Scatterpolar(
            r=[r["|F|"] for r in ok_rows],
            theta=[r["phase_deg"] for r in ok_rows],
            mode="markers+text",
            text=[f"({r['h']}{r['k']}{r['l']})" for r in ok_rows],
            textfont=dict(size=8),
            textposition="top center",
            marker=dict(size=10, color=[r["|F|"] for r in ok_rows],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="|F|", x=0.45)),
            hovertemplate="<b>(%{text})</b><br>|F|=%{r:.2f}<br>φ=%{theta:.1f}°<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        # Argand — vectors from origin
        for i, r in enumerate(ok_rows):
            fig3.add_trace(go.Scatter(
                x=[0, r["F_real"]], y=[0, r["F_imag"]],
                mode="lines",
                line=dict(color=colors[i], width=1.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=2)

        fig3.add_trace(go.Scatter(
            x=[r["F_real"] for r in ok_rows],
            y=[r["F_imag"] for r in ok_rows],
            mode="markers+text",
            marker=dict(size=9, color=[r["|F|"] for r in ok_rows],
                        colorscale="Plasma", showscale=True,
                        colorbar=dict(title="|F|", x=1.02)),
            text=[f"({r['h']}{r['k']}{r['l']})" for r in ok_rows],
            textfont=dict(size=8), textposition="top center",
            hovertemplate="<b>(%{text})</b><br>F_re=%{x:.2f} F_im=%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)

        fig3.update_layout(
            height=500, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            polar=dict(bgcolor="#0e1117",
                       radialaxis=dict(color="gray"),
                       angularaxis=dict(color="gray")),
        )
        fig3.update_xaxes(title_text="F_real", zeroline=True, zerolinecolor="gray",
                          gridcolor="#2a2a2a", row=1, col=2)
        fig3.update_yaxes(title_text="F_imag", zeroline=True, zerolinecolor="gray",
                          gridcolor="#2a2a2a", scaleanchor="x", row=1, col=2)
        st.plotly_chart(fig3, use_container_width=True)

        # Structure factor bar chart
        st.subheader("|F(hkl)| — Structure Factor Amplitudes")
        labels = [f"({r['h']}{r['k']}{r['l']})" for r in ok_rows]
        fig_bar = go.Figure(go.Bar(
            x=labels,
            y=[r["|F|"] for r in ok_rows],
            marker=dict(color=[r["|F|"] for r in ok_rows], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="|F|")),
            hovertemplate="<b>%{x}</b><br>|F|=%{y:.3f}<extra></extra>",
        ))
        fig_bar.update_layout(
            xaxis_title="(hkl)", yaxis_title="|F(hkl)|",
            height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — d-spacing match chart
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("d-Spacing: Observed vs Calculated")

    ok_rows = [r for r in indexed if r["indexed"]]
    if not ok_rows:
        st.info("No indexed peaks.")
    else:
        d_obs_arr  = [r["d_obs"]  for r in ok_rows]
        d_calc_arr = [r["d_calc"] for r in ok_rows]
        labels     = [f"({r['h']}{r['k']}{r['l']})" for r in ok_rows]
        delta_arr  = [r["delta_d"] for r in ok_rows]

        fig4 = make_subplots(rows=1, cols=2,
            subplot_titles=["Obs vs Calc d-spacing", "Δd = d_obs − d_calc"])

        # Parity plot
        d_range = [min(d_calc_arr)*0.97, max(d_calc_arr)*1.03]
        fig4.add_trace(go.Scatter(
            x=d_range, y=d_range,
            mode="lines", line=dict(color="gray", dash="dash"),
            name="Perfect match", showlegend=False,
        ), row=1, col=1)
        fig4.add_trace(go.Scatter(
            x=d_calc_arr, y=d_obs_arr,
            mode="markers+text",
            marker=dict(size=10, color=[abs(d) for d in delta_arr],
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="|Δd| Å", x=0.45),
                        cmin=0, cmax=max(abs(d) for d in delta_arr)),
            text=labels, textfont=dict(size=8), textposition="top center",
            hovertemplate="<b>%{text}</b><br>d_calc=%{x:.4f} Å<br>d_obs=%{y:.4f} Å<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        # Residuals bar
        fig4.add_trace(go.Bar(
            x=labels, y=delta_arr,
            marker_color=["#a5d6a7" if d>=0 else "#ef9a9a" for d in delta_arr],
            hovertemplate="<b>%{x}</b><br>Δd=%{y:.5f} Å<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        fig4.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        fig4.update_xaxes(title_text="d_calc (Å)", gridcolor="#2a2a2a", row=1, col=1)
        fig4.update_yaxes(title_text="d_obs (Å)",  gridcolor="#2a2a2a", row=1, col=1)
        fig4.update_xaxes(title_text="(hkl)", gridcolor="#2a2a2a", row=1, col=2)
        fig4.update_yaxes(title_text="Δd (Å)", gridcolor="#2a2a2a", row=1, col=2)
        fig4.update_layout(
            height=420, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Statistics
        st.markdown(
            f"**RMS Δd** = {np.sqrt(np.mean(np.array(delta_arr)**2)):.5f} Å  |  "
            f"**Max |Δd|** = {max(abs(d) for d in delta_arr):.5f} Å  |  "
            f"**Mean Δd** = {np.mean(delta_arr):.5f} Å (systematic shift)"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Full calculated HKL table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"All Calculated Reflections — {mineral_name}")
    st.markdown(f"{len(hkl_table)} reflections in {tt_min}–{tt_max}° (HKL max = ±{hkl_max})")

    df_hkl = pd.DataFrame([{
        "h":r["h"],"k":r["k"],"l":r["l"],
        "d_calc (Å)":   round(r["d_calc"],5),
        "2θ_calc (°)":  round(r["tt_calc"],4),
        "Mult.":        r["mult"],
        "|F(hkl)|":     round(r["|F|"],3),
        "Phase φ (°)":  round(r["phase_deg"],2),
        "F_real":       round(r["F_real"],3),
        "F_imag":       round(r["F_imag"],3),
        "I∝|F|²":       round(r["F2"],2),
        "I·Mult":       round(r["F2"]*r["mult"],2),
    } for r in hkl_table])

    st.dataframe(
        df_hkl.style.background_gradient(subset=["|F(hkl)|","I·Mult"], cmap="plasma"),
        use_container_width=True, height=520
    )

    # Calculated pattern overview
    st.subheader("Calculated Stick Pattern")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=two_theta, y=intensity,
        mode="lines", name="Measured",
        line=dict(color="#455a64", width=1),
        opacity=0.6,
    ))
    for r in hkl_table:
        fig5.add_shape(type="line",
            x0=r["tt_calc"], x1=r["tt_calc"],
            y0=0, y1=r["F2"]*r["mult"]/max(1,max(rr["F2"]*rr["mult"] for rr in hkl_table))*intensity.max()*0.9,
            line=dict(color="#ffb300", width=1.5))
    fig5.update_layout(
        height=320, xaxis_title="2θ (°)", yaxis_title="Intensity",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Export
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Export Results")

    # ── Main results CSV
    export_rows = []
    for i,r in enumerate(indexed):
        export_rows.append({
            "peak_number":  i+1,
            "2theta_obs":   round(r["tt_obs"],4),
            "2theta_calc":  round(r["tt_calc"],4) if r["indexed"] else "",
            "delta_2theta": round(r["delta_2t"],4) if r["indexed"] else "",
            "d_obs_A":      round(r["d_obs"],5),
            "d_calc_A":     round(r["d_calc"],5) if r["indexed"] else "",
            "delta_d_A":    round(r["delta_d"],5) if r["indexed"] else "",
            "h":            r["h"] if r["indexed"] else "",
            "k":            r["k"] if r["indexed"] else "",
            "l":            r["l"] if r["indexed"] else "",
            "I_obs":        round(r["I_obs"],1),
            "FWHM_deg":     round(r["fwhm"],4),
            "|F_hkl|":      round(r["|F|"],3) if r["indexed"] else "",
            "phase_deg":    round(r["phase_deg"],2) if r["indexed"] else "",
            "F_real":       round(r["F_real"],3) if r["indexed"] else "",
            "F_imag":       round(r["F_imag"],3) if r["indexed"] else "",
            "I_F2":         round(r["F2"],2) if r["indexed"] else "",
            "multiplicity": r["mult"] if r["indexed"] else "",
            "indexed":      r["indexed"],
        })

    buf = io.StringIO()
    buf.write(f"# HKL Indexing & Structure Factors — {mineral_name}\n")
    buf.write(f"# lambda = {lam} Å  |  Space group: {mineral.get('sg','—')}\n")
    buf.write(f"# Peaks found: {n_found}  |  Indexed: {n_indexed}  |  M20 = {M20}\n")
    buf.write(f"# Cell: a={mineral['a']} b={mineral['b']} c={mineral['c']} "
              f"alpha={mineral['alpha']} beta={mineral['beta']} gamma={mineral['gamma']}\n")
    pd.DataFrame(export_rows).to_csv(buf, index=False)

    # ── Full HKL table CSV
    buf_hkl = io.StringIO()
    buf_hkl.write(f"# Full HKL Table — {mineral_name}\n")
    df_hkl.to_csv(buf_hkl, index=False)

    # ── Pattern CSV
    df_pattern = pd.DataFrame({"two_theta_deg": two_theta, "intensity": intensity})

    c1,c2,c3 = st.columns(3)
    c1.download_button(
        "⬇️ Indexed Results CSV",
        buf.getvalue(),
        f"hkl_indexed_{mineral_name[:10].replace(' ','_')}.csv",
        "text/csv", type="primary",
    )
    c2.download_button(
        "⬇️ Full HKL Table CSV",
        buf_hkl.getvalue(),
        f"hkl_table_{mineral_name[:10].replace(' ','_')}.csv",
        "text/csv",
    )
    c3.download_button(
        "⬇️ Raw Pattern CSV",
        df_pattern.to_csv(index=False),
        f"pattern_{mineral_name[:10].replace(' ','_')}.csv",
        "text/csv",
    )

    st.divider()
    st.markdown("**Mineral Structure Summary**")
    st.code(
f"""Phase         : {mineral_name}
Space group   : {mineral.get("sg","—")}
Crystal system: {mineral.get("system","—")}
Z             : {mineral.get("Z","—")}

Unit Cell:
  a = {mineral['a']:.4f} Å      α = {mineral['alpha']:.3f}°
  b = {mineral['b']:.4f} Å      β = {mineral['beta']:.3f}°
  c = {mineral['c']:.4f} Å      γ = {mineral['gamma']:.3f}°

Indexing:
  λ            = {lam:.5f} Å
  2θ range     = {tt_min}–{tt_max}°
  Tolerance    = ±{tol_deg}° in 2θ
  Peaks found  = {n_found}
  Peaks indexed= {n_indexed}  ({n_indexed/max(n_found,1)*100:.0f}%)
  M₂₀          = {M20}

Structure Factors:
  Method  : 4-Gaussian ASF (Int. Tables Vol. C)
  Debye-Waller correction applied
  F(hkl) = Σ fⱼ·DWⱼ·exp(2πi(hxⱼ+kyⱼ+lzⱼ))
""", language="text")

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Peak detection: scipy find_peaks + Gaussian fit  ·  "
    "d-spacing: Bragg's law d=λ/(2sinθ)  ·  "
    "Indexing: minimum Δ2θ match against calculated HKL table  ·  "
    "Structure factors: 4-Gaussian ASF + Debye-Waller (Int. Tables Vol. C)  ·  "
    "Figure of merit: de Wolff M₂₀"
)
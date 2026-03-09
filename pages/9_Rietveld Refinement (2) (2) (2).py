"""
Rietveld Refinement + xrdfit FWHM Engine
==========================================
Vollprofil-Verfeinerung mit xrdfit (lmfit/Pseudo-Voigt) als Peak-Fitting-Backend.
Kristallitgröße via Scherrer, Williamson-Hall, Halder-Wagner.

Abhängigkeiten:
    pip install streamlit numpy scipy matplotlib pandas lmfit xrdfit

Starten:
    streamlit run rietveld_xrdfit.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import io, re, warnings, tempfile, os
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

# ── xrdfit / lmfit imports (with graceful fallback) ──
try:
    from xrdfit.spectrum_fitting import FitSpectrum, PeakParams, MaximumParams
    from xrdfit.pv_fit import do_pv_fit
    XRDFIT_AVAILABLE = True
except ImportError:
    XRDFIT_AVAILABLE = False

try:
    import lmfit
    from lmfit.models import PseudoVoigtModel, LinearModel, ConstantModel
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rietveld · xrdfit · Kristallitgröße",
    page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
)

# ── Design tokens ──
C = dict(
    bg="#07090e", card="#0c0f17", border="#181d2a",
    accent="#5CF4C4", accent2="#FF5A5F", accent3="#FFD166",
    success="#06D6A0", warn="#FFB703", muted="#44516b", text="#dde3ed",
)
DARK=C["bg"]; CARD=C["card"]; ACC=C["accent"]; ACC2=C["accent2"]
ACC3=C["accent3"]; MUTED=C["muted"]; TXT=C["text"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
/* Override Space Grotesk — using Barlow Condensed for headers */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700;800&family=Fira+Code:wght@400;500&display=swap');
:root{{
  --bg:{C['bg']};--card:{C['card']};--border:{C['border']};
  --acc:{C['accent']};--acc2:{C['accent2']};--acc3:{C['accent3']};
  --ok:{C['success']};--muted:{C['muted']};--text:{C['text']};
}}
html,body,[data-testid="stAppViewContainer"]{{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Barlow Condensed',sans-serif;font-size:15px;
}}
[data-testid="stSidebar"]{{background:var(--card)!important;border-right:1px solid var(--border);}}
h1,h2,h3,h4{{font-family:'Barlow Condensed',sans-serif;font-weight:700;}}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:var(--card);border-radius:6px;padding:3px;gap:2px;border:1px solid var(--border);}}
.stTabs [data-baseweb="tab"]{{color:var(--muted);font-family:'Fira Code',monospace;font-size:.72rem;border-radius:4px;padding:5px 12px;}}
.stTabs [aria-selected="true"]{{background:var(--acc)!important;color:#000!important;font-weight:600;}}
/* Inputs */
.stNumberInput input,.stTextInput input,.stSelectbox select{{
  background:var(--card)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;font-family:'Fira Code',monospace;border-radius:4px;}}
/* Buttons */
.stButton>button{{background:transparent;border:1px solid var(--acc);color:var(--acc);
  font-family:'Fira Code',monospace;border-radius:4px;transition:all .15s;font-size:.82rem;}}
.stButton>button:hover{{background:var(--acc);color:#000;}}
.big-btn>button{{background:var(--acc)!important;color:#000!important;font-weight:700;
  border:none!important;border-radius:6px;width:100%;padding:10px;font-size:.95rem;}}
.sec-btn>button{{background:var(--acc3)!important;color:#000!important;font-weight:700;
  border:none!important;border-radius:6px;width:100%;padding:9px;}}
/* Metrics */
div[data-testid="stMetric"]{{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;}}
div[data-testid="stMetric"] label{{color:var(--muted)!important;font-size:.68rem;
  font-family:'Fira Code',monospace;text-transform:uppercase;letter-spacing:.05em;}}
div[data-testid="stMetric"] div{{color:var(--acc)!important;font-family:'Fira Code',monospace;font-size:1.15rem;}}
/* Upload */
[data-testid="stFileUploader"]{{background:var(--card);border:1px dashed var(--border);border-radius:8px;}}
/* Custom components */
.formula{{background:#080b12;border:1px solid var(--border);border-left:3px solid var(--acc3);
  border-radius:0 6px 6px 0;padding:10px 14px;font-family:'Fira Code',monospace;
  font-size:.78rem;color:var(--text);margin:8px 0;line-height:1.7;}}
.badge{{display:inline-block;background:var(--card);border:1px solid var(--border);
  border-radius:20px;padding:2px 10px;font-family:'Fira Code',monospace;font-size:.72rem;
  color:var(--muted);margin:2px;}}
.badge.ok{{border-color:var(--ok);color:var(--ok);}}
.badge.warn{{border-color:var(--acc3);color:var(--acc3);}}
.badge.err{{border-color:var(--acc2);color:var(--acc2);}}
.badge.info{{border-color:var(--acc);color:var(--acc);}}
.sec-hdr{{font-family:'Fira Code',monospace;font-size:.65rem;font-weight:500;
  text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin:14px 0 5px 0;border-bottom:1px solid var(--border);padding-bottom:3px;}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── PHYSICS HELPERS (shared with previous version) ──
# ══════════════════════════════════════════════════════════════════════════════

def parse_cif(content):
    data={}
    def ex(k):
        m=re.search(rf"_{k}\s+([\S]+)",content,re.IGNORECASE)
        if m:
            v=m.group(1).replace("(","").replace(")","")
            try: return float(v)
            except: return v
        return None
    data["a"]=ex("cell_length_a") or 5.
    data["b"]=ex("cell_length_b") or 5.
    data["c"]=ex("cell_length_c") or 5.
    data["alpha"]=ex("cell_angle_alpha") or 90.
    data["beta"]=ex("cell_angle_beta")   or 90.
    data["gamma"]=ex("cell_angle_gamma") or 90.
    sg=ex("symmetry_space_group_name_H-M") or ex("space_group_name_H-M_alt")
    sgn=ex("symmetry_Int_Tables_number")  or ex("space_group_IT_number")
    data["space_group"]=str(sg) if sg else (f"No.{int(sgn)}" if sgn else "P 1")
    data["sg_number"]=int(sgn) if sgn else 1
    atoms=[]
    lm=re.search(r"loop_.*?_atom_site_label.*?(?=loop_|\Z)",content,re.DOTALL|re.IGNORECASE)
    if lm:
        blk=lm.group(0); hdrs=re.findall(r"_atom_site_(\w+)",blk,re.IGNORECASE)
        rows=re.findall(r"^\s+([A-Za-z][A-Za-z0-9]*\d*\s+.*?)$",blk,re.MULTILINE)
        for row in rows:
            pts=row.split()
            if len(pts)<5: continue
            at={"label":pts[0],"type":re.sub(r"\d","",pts[0])}
            try:
                ix=next(i for i,h in enumerate(hdrs) if "fract_x" in h.lower() or h.lower()=="x")
                at["x"]=float(pts[ix+1].split("(")[0])
                at["y"]=float(pts[ix+2].split("(")[0])
                at["z"]=float(pts[ix+3].split("(")[0])
            except: at["x"]=at["y"]=at["z"]=0.
            try:    at["occ"]=float(pts[-2].split("(")[0])
            except: at["occ"]=1.
            try:    at["Biso"]=float(pts[-1].split("(")[0])
            except: at["Biso"]=1.
            atoms.append(at)
    data["atoms"]=atoms
    return data

def parse_diffractogram(content):
    tt,ii,ee=[],[],[]
    for line in content.splitlines():
        line=line.strip()
        if not line or line[0] in("#","!"): continue
        pts=line.split()
        try:
            t,i=float(pts[0]),float(pts[1])
            tt.append(t); ii.append(i)
            ee.append(float(pts[2]) if len(pts)>=3 else np.sqrt(max(i,1.)))
        except: pass
    return np.array(tt),np.array(ii),np.array(ee)

def get_crystal_system(sg):
    if sg<=2: return "triclinic"
    if sg<=15: return "monoclinic"
    if sg<=74: return "orthorhombic"
    if sg<=142: return "tetragonal"
    if sg<=194: return "hexagonal"
    return "cubic"

def compute_d(h,k,l,cell):
    a,b,c=cell["a"],cell["b"],cell["c"]
    al,be,ga=cell["alpha"],cell["beta"],cell["gamma"]
    sg=cell.get("sg_number",1); sys=get_crystal_system(sg)
    try:
        if sys=="cubic":        return a/np.sqrt(h*h+k*k+l*l)
        if sys=="tetragonal":   return 1/np.sqrt((h*h+k*k)/a**2+l*l/c**2)
        if sys=="orthorhombic": return 1/np.sqrt(h*h/a**2+k*k/b**2+l*l/c**2)
        if sys=="hexagonal":    return 1/np.sqrt(4/3*(h*h+h*k+k*k)/a**2+l*l/c**2)
        if sys=="monoclinic":
            bt=np.radians(be); sb,cb=np.sin(bt),np.cos(bt)
            return 1/np.sqrt(h*h/(a*sb)**2+k*k/b**2+l*l/(c*sb)**2-2*h*l*cb/(a*c*sb**2))
        AL,BE,GA=np.radians(al),np.radians(be),np.radians(ga)
        cAL,cBE,cGA=np.cos(AL),np.cos(BE),np.cos(GA)
        V=a*b*c*np.sqrt(1-cAL**2-cBE**2-cGA**2+2*cAL*cBE*cGA)
        s11=(b*c*np.sin(AL))**2; s22=(a*c*np.sin(BE))**2; s33=(a*b*np.sin(GA))**2
        s12=a*b*c**2*(cAL*cBE-cGA); s23=a**2*b*c*(cBE*cGA-cAL); s13=a*b**2*c*(cGA*cAL-cBE)
        return V/np.sqrt(s11*h*h+s22*k*k+s33*l*l+2*s12*h*k+2*s23*k*l+2*s13*h*l)
    except: return 0.

def generate_reflections(cell,wavelength,tt_max=90.,hkl_max=8):
    refs=[]
    for h in range(-hkl_max,hkl_max+1):
     for k in range(-hkl_max,hkl_max+1):
      for l in range(-hkl_max,hkl_max+1):
        if h==k==l==0: continue
        d=compute_d(h,k,l,cell)
        if d<=0: continue
        st2=wavelength/(2*d)
        if abs(st2)>1: continue
        tt2=np.degrees(2*np.arcsin(st2))
        if 0<tt2<=tt_max:
            refs.append({"h":h,"k":k,"l":l,"d":d,"two_theta":tt2,"multiplicity":1})
    refs.sort(key=lambda x:x["two_theta"])
    merged=[refs[0]] if refs else []
    for r in refs[1:]:
        if abs(r["two_theta"]-merged[-1]["two_theta"])<0.005:
            merged[-1]["multiplicity"]+=1
        else: merged.append(r)
    return merged

def pseudo_voigt_fn(x,x0,fwhm,eta=0.5):
    x=x-x0; sig=fwhm/(2*np.sqrt(2*np.log(2)))
    return eta/(1+(x/(fwhm/2))**2)+(1-eta)*np.exp(-x**2/(2*sig**2))

def caglioti_fwhm(tt,U,V,W):
    t=np.radians(tt/2)
    return np.sqrt(max(U*np.tan(t)**2+V*np.tan(t)+W,1e-9))

def lorentz_pol(tt):
    t=np.radians(tt/2)
    return (1+np.cos(np.radians(tt))**2)/(np.sin(t)**2*np.cos(t)+1e-12)

def chebyshev_bg(x,coeffs):
    xn=2*(x-x.min())/(x.max()-x.min())-1
    res=np.zeros_like(x)
    for i,c in enumerate(coeffs):
        res+=c*np.polynomial.chebyshev.chebval(xn,[0]*i+[1])
    return res

def calc_pattern(tt,refs,p):
    wl=p.get("wavelength",1.54056)
    U,V,W=p.get("U",.01),p.get("V",-.001),p.get("W",.005)
    eta=p.get("eta",.5); scale=p.get("scale",1.); zs=p.get("zero_shift",0.)
    Biso=p.get("Biso",1.); bg=p.get("bg_coeffs",[0.]*6)
    pat=np.zeros_like(tt)
    for r in refs:
        ttk=r["two_theta"]+zs
        fwhm=caglioti_fwhm(ttk,U,V,W); lp=lorentz_pol(ttk)
        dw=np.exp(-2*Biso*(np.sin(np.radians(ttk/2))/wl)**2)
        pat+=scale*r["multiplicity"]*lp*dw*pseudo_voigt_fn(tt,ttk,fwhm,eta)
    return pat+chebyshev_bg(tt,bg)

def calc_rfactors(obs,calc,w=None):
    if w is None: w=1./np.maximum(obs,1.)
    d=obs-calc
    Rp=100*np.sum(np.abs(d))/np.sum(np.abs(obs))
    Rwp=100*np.sqrt(np.sum(w*d**2)/np.sum(w*obs**2))
    chi2=np.sum(w*d**2)/max(len(obs)-10,1)
    return Rp,Rwp,chi2

def refine_pattern(tt,obs,refs,params_in,flags,wavelength,n_cycles=5):
    names,p0,lo,hi=[],[],[],[]
    def add(n,v,l,h):
        if flags.get(n): names.append(n);p0.append(v);lo.append(l);hi.append(h)
    add("scale",params_in.get("scale",1.),0,1e9)
    add("zero_shift",params_in.get("zero_shift",0.),-1,1)
    add("U",params_in.get("U",.01),0,5)
    add("V",params_in.get("V",-.001),-5,0)
    add("W",params_in.get("W",.005),1e-7,5)
    add("eta",params_in.get("eta",.5),0,1)
    add("Biso",params_in.get("Biso",1.),0,30)
    add("a",params_in.get("a",5.),0.1,100)
    add("b",params_in.get("b",5.),0.1,100)
    add("c",params_in.get("c",5.),0.1,100)
    for i in range(6): add(f"bg_{i}",params_in.get("bg_coeffs",[0.]*6)[i],-1e6,1e6)
    if not names:
        c=calc_pattern(tt,refs,params_in)
        rp,rwp,ch=calc_rfactors(obs,c)
        return {"params":params_in,"Rp":rp,"Rwp":rwp,"chi2":ch,"calc":c,"message":"No params"}
    w=1./np.maximum(obs,1.)
    def res(pv):
        pm=dict(params_in); pm["wavelength"]=wavelength
        for n,v in zip(names,pv):
            if n.startswith("bg_"):
                bg=list(pm.get("bg_coeffs",[0.]*6)); bg[int(n[3:])]=v; pm["bg_coeffs"]=bg
            else: pm[n]=v
        return np.sqrt(w)*(obs-calc_pattern(tt,refs,pm))
    r=least_squares(res,p0,bounds=(lo,hi),method="trf",max_nfev=n_cycles*300,
                    ftol=1e-10,xtol=1e-10)
    pm_out=dict(params_in); pm_out["wavelength"]=wavelength
    for n,v in zip(names,r.x):
        if n.startswith("bg_"):
            bg=list(pm_out.get("bg_coeffs",[0.]*6)); bg[int(n[3:])]=v; pm_out["bg_coeffs"]=bg
        else: pm_out[n]=v
    c=calc_pattern(tt,refs,pm_out); rp,rwp,ch=calc_rfactors(obs,c,w)
    return {"params":pm_out,"Rp":rp,"Rwp":rwp,"chi2":ch,"calc":c,"message":r.message}

# ══════════════════════════════════════════════════════════════════════════════
# ── xrdfit / lmfit PEAK FITTING ENGINE ──
# ══════════════════════════════════════════════════════════════════════════════

def fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg=1.5,
                   profile="PseudoVoigt", bg_model="linear"):
    """
    Fit a single peak using lmfit directly (mirrors xrdfit's pv_fit.do_pv_fit).
    Supports: PseudoVoigt, Voigt, Lorentzian, Gaussian profiles.
    Background: linear or constant.
    Returns dict with fwhm, eta, center, amplitude, area, beta, errors.
    """
    if not LMFIT_AVAILABLE:
        return None
    mask=np.abs(tt_arr-tt_center)<window_deg
    if mask.sum()<8: return None
    x,y=tt_arr[mask],obs_arr[mask]
    # Choose background
    if bg_model=="linear":
        bg=LinearModel(prefix="bg_")
    else:
        bg=ConstantModel(prefix="bg_")
    # Choose peak profile
    profile_map={
        "PseudoVoigt": PseudoVoigtModel,
        "Voigt":       lmfit.models.VoigtModel,
        "Lorentzian":  lmfit.models.LorentzianModel,
        "Gaussian":    lmfit.models.GaussianModel,
    }
    PeakModel=profile_map.get(profile,PseudoVoigtModel)
    peak=PeakModel(prefix="pk_")
    model=peak+bg

    # Initial parameter guesses
    bg_est=(y[0]+y[-1])/2
    y_sub=y-bg_est
    A0=float(y_sub.max())
    if A0<=0: return None
    params=model.make_params()
    params["pk_center"].set(value=tt_center, min=tt_center-window_deg, max=tt_center+window_deg)
    params["pk_amplitude"].set(value=A0*window_deg*0.8, min=0)
    if "pk_sigma" in params:
        params["pk_sigma"].set(value=window_deg*0.2, min=1e-4, max=window_deg)
    if "pk_fraction" in params:
        params["pk_fraction"].set(value=0.5, min=0, max=1)
    if bg_model=="linear":
        params["bg_slope"].set(value=0)
        params["bg_intercept"].set(value=bg_est)
    else:
        params["bg_c"].set(value=bg_est)

    try:
        result=model.fit(y,params,x=x,method="least_squares")
        pv=result.params

        center=float(pv["pk_center"].value)
        # FWHM: lmfit stores fwhm as derived param for most models
        if "pk_fwhm" in pv:
            fwhm=float(pv["pk_fwhm"].value)
            fwhm_err=float(pv["pk_fwhm"].stderr) if pv["pk_fwhm"].stderr else 0.
        else:
            sig=float(pv["pk_sigma"].value)
            fwhm=2.355*sig
            fwhm_err=2.355*float(pv["pk_sigma"].stderr) if pv["pk_sigma"].stderr else 0.
        eta=float(pv["pk_fraction"].value) if "pk_fraction" in pv else 0.5
        amp=float(pv["pk_amplitude"].value)

        # Background at peak position
        if bg_model=="linear":
            bg_at_peak=float(pv["bg_slope"].value)*center+float(pv["bg_intercept"].value)
        else:
            bg_at_peak=float(pv["bg_c"].value)

        # Compute peak-only curve for integral breadth
        peak_only=peak.eval(pv,x=x)-bg_at_peak
        peak_max=float(peak_only.max())
        area=float(np.trapz(np.maximum(peak_only,0),x))
        beta=area/peak_max if peak_max>0 else fwhm

        # lmfit fit statistics
        redchi=float(result.redchi) if result.redchi else 0.
        aic=float(result.aic)   if result.aic    else 0.

        return {
            "two_theta": center,
            "fwhm": fwhm, "fwhm_err": fwhm_err,
            "eta": eta, "amplitude": amp,
            "background": bg_at_peak,
            "beta": beta, "area": area,
            "redchi": redchi, "aic": aic,
            "profile": profile,
            "report": result.fit_report(),
            "x_fit": x, "y_fit": y,
            "y_best": result.best_fit,
        }
    except Exception as e:
        return None

def fit_peak_xrdfit(tt_arr, obs_arr, tt_center, window_deg=1.5):
    """
    Use xrdfit's FitSpectrum / PeakParams / MaximumParams via a temp .dat file.
    Falls back to lmfit direct if xrdfit is unavailable.
    """
    if not XRDFIT_AVAILABLE:
        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg, "PseudoVoigt")

    # xrdfit expects a tab-separated file: two_theta [TAB] intensity
    mask=np.abs(tt_arr-tt_center)<window_deg*1.5
    if mask.sum()<8:
        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg)
    x_sub,y_sub=tt_arr[mask],obs_arr[mask]

    try:
        # Write temp file
        with tempfile.NamedTemporaryFile(mode="w",suffix=".dat",delete=False) as f:
            for xi,yi in zip(x_sub,y_sub):
                f.write(f"{xi:.6f}\t{yi:.6f}\n")
            tmpfile=f.name

        spec=FitSpectrum(tmpfile, first_cake_angle=90, delimiter="\t")
        lo=float(x_sub.min()); hi=float(x_sub.max())
        peak_name=f"peak_{tt_center:.2f}"
        max_lo=tt_center-window_deg*0.3
        max_hi=tt_center+window_deg*0.3
        mp=MaximumParams(peak_name,(max_lo,max_hi))
        pp=PeakParams((lo,hi),[mp])
        spec.fit_peaks(pp,cakes_to_fit=1)
        pfit=spec.get_fit(peak_name)

        # Extract results from lmfit result inside xrdfit PeakFit
        res=pfit.fit_result
        pv=res.params
        center=float(pv[f"{peak_name}_center"].value)
        fwhm=float(pv[f"{peak_name}_fwhm"].value)  if f"{peak_name}_fwhm" in pv else \
              2.355*float(pv[f"{peak_name}_sigma"].value)
        fwhm_err=0.
        if f"{peak_name}_fwhm" in pv and pv[f"{peak_name}_fwhm"].stderr:
            fwhm_err=float(pv[f"{peak_name}_fwhm"].stderr)
        elif f"{peak_name}_sigma" in pv and pv[f"{peak_name}_sigma"].stderr:
            fwhm_err=2.355*float(pv[f"{peak_name}_sigma"].stderr)

        eta=float(pv[f"{peak_name}_fraction"].value) if f"{peak_name}_fraction" in pv else 0.5
        amp=float(pv[f"{peak_name}_amplitude"].value)

        x_fit=np.array(x_sub); y_best=res.best_fit
        bg_at_peak=float(np.interp(center,x_fit,y_best-res.eval_components().get(f"{peak_name}_",0.)))
        peak_only=np.maximum(res.eval_components().get(f"{peak_name}_",y_best-y_sub.min()),0)
        peak_max=float(peak_only.max())
        area=float(np.trapz(peak_only,x_fit))
        beta=area/peak_max if peak_max>0 else fwhm

        os.unlink(tmpfile)
        return {
            "two_theta": center, "fwhm": fwhm, "fwhm_err": fwhm_err,
            "eta": eta, "amplitude": amp, "background": bg_at_peak,
            "beta": beta, "area": area,
            "redchi": float(res.redchi) if res.redchi else 0.,
            "aic": float(res.aic) if res.aic else 0.,
            "profile": "xrdfit/PseudoVoigt",
            "report": res.fit_report(),
            "x_fit": x_fit, "y_fit": y_sub, "y_best": y_best,
        }
    except Exception:
        try: os.unlink(tmpfile)
        except: pass
        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg)

def extract_all_fwhm(tt_arr, obs_arr, refs, wavelength,
                     window_deg=1.5, min_intensity_pct=2.,
                     backend="xrdfit", profile="PseudoVoigt", bg_model="linear"):
    """Iterate over all reflections and fit each peak."""
    I_max=obs_arr.max()
    results=[]
    for r in refs:
        ttk=r["two_theta"]
        if ttk<tt_arr.min() or ttk>tt_arr.max(): continue
        idx=np.argmin(np.abs(tt_arr-ttk))
        local_max=obs_arr[max(0,idx-15):min(len(obs_arr),idx+15)].max()
        if local_max<min_intensity_pct/100*I_max: continue

        if backend=="xrdfit":
            fit=fit_peak_xrdfit(tt_arr,obs_arr,ttk,window_deg)
        else:
            fit=fit_peak_lmfit(tt_arr,obs_arr,ttk,window_deg,profile,bg_model)
        if fit is None: continue

        theta_rad=np.radians(fit["two_theta"]/2)
        sin_t=np.sin(theta_rad); cos_t=np.cos(theta_rad)
        d_val=wavelength/(2*sin_t) if sin_t>0 else r["d"]
        fit.update({
            "h":r["h"],"k":r["k"],"l":r["l"],"d":d_val,
            "theta_rad":theta_rad,"sin_theta":sin_t,"cos_theta":cos_t,
            "fwhm_rad":np.radians(fit["fwhm"]),
            "beta_rad":np.radians(fit["beta"]),
        })
        results.append(fit)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# ── SCHERRER, WILLIAMSON-HALL, HALDER-WAGNER ──
# ══════════════════════════════════════════════════════════════════════════════

def scherrer_analysis(peaks,wavelength,K=0.9,use_beta=False):
    rows=[]
    for p in peaks:
        beta=p["beta_rad"] if use_beta else p["fwhm_rad"]
        if beta<=0 or p["cos_theta"]<=0: continue
        D=K*wavelength/(beta*p["cos_theta"])
        rows.append({"hkl":f"{p['h']}{p['k']}{p['l']}",
                     "2θ (°)":round(p["two_theta"],3),
                     "FWHM (°)":round(p["fwhm"],5),
                     "β (°)":round(p["beta"],5),
                     "D (nm)":round(D*10,2)})
    if not rows: return pd.DataFrame(),None
    df=pd.DataFrame(rows)
    return df,{"D_mean_nm":df["D (nm)"].mean(),"D_std_nm":df["D (nm)"].std()}

def williamson_hall(peaks,wavelength,use_beta=False):
    xs,ys,lbls=[],[],[]
    for p in peaks:
        beta=p["beta_rad"] if use_beta else p["fwhm_rad"]
        if beta<=0: continue
        xs.append(4*p["sin_theta"]/wavelength)
        ys.append(beta*p["cos_theta"]/wavelength)
        lbls.append(f"{p['h']}{p['k']}{p['l']}")
    if len(xs)<2: return None,None,None,xs,ys,lbls
    xa,ya=np.array(xs),np.array(ys)
    coeffs,cov=np.polyfit(xa,ya,1,cov=True)
    sl,ic=coeffs; se=np.sqrt(cov[0,0]); ie=np.sqrt(cov[1,1])
    D_nm=(1/ic)*10 if ic>0 else np.nan
    return {"D_nm":D_nm,"eps":sl,"slope":sl,"intercept":ic,
            "slope_err":se,"int_err":ie},np.polyval(coeffs,xa),coeffs,xs,ys,lbls

def halder_wagner(peaks,wavelength,use_beta=False):
    xs,ys,lbls=[],[],[]
    for p in peaks:
        beta=p["beta_rad"] if use_beta else p["fwhm_rad"]
        if beta<=0: continue
        d_s=2*p["sin_theta"]/wavelength
        b_s=2*p["cos_theta"]*beta/wavelength
        if d_s==0: continue
        xs.append(b_s/d_s**2); ys.append((b_s/d_s)**2)
        lbls.append(f"{p['h']}{p['k']}{p['l']}")
    if len(xs)<2: return None,None,None,xs,ys,lbls
    xa,ya=np.array(xs),np.array(ys)
    coeffs,cov=np.polyfit(xa,ya,1,cov=True)
    sl,ic=coeffs; se=np.sqrt(cov[0,0]); ie=np.sqrt(cov[1,1])
    D_nm=(1/sl)*10 if sl>0 else np.nan
    eps=np.sqrt(max(ic,0))/2
    return {"D_nm":D_nm,"eps":eps,"slope":sl,"intercept":ic,
            "slope_err":se,"int_err":ie},np.polyval(coeffs,xa),coeffs,xs,ys,lbls

# ══════════════════════════════════════════════════════════════════════════════
# ── PLOTTING ──
# ══════════════════════════════════════════════════════════════════════════════

def _ax(ax,xl="",yl="",title=""):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(MUTED)
    ax.tick_params(colors=MUTED,which="both",length=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if xl: ax.set_xlabel(xl,color=MUTED,fontsize=9)
    if yl: ax.set_ylabel(yl,color=MUTED,fontsize=9)
    if title: ax.set_title(title,color=TXT,fontsize=10,fontweight="bold")

def fig_buf(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=DARK)
    plt.close(fig); buf.seek(0); return buf

def plot_rietveld(tt,obs,calc,refs,title="Rietveld"):
    fig=plt.figure(figsize=(13,7),facecolor=DARK)
    gs=gridspec.GridSpec(2,1,height_ratios=[4,1],hspace=.05)
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
    _ax(ax1,yl="Intensity",title=title); _ax(ax2,xl="2θ (°)",yl="Δ")
    ax1.plot(tt,obs,color=TXT,lw=.9,label="Observed")
    ax1.plot(tt,calc,color=ACC,lw=1.5,label="Calculated",zorder=3)
    ax1.fill_between(tt,obs,calc,alpha=.12,color=ACC2)
    if refs:
        ttk=[r["two_theta"] for r in refs if tt.min()<=r["two_theta"]<=tt.max()]
        y0=obs.min()-(obs.max()-obs.min())*.06
        ax1.vlines(ttk,y0*.95,y0*.82,color=C["success"],lw=.7,alpha=.8,label="hkl")
    ax1.legend(facecolor=CARD,edgecolor=MUTED,labelcolor=TXT,fontsize=8)
    ax1.set_xlim(tt.min(),tt.max())
    plt.setp(ax1.get_xticklabels(),visible=False)
    ax2.plot(tt,obs-calc,color=ACC2,lw=.8)
    ax2.axhline(0,color=MUTED,lw=.5,ls="--")
    ax2.set_xlim(tt.min(),tt.max())
    return fig_buf(fig)

def plot_single_peak_fit(pfit):
    """Show xrdfit/lmfit fit result for one peak."""
    if pfit is None or "x_fit" not in pfit: return None
    fig,ax=plt.subplots(figsize=(6,3.5),facecolor=DARK)
    _ax(ax,"2θ (°)","Intensity",f"Peak fit — {pfit.get('profile','?')}")
    ax.plot(pfit["x_fit"],pfit["y_fit"],color=TXT,lw=1.,label="Data",marker="o",ms=2)
    ax.plot(pfit["x_fit"],pfit["y_best"],color=ACC,lw=1.8,label="Fit")
    ax.axvline(pfit["two_theta"],color=ACC3,lw=.7,ls="--",alpha=.8,label="Center")
    # FWHM bar
    fwhm=pfit["fwhm"]
    y_half=pfit["amplitude"]/2+pfit["background"]
    ax.annotate("",xy=(pfit["two_theta"]+fwhm/2,y_half),
                xytext=(pfit["two_theta"]-fwhm/2,y_half),
                arrowprops=dict(arrowstyle="<->",color=ACC2,lw=1.2))
    ax.text(pfit["two_theta"],y_half*1.02,f"FWHM={fwhm:.4f}°",
            color=ACC2,ha="center",fontsize=7,fontfamily="monospace")
    ax.legend(facecolor=CARD,edgecolor=MUTED,labelcolor=TXT,fontsize=8)
    fig.tight_layout(); return fig_buf(fig)

def plot_fwhm_overview(peaks,U,V,W,tt_range):
    tt_line=np.linspace(tt_range[0],tt_range[1],500)
    fw_line=np.array([caglioti_fwhm(t,U,V,W) for t in tt_line])
    fig,axes=plt.subplots(1,2,figsize=(13,4),facecolor=DARK)
    # Panel 1: FWHM vs 2θ
    ax=axes[0]; _ax(ax,"2θ (°)","FWHM (°)","FWHM²(2θ) — Caglioti vs Messwerte")
    ax.plot(tt_line,fw_line,color=ACC,lw=1.5,label="Caglioti (verfeinert)")
    tt_pts=[p["two_theta"] for p in peaks]; fw_pts=[p["fwhm"] for p in peaks]
    fw_err=[p["fwhm_err"] for p in peaks]
    ax.errorbar(tt_pts,fw_pts,yerr=fw_err,fmt="o",color=ACC2,ms=5,
                ecolor=MUTED,elinewidth=.8,capsize=2,label="xrdfit/lmfit Fits")
    ax.legend(facecolor=CARD,edgecolor=MUTED,labelcolor=TXT,fontsize=8)
    # Panel 2: FWHM² vs tan²θ
    ax2=axes[1]; _ax(ax2,"tan²θ","FWHM² (°²)","Linearisiert: FWHM²=U·tan²θ+V·tanθ+W")
    tan2_line=np.tan(np.radians(tt_line/2))**2
    ax2.plot(tan2_line,fw_line**2,color=ACC,lw=1.5)
    tan2_pts=[np.tan(np.radians(t/2))**2 for t in tt_pts]
    fw2_pts=[f**2 for f in fw_pts]
    ax2.scatter(tan2_pts,fw2_pts,color=ACC2,s=35,zorder=4)
    for t2,f2,p in zip(tan2_pts,fw2_pts,peaks):
        ax2.annotate(f"{p['h']}{p['k']}{p['l']}",(t2,f2),
                     textcoords="offset points",xytext=(4,3),
                     fontsize=6.5,color=MUTED,fontfamily="monospace")
    fig.tight_layout(pad=1.5); return fig_buf(fig)

def plot_linear_analysis(xs,ys,lbls,coeffs,fit_y,xlabel,ylabel,title,color_data=ACC2,color_fit=ACC):
    fig,ax=plt.subplots(figsize=(8,5),facecolor=DARK)
    _ax(ax,xlabel,ylabel,title)
    ax.scatter(xs,ys,color=color_data,s=65,zorder=4,label="Datenpunkte")
    if fit_y is not None:
        xs_s=np.sort(xs)
        ax.plot(xs_s,np.polyval(coeffs,xs_s),color=color_fit,lw=1.5,label="Linearer Fit")
    for x,y,lbl in zip(xs,ys,lbls):
        ax.annotate(lbl,(x,y),textcoords="offset points",xytext=(5,4),
                    fontsize=7.5,color=MUTED,fontfamily="monospace")
    ax.legend(facecolor=CARD,edgecolor=MUTED,labelcolor=TXT,fontsize=8)
    fig.tight_layout(); return fig_buf(fig)

def plot_size_comparison(methods):
    names=[k for k,v in methods.items() if v is not None and not np.isnan(v)]
    vals=[methods[k] for k in names]
    if not vals: return None
    fig,ax=plt.subplots(figsize=(7,3.5),facecolor=DARK)
    _ax(ax,"","Kristallitgröße D (nm)","Vergleich aller Methoden")
    cols=[ACC,ACC2,ACC3,C["success"]][:len(names)]
    bars=ax.bar(names,vals,color=cols,width=.4)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+.5,f"{v:.1f} nm",
                ha="center",va="bottom",color=TXT,fontsize=9,fontfamily="monospace")
    ax.set_ylim(0,max(vals)*1.3)
    fig.tight_layout(); return fig_buf(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ── SESSION STATE ──
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFS={
    "cell":{},"atoms":[],"two_theta":None,"obs":None,"error":None,
    "reflections":[],
    "params":{"scale":1.,"zero_shift":0.,"U":.01,"V":-.001,"W":.005,
              "eta":.5,"Biso":1.,"bg_coeffs":[0.]*6},
    "calc":None,"R_vals":None,"peak_fits":[],
    "wh_result":None,"hw_result":None,"scherrer_result":None,
}
for k,v in _SS_DEFS.items():
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# ── HEADER ──
# ══════════════════════════════════════════════════════════════════════════════
# Backend status badges
xrdfit_badge=("<span class='badge ok'>✓ xrdfit</span>" if XRDFIT_AVAILABLE
              else "<span class='badge warn'>⚠ xrdfit nicht installiert</span>")
lmfit_badge=("<span class='badge ok'>✓ lmfit</span>" if LMFIT_AVAILABLE
             else "<span class='badge err'>✗ lmfit fehlt</span>")
st.markdown(f"""
<div style='padding:6px 0 16px 0;'>
  <span style='font-family:Barlow Condensed,sans-serif;font-size:2.1rem;font-weight:800;
               color:{ACC};letter-spacing:-.5px;'>Rietveld · xrdfit</span>
  <span style='font-family:Fira Code,monospace;font-size:.72rem;color:{MUTED};margin-left:12px;'>
    Vollprofil-FWHM · Scherrer · Williamson-Hall · Halder-Wagner
  </span><br>
  <div style='margin-top:6px;'>{xrdfit_badge} {lmfit_badge}
    <span class='badge info'>lmfit backend always active</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ──
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sec-hdr">Datei-Import</div>', unsafe_allow_html=True)
    cif_f=st.file_uploader("CIF-Datei",type=["cif"])
    diff_f=st.file_uploader("Diffraktogramm (.xy .dat .xye .txt)",type=["xy","dat","xye","txt","csv"])

    if cif_f:
        raw=cif_f.read().decode("utf-8",errors="ignore")
        p=parse_cif(raw); st.session_state.cell=p; st.session_state.atoms=p.get("atoms",[])
        for k in ("a","b","c","alpha","beta","gamma"):
            st.session_state.params[k]=p.get(k,5.)
        st.success(f"✓ {p.get('space_group','?')}")

    if diff_f:
        raw=diff_f.read().decode("utf-8",errors="ignore")
        tt,obs,err=parse_diffractogram(raw)
        if len(tt)>5:
            st.session_state.two_theta=tt; st.session_state.obs=obs; st.session_state.error=err
            st.session_state.params["scale"]=float(obs.max())*.01
            st.success(f"✓ {len(tt)} Punkte")
        else: st.error("Parse-Fehler")

    st.markdown('<div class="sec-hdr">Messparameter</div>', unsafe_allow_html=True)
    wavelength=st.number_input("λ (Å)",value=1.54056,step=1e-5,format="%.5f",
                                help="CuKα1=1.54056 · MoKα1=0.70930 · CoKα1=1.78897")
    K_sch=st.number_input("Scherrer K",value=0.9,step=.01,format="%.3f")
    use_beta=st.checkbox("Integrale Breite β statt FWHM",value=False)
    hkl_max=st.slider("hkl-Maximum",2,15,8)
    tt_max=st.slider("2θ-Maximum (°)",30.,150.,90.,5.)

    if st.button("🔄 Reflexliste erzeugen"):
        if st.session_state.cell:
            c2={**st.session_state.cell,
                "a":st.session_state.params.get("a",st.session_state.cell.get("a",5)),
                "b":st.session_state.params.get("b",st.session_state.cell.get("b",5)),
                "c":st.session_state.params.get("c",st.session_state.cell.get("c",5))}
            with st.spinner("…"):
                refs=generate_reflections(c2,wavelength,tt_max,hkl_max)
            st.session_state.reflections=refs
            st.success(f"{len(refs)} Reflexe")
        else: st.warning("CIF zuerst laden")

    st.markdown('<div class="sec-hdr">Rietveld-Verfeinerung</div>', unsafe_allow_html=True)
    n_cyc=st.slider("Zyklen",1,30,8)
    rf={}
    fp=[("scale","Skalierung"),("zero_shift","Nullpunkt"),
        ("U","U"),("V","V"),("W","W"),("eta","η"),("Biso","B_iso"),
        ("a","a"),("b","b"),("c","c")]
    cc=st.columns(2)
    for i,(k,lbl) in enumerate(fp):
        with cc[i%2]: rf[k]=st.checkbox(lbl,value=(k in ("scale","zero_shift","W","Biso")),key=f"rf_{k}")
    rf_bg=st.checkbox("Hintergrund",value=True)
    for i in range(6): rf[f"bg_{i}"]=rf_bg

    st.markdown('<div class="sec-hdr">xrdfit / lmfit Peak-Fit</div>', unsafe_allow_html=True)

    backend_choice=st.selectbox(
        "Peak-Fit Backend",
        ["xrdfit (PseudoVoigt)" if XRDFIT_AVAILABLE else "xrdfit (nicht verf. → lmfit)",
         "lmfit – PseudoVoigt","lmfit – Voigt","lmfit – Lorentzian","lmfit – Gaussian"],
        index=0
    )
    if "xrdfit" in backend_choice and XRDFIT_AVAILABLE:
        fit_backend="xrdfit"; fit_profile="PseudoVoigt"
    else:
        fit_backend="lmfit"
        fit_profile=backend_choice.split("– ")[-1] if "–" in backend_choice else "PseudoVoigt"

    bg_choice=st.selectbox("Hintergrund-Modell",["linear","constant"],index=0)
    fwhm_window=st.slider("Fit-Fenster (°)",0.3,3.0,1.0,.1)
    min_int=st.slider("Min. Intensität (%)",0.5,20.,3.,.5)

# ══════════════════════════════════════════════════════════════════════════════
# ── TABS ──
# ══════════════════════════════════════════════════════════════════════════════
tabs=st.tabs(["📊 Daten","🔬 Struktur","⚙️ Parameter","▶ Rietveld",
               "🔬 xrdfit FWHM","📐 Caglioti","🔭 Scherrer",
               "📈 Williamson-Hall","📉 Halder-Wagner","📋 Ergebnisse"])
(t_data,t_struct,t_param,t_riet,
 t_xrdfit,t_cag,t_sch,t_wh,t_hw,t_res)=tabs

# ─────────── TAB: Daten ───────────
with t_data:
    if st.session_state.obs is not None:
        tt=st.session_state.two_theta; obs=st.session_state.obs
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Punkte",f"{len(tt)}")
        c2.metric("2θ",f"{tt.min():.2f}° – {tt.max():.2f}°")
        c3.metric("I_max",f"{obs.max():.0f}")
        c4.metric("SNR (est.)",f"{obs.max()/np.sqrt(obs.max()+1):.1f}")
        ca,cb=st.columns(2)
        with ca: tlo=st.number_input("von (°)",value=float(tt.min()),step=.1,key="tlo")
        with cb: thi=st.number_input("bis (°)",value=float(tt.max()),step=.1,key="thi")
        mask=(tt>=tlo)&(tt<=thi)
        if mask.sum()>5:
            st.session_state.two_theta=tt[mask]; st.session_state.obs=obs[mask]
            if st.session_state.error is not None:
                st.session_state.error=st.session_state.error[mask]
        fig,ax=plt.subplots(figsize=(12,3.5),facecolor=DARK)
        _ax(ax,"2θ (°)","Intensity","Diffraktogramm")
        ax.plot(st.session_state.two_theta,st.session_state.obs,color=ACC,lw=.8)
        fig.tight_layout(); st.image(fig_buf(fig),use_container_width=True)

        if st.checkbox("🔍 Automatische Peaksuche"):
            ht=st.slider("Min. I (%)",1,50,5)/100*obs.max()
            pks,_=find_peaks(st.session_state.obs,height=ht,distance=5)
            pt=st.session_state.two_theta[pks]
            st.dataframe(pd.DataFrame({"2θ (°)":pt.round(4),
                "d (Å)":(wavelength/(2*np.sin(np.radians(pt/2)))).round(4),
                "I":st.session_state.obs[pks].round(0)}),use_container_width=True)
    else:
        st.info("Diffraktogramm in der Seitenleiste laden.")

# ─────────── TAB: Struktur ───────────
with t_struct:
    if st.session_state.cell:
        cell=st.session_state.cell; sgn=cell.get("sg_number",1)
        st.markdown(f"""<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;'>
          <span class='badge ok'>{cell.get('space_group','?')}</span>
          <span class='badge'>Nr.{sgn}</span>
          <span class='badge'>{get_crystal_system(sgn).upper()}</span></div>""",
                    unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            a_v=st.number_input("a (Å)",value=float(st.session_state.params.get("a",cell.get("a",5.))),step=.001,format="%.5f",key="sa")
            b_v=st.number_input("b (Å)",value=float(st.session_state.params.get("b",cell.get("b",5.))),step=.001,format="%.5f",key="sb")
            c_v=st.number_input("c (Å)",value=float(st.session_state.params.get("c",cell.get("c",5.))),step=.001,format="%.5f",key="sc")
        with c2:
            al_v=st.number_input("α (°)",value=float(cell.get("alpha",90.)),step=.01,format="%.4f")
            be_v=st.number_input("β (°)",value=float(cell.get("beta",90.)),step=.01,format="%.4f")
            ga_v=st.number_input("γ (°)",value=float(cell.get("gamma",90.)),step=.01,format="%.4f")
        for k,v in [("a",a_v),("b",b_v),("c",c_v),("alpha",al_v),("beta",be_v),("gamma",ga_v)]:
            st.session_state.params[k]=v; st.session_state.cell[k]=v
        AL,BE,GA=np.radians(al_v),np.radians(be_v),np.radians(ga_v)
        vol=a_v*b_v*c_v*np.sqrt(1-np.cos(AL)**2-np.cos(BE)**2-np.cos(GA)**2+2*np.cos(AL)*np.cos(BE)*np.cos(GA))
        st.metric("Volumen (ų)",f"{vol:.4f}")
        if st.session_state.atoms:
            df_at=pd.DataFrame([{"Label":a["label"],"Typ":a["type"],
                                  "x":a["x"],"y":a["y"],"z":a["z"],
                                  "Occ":a["occ"],"B_iso":a["Biso"]}
                                 for a in st.session_state.atoms])
            ed=st.data_editor(df_at,use_container_width=True,num_rows="dynamic")
            st.session_state.atoms=[{"label":r.Label,"type":r.Typ,
                                      "x":r.x,"y":r.y,"z":r.z,
                                      "occ":r.Occ,"Biso":r.B_iso}
                                     for _,r in ed.iterrows()]
        if st.session_state.reflections:
            ref_df=pd.DataFrame([{"h":r["h"],"k":r["k"],"l":r["l"],
                                   "d(Å)":round(r["d"],4),"2θ(°)":round(r["two_theta"],4),
                                   "m":r["multiplicity"]}
                                  for r in st.session_state.reflections])
            st.dataframe(ref_df,use_container_width=True,height=280)
    else:
        st.info("CIF-Datei laden.")

# ─────────── TAB: Parameter ───────────
with t_param:
    st.markdown('<div class="sec-hdr">Caglioti Profilparameter</div>', unsafe_allow_html=True)
    st.markdown("""<div class='formula'>
FWHM²(θ) = U·tan²θ + V·tanθ + W<br>
Pseudo-Voigt: f(x) = η·L(x) + (1-η)·G(x)   η=0→Gauß  η=1→Lorentz
    </div>""", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: U_v=st.number_input("U",value=float(st.session_state.params.get("U",.01)),step=.001,format="%.5f")
    with c2: V_v=st.number_input("V",value=float(st.session_state.params.get("V",-.001)),step=.001,format="%.5f")
    with c3: W_v=st.number_input("W",value=float(st.session_state.params.get("W",.005)),step=.0001,format="%.6f")
    eta_v=st.slider("η",0.,1.,float(st.session_state.params.get("eta",.5)),.01)
    for k,v in [("U",U_v),("V",V_v),("W",W_v),("eta",eta_v)]: st.session_state.params[k]=v

    tt_line=np.linspace(5,90,500)
    fw_line=np.array([caglioti_fwhm(t,U_v,V_v,W_v) for t in tt_line])
    fig2,ax2=plt.subplots(figsize=(10,2.8),facecolor=DARK)
    _ax(ax2,"2θ (°)","FWHM (°)","Caglioti-Verlauf")
    ax2.plot(tt_line,fw_line,color=ACC,lw=1.5)
    ax2.fill_between(tt_line,fw_line,alpha=.12,color=ACC)
    fig2.tight_layout(); st.image(fig_buf(fig2),use_container_width=True)

    cs,cz=st.columns(2)
    with cs: sc_v=st.number_input("Skalierung",value=float(st.session_state.params.get("scale",1.)),step=.01,format="%.4f")
    with cz: zs_v=st.number_input("Nullpunkt (°)",value=float(st.session_state.params.get("zero_shift",0.)),step=.001,format="%.4f")
    st.session_state.params["scale"]=sc_v; st.session_state.params["zero_shift"]=zs_v
    bi_v=st.number_input("B_iso (Å²)",value=float(st.session_state.params.get("Biso",1.)),step=.1,format="%.3f")
    st.session_state.params["Biso"]=bi_v
    bg=st.session_state.params.get("bg_coeffs",[0.]*6); new_bg=[]
    bgc=st.columns(3)
    for i in range(6):
        with bgc[i%3]:
            nv=st.number_input(f"c_{i}",value=float(bg[i]),step=1.,format="%.2f",key=f"bgc_{i}")
            new_bg.append(nv)
    st.session_state.params["bg_coeffs"]=new_bg

# ─────────── TAB: Rietveld ───────────
with t_riet:
    ready=st.session_state.obs is not None and len(st.session_state.reflections)>0
    if not ready:
        st.warning("(1) CIF · (2) Diffraktogramm · (3) Reflexliste erzeugen")
    else:
        st.markdown(f"""<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;'>
          <span class='badge ok'>✓ {len(st.session_state.two_theta)} Punkte</span>
          <span class='badge ok'>✓ {len(st.session_state.reflections)} Reflexe</span>
          <span class='badge info'>λ={wavelength:.5f} Å</span></div>""",unsafe_allow_html=True)

        with st.expander("👁 Vorschau",expanded=True):
            p0=dict(st.session_state.params); p0["wavelength"]=wavelength
            c0=calc_pattern(st.session_state.two_theta,st.session_state.reflections,p0)
            Rp0,Rwp0,ch0=calc_rfactors(st.session_state.obs,c0)
            m1,m2,m3=st.columns(3)
            m1.metric("R_p",f"{Rp0:.3f}%"); m2.metric("R_wp",f"{Rwp0:.3f}%"); m3.metric("χ²",f"{ch0:.4f}")
            st.image(plot_rietveld(st.session_state.two_theta,st.session_state.obs,
                                   c0,st.session_state.reflections,"Vorschau"),use_container_width=True)

        st.markdown('<div class="big-btn">',unsafe_allow_html=True)
        if st.button("▶  Rietveld-Verfeinerung",key="run_riet"):
            pm=dict(st.session_state.params); pm["wavelength"]=wavelength
            with st.spinner(f"Verfeinere ({n_cyc} Zyklen)…"):
                res=refine_pattern(st.session_state.two_theta,st.session_state.obs,
                                   st.session_state.reflections,pm,rf,wavelength,n_cyc)
            st.session_state.params.update(res["params"])
            st.session_state.R_vals=(res["Rp"],res["Rwp"],res["chi2"])
            st.session_state.calc=res["calc"]
            st.success(f"✓ {res['message']}")
        st.markdown("</div>",unsafe_allow_html=True)

        if st.session_state.calc is not None:
            Rp,Rwp,chi2=st.session_state.R_vals
            m1,m2,m3=st.columns(3)
            m1.metric("R_p",f"{Rp:.4f}%"); m2.metric("R_wp",f"{Rwp:.4f}%"); m3.metric("χ²",f"{chi2:.5f}")
            st.image(plot_rietveld(st.session_state.two_theta,st.session_state.obs,
                                   st.session_state.calc,st.session_state.reflections,
                                   "Rietveld-Verfeinerung"),use_container_width=True)

# ─────────── TAB: xrdfit FWHM ───────────
with t_xrdfit:
    st.markdown("### xrdfit / lmfit — Vollprofil-Peak-Fit")
    be_label="xrdfit (PseudoVoigt)" if fit_backend=="xrdfit" else f"lmfit – {fit_profile}"
    st.markdown(f"""<div class='formula'>
Backend: <b>{be_label}</b> · Hintergrund: <b>{bg_choice}</b><br>
Jeder Reflex → individueller Pseudo-Voigt-Fit → FWHM, β, η, σ_FWHM<br>
Integrale Breite: β = ∫I dθ / I_max  (enthält Untergrund-Korrektion)
    </div>""", unsafe_allow_html=True)

    if st.session_state.obs is None or not st.session_state.reflections:
        st.info("Diffraktogramm + Reflexliste benötigt.")
    else:
        if not LMFIT_AVAILABLE:
            st.error("lmfit nicht installiert. `pip install lmfit xrdfit`")
        else:
            st.markdown('<div class="sec-btn">',unsafe_allow_html=True)
            run_fwhm=st.button(f"🔬 Alle Peaks fitten ({be_label})",key="run_fwhm")
            st.markdown("</div>",unsafe_allow_html=True)

            if run_fwhm:
                with st.spinner("Fitte Peaks mit lmfit/xrdfit…"):
                    pf=extract_all_fwhm(
                        st.session_state.two_theta,st.session_state.obs,
                        st.session_state.reflections,wavelength,
                        window_deg=fwhm_window,min_intensity_pct=min_int,
                        backend=fit_backend,profile=fit_profile,bg_model=bg_choice)
                st.session_state.peak_fits=pf
                st.success(f"✓ {len(pf)} Peaks erfolgreich gefittet")

            if st.session_state.peak_fits:
                pf=st.session_state.peak_fits

                # Overview table
                df_fwhm=pd.DataFrame([{
                    "hkl":f"{p['h']}{p['k']}{p['l']}",
                    "2θ (°)":round(p["two_theta"],4),
                    "FWHM (°)":round(p["fwhm"],5),
                    "±FWHM":round(p["fwhm_err"],5),
                    "β (°)":round(p["beta"],5),
                    "η":round(p["eta"],4),
                    "A":round(p["amplitude"],1),
                    "χ²_red":round(p.get("redchi",0),4),
                    "Profil":p.get("profile","?"),
                } for p in pf])
                st.dataframe(df_fwhm,use_container_width=True)

                # Peak selector for individual fit view
                st.markdown("#### Einzelner Peak-Fit anzeigen")
                hkl_labels=[f"{p['h']}{p['k']}{p['l']} @ {p['two_theta']:.3f}°" for p in pf]
                sel=st.selectbox("Peak auswählen",hkl_labels,key="pk_sel")
                sel_idx=hkl_labels.index(sel)
                sel_pfit=pf[sel_idx]
                buf_single=plot_single_peak_fit(sel_pfit)
                if buf_single:
                    st.image(buf_single,use_container_width=False)
                # Show lmfit report
                with st.expander("lmfit-Fitbericht"):
                    st.code(sel_pfit.get("report","—"),language="text")

                # Download
                csv_fwhm=df_fwhm.to_csv(index=False).encode()
                st.download_button("⬇ FWHM-Tabelle (.csv)",csv_fwhm,"xrdfit_fwhm.csv","text/csv")

# ─────────── TAB: Caglioti ───────────
with t_cag:
    st.markdown("### Caglioti-Fit an beobachteten FWHM-Werten")
    st.markdown("""<div class='formula'>
FWHM²(θ) = U·tan²θ + V·tanθ + W<br>
Verfeinerte U,V,W können in ⚙️ Parameter übernommen werden.
    </div>""", unsafe_allow_html=True)

    if not st.session_state.peak_fits:
        st.info("Zuerst xrdfit FWHM-Tab ausführen.")
    else:
        pf=st.session_state.peak_fits
        p=st.session_state.params
        U_c,V_c,W_c=p.get("U",.01),p.get("V",-.001),p.get("W",.005)

        st.image(plot_fwhm_overview(pf,U_c,V_c,W_c,
                                     (st.session_state.two_theta.min(),
                                      st.session_state.two_theta.max())),
                 use_container_width=True)

        tt_obs=np.array([p_["two_theta"] for p_ in pf])
        fw_obs=np.array([p_["fwhm"]      for p_ in pf])
        try:
            def cag(tt,U,V,W):
                t=np.radians(tt/2)
                return np.sqrt(np.maximum(U*np.tan(t)**2+V*np.tan(t)+W,1e-9))
            popt,pcov=curve_fit(cag,tt_obs,fw_obs,p0=[U_c,V_c,W_c],
                                bounds=([0,-5,1e-7],[5,0,5]))
            perr=np.sqrt(np.diag(pcov))
            U_fit,V_fit,W_fit=popt
            m1,m2,m3=st.columns(3)
            m1.metric("U",f"{U_fit:.5f} ± {perr[0]:.5f}")
            m2.metric("V",f"{V_fit:.5f} ± {perr[1]:.5f}")
            m3.metric("W",f"{W_fit:.6f} ± {perr[2]:.6f}")
            if st.button("↑ Caglioti-Parameter → ⚙️ übernehmen"):
                st.session_state.params["U"]=float(U_fit)
                st.session_state.params["V"]=float(V_fit)
                st.session_state.params["W"]=float(W_fit)
                st.success("Übernommen!")
        except Exception as e:
            st.warning(f"Caglioti-Fit fehlgeschlagen: {e}")

# ─────────── TAB: Scherrer ───────────
with t_sch:
    st.markdown("### Scherrer-Analyse")
    st.markdown(f"""<div class='formula'>
D = K · λ / (β · cosθ)<br>
D = Kristallitgröße (nm) · K = {K_sch} · β = {'integrale Breite' if use_beta else 'FWHM'} (rad)<br>
Gilt nur bei vernachlässigbarer Gitterverzerrung ε ≈ 0
    </div>""", unsafe_allow_html=True)

    if not st.session_state.peak_fits:
        st.info("Zuerst xrdfit FWHM-Tab ausführen.")
    else:
        pf=st.session_state.peak_fits
        df_sch,stats=scherrer_analysis(pf,wavelength,K=K_sch,use_beta=use_beta)
        if stats:
            st.session_state.scherrer_result=stats["D_mean_nm"]
            m1,m2,m3=st.columns(3)
            m1.metric("⟨D⟩ (nm)",f"{stats['D_mean_nm']:.2f}")
            m2.metric("σ_D (nm)",f"{stats['D_std_nm']:.2f}")
            m3.metric("Peaks",f"{len(df_sch)}")

            fig,ax=plt.subplots(figsize=(10,3.5),facecolor=DARK)
            _ax(ax,"hkl","D (nm)","Scherrer D pro Reflex")
            bar_c=[ACC if abs(v-stats["D_mean_nm"])<=stats["D_std_nm"] else ACC2
                   for v in df_sch["D (nm)"]]
            bars=ax.bar(df_sch["hkl"],df_sch["D (nm)"],color=bar_c,width=.6)
            ax.axhline(stats["D_mean_nm"],color=ACC3,lw=1.2,ls="--",
                       label=f"⟨D⟩={stats['D_mean_nm']:.1f} nm")
            ax.fill_between(range(len(df_sch)),
                             stats["D_mean_nm"]-stats["D_std_nm"],
                             stats["D_mean_nm"]+stats["D_std_nm"],
                             color=ACC3,alpha=.1)
            ax.legend(facecolor=CARD,edgecolor=MUTED,labelcolor=TXT,fontsize=8)
            ax.set_xticklabels(df_sch["hkl"],color=MUTED,fontsize=8,rotation=45)
            fig.tight_layout(); st.image(fig_buf(fig),use_container_width=True)
            st.dataframe(df_sch,use_container_width=True)
        else:
            st.warning("Zu wenige Peaks für Scherrer.")

# ─────────── TAB: Williamson-Hall ───────────
with t_wh:
    st.markdown("### Williamson-Hall-Plot")
    st.markdown("""<div class='formula'>
β·cosθ / λ  =  1/D  +  4ε·sinθ / λ<br>
y = β·cosθ/λ  [Å⁻¹]   ·   x = 4·sinθ/λ  [Å⁻¹]<br>
Steigung → Mikrodehnung ε  |  Achsenabschnitt → 1/D
    </div>""", unsafe_allow_html=True)

    if not st.session_state.peak_fits:
        st.info("Zuerst xrdfit FWHM-Tab ausführen.")
    else:
        wh,fit_wh,cfs_wh,xs_wh,ys_wh,lbs_wh=williamson_hall(
            st.session_state.peak_fits,wavelength,use_beta)
        if wh:
            st.session_state.wh_result=wh
            m1,m2,m3,m4=st.columns(4)
            m1.metric("D (nm)",f"{wh['D_nm']:.2f}" if not np.isnan(wh['D_nm']) else "n/a")
            m2.metric("ε",f"{wh['eps']:.5f}")
            m3.metric("Steigung ± σ",f"{wh['slope']:.4f} ± {wh['slope_err']:.4f}")
            m4.metric("Intercept ± σ",f"{wh['intercept']:.4f} ± {wh['int_err']:.4f}")
            st.image(plot_linear_analysis(
                xs_wh,ys_wh,lbs_wh,cfs_wh,fit_wh,
                "4·sinθ / λ  (Å⁻¹)","β·cosθ / λ  (Å⁻¹)","Williamson-Hall"),
                     use_container_width=True)
            wh_df=pd.DataFrame({"hkl":lbs_wh,
                                  "4sinθ/λ":[round(x,6) for x in xs_wh],
                                  "βcosθ/λ":[round(y,7) for y in ys_wh]})
            st.dataframe(wh_df,use_container_width=True)
            st.markdown(f"""<div class='formula'>
D(WH) = {wh['D_nm']:.2f} nm  ·  ε = {wh['eps']:.5f}<br>
Fit: y = {wh['slope']:.5e}·x + {wh['intercept']:.5e}
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Min. 2 Reflexe für Williamson-Hall nötig.")

# ─────────── TAB: Halder-Wagner ───────────
with t_hw:
    st.markdown("### Halder-Wagner-Plot")
    st.markdown("""<div class='formula'>
Modifizierter Halder-Wagner:  (β*/d*)²  =  (β*/d*)/D  +  (2ε)²<br>
β* = 2·cosθ·β/λ   ·   d* = 2·sinθ/λ<br>
y = (β*/d*)²   ·   x = β*/d*²<br>
Steigung → 1/D  |  Achsenabschnitt → (2ε)²  →  ε = √intercept / 2
    </div>""", unsafe_allow_html=True)

    if not st.session_state.peak_fits:
        st.info("Zuerst xrdfit FWHM-Tab ausführen.")
    else:
        hw,fit_hw,cfs_hw,xs_hw,ys_hw,lbs_hw=halder_wagner(
            st.session_state.peak_fits,wavelength,use_beta)
        if hw:
            st.session_state.hw_result=hw
            m1,m2,m3,m4=st.columns(4)
            m1.metric("D (nm)",f"{hw['D_nm']:.2f}" if not np.isnan(hw['D_nm']) else "n/a")
            m2.metric("ε",f"{hw['eps']:.5f}")
            m3.metric("Steigung ± σ",f"{hw['slope']:.3e} ± {hw['slope_err']:.2e}")
            m4.metric("Intercept",f"{hw['intercept']:.3e} ± {hw['int_err']:.2e}")
            st.image(plot_linear_analysis(
                xs_hw,ys_hw,lbs_hw,cfs_hw,fit_hw,
                "β*/d*²  (Å)","(β*/d*)²  (Å⁻²)","Halder-Wagner",
                color_data=ACC3,color_fit=ACC2),
                     use_container_width=True)
            hw_df=pd.DataFrame({"hkl":lbs_hw,
                                  "β*/d*²":[round(x,6) for x in xs_hw],
                                  "(β*/d*)²":[round(y,7) for y in ys_hw]})
            st.dataframe(hw_df,use_container_width=True)
            st.markdown(f"""<div class='formula'>
D(HW) = {hw['D_nm']:.2f} nm  ·  ε = {hw['eps']:.5f}<br>
Fit: y = {hw['slope']:.4e}·x + {hw['intercept']:.4e}
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Min. 2 Reflexe für Halder-Wagner nötig.")

# ─────────── TAB: Ergebnisse ───────────
with t_res:
    st.markdown("### Gesamtübersicht")

    if st.session_state.R_vals:
        st.markdown('<div class="sec-hdr">Rietveld Gütefaktoren</div>',unsafe_allow_html=True)
        Rp,Rwp,chi2=st.session_state.R_vals
        m1,m2,m3=st.columns(3)
        m1.metric("R_p (%)",f"{Rp:.4f}"); m2.metric("R_wp (%)",f"{Rwp:.4f}"); m3.metric("χ²",f"{chi2:.5f}")

    if st.session_state.cell:
        st.markdown('<div class="sec-hdr">Verfeinerte Gitterparameter</div>',unsafe_allow_html=True)
        p=st.session_state.params
        st.dataframe(pd.DataFrame({
            "Parameter":["a (Å)","b (Å)","c (Å)","α (°)","β (°)","γ (°)",
                          "Skalierung","Nullpunkt (°)","U","V","W","η","B_iso (Å²)"],
            "Wert":[p.get("a","—"),p.get("b","—"),p.get("c","—"),
                    p.get("alpha","—"),p.get("beta","—"),p.get("gamma","—"),
                    p.get("scale","—"),p.get("zero_shift","—"),
                    p.get("U","—"),p.get("V","—"),p.get("W","—"),
                    p.get("eta","—"),p.get("Biso","—")]
        }),use_container_width=True)

    # Size comparison
    st.markdown('<div class="sec-hdr">Kristallitgröße – Methodenvergleich</div>',unsafe_allow_html=True)
    methods={
        "Scherrer":       st.session_state.scherrer_result,
        "Williamson-Hall":st.session_state.wh_result["D_nm"] if st.session_state.wh_result else None,
        "Halder-Wagner":  st.session_state.hw_result["D_nm"]  if st.session_state.hw_result  else None,
    }
    avail={k:v for k,v in methods.items() if v is not None and not np.isnan(v)}
    if avail:
        mc=st.columns(len(avail))
        for col,(k,v) in zip(mc,avail.items()): col.metric(f"D_{k[:3]} (nm)",f"{v:.2f}")
        buf_cmp=plot_size_comparison(avail)
        if buf_cmp: st.image(buf_cmp,use_container_width=True)

        st.markdown('<div class="sec-hdr">Mikrodehnungsanalyse</div>',unsafe_allow_html=True)
        rows_eps=[]
        if st.session_state.wh_result:
            w=st.session_state.wh_result
            rows_eps.append({"Methode":"Williamson-Hall","D (nm)":round(w["D_nm"],2),"ε":f"{w['eps']:.5f}"})
        if st.session_state.hw_result:
            h=st.session_state.hw_result
            rows_eps.append({"Methode":"Halder-Wagner","D (nm)":round(h["D_nm"],2),"ε":f"{h['eps']:.5f}"})
        if rows_eps: st.dataframe(pd.DataFrame(rows_eps),use_container_width=True)
    else:
        st.info("Analysemethoden (Tabs Scherrer/WH/HW) noch nicht ausgeführt.")

    # Export
    st.markdown('<div class="sec-hdr">Export</div>',unsafe_allow_html=True)
    p=st.session_state.params; sg=st.session_state.cell.get("space_group","?") if st.session_state.cell else "—"
    lines=["Rietveld + xrdfit FWHM Analysis","="*48,
           f"Backend      : {be_label}",
           f"Raumgruppe   : {sg}",
           f"Wellenlänge  : {wavelength:.5f} Å",
           f"Scherrer K   : {K_sch}",""]
    if st.session_state.R_vals:
        Rp,Rwp,chi2=st.session_state.R_vals
        lines+=["── Gütefaktoren ──",f"  Rp={Rp:.4f}%  Rwp={Rwp:.4f}%  χ²={chi2:.5f}",""]
    if avail:
        lines+=["── Kristallitgrößen ──"]
        for k,v in avail.items(): lines.append(f"  {k}: D={v:.2f} nm")
        lines+=[""]
    if st.session_state.wh_result:
        w=st.session_state.wh_result
        lines+=["── Williamson-Hall ──",f"  D={w['D_nm']:.2f} nm  ε={w['eps']:.5f}",""]
    if st.session_state.hw_result:
        h=st.session_state.hw_result
        lines+=["── Halder-Wagner ──",f"  D={h['D_nm']:.2f} nm  ε={h['eps']:.5f}",""]
    summary="\n".join(lines)

    e1,e2,e3=st.columns(3)
    with e1:
        st.download_button("⬇ Summary (.txt)",summary.encode(),"rietveld_xrdfit_summary.txt","text/plain")
    with e2:
        if st.session_state.calc is not None:
            out_df=pd.DataFrame({"2theta":st.session_state.two_theta,"obs":st.session_state.obs,
                                  "calc":st.session_state.calc,
                                  "diff":st.session_state.obs-st.session_state.calc})
            st.download_button("⬇ Muster (.csv)",out_df.to_csv(index=False).encode(),
                               "rietveld_pattern.csv","text/csv")
    with e3:
        if st.session_state.peak_fits:
            pf=st.session_state.peak_fits
            fdf=pd.DataFrame([{"hkl":f"{p['h']}{p['k']}{p['l']}",
                                "2theta":round(p["two_theta"],4),
                                "FWHM_deg":round(p["fwhm"],5),
                                "FWHM_err":round(p["fwhm_err"],5),
                                "beta_deg":round(p["beta"],5),
                                "eta":round(p["eta"],4),
                                "d_A":round(p["d"],5),
                                "redchi":round(p.get("redchi",0),4),
                                "profile":p.get("profile","?")} for p in pf])
            st.download_button("⬇ FWHM xrdfit (.csv)",fdf.to_csv(index=False).encode(),
                               "xrdfit_fwhm_results.csv","text/csv")
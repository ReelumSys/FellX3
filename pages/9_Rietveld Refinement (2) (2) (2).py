
import streamlit as st
import pandas as pd

# Data from Main Page
main_df = st.session_state.get('main_df')
comp_df = st.session_state.get('comp_df')
cif_data = st.session_state.get('cif_data')

if main_df is None:
    st.warning("Main XRD pattern missing. Please upload it on the Main Page.")
    st.stop()

     1|﻿"""
     2|Rietveld Refinement + xrdfit FWHM Engine
     3|==========================================
     4|Vollprofil-Verfeinerung mit xrdfit (lmfit/Pseudo-Voigt) als Peak-Fitting-Backend.
     5|Kristallitgröße via Scherrer, Williamson-Hall, Halder-Wagner.
     6|
     7|Abhängigkeiten:
     8|    pip install streamlit numpy scipy matplotlib pandas lmfit xrdfit
     9|
    10|Starten:
    11|    streamlit run rietveld_xrdfit.py
    12|"""
    13|
    14|import streamlit as st
    15|import numpy as np
    16|import matplotlib
    17|matplotlib.use("Agg")
    18|import matplotlib.pyplot as plt
    19|import matplotlib.gridspec as gridspec
    20|from matplotlib.ticker import AutoMinorLocator
    21|import io, re, warnings, tempfile, os
    22|import pandas as pd
    23|from scipy.optimize import least_squares, curve_fit
    24|from scipy.signal import find_peaks
    25|warnings.filterwarnings("ignore")
    26|
    27|# ── xrdfit / lmfit imports (with graceful fallback) ──
    28|try:
    29|    from xrdfit.spectrum_fitting import FitSpectrum, PeakParams, MaximumParams
    30|    from xrdfit.pv_fit import do_pv_fit
    31|    XRDFIT_AVAILABLE = True
    32|except ImportError:
    33|    XRDFIT_AVAILABLE = False
    34|
    35|try:
    36|    import lmfit
    37|    from lmfit.models import PseudoVoigtModel, LinearModel, ConstantModel
    38|    LMFIT_AVAILABLE = True
    39|except ImportError:
    40|    LMFIT_AVAILABLE = False
    41|
    42|# ══════════════════════════════════════════════════════════════════════════════
    43|st.set_page_config(
    44|    page_title="Rietveld · xrdfit · Kristallitgröße",
    45|    page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
    46|)
    47|
    48|# ── Design tokens ──
    49|C = dict(
    50|    bg="#07090e", card="#0c0f17", border="#181d2a",
    51|    accent="#5CF4C4", accent2="#FF5A5F", accent3="#FFD166",
    52|    success="#06D6A0", warn="#FFB703", muted="#44516b", text="#dde3ed",
    53|)
    54|DARK=C["bg"]; CARD=C["card"]; ACC=C["accent"]; ACC2=C["accent2"]
    55|ACC3=C["accent3"]; MUTED=C["muted"]; TXT=C["text"]
    56|
    57|st.markdown(f"""
    58|<style>
    59|@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
    60|/* Override Space Grotesk — using Barlow Condensed for headers */
    61|@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700;800&family=Fira+Code:wght@400;500&display=swap');
    62|:root{{
    63|  --bg:{C['bg']};--card:{C['card']};--border:{C['border']};
    64|  --acc:{C['accent']};--acc2:{C['accent2']};--acc3:{C['accent3']};
    65|  --ok:{C['success']};--muted:{C['muted']};--text:{C['text']};
    66|}}
    67|html,body,[data-testid="stAppViewContainer"]{{
    68|  background:var(--bg)!important;color:var(--text)!important;
    69|  font-family:'Barlow Condensed',sans-serif;font-size:15px;
    70|}}
    71|[data-testid="stSidebar"]{{background:var(--card)!important;border-right:1px solid var(--border);}}
    72|h1,h2,h3,h4{{font-family:'Barlow Condensed',sans-serif;font-weight:700;}}
    73|/* Tabs */
    74|.stTabs [data-baseweb="tab-list"]{{background:var(--card);border-radius:6px;padding:3px;gap:2px;border:1px solid var(--border);}}
    75|.stTabs [data-baseweb="tab"]{{color:var(--muted);font-family:'Fira Code',monospace;font-size:.72rem;border-radius:4px;padding:5px 12px;}}
    76|.stTabs [aria-selected="true"]{{background:var(--acc)!important;color:#000!important;font-weight:600;}}
    77|/* Inputs */
    78|.stNumberInput input,.stTextInput input,.stSelectbox select{{
    79|  background:var(--card)!important;color:var(--text)!important;
    80|  border:1px solid var(--border)!important;font-family:'Fira Code',monospace;border-radius:4px;}}
    81|/* Buttons */
    82|.stButton>button{{background:transparent;border:1px solid var(--acc);color:var(--acc);
    83|  font-family:'Fira Code',monospace;border-radius:4px;transition:all .15s;font-size:.82rem;}}
    84|.stButton>button:hover{{background:var(--acc);color:#000;}}
    85|.big-btn>button{{background:var(--acc)!important;color:#000!important;font-weight:700;
    86|  border:none!important;border-radius:6px;width:100%;padding:10px;font-size:.95rem;}}
    87|.sec-btn>button{{background:var(--acc3)!important;color:#000!important;font-weight:700;
    88|  border:none!important;border-radius:6px;width:100%;padding:9px;}}
    89|/* Metrics */
    90|div[data-testid="stMetric"]{{background:var(--card);border:1px solid var(--border);
    91|  border-radius:8px;padding:10px 14px;}}
    92|div[data-testid="stMetric"] label{{color:var(--muted)!important;font-size:.68rem;
    93|  font-family:'Fira Code',monospace;text-transform:uppercase;letter-spacing:.05em;}}
    94|div[data-testid="stMetric"] div{{color:var(--acc)!important;font-family:'Fira Code',monospace;font-size:1.15rem;}}
    95|/* Upload */
    96|[data-testid="stFileUploader"]{{background:var(--card);border:1px dashed var(--border);border-radius:8px;}}
    97|/* Custom components */
    98|.formula{{background:#080b12;border:1px solid var(--border);border-left:3px solid var(--acc3);
    99|  border-radius:0 6px 6px 0;padding:10px 14px;font-family:'Fira Code',monospace;
   100|  font-size:.78rem;color:var(--text);margin:8px 0;line-height:1.7;}}
   101|.badge{{display:inline-block;background:var(--card);border:1px solid var(--border);
   102|  border-radius:20px;padding:2px 10px;font-family:'Fira Code',monospace;font-size:.72rem;
   103|  color:var(--muted);margin:2px;}}
   104|.badge.ok{{border-color:var(--ok);color:var(--ok);}}
   105|.badge.warn{{border-color:var(--acc3);color:var(--acc3);}}
   106|.badge.err{{border-color:var(--acc2);color:var(--acc2);}}
   107|.badge.info{{border-color:var(--acc);color:var(--acc);}}
   108|.sec-hdr{{font-family:'Fira Code',monospace;font-size:.65rem;font-weight:500;
   109|  text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
   110|  margin:14px 0 5px 0;border-bottom:1px solid var(--border);padding-bottom:3px;}}
   111|</style>
   112|""", unsafe_allow_html=True)
   113|
   114|# ══════════════════════════════════════════════════════════════════════════════
   115|# ── PHYSICS HELPERS (shared with previous version) ──
   116|# ══════════════════════════════════════════════════════════════════════════════
   117|
   118|def parse_cif(content):
   119|    data={}
   120|    def ex(k):
   121|        m=re.search(rf"_{k}\s+([\S]+)",content,re.IGNORECASE)
   122|        if m:
   123|            v=m.group(1).replace("(","").replace(")","")
   124|            try: return float(v)
   125|            except: return v
   126|        return None
   127|    data["a"]=ex("cell_length_a") or 5.
   128|    data["b"]=ex("cell_length_b") or 5.
   129|    data["c"]=ex("cell_length_c") or 5.
   130|    data["alpha"]=ex("cell_angle_alpha") or 90.
   131|    data["beta"]=ex("cell_angle_beta")   or 90.
   132|    data["gamma"]=ex("cell_angle_gamma") or 90.
   133|    sg=ex("symmetry_space_group_name_H-M") or ex("space_group_name_H-M_alt")
   134|    sgn=ex("symmetry_Int_Tables_number")  or ex("space_group_IT_number")
   135|    data["space_group"]=str(sg) if sg else (f"No.{int(sgn)}" if sgn else "P 1")
   136|    data["sg_number"]=int(sgn) if sgn else 1
   137|    atoms=[]
   138|    lm=re.search(r"loop_.*?_atom_site_label.*?(?=loop_|\Z)",content,re.DOTALL|re.IGNORECASE)
   139|    if lm:
   140|        blk=lm.group(0); hdrs=re.findall(r"_atom_site_(\w+)",blk,re.IGNORECASE)
   141|        rows=re.findall(r"^\s+([A-Za-z][A-Za-z0-9]*\d*\s+.*?)$",blk,re.MULTILINE)
   142|        for row in rows:
   143|            pts=row.split()
   144|            if len(pts)<5: continue
   145|            at={"label":pts[0],"type":re.sub(r"\d","",pts[0])}
   146|            try:
   147|                ix=next(i for i,h in enumerate(hdrs) if "fract_x" in h.lower() or h.lower()=="x")
   148|                at["x"]=float(pts[ix+1].split("(")[0])
   149|                at["y"]=float(pts[ix+2].split("(")[0])
   150|                at["z"]=float(pts[ix+3].split("(")[0])
   151|            except: at["x"]=at["y"]=at["z"]=0.
   152|            try:    at["occ"]=float(pts[-2].split("(")[0])
   153|            except: at["occ"]=1.
   154|            try:    at["Biso"]=float(pts[-1].split("(")[0])
   155|            except: at["Biso"]=1.
   156|            atoms.append(at)
   157|    data["atoms"]=atoms
   158|    return data
   159|
   160|def parse_diffractogram(content):
   161|    tt,ii,ee=[],[],[]
   162|    for line in content.splitlines():
   163|        line=line.strip()
   164|        if not line or line[0] in("#","!"): continue
   165|        pts=line.split()
   166|        try:
   167|            t,i=float(pts[0]),float(pts[1])
   168|            tt.append(t); ii.append(i)
   169|            ee.append(float(pts[2]) if len(pts)>=3 else np.sqrt(max(i,1.)))
   170|        except: pass
   171|    return np.array(tt),np.array(ii),np.array(ee)
   172|
   173|def get_crystal_system(sg):
   174|    if sg<=2: return "triclinic"
   175|    if sg<=15: return "monoclinic"
   176|    if sg<=74: return "orthorhombic"
   177|    if sg<=142: return "tetragonal"
   178|    if sg<=194: return "hexagonal"
   179|    return "cubic"
   180|
   181|def compute_d(h,k,l,cell):
   182|    a,b,c=cell["a"],cell["b"],cell["c"]
   183|    al,be,ga=cell["alpha"],cell["beta"],cell["gamma"]
   184|    sg=cell.get("sg_number",1); sys=get_crystal_system(sg)
   185|    try:
   186|        if sys=="cubic":        return a/np.sqrt(h*h+k*k+l*l)
   187|        if sys=="tetragonal":   return 1/np.sqrt((h*h+k*k)/a**2+l*l/c**2)
   188|        if sys=="orthorhombic": return 1/np.sqrt(h*h/a**2+k*k/b**2+l*l/c**2)
   189|        if sys=="hexagonal":    return 1/np.sqrt(4/3*(h*h+h*k+k*k)/a**2+l*l/c**2)
   190|        if sys=="monoclinic":
   191|            bt=np.radians(be); sb,cb=np.sin(bt),np.cos(bt)
   192|            return 1/np.sqrt(h*h/(a*sb)**2+k*k/b**2+l*l/(c*sb)**2-2*h*l*cb/(a*c*sb**2))
   193|        AL,BE,GA=np.radians(al),np.radians(be),np.radians(ga)
   194|        cAL,cBE,cGA=np.cos(AL),np.cos(BE),np.cos(GA)
   195|        V=a*b*c*np.sqrt(1-cAL**2-cBE**2-cGA**2+2*cAL*cBE*cGA)
   196|        s11=(b*c*np.sin(AL))**2; s22=(a*c*np.sin(BE))**2; s33=(a*b*np.sin(GA))**2
   197|        s12=a*b*c**2*(cAL*cBE-cGA); s23=a**2*b*c*(cBE*cGA-cAL); s13=a*b**2*c*(cGA*cAL-cBE)
   198|        return V/np.sqrt(s11*h*h+s22*k*k+s33*l*l+2*s12*h*k+2*s23*k*l+2*s13*h*l)
   199|    except: return 0.
   200|
   201|def generate_reflections(cell,wavelength,tt_max=90.,hkl_max=8):
   202|    refs=[]
   203|    for h in range(-hkl_max,hkl_max+1):
   204|     for k in range(-hkl_max,hkl_max+1):
   205|      for l in range(-hkl_max,hkl_max+1):
   206|        if h==k==l==0: continue
   207|        d=compute_d(h,k,l,cell)
   208|        if d<=0: continue
   209|        st2=wavelength/(2*d)
   210|        if abs(st2)>1: continue
   211|        tt2=np.degrees(2*np.arcsin(st2))
   212|        if 0<tt2<=tt_max:
   213|            refs.append({"h":h,"k":k,"l":l,"d":d,"two_theta":tt2,"multiplicity":1})
   214|    refs.sort(key=lambda x:x["two_theta"])
   215|    merged=[refs[0]] if refs else []
   216|    for r in refs[1:]:
   217|        if abs(r["two_theta"]-merged[-1]["two_theta"])<0.005:
   218|            merged[-1]["multiplicity"]+=1
   219|        else: merged.append(r)
   220|    return merged
   221|
   222|def pseudo_voigt_fn(x,x0,fwhm,eta=0.5):
   223|    x=x-x0; sig=fwhm/(2*np.sqrt(2*np.log(2)))
   224|    return eta/(1+(x/(fwhm/2))**2)+(1-eta)*np.exp(-x**2/(2*sig**2))
   225|
   226|def caglioti_fwhm(tt,U,V,W):
   227|    t=np.radians(tt/2)
   228|    return np.sqrt(max(U*np.tan(t)**2+V*np.tan(t)+W,1e-9))
   229|
   230|def lorentz_pol(tt):
   231|    t=np.radians(tt/2)
   232|    return (1+np.cos(np.radians(tt))**2)/(np.sin(t)**2*np.cos(t)+1e-12)
   233|
   234|def chebyshev_bg(x,coeffs):
   235|    xn=2*(x-x.min())/(x.max()-x.min())-1
   236|    res=np.zeros_like(x)
   237|    for i,c in enumerate(coeffs):
   238|        res+=c*np.polynomial.chebyshev.chebval(xn,[0]*i+[1])
   239|    return res
   240|
   241|def calc_pattern(tt,refs,p):
   242|    wl=p.get("wavelength",1.54056)
   243|    U,V,W=p.get("U",.01),p.get("V",-.001),p.get("W",.005)
   244|    eta=p.get("eta",.5); scale=p.get("scale",1.); zs=p.get("zero_shift",0.)
   245|    Biso=p.get("Biso",1.); bg=p.get("bg_coeffs",[0.]*6)
   246|    pat=np.zeros_like(tt)
   247|    for r in refs:
   248|        ttk=r["two_theta"]+zs
   249|        fwhm=caglioti_fwhm(ttk,U,V,W); lp=lorentz_pol(ttk)
   250|        dw=np.exp(-2*Biso*(np.sin(np.radians(ttk/2))/wl)**2)
   251|        pat+=scale*r["multiplicity"]*lp*dw*pseudo_voigt_fn(tt,ttk,fwhm,eta)
   252|    return pat+chebyshev_bg(tt,bg)
   253|
   254|def calc_rfactors(obs,calc,w=None):
   255|    if w is None: w=1./np.maximum(obs,1.)
   256|    d=obs-calc
   257|    Rp=100*np.sum(np.abs(d))/np.sum(np.abs(obs))
   258|    Rwp=100*np.sqrt(np.sum(w*d**2)/np.sum(w*obs**2))
   259|    chi2=np.sum(w*d**2)/max(len(obs)-10,1)
   260|    return Rp,Rwp,chi2
   261|
   262|def refine_pattern(tt,obs,refs,params_in,flags,wavelength,n_cycles=5):
   263|    names,p0,lo,hi=[],[],[],[]
   264|    def add(n,v,l,h):
   265|        if flags.get(n): names.append(n);p0.append(v);lo.append(l);hi.append(h)
   266|    add("scale",params_in.get("scale",1.),0,1e9)
   267|    add("zero_shift",params_in.get("zero_shift",0.),-1,1)
   268|    add("U",params_in.get("U",.01),0,5)
   269|    add("V",params_in.get("V",-.001),-5,0)
   270|    add("W",params_in.get("W",.005),1e-7,5)
   271|    add("eta",params_in.get("eta",.5),0,1)
   272|    add("Biso",params_in.get("Biso",1.),0,30)
   273|    add("a",params_in.get("a",5.),0.1,100)
   274|    add("b",params_in.get("b",5.),0.1,100)
   275|    add("c",params_in.get("c",5.),0.1,100)
   276|    for i in range(6): add(f"bg_{i}",params_in.get("bg_coeffs",[0.]*6)[i],-1e6,1e6)
   277|    if not names:
   278|        c=calc_pattern(tt,refs,params_in)
   279|        rp,rwp,ch=calc_rfactors(obs,c)
   280|        return {"params":params_in,"Rp":rp,"Rwp":rwp,"chi2":ch,"calc":c,"message":"No params"}
   281|    w=1./np.maximum(obs,1.)
   282|    def res(pv):
   283|        pm=dict(params_in); pm["wavelength"]=wavelength
   284|        for n,v in zip(names,pv):
   285|            if n.startswith("bg_"):
   286|                bg=list(pm.get("bg_coeffs",[0.]*6)); bg[int(n[3:])]=v; pm["bg_coeffs"]=bg
   287|            else: pm[n]=v
   288|        return np.sqrt(w)*(obs-calc_pattern(tt,refs,pm))
   289|    r=least_squares(res,p0,bounds=(lo,hi),method="trf",max_nfev=n_cycles*300,
   290|                    ftol=1e-10,xtol=1e-10)
   291|    pm_out=dict(params_in); pm_out["wavelength"]=wavelength
   292|    for n,v in zip(names,r.x):
   293|        if n.startswith("bg_"):
   294|            bg=list(pm_out.get("bg_coeffs",[0.]*6)); bg[int(n[3:])]=v; pm_out["bg_coeffs"]=bg
   295|        else: pm_out[n]=v
   296|    c=calc_pattern(tt,refs,pm_out); rp,rwp,ch=calc_rfactors(obs,c,w)
   297|    return {"params":pm_out,"Rp":rp,"Rwp":rwp,"chi2":ch,"calc":c,"message":r.message}
   298|
   299|# ══════════════════════════════════════════════════════════════════════════════
   300|# ── xrdfit / lmfit PEAK FITTING ENGINE ──
   301|# ══════════════════════════════════════════════════════════════════════════════
   302|
   303|def fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg=1.5,
   304|                   profile="PseudoVoigt", bg_model="linear"):
   305|    """
   306|    Fit a single peak using lmfit directly (mirrors xrdfit's pv_fit.do_pv_fit).
   307|    Supports: PseudoVoigt, Voigt, Lorentzian, Gaussian profiles.
   308|    Background: linear or constant.
   309|    Returns dict with fwhm, eta, center, amplitude, area, beta, errors.
   310|    """
   311|    if not LMFIT_AVAILABLE:
   312|        return None
   313|    mask=np.abs(tt_arr-tt_center)<window_deg
   314|    if mask.sum()<8: return None
   315|    x,y=tt_arr[mask],obs_arr[mask]
   316|    # Choose background
   317|    if bg_model=="linear":
   318|        bg=LinearModel(prefix="bg_")
   319|    else:
   320|        bg=ConstantModel(prefix="bg_")
   321|    # Choose peak profile
   322|    profile_map={
   323|        "PseudoVoigt": PseudoVoigtModel,
   324|        "Voigt":       lmfit.models.VoigtModel,
   325|        "Lorentzian":  lmfit.models.LorentzianModel,
   326|        "Gaussian":    lmfit.models.GaussianModel,
   327|    }
   328|    PeakModel=profile_map.get(profile,PseudoVoigtModel)
   329|    peak=PeakModel(prefix="pk_")
   330|    model=peak+bg
   331|
   332|    # Initial parameter guesses
   333|    bg_est=(y[0]+y[-1])/2
   334|    y_sub=y-bg_est
   335|    A0=float(y_sub.max())
   336|    if A0<=0: return None
   337|    params=model.make_params()
   338|    params["pk_center"].set(value=tt_center, min=tt_center-window_deg, max=tt_center+window_deg)
   339|    params["pk_amplitude"].set(value=A0*window_deg*0.8, min=0)
   340|    if "pk_sigma" in params:
   341|        params["pk_sigma"].set(value=window_deg*0.2, min=1e-4, max=window_deg)
   342|    if "pk_fraction" in params:
   343|        params["pk_fraction"].set(value=0.5, min=0, max=1)
   344|    if bg_model=="linear":
   345|        params["bg_slope"].set(value=0)
   346|        params["bg_intercept"].set(value=bg_est)
   347|    else:
   348|        params["bg_c"].set(value=bg_est)
   349|
   350|    try:
   351|        result=model.fit(y,params,x=x,method="least_squares")
   352|        pv=result.params
   353|
   354|        center=float(pv["pk_center"].value)
   355|        # FWHM: lmfit stores fwhm as derived param for most models
   356|        if "pk_fwhm" in pv:
   357|            fwhm=float(pv["pk_fwhm"].value)
   358|            fwhm_err=float(pv["pk_fwhm"].stderr) if pv["pk_fwhm"].stderr else 0.
   359|        else:
   360|            sig=float(pv["pk_sigma"].value)
   361|            fwhm=2.355*sig
   362|            fwhm_err=2.355*float(pv["pk_sigma"].stderr) if pv["pk_sigma"].stderr else 0.
   363|        eta=float(pv["pk_fraction"].value) if "pk_fraction" in pv else 0.5
   364|        amp=float(pv["pk_amplitude"].value)
   365|
   366|        # Background at peak position
   367|        if bg_model=="linear":
   368|            bg_at_peak=float(pv["bg_slope"].value)*center+float(pv["bg_intercept"].value)
   369|        else:
   370|            bg_at_peak=float(pv["bg_c"].value)
   371|
   372|        # Compute peak-only curve for integral breadth
   373|        peak_only=peak.eval(pv,x=x)-bg_at_peak
   374|        peak_max=float(peak_only.max())
   375|        area=float(np.trapz(np.maximum(peak_only,0),x))
   376|        beta=area/peak_max if peak_max>0 else fwhm
   377|
   378|        # lmfit fit statistics
   379|        redchi=float(result.redchi) if result.redchi else 0.
   380|        aic=float(result.aic)   if result.aic    else 0.
   381|
   382|        return {
   383|            "two_theta": center,
   384|            "fwhm": fwhm, "fwhm_err": fwhm_err,
   385|            "eta": eta, "amplitude": amp,
   386|            "background": bg_at_peak,
   387|            "beta": beta, "area": area,
   388|            "redchi": redchi, "aic": aic,
   389|            "profile": profile,
   390|            "report": result.fit_report(),
   391|            "x_fit": x, "y_fit": y,
   392|            "y_best": result.best_fit,
   393|        }
   394|    except Exception as e:
   395|        return None
   396|
   397|def fit_peak_xrdfit(tt_arr, obs_arr, tt_center, window_deg=1.5):
   398|    """
   399|    Use xrdfit's FitSpectrum / PeakParams / MaximumParams via a temp .dat file.
   400|    Falls back to lmfit direct if xrdfit is unavailable.
   401|    """
   402|    if not XRDFIT_AVAILABLE:
   403|        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg, "PseudoVoigt")
   404|
   405|    # xrdfit expects a tab-separated file: two_theta [TAB] intensity
   406|    mask=np.abs(tt_arr-tt_center)<window_deg*1.5
   407|    if mask.sum()<8:
   408|        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg)
   409|    x_sub,y_sub=tt_arr[mask],obs_arr[mask]
   410|
   411|    try:
   412|        # Write temp file
   413|        with tempfile.NamedTemporaryFile(mode="w",suffix=".dat",delete=False) as f:
   414|            for xi,yi in zip(x_sub,y_sub):
   415|                f.write(f"{xi:.6f}\t{yi:.6f}\n")
   416|            tmpfile=f.name
   417|
   418|        spec=FitSpectrum(tmpfile, first_cake_angle=90, delimiter="\t")
   419|        lo=float(x_sub.min()); hi=float(x_sub.max())
   420|        peak_name=f"peak_{tt_center:.2f}"
   421|        max_lo=tt_center-window_deg*0.3
   422|        max_hi=tt_center+window_deg*0.3
   423|        mp=MaximumParams(peak_name,(max_lo,max_hi))
   424|        pp=PeakParams((lo,hi),[mp])
   425|        spec.fit_peaks(pp,cakes_to_fit=1)
   426|        pfit=spec.get_fit(peak_name)
   427|
   428|        # Extract results from lmfit result inside xrdfit PeakFit
   429|        res=pfit.fit_result
   430|        pv=res.params
   431|        center=float(pv[f"{peak_name}_center"].value)
   432|        fwhm=float(pv[f"{peak_name}_fwhm"].value)  if f"{peak_name}_fwhm" in pv else \
   433|              2.355*float(pv[f"{peak_name}_sigma"].value)
   434|        fwhm_err=0.
   435|        if f"{peak_name}_fwhm" in pv and pv[f"{peak_name}_fwhm"].stderr:
   436|            fwhm_err=float(pv[f"{peak_name}_fwhm"].stderr)
   437|        elif f"{peak_name}_sigma" in pv and pv[f"{peak_name}_sigma"].stderr:
   438|            fwhm_err=2.355*float(pv[f"{peak_name}_sigma"].stderr)
   439|
   440|        eta=float(pv[f"{peak_name}_fraction"].value) if f"{peak_name}_fraction" in pv else 0.5
   441|        amp=float(pv[f"{peak_name}_amplitude"].value)
   442|
   443|        x_fit=np.array(x_sub); y_best=res.best_fit
   444|        bg_at_peak=float(np.interp(center,x_fit,y_best-res.eval_components().get(f"{peak_name}_",0.)))
   445|        peak_only=np.maximum(res.eval_components().get(f"{peak_name}_",y_best-y_sub.min()),0)
   446|        peak_max=float(peak_only.max())
   447|        area=float(np.trapz(peak_only,x_fit))
   448|        beta=area/peak_max if peak_max>0 else fwhm
   449|
   450|        os.unlink(tmpfile)
   451|        return {
   452|            "two_theta": center, "fwhm": fwhm, "fwhm_err": fwhm_err,
   453|            "eta": eta, "amplitude": amp, "background": bg_at_peak,
   454|            "beta": beta, "area": area,
   455|            "redchi": float(res.redchi) if res.redchi else 0.,
   456|            "aic": float(res.aic) if res.aic else 0.,
   457|            "profile": "xrdfit/PseudoVoigt",
   458|            "report": res.fit_report(),
   459|            "x_fit": x_fit, "y_fit": y_sub, "y_best": y_best,
   460|        }
   461|    except Exception:
   462|        try: os.unlink(tmpfile)
   463|        except: pass
   464|        return fit_peak_lmfit(tt_arr, obs_arr, tt_center, window_deg)
   465|
   466|def extract_all_fwhm(tt_arr, obs_arr, refs, wavelength,
   467|                     window_deg=1.5, min_intensity_pct=2.,
   468|                     backend="xrdfit", profile="PseudoVoigt", bg_model="linear"):
   469|    """Iterate over all reflections and fit each peak."""
   470|    I_max=obs_arr.max()
   471|    results=[]
   472|    for r in refs:
   473|        ttk=r["two_theta"]
   474|        if ttk<tt_arr.min() or ttk>tt_arr.max(): continue
   475|        idx=np.argmin(np.abs(tt_arr-ttk))
   476|        local_max=obs_arr[max(0,idx-15):min(len(obs_arr),idx+15)].max()
   477|        if local_max<min_intensity_pct/100*I_max: continue
   478|
   479|        if backend=="xrdfit":
   480|            fit=fit_peak_xrdfit(tt_arr,obs_arr,ttk,window_deg)
   481|        else:
   482|            fit=fit_peak_lmfit(tt_arr,obs_arr,ttk,window_deg,profile,bg_model)
   483|        if fit is None: continue
   484|
   485|        theta_rad=np.radians(fit["two_theta"]/2)
   486|        sin_t=np.sin(theta_rad); cos_t=np.cos(theta_rad)
   487|        d_val=wavelength/(2*sin_t) if sin_t>0 else r["d"]
   488|        fit.update({
   489|            "h":r["h"],"k":r["k"],"l":r["l"],"d":d_val,
   490|            "theta_rad":theta_rad,"sin_theta":sin_t,"cos_theta":cos_t,
   491|            "fwhm_rad":np.radians(fit["fwhm"]),
   492|            "beta_rad":np.radians(fit["beta"]),
   493|        })
   494|        results.append(fit)
   495|    return results
   496|
   497|# ══════════════════════════════════════════════════════════════════════════════
   498|# ── SCHERRER, WILLIAMSON-HALL, HALDER-WAGNER ──
   499|# ══════════════════════════════════════════════════════════════════════════════
   500|
   501|
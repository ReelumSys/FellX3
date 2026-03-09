"""
Bravais-Gitter Visualisierung aus CIF-Datei
============================================
Interaktive 3D-Darstellung mit Three.js (via Streamlit Components).

Abhängigkeiten:
    pip install streamlit numpy

Starten:
    streamlit run bravais_lattice.py
"""

import streamlit as st
import numpy as np
import re
import json

st.set_page_config(
    page_title="Bravais Lattice Viewer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;800&family=Share+Tech+Mono&display=swap');
:root{
  --bg:#060810; --card:#0b0e18; --border:#151b2e;
  --acc:#00BFFF; --acc2:#FF6B6B; --acc3:#FFE66D;
  --ok:#39FF8A; --muted:#3d4a66; --text:#d8e0f0;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important; color:var(--text)!important;
  font-family:'Exo 2',sans-serif;
}
[data-testid="stSidebar"]{
  background:var(--card)!important;
  border-right:1px solid var(--border);
}
h1,h2,h3{font-family:'Exo 2',sans-serif;font-weight:800;}
.stTabs [data-baseweb="tab-list"]{
  background:var(--card);border-radius:6px;padding:3px;
  border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  color:var(--muted);font-family:'Share Tech Mono',monospace;
  font-size:.75rem;border-radius:4px;padding:5px 14px;
}
.stTabs [aria-selected="true"]{background:var(--acc)!important;color:#000!important;font-weight:700;}
div[data-testid="stMetric"]{
  background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;
}
div[data-testid="stMetric"] label{color:var(--muted)!important;font-size:.68rem;
  font-family:'Share Tech Mono',monospace;text-transform:uppercase;}
div[data-testid="stMetric"] div{color:var(--acc)!important;
  font-family:'Share Tech Mono',monospace;font-size:1.1rem;}
[data-testid="stFileUploader"]{
  background:var(--card);border:1px dashed var(--border);border-radius:8px;
}
.stButton>button{
  background:transparent;border:1px solid var(--acc);color:var(--acc);
  font-family:'Share Tech Mono',monospace;border-radius:5px;
  transition:all .15s;font-size:.82rem;
}
.stButton>button:hover{background:var(--acc);color:#000;}
.stNumberInput input,.stSelectbox select{
  background:var(--card)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;
  font-family:'Share Tech Mono',monospace;border-radius:4px;
}
.info-card{
  background:var(--card);border:1px solid var(--border);
  border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
  padding:10px 14px;font-family:'Share Tech Mono',monospace;
  font-size:.78rem;margin:6px 0;line-height:1.8;
}
.lattice-badge{
  display:inline-block;background:var(--card);
  border:1px solid var(--acc);border-radius:20px;
  padding:3px 14px;font-family:'Share Tech Mono',monospace;
  font-size:.78rem;color:var(--acc);margin:3px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CIF PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_cif(content: str) -> dict:
    def extr(key):
        m = re.search(rf"_{key}\s+([\S]+)", content, re.IGNORECASE)
        if m:
            v = m.group(1).replace("(", "").replace(")", "")
            try: return float(v)
            except: return v
        return None

    data = {}
    data["a"]     = extr("cell_length_a")    or 5.0
    data["b"]     = extr("cell_length_b")    or 5.0
    data["c"]     = extr("cell_length_c")    or 5.0
    data["alpha"] = extr("cell_angle_alpha") or 90.0
    data["beta"]  = extr("cell_angle_beta")  or 90.0
    data["gamma"] = extr("cell_angle_gamma") or 90.0

    sg  = (extr("symmetry_space_group_name_H-M") or
           extr("space_group_name_H-M_alt") or
           extr("symmetry_space_group_name_H_M"))
    sgn = (extr("symmetry_Int_Tables_number") or
           extr("space_group_IT_number"))
    data["space_group"] = str(sg).strip("'\" ") if sg else "P 1"
    data["sg_number"]   = int(sgn) if sgn else 1

    # formula
    formula = extr("chemical_formula_sum") or extr("chemical_formula_structural") or ""
    data["formula"] = str(formula).strip("'\" ") if formula else "?"

    # compound name
    name = extr("chemical_name_mineral") or extr("chemical_name_common") or extr("chemical_name_systematic") or ""
    data["name"] = str(name).strip("'\" ") if name else "?"

    # Z
    z = extr("cell_formula_units_Z")
    data["Z"] = int(z) if z else 1

    # atoms
    atoms = []
    loop_m = re.search(r"loop_.*?_atom_site_label.*?(?=loop_|\Z)", content, re.DOTALL | re.IGNORECASE)
    if loop_m:
        blk  = loop_m.group(0)
        hdrs = re.findall(r"_atom_site_(\w+)", blk, re.IGNORECASE)
        rows = re.findall(r"^\s{0,4}([A-Za-z][A-Za-z0-9]*\d*\s+.+)$", blk, re.MULTILINE)
        for row in rows:
            pts = row.split()
            if len(pts) < 4: continue
            at = {"label": pts[0], "type": re.sub(r"[0-9+\-]", "", pts[0])[:2]}
            try:
                ix = next(i for i, h in enumerate(hdrs)
                          if "fract_x" in h.lower() or h.lower() == "x")
                at["x"] = float(pts[ix+1].split("(")[0])
                at["y"] = float(pts[ix+2].split("(")[0])
                at["z"] = float(pts[ix+3].split("(")[0])
            except:
                at["x"] = at["y"] = at["z"] = 0.0
            try:    at["occ"]  = float(pts[-2].split("(")[0])
            except: at["occ"]  = 1.0
            try:    at["Biso"] = float(pts[-1].split("(")[0])
            except: at["Biso"] = 1.0
            atoms.append(at)
    data["atoms"] = atoms
    return data

# ══════════════════════════════════════════════════════════════════════════════
# CRYSTAL SYSTEM & BRAVAIS LATTICE
# ══════════════════════════════════════════════════════════════════════════════
def get_bravais(sg_number: int, sg_name: str) -> dict:
    """Determine Bravais lattice type from space group."""
    sg = sg_name.strip()
    # Centering from space group symbol
    centering = sg[0] if sg else "P"
    if centering not in "PIFABCR": centering = "P"

    if sg_number <= 2:   system = "Triklin"
    elif sg_number <= 15: system = "Monoklin"
    elif sg_number <= 74: system = "Orthorhombisch"
    elif sg_number <= 142: system = "Tetragonal"
    elif sg_number <= 167: system = "Trigonal"
    elif sg_number <= 194: system = "Hexagonal"
    else:                 system = "Kubisch"

    bravais_map = {
        ("Triklin",      "P"): ("aP", "Triklin primitiv"),
        ("Monoklin",     "P"): ("mP", "Monoklin primitiv"),
        ("Monoklin",     "C"): ("mS", "Monoklin basiszentriert"),
        ("Orthorhombisch","P"): ("oP", "Orthorhombisch primitiv"),
        ("Orthorhombisch","C"): ("oS", "Orthorhombisch basiszentriert"),
        ("Orthorhombisch","I"): ("oI", "Orthorhombisch raumzentriert"),
        ("Orthorhombisch","F"): ("oF", "Orthorhombisch flächenzentriert"),
        ("Tetragonal",   "P"): ("tP", "Tetragonal primitiv"),
        ("Tetragonal",   "I"): ("tI", "Tetragonal raumzentriert"),
        ("Trigonal",     "P"): ("hR", "Rhomboedrisch"),
        ("Trigonal",     "R"): ("hR", "Rhomboedrisch"),
        ("Hexagonal",    "P"): ("hP", "Hexagonal primitiv"),
        ("Kubisch",      "P"): ("cP", "Kubisch primitiv"),
        ("Kubisch",      "I"): ("cI", "Kubisch raumzentriert (BCC)"),
        ("Kubisch",      "F"): ("cF", "Kubisch flächenzentriert (FCC)"),
    }
    key  = (system, centering)
    sym, name = bravais_map.get(key, ("?", f"{system} ({centering})"))
    return {"system": system, "centering": centering, "symbol": sym, "name": name}

# ══════════════════════════════════════════════════════════════════════════════
# LATTICE VECTOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def cell_vectors(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Compute Cartesian lattice vectors."""
    al = np.radians(alpha_deg)
    be = np.radians(beta_deg)
    ga = np.radians(gamma_deg)
    # a along x
    ax = a
    # b in xy plane
    bx = b * np.cos(ga)
    by = b * np.sin(ga)
    # c general
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be)*np.cos(ga)) / np.sin(ga)
    cz_sq = c**2 - cx**2 - cy**2
    cz = np.sqrt(max(cz_sq, 0))
    return (
        np.array([ax, 0,  0]),
        np.array([bx, by, 0]),
        np.array([cx, cy, cz])
    )

def centering_translations(centering: str):
    """Return fractional translation vectors for centering."""
    t = {
        "P": [[0,0,0]],
        "I": [[0,0,0],[.5,.5,.5]],
        "F": [[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]],
        "A": [[0,0,0],[0,.5,.5]],
        "B": [[0,0,0],[.5,0,.5]],
        "C": [[0,0,0],[.5,.5,0]],
        "R": [[0,0,0],[2/3,1/3,1/3],[1/3,2/3,2/3]],
    }
    return t.get(centering, [[0,0,0]])

def element_color(symbol: str) -> str:
    """CPK color scheme."""
    colors = {
        "H":"#FFFFFF","He":"#D9FFFF","Li":"#CC80FF","Be":"#C2FF00",
        "B":"#FFB5B5","C":"#909090","N":"#3050F8","O":"#FF0D0D",
        "F":"#90E050","Ne":"#B3E3F5","Na":"#AB5CF2","Mg":"#8AFF00",
        "Al":"#BFA6A6","Si":"#F0C8A0","P":"#FF8000","S":"#FFFF30",
        "Cl":"#1FF01F","Ar":"#80D1E3","K":"#8F40D4","Ca":"#3DFF00",
        "Ti":"#BFC2C7","V":"#A6A6AB","Cr":"#8A99C7","Mn":"#9C7AC7",
        "Fe":"#E06633","Co":"#F090A0","Ni":"#50D050","Cu":"#C88033",
        "Zn":"#7D80B0","Ga":"#C28F8F","Ge":"#668F8F","As":"#BD80E3",
        "Se":"#FFA100","Br":"#A62929","Kr":"#5CB8D1","Rb":"#702EB0",
        "Sr":"#00FF00","Y":"#94FFFF","Zr":"#94E0E0","Nb":"#73C2C9",
        "Mo":"#54B5B5","Ag":"#C0C0C0","Ba":"#00C900","La":"#70D4FF",
        "Ce":"#FFFFC7","Pb":"#575961","Bi":"#9E4FB5","Au":"#FFD123",
        "Pt":"#D0D0E0","Pd":"#006985","W":"#2194D6","Ta":"#4DA6FF",
    }
    return colors.get(symbol.capitalize(), "#FF69B4")

def element_radius(symbol: str) -> float:
    """Van der Waals radii in Å (scaled for display)."""
    radii = {
        "H":.53,"He":.31,"Li":1.67,"Be":1.12,"B":.87,"C":.67,
        "N":.56,"O":.48,"F":.42,"Na":1.90,"Mg":1.45,"Al":1.18,
        "Si":1.11,"P":.98,"S":.88,"Cl":.79,"K":2.43,"Ca":1.94,
        "Ti":1.76,"V":1.71,"Cr":1.66,"Mn":1.61,"Fe":1.56,"Co":1.52,
        "Ni":1.49,"Cu":1.45,"Zn":1.42,"Ge":1.22,"As":1.19,"Se":1.16,
        "Br":1.14,"Rb":2.65,"Sr":2.19,"Y":2.12,"Zr":2.06,"Nb":1.98,
        "Mo":1.90,"Ag":1.72,"Ba":2.53,"La":2.57,"Ce":2.58,"Au":1.66,
        "Pt":1.77,"Pb":2.02,"Bi":2.07,"W":1.93,"Ta":2.00,"Pd":1.63,
    }
    return radii.get(symbol.capitalize(), 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD GEOMETRY DATA FOR Three.js
# ══════════════════════════════════════════════════════════════════════════════
def build_geometry(cell: dict, supercell=(1,1,1), show_atoms=True, show_bonds=True,
                   bond_cutoff=3.5, show_lattice_points=True) -> dict:
    a,b,c = cell["a"], cell["b"], cell["c"]
    al,be,ga = cell["alpha"], cell["beta"], cell["gamma"]
    va, vb, vc = cell_vectors(a, b, c, al, be, ga)
    centering   = cell.get("centering", "P")
    translations= centering_translations(centering)
    atoms_frac  = cell.get("atoms", [])
    nx, ny, nz  = supercell

    # ── Unit cell edges (12 edges × 2 endpoints) ──
    def corner(i, j, k):
        v = i*va + j*vb + k*vc
        return v.tolist()

    edges = []
    for ix in range(nx):
     for iy in range(ny):
      for iz in range(nz):
        o = ix*va + iy*vb + iz*vc
        for (p1,p2) in [((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
                         ((1,0,0),(1,1,0)),((1,0,0),(1,0,1)),
                         ((0,1,0),(1,1,0)),((0,1,0),(0,1,1)),
                         ((0,0,1),(1,0,1)),((0,0,1),(0,1,1)),
                         ((1,1,0),(1,1,1)),((1,0,1),(1,1,1)),((0,1,1),(1,1,1))]:
            s = o + p1[0]*va + p1[1]*vb + p1[2]*vc
            e = o + p2[0]*va + p2[1]*vb + p2[2]*vc
            edges.append({"s": s.tolist(), "e": e.tolist()})

    # ── Atom positions ──
    atom_spheres = []
    if show_atoms and atoms_frac:
        for ix in range(nx):
         for iy in range(ny):
          for iz in range(nz):
            offset = ix*va + iy*vb + iz*vc
            for at in atoms_frac:
                for tr in translations:
                    fx = at["x"] + tr[0]
                    fy = at["y"] + tr[1]
                    fz = at["z"] + tr[2]
                    # Only keep atoms inside unit cell [0,1)
                    fx = fx % 1.0; fy = fy % 1.0; fz = fz % 1.0
                    pos = offset + fx*va + fy*vb + fz*vc
                    sym = at["type"][:2].strip()
                    atom_spheres.append({
                        "pos": pos.tolist(),
                        "color": element_color(sym),
                        "radius": element_radius(sym) * 0.25,
                        "label": at["label"],
                        "element": sym,
                        "occ": at.get("occ", 1.0),
                    })

    # ── Lattice centering points (for lattice-point-only view) ──
    lattice_pts = []
    if show_lattice_points:
        for ix in range(nx+1):
         for iy in range(ny+1):
          for iz in range(nz+1):
            for tr in translations:
                fx = ix + tr[0]; fy = iy + tr[1]; fz = iz + tr[2]
                # filter duplicates at supercell boundary
                if fx > nx+1e-6 or fy > ny+1e-6 or fz > nz+1e-6: continue
                pos = fx*va + fy*vb + fz*vc
                lattice_pts.append(pos.tolist())

    # ── Bonds (simple distance cutoff) ──
    bonds = []
    if show_bonds and len(atom_spheres) > 1:
        positions = [np.array(s["pos"]) for s in atom_spheres]
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if 0.3 < dist < bond_cutoff:
                    bonds.append({
                        "s": positions[i].tolist(),
                        "e": positions[j].tolist(),
                    })

    # ── Axes vectors ──
    axes = {
        "a": va.tolist(), "b": vb.tolist(), "c": vc.tolist(),
        "a_len": a, "b_len": b, "c_len": c,
    }

    return {
        "edges": edges,
        "atoms": atom_spheres,
        "lattice_pts": lattice_pts,
        "bonds": bonds,
        "axes": axes,
        "cell": {"a":a,"b":b,"c":c,"alpha":al,"beta":be,"gamma":ga},
    }

# ══════════════════════════════════════════════════════════════════════════════
# THREE.js HTML COMPONENT
# ══════════════════════════════════════════════════════════════════════════════
def make_threejs_html(geo: dict, bravais: dict, height=650) -> str:
    geo_json = json.dumps(geo)
    bv_json  = json.dumps(bravais)
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{margin:0;padding:0;box-sizing:border-box;}}
  body {{background:#060810;overflow:hidden;font-family:'Share Tech Mono',monospace;}}
  #canvas-wrap {{width:100%;height:{height}px;position:relative;}}
  canvas {{display:block;}}
  #overlay {{
    position:absolute;top:10px;left:12px;
    color:#00BFFF;font-size:.72rem;line-height:1.7;
    text-shadow:0 0 8px #00BFFF88;pointer-events:none;
  }}
  #legend {{
    position:absolute;bottom:10px;right:12px;
    background:rgba(6,8,16,.85);border:1px solid #151b2e;
    border-radius:8px;padding:8px 12px;color:#d8e0f0;
    font-size:.7rem;max-width:180px;
  }}
  #legend h4{{color:#00BFFF;margin-bottom:4px;font-size:.75rem;}}
  .leg-item{{display:flex;align-items:center;gap:6px;margin:2px 0;}}
  .leg-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  #controls-hint{{
    position:absolute;bottom:10px;left:12px;
    color:#3d4a66;font-size:.65rem;line-height:1.6;pointer-events:none;
  }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
</head>
<body>
<div id="canvas-wrap">
  <div id="overlay"></div>
  <div id="legend"><h4>Legende</h4><div id="legend-items"></div></div>
  <div id="controls-hint">
    🖱 Linksklick: Drehen<br>
    🖱 Rechtsklick: Verschieben<br>
    🖱 Scroll: Zoom
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const GEO = {geo_json};
const BV  = {bv_json};

// ── Scene setup ──
const wrap   = document.getElementById('canvas-wrap');
const W = wrap.clientWidth, H = wrap.clientHeight;
const renderer = new THREE.WebGLRenderer({{antialias:true,alpha:true}});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
wrap.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W/H, 0.01, 500);

// ── Lighting ──
scene.add(new THREE.AmbientLight(0x334466, 1.2));
const dLight = new THREE.DirectionalLight(0xffffff, 1.0);
dLight.position.set(10, 15, 10);
scene.add(dLight);
const dLight2 = new THREE.DirectionalLight(0x00BFFF, 0.4);
dLight2.position.set(-8,-5,8);
scene.add(dLight2);

// ── Materials ──
const edgeMat   = new THREE.LineBasicMaterial({{color:0x00BFFF, linewidth:1.5, transparent:true, opacity:.7}});
const bondMat   = new THREE.MeshPhongMaterial({{color:0xaabbcc, transparent:true, opacity:.45, shininess:40}});
const latPtMat  = new THREE.MeshPhongMaterial({{color:0x00BFFF, emissive:0x004466, shininess:80}});

const group = new THREE.Group();
scene.add(group);

// ── Helpers ──
function vec3(arr){{ return new THREE.Vector3(arr[0],arr[1],arr[2]); }}

function makeCylinder(p1, p2, mat, radius=0.04){{
  const dir = vec3(p2).sub(vec3(p1));
  const len = dir.length();
  if(len < 1e-6) return null;
  const mid = vec3(p1).add(vec3(p2)).multiplyScalar(.5);
  const geo = new THREE.CylinderGeometry(radius,radius,len,8,1);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0,1,0), dir.normalize()
  );
  return mesh;
}}

// ── Unit cell edges ──
GEO.edges.forEach(e => {{
  const pts = [vec3(e.s), vec3(e.e)];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  group.add(new THREE.Line(geo, edgeMat));
}});

// ── Lattice points ──
const lpGeo = new THREE.SphereGeometry(.08, 12, 12);
GEO.lattice_pts.forEach(p => {{
  const mesh = new THREE.Mesh(lpGeo, latPtMat);
  mesh.position.copy(vec3(p));
  group.add(mesh);
}});

// ── Atom spheres ──
const legendElements = {{}};
GEO.atoms.forEach(at => {{
  const geo = new THREE.SphereGeometry(at.radius, 20, 20);
  const mat = new THREE.MeshPhongMaterial({{
    color: new THREE.Color(at.color),
    emissive: new THREE.Color(at.color).multiplyScalar(.15),
    shininess: 90,
    transparent: at.occ < 1.0,
    opacity: Math.max(at.occ, .3),
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(vec3(at.pos));
  mesh.userData = {{label: at.label, element: at.element}};
  group.add(mesh);
  legendElements[at.element] = at.color;
}});

// ── Bonds ──
GEO.bonds.forEach(b => {{
  const cyl = makeCylinder(b.s, b.e, bondMat, 0.03);
  if(cyl) group.add(cyl);
}});

// ── Axes arrows ──
const axColors = [0xFF4444, 0x44FF44, 0x4488FF];
const axLabels = ['a','b','c'];
const axVecs   = [GEO.axes.a, GEO.axes.b, GEO.axes.c];
axVecs.forEach((v, i) => {{
  const dir = new THREE.Vector3(v[0],v[1],v[2]).normalize();
  const len = new THREE.Vector3(v[0],v[1],v[2]).length();
  const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0),
                                       len*1.15, axColors[i], len*.12, .08);
  group.add(arrow);
}});

// ── Center group ──
const box = new THREE.Box3().setFromObject(group);
const center = new THREE.Vector3();
box.getCenter(center);
group.position.sub(center);
const size = box.getSize(new THREE.Vector3()).length();
camera.position.set(size*.7, size*.5, size*1.1);
camera.lookAt(0,0,0);

// ── Overlay info ──
const ov = document.getElementById('overlay');
ov.innerHTML = `
  <div style="font-size:.9rem;color:#00BFFF;font-weight:700;margin-bottom:4px;">
    ${{BV.name}}
  </div>
  <div>Symbol: <b style="color:#FFE66D">${{BV.symbol}}</b></div>
  <div>System: ${{BV.system}}</div>
  <div>Zentrierung: ${{BV.centering}}</div>
  <hr style="border-color:#151b2e;margin:5px 0;">
  <div>a = ${{GEO.cell.a.toFixed(4)}} Å</div>
  <div>b = ${{GEO.cell.b.toFixed(4)}} Å</div>
  <div>c = ${{GEO.cell.c.toFixed(4)}} Å</div>
  <div>α = ${{GEO.cell.alpha.toFixed(3)}}°</div>
  <div>β = ${{GEO.cell.beta.toFixed(3)}}°</div>
  <div>γ = ${{GEO.cell.gamma.toFixed(3)}}°</div>
`;

// ── Legend ──
const legDiv = document.getElementById('legend-items');
const axLegend = `
  <div class="leg-item"><div class="leg-dot" style="background:#FF4444;border-radius:0;height:3px;width:16px;"></div><span>a-Achse</span></div>
  <div class="leg-item"><div class="leg-dot" style="background:#44FF44;border-radius:0;height:3px;width:16px;"></div><span>b-Achse</span></div>
  <div class="leg-item"><div class="leg-dot" style="background:#4488FF;border-radius:0;height:3px;width:16px;"></div><span>c-Achse</span></div>
  <div class="leg-item"><div class="leg-dot" style="background:#00BFFF;"></div><span>Gitterpunkt</span></div>
`;
legDiv.innerHTML = axLegend;
Object.entries(legendElements).forEach(([el, col]) => {{
  const d = document.createElement('div');
  d.className = 'leg-item';
  d.innerHTML = `<div class="leg-dot" style="background:${{col}};"></div><span>${{el}}</span>`;
  legDiv.appendChild(d);
}});

// ── Mouse orbit controls ──
let isDragging = false, isRightDrag = false;
let prevX = 0, prevY = 0;
const spherical = new THREE.Spherical().setFromVector3(camera.position);

renderer.domElement.addEventListener('mousedown', e => {{
  isDragging = true; isRightDrag = e.button === 2;
  prevX = e.clientX; prevY = e.clientY;
}});
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
renderer.domElement.addEventListener('mouseup',   () => isDragging = false);
renderer.domElement.addEventListener('mouseleave',() => isDragging = false);
renderer.domElement.addEventListener('mousemove', e => {{
  if(!isDragging) return;
  const dx = (e.clientX - prevX) * .008;
  const dy = (e.clientY - prevY) * .008;
  prevX = e.clientX; prevY = e.clientY;
  if(isRightDrag){{
    // Pan
    const right = new THREE.Vector3().crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
    group.position.addScaledVector(right, -dx*size*.3);
    group.position.addScaledVector(camera.up, dy*size*.3);
  }} else {{
    // Orbit
    spherical.theta -= dx;
    spherical.phi   = Math.max(.1, Math.min(Math.PI-.1, spherical.phi + dy));
    camera.position.setFromSpherical(spherical);
    camera.lookAt(0,0,0);
  }}
}});
renderer.domElement.addEventListener('wheel', e => {{
  spherical.radius = Math.max(.5, spherical.radius + e.deltaY * .01);
  camera.position.setFromSpherical(spherical);
  camera.lookAt(0,0,0);
}});

// ── Touch support ──
let lastTouchDist = 0;
renderer.domElement.addEventListener('touchstart', e => {{
  if(e.touches.length === 1){{
    isDragging = true; prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
  }} else if(e.touches.length === 2){{
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    lastTouchDist = Math.sqrt(dx*dx+dy*dy);
  }}
}});
renderer.domElement.addEventListener('touchmove', e => {{
  e.preventDefault();
  if(e.touches.length === 1 && isDragging){{
    const dx = (e.touches[0].clientX - prevX)*.008;
    const dy = (e.touches[0].clientY - prevY)*.008;
    prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
    spherical.theta -= dx;
    spherical.phi = Math.max(.1, Math.min(Math.PI-.1, spherical.phi+dy));
    camera.position.setFromSpherical(spherical);
    camera.lookAt(0,0,0);
  }} else if(e.touches.length === 2){{
    const dx = e.touches[0].clientX-e.touches[1].clientX;
    const dy = e.touches[0].clientY-e.touches[1].clientY;
    const dist = Math.sqrt(dx*dx+dy*dy);
    spherical.radius = Math.max(.5, spherical.radius*(lastTouchDist/dist));
    lastTouchDist = dist;
    camera.position.setFromSpherical(spherical);
    camera.lookAt(0,0,0);
  }}
}},{{passive:false}});
renderer.domElement.addEventListener('touchend', () => isDragging=false);

// ── Resize ──
window.addEventListener('resize', () => {{
  const W2 = wrap.clientWidth;
  renderer.setSize(W2, H);
  camera.aspect = W2/H;
  camera.updateProjectionMatrix();
}});

// ── Animate ──
let autoRotate = true;
document.addEventListener('keydown', e => {{ if(e.key===' ') autoRotate=!autoRotate; }});

function animate(){{
  requestAnimationFrame(animate);
  if(autoRotate){{
    spherical.theta += .003;
    camera.position.setFromSpherical(spherical);
    camera.lookAt(0,0,0);
  }}
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "cell" not in st.session_state: st.session_state.cell = {}
if "bravais" not in st.session_state: st.session_state.bravais = {}

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:4px 0 18px 0;'>
  <span style='font-family:Exo 2,sans-serif;font-size:2.2rem;font-weight:800;
               color:#00BFFF;letter-spacing:-1px;'>Bravais-Gitter</span>
  <span style='font-family:Share Tech Mono,monospace;font-size:.8rem;
               color:#3d4a66;margin-left:14px;'>3D-Visualisierung aus CIF-Datei</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 CIF-Import")
    cif_file = st.file_uploader("CIF-Datei hochladen", type=["cif"])

    if cif_file:
        content = cif_file.read().decode("utf-8", errors="ignore")
        cell = parse_cif(content)
        bv   = get_bravais(cell["sg_number"], cell["space_group"])
        cell["centering"] = bv["centering"]
        st.session_state.cell    = cell
        st.session_state.bravais = bv
        st.success(f"✓ {cell.get('name','?')} · {bv['name']}")

    st.divider()
    st.markdown("### ⚙️ Darstellung")

    supercell_n = st.slider("Superzelle (n×n×n)", 1, 3, 1)
    supercell   = (supercell_n, supercell_n, supercell_n)

    show_atoms  = st.checkbox("Atome anzeigen",          value=True)
    show_bonds  = st.checkbox("Bindungen anzeigen",       value=True)
    show_lp     = st.checkbox("Gitterpunkte hervorheben", value=True)
    bond_cutoff = st.slider("Bindungs-Cutoff (Å)", 1.0, 6.0, 3.5, 0.1)
    viewer_h    = st.slider("Viewer-Höhe (px)", 400, 900, 650, 50)

    st.divider()
    st.markdown("### 🔧 Manuelle Gitterparameter")
    st.caption("Überschreibt CIF-Werte")

    if st.session_state.cell:
        c = st.session_state.cell
    else:
        c = {"a":5.,"b":5.,"c":5.,"alpha":90.,"beta":90.,"gamma":90.,
             "sg_number":225,"space_group":"F m -3 m","centering":"F",
             "atoms":[],"formula":"?","name":"?","Z":4}

    col1,col2 = st.columns(2)
    with col1:
        a_v = st.number_input("a (Å)", value=float(c.get("a",5.)), step=.01, format="%.4f")
        b_v = st.number_input("b (Å)", value=float(c.get("b",5.)), step=.01, format="%.4f")
        c_v = st.number_input("c (Å)", value=float(c.get("c",5.)), step=.01, format="%.4f")
    with col2:
        al_v = st.number_input("α (°)", value=float(c.get("alpha",90.)), step=.1, format="%.3f")
        be_v = st.number_input("β (°)", value=float(c.get("beta",90.)),  step=.1, format="%.3f")
        ga_v = st.number_input("γ (°)", value=float(c.get("gamma",90.)), step=.1, format="%.3f")

    centering_manual = st.selectbox("Zentrierung",
        ["P","I","F","A","B","C","R"],
        index=["P","I","F","A","B","C","R"].index(c.get("centering","P")))

    # Apply manual values
    c_display = dict(c)
    c_display.update({"a":a_v,"b":b_v,"c":c_v,
                       "alpha":al_v,"beta":be_v,"gamma":ga_v,
                       "centering":centering_manual})

    bv_display = get_bravais(c_display.get("sg_number",1), c_display["centering"]+"  ")
    bv_display["centering"] = centering_manual


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
tab_3d, tab_info, tab_atoms, tab_all14 = st.tabs([
    "🔷 3D-Gitter", "ℹ️ Kristallinfo", "⚛️ Atome", "📚 Alle 14 Bravais-Gitter"
])

# ─── TAB 1: 3D Viewer ───
with tab_3d:
    geo = build_geometry(
        c_display, supercell,
        show_atoms=show_atoms,
        show_bonds=show_bonds,
        bond_cutoff=bond_cutoff,
        show_lattice_points=show_lp,
    )
    html_code = make_threejs_html(geo, bv_display, height=viewer_h)
    st.components.v1.html(html_code, height=viewer_h+10, scrolling=False)

    # Metrics below viewer
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("a (Å)", f"{a_v:.4f}")
    m2.metric("b (Å)", f"{b_v:.4f}")
    m3.metric("c (Å)", f"{c_v:.4f}")
    m4.metric("α (°)", f"{al_v:.3f}")
    m5.metric("β (°)", f"{be_v:.3f}")
    m6.metric("γ (°)", f"{ga_v:.3f}")

    st.markdown(f"""
    <div style='margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;'>
      <span class='lattice-badge'>🔷 {bv_display['name']}</span>
      <span class='lattice-badge'>Symbol: {bv_display['symbol']}</span>
      <span class='lattice-badge'>System: {bv_display['system']}</span>
      <span class='lattice-badge'>Zentrierung: {bv_display['centering']}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── TAB 2: Crystal Info ───
with tab_info:
    cell_info = c_display
    bv_info   = bv_display

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Kristallographische Daten")
        st.markdown(f"""
        <div class='info-card'>
        Verbindung  : <b>{cell_info.get('name','?')}</b><br>
        Formel      : <b>{cell_info.get('formula','?')}</b><br>
        Raumgruppe  : <b>{cell_info.get('space_group','?')}</b><br>
        SG-Nummer   : <b>{cell_info.get('sg_number','?')}</b><br>
        Z           : <b>{cell_info.get('Z','?')}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Gitterparameter")
        AL = np.radians(al_v); BE = np.radians(be_v); GA = np.radians(ga_v)
        vol = a_v*b_v*c_v*np.sqrt(
            1 - np.cos(AL)**2 - np.cos(BE)**2 - np.cos(GA)**2
            + 2*np.cos(AL)*np.cos(BE)*np.cos(GA)
        )
        rho_est = (cell_info.get("Z",1) * 100) / (vol * 0.6022) if vol > 0 else 0

        st.markdown(f"""
        <div class='info-card'>
        a = {a_v:.5f} Å &nbsp;|&nbsp; b = {b_v:.5f} Å &nbsp;|&nbsp; c = {c_v:.5f} Å<br>
        α = {al_v:.4f}° &nbsp;|&nbsp; β = {be_v:.4f}° &nbsp;|&nbsp; γ = {ga_v:.4f}°<br>
        <hr style="border-color:#151b2e;margin:5px 0;">
        Volumen : <b>{vol:.4f} Å³</b><br>
        Dichte (Abschätzung) : <b>{rho_est:.3f} g/cm³</b>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Bravais-Gitter")
        st.markdown(f"""
        <div class='info-card'>
        Kristallsystem : <b>{bv_info['system']}</b><br>
        Gittertyp      : <b>{bv_info['name']}</b><br>
        Pearson-Symbol : <b>{bv_info['symbol']}</b><br>
        Zentrierung    : <b>{bv_info['centering']}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Zentrierungs-Translationen")
        trans = centering_translations(bv_info['centering'])
        tr_df = pd.DataFrame(trans, columns=["x","y","z"])
        import pandas as pd
        tr_df = pd.DataFrame(trans, columns=["x","y","z"])
        st.dataframe(tr_df, use_container_width=True)

        # Lattice vectors
        va, vb, vc = cell_vectors(a_v, b_v, c_v, al_v, be_v, ga_v)
        st.markdown("#### Kartesische Gittervektoren (Å)")
        st.markdown(f"""
        <div class='info-card'>
        <b>a</b> = [{va[0]:.4f}, {va[1]:.4f}, {va[2]:.4f}]<br>
        <b>b</b> = [{vb[0]:.4f}, {vb[1]:.4f}, {vb[2]:.4f}]<br>
        <b>c</b> = [{vc[0]:.4f}, {vc[1]:.4f}, {vc[2]:.4f}]
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 3: Atoms ───
with tab_atoms:
    import pandas as pd
    st.markdown("#### Atompositionen (aus CIF)")
    if cell_info.get("atoms"):
        atoms_df = pd.DataFrame([{
            "Label":  at["label"],
            "Element": at["type"],
            "x (frac)": round(at["x"],5),
            "y (frac)": round(at["y"],5),
            "z (frac)": round(at["z"],5),
            "Occ.":    round(at["occ"],3),
            "B_iso":   round(at["Biso"],3),
            "Farbe":   element_color(at["type"]),
        } for at in cell_info["atoms"]])

        # Color swatches
        def color_cell(val):
            return f"background-color:{val};color:{val};"

        st.dataframe(atoms_df.drop(columns=["Farbe"]),
                     use_container_width=True, height=350)

        # Element summary
        st.markdown("#### Elementübersicht")
        from collections import Counter
        elem_counts = Counter(at["type"] for at in cell_info["atoms"])
        ec_df = pd.DataFrame([{
            "Element": el,
            "Anzahl": cnt,
            "Farbe (CPK)": element_color(el),
            "Radius (Å)": element_radius(el),
        } for el, cnt in sorted(elem_counts.items())])
        st.dataframe(ec_df, use_container_width=True)
    else:
        st.info("Keine Atompositionen in der CIF-Datei gefunden.")
        st.markdown("Lade eine vollständige CIF-Datei mit `_atom_site_*`-Einträgen.")

# ─── TAB 4: All 14 Bravais ───
with tab_all14:
    import pandas as pd
    st.markdown("### Die 14 Bravais-Gitter")

    bravais_14 = [
        ("aP","Triklin primitiv","Triklin","P",1,[90,90,90]),
        ("mP","Monoklin primitiv","Monoklin","P",3,[90,100,90]),
        ("mS","Monoklin basiszentriert","Monoklin","C",5,[90,100,90]),
        ("oP","Orthorhombisch primitiv","Orthorhombisch","P",16,[90,90,90]),
        ("oS","Orthorhombisch basiszentriert","Orthorhombisch","C",20,[90,90,90]),
        ("oI","Orthorhombisch raumzentriert","Orthorhombisch","I",23,[90,90,90]),
        ("oF","Orthorhombisch flächenzentriert","Orthorhombisch","F",22,[90,90,90]),
        ("tP","Tetragonal primitiv","Tetragonal","P",75,[90,90,90]),
        ("tI","Tetragonal raumzentriert","Tetragonal","I",79,[90,90,90]),
        ("hR","Rhomboedrisch","Trigonal","R",146,[60,60,60]),
        ("hP","Hexagonal primitiv","Hexagonal","P",168,[90,90,120]),
        ("cP","Kubisch primitiv (SC)","Kubisch","P",195,[90,90,90]),
        ("cI","Kubisch raumzentriert (BCC)","Kubisch","I",197,[90,90,90]),
        ("cF","Kubisch flächenzentriert (FCC)","Kubisch","F",196,[90,90,90]),
    ]

    df14 = pd.DataFrame(bravais_14,
                         columns=["Symbol","Name","Kristallsystem","Zentrierung",
                                  "SG-Nr. (Beispiel)","Winkel [α,β,γ]"])
    st.dataframe(df14, use_container_width=True, height=420)

    st.divider()

    # Quick preview selector
    st.markdown("#### Schnellvorschau")
    sel_bv = st.selectbox("Gittertyp auswählen",
                           [f"{b[0]} — {b[1]}" for b in bravais_14])
    sel_idx = [f"{b[0]} — {b[1]}" for b in bravais_14].index(sel_bv)
    sb = bravais_14[sel_idx]

    # Build preview cell
    prev_angles = sb[5]
    prev_cell = {
        "a":4.0, "b":5.0 if sb[2]=="Orthorhombisch" else 4.0,
        "c":6.0 if sb[2] in ("Orthorhombisch","Tetragonal","Hexagonal") else 4.0,
        "alpha":prev_angles[0], "beta":prev_angles[1], "gamma":prev_angles[2],
        "centering":sb[3], "atoms":[],
        "space_group":sb[3]+" ", "sg_number":sb[4],
        "name":sb[1], "formula":"?", "Z":1,
    }
    prev_bv = {"system":sb[2],"centering":sb[3],"symbol":sb[0],"name":sb[1]}
    prev_geo = build_geometry(prev_cell,(1,1,1),
                               show_atoms=False,show_bonds=False,
                               show_lattice_points=True)
    prev_html = make_threejs_html(prev_geo, prev_bv, height=420)
    st.components.v1.html(prev_html, height=430, scrolling=False)
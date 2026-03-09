"""
Bravais-Gitter Visualisierung aus einer CIF-Datei
Benötigt: pip install pymatgen matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import io
from main import cif_file2
# ----------------------------------------------------------------
# Hier deine CIF-Variable einsetzen
# cif_file2 kann sein:
#   - ein Dateipfad (str):     cif_file2 = "pfad/zur/datei.cif"
#   - ein CIF-String (str):    cif_file2 = "data_...\n_cell_length_a ..."
#   - ein bytes-Objekt:        cif_file2 = b"data_..."
# ----------------------------------------------------------------
# cif_file2 = "deine_datei.cif"   # <-- hier einsetzen oder von außen übergeben
# ----------------------------------------------------------------

def load_structure(cif_file2):
    """Lädt eine Struktur aus einem Dateipfad, CIF-String oder bytes."""
    if isinstance(cif_file2, (bytes, bytearray)):
        cif_file2 = cif_file2.decode("utf-8")
    
    if isinstance(cif_file2, str):
        # Prüfen ob es ein Dateipfad oder ein CIF-Inhalt ist
        if "\n" not in cif_file2 and cif_file2.endswith(".cif"):
            return Structure.from_file(cif_file2)
        else:
            return Structure.from_str(cif_file2, fmt="cif")
    
    raise ValueError("cif_file2 muss ein Dateipfad (str), CIF-String (str) oder bytes sein.")


def get_bravais_info(structure):
    """Extrahiert Bravais-Gitter-Informationen aus der Struktur."""
    analyzer = SpacegroupAnalyzer(structure)
    
    lattice = structure.lattice
    spacegroup = analyzer.get_space_group_symbol()
    spacegroup_number = analyzer.get_space_group_number()
    crystal_system = analyzer.get_crystal_system()
    lattice_type = analyzer.get_lattice_type()
    
    info = {
        "Raumgruppe":        f"{spacegroup} (#{spacegroup_number})",
        "Kristallsystem":    crystal_system,
        "Bravais-Gittertyp": lattice_type,
        "a (Å)":             f"{lattice.a:.4f}",
        "b (Å)":             f"{lattice.b:.4f}",
        "c (Å)":             f"{lattice.c:.4f}",
        "α (°)":             f"{lattice.alpha:.4f}",
        "β (°)":             f"{lattice.beta:.4f}",
        "γ (°)":             f"{lattice.gamma:.4f}",
        "Volumen (ų)":      f"{lattice.volume:.4f}",
        "Anzahl Atome":      len(structure),
    }
    return info, analyzer


def draw_parallelepiped(ax, matrix, color="steelblue", alpha=0.15, linecolor="navy"):
    """Zeichnet die Einheitszelle als Parallelepiped."""
    a, b, c = matrix[0], matrix[1], matrix[2]
    origin = np.zeros(3)

    # 8 Ecken der Einheitszelle
    corners = np.array([
        origin,
        a,
        b,
        c,
        a + b,
        a + c,
        b + c,
        a + b + c,
    ])

    # 12 Kanten
    edges = [
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,4),(2,6),
        (3,5),(3,6),
        (4,7),(5,7),(6,7),
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=linecolor, linewidth=1.5)

    # 6 Flächen
    faces = [
        [corners[0], corners[1], corners[4], corners[2]],
        [corners[3], corners[5], corners[7], corners[6]],
        [corners[0], corners[1], corners[5], corners[3]],
        [corners[2], corners[4], corners[7], corners[6]],
        [corners[0], corners[2], corners[6], corners[3]],
        [corners[1], corners[4], corners[7], corners[5]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=linecolor, linewidth=0.5)
    ax.add_collection3d(poly)

    return corners


def plot_bravais_lattice(cif_file2, supercell=(2, 2, 2), show_atoms=True):
    """
    Hauptfunktion: Liest CIF und plottet das Bravais-Gitter.
    
    Parameter:
        cif_file2   : Dateipfad, CIF-String oder bytes
        supercell   : Tuple (nx, ny, nz) für Wiederholungen der Einheitszelle
        show_atoms  : Atompositionen einzeichnen
    """
    structure = load_structure(cif_file2)
    info, analyzer = get_bravais_info(structure)

    matrix = structure.lattice.matrix  # 3x3 Matrix der Gittervektoren
    a_vec, b_vec, c_vec = matrix[0], matrix[1], matrix[2]

    # Farben pro Element
    element_colors = {}
    color_palette = plt.cm.Set1.colors
    elements = list(dict.fromkeys(str(s.specie) for s in structure))
    for i, el in enumerate(elements):
        element_colors[el] = color_palette[i % len(color_palette)]

    fig = plt.figure(figsize=(14, 8))

    # --- 3D Plot ---
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("Einheitszelle & Bravais-Gitter", fontsize=12, fontweight="bold", pad=12)

    nx, ny, nz = supercell
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = ix * a_vec + iy * b_vec + iz * c_vec
                shifted = matrix.copy()
                # Verschiebe Ecken
                draw_parallelepiped(
                    ax,
                    matrix,
                    color="steelblue" if (ix + iy + iz) % 2 == 0 else "lightsteelblue",
                    alpha=0.08,
                )
                # Verschiebe alle Punkte
                if ix == 0 and iy == 0 and iz == 0:
                    draw_parallelepiped(ax, matrix, color="steelblue", alpha=0.2, linecolor="navy")

    # Atome zeichnen
    if show_atoms:
        for site in structure:
            frac = site.frac_coords
            cart = structure.lattice.get_cartesian_coords(frac)
            el = str(site.specie)
            ax.scatter(*cart, color=element_colors[el], s=80, zorder=5,
                       edgecolors="black", linewidths=0.5, label=el)

        # Gitterpunkte der Superzelle
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz + 1):
                    pt = ix * a_vec + iy * b_vec + iz * c_vec
                    ax.scatter(*pt, color="gray", s=15, alpha=0.4, zorder=1)

    # Achsen-Beschriftung mit Gittervektoren
    scale = 1.2
    origin = np.zeros(3)
    for vec, label, color in zip(
        [a_vec, b_vec, c_vec], ["a", "b", "c"], ["red", "green", "blue"]
    ):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.15, linewidth=2)
        ax.text(*(vec * scale), label, color=color, fontsize=12, fontweight="bold")

    # Legende (Elemente)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=element_colors[el], markersize=9,
                   markeredgecolor="black", label=el)
        for el in elements
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, title="Elemente")

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.grid(False)

    # --- Info-Panel ---
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    ax2.set_title("Kristallografische Parameter", fontsize=12, fontweight="bold")

    rows = [[k, v] for k, v in info.items()]
    table = ax2.table(
        cellText=rows,
        colLabels=["Parameter", "Wert"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header-Zeile stylen
    for col in range(2):
        table[0, col].set_facecolor("#1a3a5c")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # Zeilen abwechselnd färben
    for row in range(1, len(rows) + 1):
        for col in range(2):
            table[row, col].set_facecolor("#eaf0fb" if row % 2 == 0 else "white")

    fig.suptitle(
        f"Bravais-Gitter: {info['Bravais-Gittertyp'].upper()}  |  "
        f"Raumgruppe: {info['Raumgruppe']}",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig("bravais_gitter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n--- Bravais-Gitter Info ---")
    for k, v in info.items():
        print(f"  {k:<22}: {v}")
    print("\nPlot gespeichert als: bravais_gitter.png")


# ----------------------------------------------------------------
# Aufruf – cif_file2 hier einsetzen:
# ----------------------------------------------------------------
if __name__ == "__main__":
    plot_bravais_lattice(cif_file2, supercell=(2, 2, 2), show_atoms=True)
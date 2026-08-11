#!/usr/bin/env python3
"""
Parametric generator for a Krause CORDA-style mobile assembly scaffold
as a Gazebo / gz-sim model.

Reference article: KRAUSE CORDA MontageGeruest, Art.-No. 10091497
    working height   ca. 5.00 m
    platform height  ca. 3.00 m
    scaffold height  ca. 4.20 m   <- "4.2 m" model
    base footprint   ca. 1.65 x 1.35 m (over stabilisers)
    platform         ca. 1.50 x 0.56 m
    mass             48.0 kg
    deck load        150 kg/m^2

The model is generated from primitive elements (tubes = cylinders,
boards = boxes). The same element list is emitted three ways:

  * model.sdf          - primitive <visual>/<collision> geometry, no mesh
                         dependency, works in Gazebo Classic and gz-sim.
  * meshes/*.obj/.mtl  - single triangulated visual mesh, if you prefer
                         one draw call / want to import into Blender.
  * inertial block     - mass properties integrated over all elements
                         (parallel-axis summed), not a hand-waved box.

Usage:  python3 corda_gen.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Parameters -- edit these to retarget the model (e.g. the 4.00 m / 27.5 kg
# variant, Art.-No. 22724915: height=3.10, platform_z=2.00, mass=27.5)
# --------------------------------------------------------------------------


@dataclass
class Params:
    name: str = "krause_corda_4200"

    height: float = 4.20          # top of guardrail above ground
    platform_z: float = 3.00      # walking surface height
    frame_span_x: float = 1.50    # distance between the two ladder frames
    frame_width_y: float = 0.60   # ladder frame width (rung length)

    base_x: float = 1.65          # stabiliser footprint, long axis
    base_y: float = 1.35          # stabiliser footprint, short axis

    deck_x: float = 1.50
    deck_y: float = 0.56
    deck_t: float = 0.045

    upright_d: float = 0.050      # vertical tube outer diameter
    rung_d: float = 0.030
    brace_d: float = 0.025
    rail_d: float = 0.035
    outrigger_d: float = 0.040

    rung_pitch: float = 0.280     # ladder rung spacing
    castor_d: float = 0.125       # castor wheel diameter
    castor_w: float = 0.040

    toeboard_h: float = 0.150
    toeboard_t: float = 0.020
    guard_top_off: float = 0.05   # top rail below scaffold height
    guard_mid_off: float = 0.60   # mid rail below top rail

    mass: float = 48.0            # kg, from datasheet

    second_deck: bool = False     # optional intermediate platform
    second_deck_z: float = 1.40


# --------------------------------------------------------------------------
# Element primitives
# --------------------------------------------------------------------------


@dataclass
class Element:
    kind: str                 # "cylinder" | "box"
    dims: tuple               # cylinder: (radius, length); box: (sx, sy, sz)
    xyz: np.ndarray           # centre position
    rpy: np.ndarray           # extrinsic XYZ euler (SDF convention)
    material: str
    collide: bool = True
    tag: str = ""

    def volume(self) -> float:
        if self.kind == "cylinder":
            r, l = self.dims
            return math.pi * r * r * l
        sx, sy, sz = self.dims
        return sx * sy * sz


def rot_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy_ = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tube(p0, p1, diameter, material, collide=True, tag="") -> Element:
    """Cylinder spanning two points; SDF cylinders are +Z aligned."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        raise ValueError("degenerate tube")
    dz = d / L
    pitch = math.acos(max(-1.0, min(1.0, dz[2])))
    yaw = math.atan2(dz[1], dz[0])
    return Element("cylinder", (diameter / 2.0, L), (p0 + p1) / 2.0,
                   np.array([0.0, pitch, yaw]), material, collide, tag)


def box(center, size, material, rpy=(0, 0, 0), collide=True, tag="") -> Element:
    return Element("box", tuple(float(s) for s in size),
                   np.asarray(center, dtype=float),
                   np.asarray(rpy, dtype=float), material, collide, tag)


# --------------------------------------------------------------------------
# Scaffold assembly
# --------------------------------------------------------------------------


def build(p: Params) -> list[Element]:
    e: list[Element] = []

    hx = p.frame_span_x / 2.0      # +-0.75
    hy = p.frame_width_y / 2.0     # +-0.30
    bx = p.base_x / 2.0            # +-0.825
    by = p.base_y / 2.0            # +-0.675

    r_cast = p.castor_d / 2.0
    z_frame = 2.0 * r_cast + 0.02  # bottom of the vertical frames
    z_top = p.height

    ALU = "alu"
    DECK = "wood"
    RED = "red"
    DARK = "dark"

    # ---- castors + splayed stabiliser legs -------------------------------
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * bx, sy * by
            # wheel: cylinder lying on its side, axis along X
            e.append(Element("cylinder", (r_cast, p.castor_w),
                             np.array([cx, cy, r_cast]),
                             np.array([0.0, math.pi / 2, 0.0]),
                             DARK, True, "castor"))
            # fork / swivel head
            e.append(tube((cx, cy, r_cast), (cx, cy, 2 * r_cast + 0.05),
                          0.045, RED, True, "castor_fork"))
            # stabiliser arm from the foot of the upright out to the castor
            e.append(tube((sx * hx, sy * hy, z_frame + 0.16),
                          (cx, cy, 2 * r_cast + 0.05),
                          p.outrigger_d, ALU, True, "outrigger"))

    # ---- four vertical uprights ------------------------------------------
    for sx in (-1, 1):
        for sy in (-1, 1):
            e.append(tube((sx * hx, sy * hy, z_frame),
                          (sx * hx, sy * hy, z_top),
                          p.upright_d, ALU, True, "upright"))

    # ---- ladder rungs on both end frames (x = +-hx) ----------------------
    z = z_frame + 0.25
    while z <= p.platform_z + 0.9:
        for sx in (-1, 1):
            e.append(tube((sx * hx, -hy, z), (sx * hx, hy, z),
                          p.rung_d, ALU, True, "rung"))
        z += p.rung_pitch

    # ---- horizontal ledgers on the long faces ----------------------------
    ledger_z = [z_frame + 0.10, 1.55, p.platform_z, p.platform_z + 1.0]
    if p.second_deck:
        ledger_z.append(p.second_deck_z)
    for zl in sorted(set(ledger_z)):
        if zl > z_top:
            continue
        for sy in (-1, 1):
            e.append(tube((-hx, sy * hy, zl), (hx, sy * hy, zl),
                          p.brace_d, ALU, True, "ledger"))

    # ---- diagonal bracing, both long faces, two lifts ---------------------
    for (z0, z1) in ((z_frame + 0.10, 1.55), (1.55, p.platform_z)):
        for sy in (-1, 1):
            e.append(tube((-hx, sy * hy, z0), (hx, sy * hy, z1),
                          p.brace_d, ALU, True, "brace"))
            e.append(tube((-hx, sy * hy, z1), (hx, sy * hy, z0),
                          p.brace_d, ALU, True, "brace"))

    # ---- platform(s) ------------------------------------------------------
    def deck(zt):
        e.append(box((0.0, 0.0, zt - p.deck_t / 2.0),
                     (p.deck_x, p.deck_y, p.deck_t), DECK, tag="deck"))

    deck(p.platform_z)
    if p.second_deck:
        deck(p.second_deck_z)

    # ---- guardrails + toeboards around the working platform --------------
    z_top_rail = z_top - p.guard_top_off
    z_mid_rail = z_top_rail - p.guard_mid_off
    for zr in (z_top_rail, z_mid_rail):
        for sy in (-1, 1):
            e.append(tube((-hx, sy * hy, zr), (hx, sy * hy, zr),
                          p.rail_d, ALU, True, "guardrail"))
        for sx in (-1, 1):
            e.append(tube((sx * hx, -hy, zr), (sx * hx, hy, zr),
                          p.rail_d, ALU, True, "guardrail"))

    tb_z = p.platform_z + p.toeboard_h / 2.0
    for sy in (-1, 1):
        e.append(box((0.0, sy * (p.deck_y / 2.0 + p.toeboard_t / 2.0), tb_z),
                     (p.deck_x, p.toeboard_t, p.toeboard_h), DECK,
                     tag="toeboard"))
    for sx in (-1, 1):
        e.append(box((sx * (p.deck_x / 2.0 + p.toeboard_t / 2.0), 0.0, tb_z),
                     (p.toeboard_t, p.deck_y + 2 * p.toeboard_t,
                      p.toeboard_h), DECK, tag="toeboard"))

    return e


# --------------------------------------------------------------------------
# Mass properties: sum of element inertias about the model origin
# --------------------------------------------------------------------------


# Effective densities [kg/m^3]. The elements are modelled as solids, so these
# are *apparent* densities: alu tubes are hollow (~2.0 mm wall -> roughly 35 %
# of the solid section), the deck is a plywood panel in a light alu frame.
DENSITY = {"alu": 950.0, "wood": 480.0, "red": 1200.0, "dark": 1150.0}


def inertial(elements: list[Element], total_mass: float):
    vols = np.array([el.volume() * DENSITY[el.material] for el in elements])
    masses = vols / vols.sum() * total_mass

    com = np.zeros(3)
    for m, el in zip(masses, elements):
        com += m * el.xyz
    com /= total_mass

    I = np.zeros((3, 3))
    for m, el in zip(masses, elements):
        if el.kind == "cylinder":
            r, L = el.dims
            ixx = iyy = m * (3 * r * r + L * L) / 12.0
            izz = m * r * r / 2.0
        else:
            sx, sy, sz = el.dims
            ixx = m * (sy * sy + sz * sz) / 12.0
            iyy = m * (sx * sx + sz * sz) / 12.0
            izz = m * (sx * sx + sy * sy) / 12.0
        Il = np.diag([ixx, iyy, izz])
        R = rot_matrix(el.rpy)
        Iw = R @ Il @ R.T
        d = el.xyz - com
        I += Iw + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I[np.abs(I) < 1e-9] = 0.0          # kill numerical dust on the off-diagonals
    com[np.abs(com) < 1e-9] = 0.0
    return com, I


# --------------------------------------------------------------------------
# SDF emission
# --------------------------------------------------------------------------

MATERIALS = {
    #          ambient              diffuse              specular       classic script
    "alu":  ((0.55, 0.57, 0.60), (0.72, 0.74, 0.78), (0.45, 0.45, 0.45), "Gazebo/Grey"),
    "wood": ((0.42, 0.34, 0.24), (0.66, 0.54, 0.38), (0.05, 0.05, 0.05), "Gazebo/Wood"),
    "red":  ((0.35, 0.05, 0.12), (0.72, 0.10, 0.25), (0.20, 0.20, 0.20), "Gazebo/Red"),
    "dark": ((0.06, 0.06, 0.07), (0.15, 0.15, 0.16), (0.10, 0.10, 0.10), "Gazebo/Black"),
}


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def geom_xml(el: Element, indent: str) -> str:
    if el.kind == "cylinder":
        r, L = el.dims
        g = (f"{indent}  <cylinder>\n"
             f"{indent}    <radius>{r:.6g}</radius>\n"
             f"{indent}    <length>{L:.6g}</length>\n"
             f"{indent}  </cylinder>\n")
    else:
        g = (f"{indent}  <box>\n"
             f"{indent}    <size>{fmt(el.dims)}</size>\n"
             f"{indent}  </box>\n")
    return f"{indent}<geometry>\n{g}{indent}</geometry>\n"


def material_xml(name: str, indent: str) -> str:
    amb, dif, spec, script = MATERIALS[name]
    return (f"{indent}<material>\n"
            f"{indent}  <script>\n"
            f"{indent}    <uri>file://media/materials/scripts/gazebo.material</uri>\n"
            f"{indent}    <name>{script}</name>\n"
            f"{indent}  </script>\n"
            f"{indent}  <ambient>{fmt(amb)} 1</ambient>\n"
            f"{indent}  <diffuse>{fmt(dif)} 1</diffuse>\n"
            f"{indent}  <specular>{fmt(spec)} 1</specular>\n"
            f"{indent}</material>\n")


def emit_sdf(p: Params, elements: list[Element], static: bool = True) -> str:
    com, I = inertial(elements, p.mass)

    out = []
    out.append('<?xml version="1.0" ?>\n')
    out.append('<sdf version="1.7">\n')
    out.append(f'  <model name="{p.name}">\n')
    out.append(f'    <static>{"true" if static else "false"}</static>\n')
    out.append('    <link name="base_link">\n')
    out.append('      <inertial>\n')
    out.append(f'        <pose>{fmt(com)} 0 0 0</pose>\n')
    out.append(f'        <mass>{p.mass:.4g}</mass>\n')
    out.append('        <inertia>\n')
    out.append(f'          <ixx>{I[0,0]:.6g}</ixx>\n')
    out.append(f'          <ixy>{I[0,1]:.6g}</ixy>\n')
    out.append(f'          <ixz>{I[0,2]:.6g}</ixz>\n')
    out.append(f'          <iyy>{I[1,1]:.6g}</iyy>\n')
    out.append(f'          <iyz>{I[1,2]:.6g}</iyz>\n')
    out.append(f'          <izz>{I[2,2]:.6g}</izz>\n')
    out.append('        </inertia>\n')
    out.append('      </inertial>\n')

    for i, el in enumerate(elements):
        n = f"{el.tag or el.kind}_{i:03d}"
        out.append(f'      <visual name="v_{n}">\n')
        out.append(f'        <pose>{fmt(el.xyz)} {fmt(el.rpy)}</pose>\n')
        out.append(geom_xml(el, "        "))
        out.append(material_xml(el.material, "        "))
        out.append('      </visual>\n')
        if el.collide:
            out.append(f'      <collision name="c_{n}">\n')
            out.append(f'        <pose>{fmt(el.xyz)} {fmt(el.rpy)}</pose>\n')
            out.append(geom_xml(el, "        "))
            out.append('        <surface>\n'
                       '          <friction><ode>\n'
                       '            <mu>0.8</mu><mu2>0.8</mu2>\n'
                       '          </ode></friction>\n'
                       '        </surface>\n')
            out.append('      </collision>\n')

    out.append('    </link>\n')
    out.append('  </model>\n')
    out.append('</sdf>\n')
    return "".join(out)


def emit_sdf_mesh(p: Params, elements: list[Element], static: bool = True) -> str:
    """Variant that uses the OBJ for visuals and coarse boxes for collision."""
    com, I = inertial(elements, p.mass)
    hx, hy = p.frame_span_x / 2.0, p.frame_width_y / 2.0
    z_frame = p.castor_d + 0.02

    coll = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            coll.append(((sx * hx, sy * hy, (z_frame + p.height) / 2.0),
                         (0.06, 0.06, p.height - z_frame)))
    coll.append(((0, 0, p.platform_z - p.deck_t / 2.0),
                 (p.deck_x, p.deck_y, p.deck_t)))
    zt = p.height - p.guard_top_off
    for sy in (-1, 1):
        coll.append(((0, sy * hy, zt), (p.frame_span_x, 0.04, 0.04)))
    coll.append(((0, 0, p.castor_d / 2.0), (p.base_x, p.base_y, p.castor_d)))

    out = ['<?xml version="1.0" ?>\n', '<sdf version="1.7">\n',
           f'  <model name="{p.name}">\n',
           f'    <static>{"true" if static else "false"}</static>\n',
           '    <link name="base_link">\n',
           '      <inertial>\n',
           f'        <pose>{fmt(com)} 0 0 0</pose>\n',
           f'        <mass>{p.mass:.4g}</mass>\n',
           '        <inertia>\n',
           f'          <ixx>{I[0,0]:.6g}</ixx><ixy>{I[0,1]:.6g}</ixy>'
           f'<ixz>{I[0,2]:.6g}</ixz>\n',
           f'          <iyy>{I[1,1]:.6g}</iyy><iyz>{I[1,2]:.6g}</iyz>'
           f'<izz>{I[2,2]:.6g}</izz>\n',
           '        </inertia>\n      </inertial>\n',
           '      <visual name="visual">\n',
           f'        <geometry><mesh><uri>model://{p.name}/meshes/'
           f'{p.name}.obj</uri></mesh></geometry>\n',
           '      </visual>\n']
    for i, (c, s) in enumerate(coll):
        out.append(f'      <collision name="c_{i:02d}">\n'
                   f'        <pose>{fmt(c)} 0 0 0</pose>\n'
                   f'        <geometry><box><size>{fmt(s)}</size></box></geometry>\n'
                   f'      </collision>\n')
    out += ['    </link>\n', '  </model>\n', '</sdf>\n']
    return "".join(out)


# --------------------------------------------------------------------------
# OBJ / MTL emission
# --------------------------------------------------------------------------


def tessellate(el: Element, segments: int = 12):
    """Return (vertices Nx3 in world frame, faces list of index triples)."""
    R = rot_matrix(el.rpy)
    if el.kind == "box":
        sx, sy, sz = (d / 2.0 for d in el.dims)
        v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz],
                      [-sx, sy, -sz], [-sx, -sy, sz], [sx, -sy, sz],
                      [sx, sy, sz], [-sx, sy, sz]])
        f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    else:
        r, L = el.dims
        h = L / 2.0
        ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        bot = np.stack([r * np.cos(ang), r * np.sin(ang),
                        np.full(segments, -h)], axis=1)
        top = np.stack([r * np.cos(ang), r * np.sin(ang),
                        np.full(segments, h)], axis=1)
        v = np.vstack([bot, top, [[0, 0, -h]], [[0, 0, h]]])
        cb, ct = 2 * segments, 2 * segments + 1
        f = []
        for i in range(segments):
            j = (i + 1) % segments
            f += [(i, j, segments + j), (i, segments + j, segments + i)]
            f += [(cb, j, i), (ct, segments + i, segments + j)]
    return (R @ v.T).T + el.xyz, f


def emit_obj(p: Params, elements: list[Element]):
    verts, lines = [], [f"mtllib {p.name}.mtl\n"]
    by_mat: dict[str, list] = {}
    for el in elements:
        v, f = tessellate(el)
        base = len(verts)
        verts.extend(v.tolist())
        by_mat.setdefault(el.material, []).extend(
            [(a + base + 1, b + base + 1, c + base + 1) for a, b, c in f])
    for v in verts:
        lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
    for mat, faces in by_mat.items():
        lines.append(f"g {mat}\nusemtl {mat}\n")
        for a, b, c in faces:
            lines.append(f"f {a} {b} {c}\n")

    mtl = []
    for name, (amb, dif, spec, _) in MATERIALS.items():
        mtl.append(f"newmtl {name}\n"
                   f"Ka {fmt(amb)}\nKd {fmt(dif)}\nKs {fmt(spec)}\n"
                   f"Ns 40\nd 1.0\nillum 2\n\n")
    return "".join(lines), "".join(mtl), len(verts)


MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>Krause CORDA 4.2 m mobile scaffold</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>generated by corda_gen.py</name>
    <email>-</email>
  </author>
  <description>
    Mobile assembly scaffold approximating KRAUSE CORDA Art.-No. 10091497:
    4.20 m scaffold height, 3.00 m platform height, 5.00 m working height,
    1.65 x 1.35 m stabiliser footprint, 1.50 x 0.56 m deck, 48 kg.
    Dimensionally faithful to the manufacturer datasheet; joint detail,
    profile cross-sections and hardware are approximated. Not a CAD model
    and not suitable for structural analysis.
  </description>
</model>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--dynamic", action="store_true",
                    help="emit <static>false</static>")
    args = ap.parse_args()

    p = Params()
    els = build(p)
    root = os.path.join(args.outdir, p.name)
    os.makedirs(os.path.join(root, "meshes"), exist_ok=True)

    with open(os.path.join(root, "model.sdf"), "w") as f:
        f.write(emit_sdf(p, els, static=not args.dynamic))
    with open(os.path.join(root, "model_mesh.sdf"), "w") as f:
        f.write(emit_sdf_mesh(p, els, static=not args.dynamic))
    with open(os.path.join(root, "model.config"), "w") as f:
        f.write(MODEL_CONFIG)

    obj, mtl, nv = emit_obj(p, els)
    with open(os.path.join(root, "meshes", f"{p.name}.obj"), "w") as f:
        f.write(obj)
    with open(os.path.join(root, "meshes", f"{p.name}.mtl"), "w") as f:
        f.write(mtl)

    com, I = inertial(els, p.mass)
    print(f"elements : {len(els)}")
    print(f"vertices : {nv}")
    print(f"CoM      : {com.round(4)}")
    print("inertia  :\n", I.round(4))
    return p, els


if __name__ == "__main__":
    main()

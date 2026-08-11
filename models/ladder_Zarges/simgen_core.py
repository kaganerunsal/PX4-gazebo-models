#!/usr/bin/env python3
"""
simgen_core.py -- shared machinery for building Gazebo props out of
primitive elements (tubes and boxes).

Extracted from corda_gen.py and generalised to multi-link models with
joints. Provides:

    Element / tube() / box()      primitive construction
    transform()                   rigid-body transform of an element list
    rot_matrix() / rpy_from_R()   SDF extrinsic-XYZ euler helpers
    inertial()                    integrated mass properties of an element list
    emit_sdf()                    multi-link SDF with primitive geometry
    emit_obj()                    triangulated OBJ + MTL

Conventions: SI units, SDF <pose> is "x y z roll pitch yaw" with extrinsic
XYZ euler angles, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------
# Materials. Each entry: ambient, diffuse, specular, Gazebo-Classic script.
# --------------------------------------------------------------------------

MATERIALS = {
    "alu":     ((0.55, 0.57, 0.60), (0.72, 0.74, 0.78), (0.45, 0.45, 0.45), "Gazebo/Grey"),
    "alu_dark":((0.40, 0.42, 0.45), (0.55, 0.57, 0.60), (0.35, 0.35, 0.35), "Gazebo/DarkGrey"),
    "wood":    ((0.42, 0.34, 0.24), (0.66, 0.54, 0.38), (0.05, 0.05, 0.05), "Gazebo/Wood"),
    "red":     ((0.35, 0.05, 0.12), (0.72, 0.10, 0.25), (0.20, 0.20, 0.20), "Gazebo/Red"),
    "dark":    ((0.06, 0.06, 0.07), (0.15, 0.15, 0.16), (0.10, 0.10, 0.10), "Gazebo/Black"),
    "blue":    ((0.05, 0.18, 0.38), (0.10, 0.35, 0.70), (0.15, 0.15, 0.15), "Gazebo/Blue"),
    "yellow":  ((0.45, 0.40, 0.05), (0.90, 0.80, 0.10), (0.15, 0.15, 0.15), "Gazebo/Yellow"),
}

# Apparent densities [kg/m^3]. Elements are modelled as solids, so hollow
# extrusions get a reduced value. Only ratios matter -- the totals are
# normalised to the datasheet mass.
DENSITY = {
    "alu": 950.0, "alu_dark": 950.0, "wood": 480.0,
    "red": 1200.0, "dark": 1150.0, "blue": 1200.0, "yellow": 1200.0,
}


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


@dataclass
class Element:
    kind: str                 # "cylinder" | "box"
    dims: tuple               # cylinder: (radius, length); box: (sx, sy, sz)
    xyz: np.ndarray
    rpy: np.ndarray
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


def rpy_from_R(R: np.ndarray) -> np.ndarray:
    """Inverse of rot_matrix for the non-degenerate case."""
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def rot_y(a: float) -> np.ndarray:
    return rot_matrix([0.0, a, 0.0])


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


def beam(p0, p1, cross, material, collide=True, tag="") -> Element:
    """Box beam spanning two points. `cross` = (width, depth) of the section,
    applied to the local X and Y axes; the long axis of the beam is local Z."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    dz = d / L
    pitch = math.acos(max(-1.0, min(1.0, dz[2])))
    yaw = math.atan2(dz[1], dz[0])
    return Element("box", (cross[0], cross[1], L), (p0 + p1) / 2.0,
                   np.array([0.0, pitch, yaw]), material, collide, tag)


def transform(elements: list[Element], R: np.ndarray, t) -> list[Element]:
    """Rigidly move a list of elements: p' = R p + t."""
    t = np.asarray(t, dtype=float)
    out = []
    for el in elements:
        Rw = R @ rot_matrix(el.rpy)
        out.append(replace(el, xyz=R @ el.xyz + t, rpy=rpy_from_R(Rw)))
    return out


# --------------------------------------------------------------------------
# Mass properties
# --------------------------------------------------------------------------


def inertial(elements: list[Element], total_mass: float):
    """Return (CoM, inertia tensor about the CoM) for the element list."""
    w = np.array([el.volume() * DENSITY[el.material] for el in elements])
    masses = w / w.sum() * total_mass

    com = np.zeros(3)
    for m, el in zip(masses, elements):
        com += m * el.xyz
    com /= total_mass

    I = np.zeros((3, 3))
    for m, el in zip(masses, elements):
        if el.kind == "cylinder":
            r, L = el.dims
            Il = np.diag([m * (3 * r * r + L * L) / 12.0,
                          m * (3 * r * r + L * L) / 12.0,
                          m * r * r / 2.0])
        else:
            sx, sy, sz = el.dims
            Il = np.diag([m * (sy * sy + sz * sz) / 12.0,
                          m * (sx * sx + sz * sz) / 12.0,
                          m * (sx * sx + sy * sy) / 12.0])
        R = rot_matrix(el.rpy)
        d = el.xyz - com
        I += R @ Il @ R.T + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    I[np.abs(I) < 1e-9] = 0.0
    com[np.abs(com) < 1e-9] = 0.0
    return com, I


def mass_split(links: dict, total_mass: float) -> dict:
    """Distribute a total mass over several links by weighted volume."""
    w = {k: sum(e.volume() * DENSITY[e.material] for e in els)
         for k, els in links.items()}
    s = sum(w.values())
    return {k: v / s * total_mass for k, v in w.items()}


# --------------------------------------------------------------------------
# SDF emission
# --------------------------------------------------------------------------


def fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _geom(el: Element, ind: str) -> str:
    if el.kind == "cylinder":
        r, L = el.dims
        g = (f"{ind}  <cylinder>\n{ind}    <radius>{r:.6g}</radius>\n"
             f"{ind}    <length>{L:.6g}</length>\n{ind}  </cylinder>\n")
    else:
        g = f"{ind}  <box>\n{ind}    <size>{fmt(el.dims)}</size>\n{ind}  </box>\n"
    return f"{ind}<geometry>\n{g}{ind}</geometry>\n"


def _material(name: str, ind: str) -> str:
    amb, dif, spec, script = MATERIALS[name]
    return (f"{ind}<material>\n"
            f"{ind}  <script>\n"
            f"{ind}    <uri>file://media/materials/scripts/gazebo.material</uri>\n"
            f"{ind}    <name>{script}</name>\n"
            f"{ind}  </script>\n"
            f"{ind}  <ambient>{fmt(amb)} 1</ambient>\n"
            f"{ind}  <diffuse>{fmt(dif)} 1</diffuse>\n"
            f"{ind}  <specular>{fmt(spec)} 1</specular>\n"
            f"{ind}</material>\n")


def emit_sdf(model_name: str, links: dict, masses: dict,
             joints: list = (), static: bool = True,
             mu: float = 0.7) -> str:
    """
    links  : {link_name: [Element, ...]}   elements in MODEL frame
    masses : {link_name: kg}
    joints : [dict(name=, type=, parent=, child=, pose=(x,y,z,r,p,y),
                   axis=(x,y,z), limit=(lo, hi))]
    """
    out = ['<?xml version="1.0" ?>\n', '<sdf version="1.7">\n',
           f'  <model name="{model_name}">\n',
           f'    <static>{"true" if static else "false"}</static>\n']

    for link_name, els in links.items():
        com, I = inertial(els, masses[link_name])
        out += [f'    <link name="{link_name}">\n',
                '      <inertial>\n',
                f'        <pose>{fmt(com)} 0 0 0</pose>\n',
                f'        <mass>{masses[link_name]:.4g}</mass>\n',
                '        <inertia>\n',
                f'          <ixx>{I[0,0]:.6g}</ixx><ixy>{I[0,1]:.6g}</ixy>'
                f'<ixz>{I[0,2]:.6g}</ixz>\n',
                f'          <iyy>{I[1,1]:.6g}</iyy><iyz>{I[1,2]:.6g}</iyz>'
                f'<izz>{I[2,2]:.6g}</izz>\n',
                '        </inertia>\n      </inertial>\n']
        for i, el in enumerate(els):
            n = f"{el.tag or el.kind}_{i:03d}"
            out += [f'      <visual name="v_{n}">\n',
                    f'        <pose>{fmt(el.xyz)} {fmt(el.rpy)}</pose>\n',
                    _geom(el, "        "),
                    _material(el.material, "        "),
                    '      </visual>\n']
            if el.collide:
                out += [f'      <collision name="c_{n}">\n',
                        f'        <pose>{fmt(el.xyz)} {fmt(el.rpy)}</pose>\n',
                        _geom(el, "        "),
                        '        <surface><friction><ode>\n'
                        f'          <mu>{mu}</mu><mu2>{mu}</mu2>\n'
                        '        </ode></friction></surface>\n',
                        '      </collision>\n']
        out.append('    </link>\n')

    for j in joints:
        out += [f'    <joint name="{j["name"]}" type="{j["type"]}">\n',
                f'      <parent>{j["parent"]}</parent>\n',
                f'      <child>{j["child"]}</child>\n',
                f'      <pose>{fmt(j["pose"])}</pose>\n',
                '      <axis>\n',
                f'        <xyz>{fmt(j["axis"])}</xyz>\n']
        if "limit" in j:
            lo, hi = j["limit"]
            out.append(f'        <limit><lower>{lo:.6g}</lower>'
                       f'<upper>{hi:.6g}</upper></limit>\n')
        out += ['        <dynamics><damping>0.5</damping></dynamics>\n',
                '      </axis>\n    </joint>\n']

    out += ['  </model>\n', '</sdf>\n']
    return "".join(out)


# --------------------------------------------------------------------------
# OBJ / MTL emission
# --------------------------------------------------------------------------


def tessellate(el: Element, segments: int = 12):
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
        a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        bot = np.stack([r * np.cos(a), r * np.sin(a), np.full(segments, -h)], 1)
        top = np.stack([r * np.cos(a), r * np.sin(a), np.full(segments, h)], 1)
        v = np.vstack([bot, top, [[0, 0, -h]], [[0, 0, h]]])
        cb, ct = 2 * segments, 2 * segments + 1
        f = []
        for i in range(segments):
            j = (i + 1) % segments
            f += [(i, j, segments + j), (i, segments + j, segments + i),
                  (cb, j, i), (ct, segments + i, segments + j)]
    return (R @ v.T).T + el.xyz, f


def emit_obj(mtl_name: str, elements: list[Element]):
    verts, lines = [], [f"mtllib {mtl_name}.mtl\n"]
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

    mtl = "".join(
        f"newmtl {n}\nKa {fmt(a)}\nKd {fmt(d)}\nKs {fmt(s)}\nNs 40\nd 1.0\n"
        f"illum 2\n\n" for n, (a, d, s, _) in MATERIALS.items())
    return "".join(lines), mtl, len(verts)

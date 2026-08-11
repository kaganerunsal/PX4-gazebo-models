#!/usr/bin/env python3
"""
Parametric generator for a ZARGES Z600 two-part combination ladder
(Vielzweckleiter / Schiebeleiter) as a Gazebo / gz-sim model.

Identified from the rating plate:
    Art.-No.   40284
    EAN        4003866402847
    Series     ZARGES Z600, Zarges GmbH, D-82362 Weilheim
    Built      12/2015
    Mass       15.00 kg
    A-frame    ca. 3.60 m   (pictogram, ladder length as Stehleiter)
    Leaning    ca. 6.10 m   (pictogram, ladder length extended)
    Standard   EN 131, max. 150 kg

The 3.60 / 6.10 pair fixes the configuration: ZARGES lists 12 rungs =
3.61 m ladder length for the Z600 rung series, so this is 2 x 12 rungs,
two 3.61 m sections extended with a 1.12 m (4-rung) overlap -> 6.10 m.

Four poses are emitted from one element set:
    a_frame      both sections hinged at the apex, standing (default)
    leaning      extended to 6.10 m, leaning at 70 deg
    folded       closed, lying flat on the ground (transport / hazard prop)
    articulated  two links + revolute apex joint, spread angle drivable

Usage:  python3 zarges_gen.py [--outdir DIR] [--dynamic]
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from simgen_core import (beam, box, emit_obj, emit_sdf, inertial, mass_split,
                         rot_y, transform)

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


class Params:
    name = "zarges_z600_40284"

    section_len = 3.61        # one section, ZARGES table for 12 rungs
    extended_len = 6.10       # pictogram
    rungs = 12
    rung_pitch = 0.280        # EN 131
    first_rung = 0.245        # from the foot end of the stile

    outer_width_lo = 0.420    # Z600 rung ladder outer width
    outer_width_up = 0.360    # upper section nests inside the lower
    stile_w = 0.025           # stile wall thickness (width direction)
    stile_d_lo = 0.075        # stile profile depth
    stile_d_up = 0.065

    rung_depth = 0.030        # ZARGES: Sprossentiefe 30 mm
    rung_thick = 0.025

    foot_h = 0.085            # plastic shoe
    foot_d = 0.090
    foot_w = 0.035

    spread_half = math.radians(14.0)   # A-frame half opening angle
    lean_angle = math.radians(70.0)    # leaning ladder, EN 131 recommends 65-75
    strap_z_frac = 0.62                # restraint straps up the A-frame legs

    mass = 15.0               # kg, rating plate

    traverse = False          # stabiliser bar; not fitted on a 12/2015 unit
    traverse_w = 1.000

    label = True              # the blue/yellow rating plate on the lower stile
    label_z = 2.30            # up the lower section, matching the photo


# --------------------------------------------------------------------------
# One ladder section, built in its own frame:
#   +Z = up the ladder,  +-Y = width (rungs run along Y),  +X = profile depth
# --------------------------------------------------------------------------


def section(p: Params, upper: bool) -> list[Element]:  # noqa: F821
    w = p.outer_width_up if upper else p.outer_width_lo
    d = p.stile_d_up if upper else p.stile_d_lo
    L = p.section_len
    e = []

    y_stile = w / 2.0 - p.stile_w / 2.0
    for sy in (-1, 1):
        e.append(beam((0, sy * y_stile, 0.0), (0, sy * y_stile, L),
                      (d, p.stile_w), "alu", True, "stile"))

    clear = w - 2 * p.stile_w
    for i in range(p.rungs):
        z = p.first_rung + i * p.rung_pitch
        if z > L - 0.10:
            break
        e.append(box((0.0, 0.0, z), (p.rung_depth, clear, p.rung_thick),
                     "alu", tag="rung"))

    # plastic shoes at the foot end
    for sy in (-1, 1):
        e.append(box((0.0, sy * y_stile, -p.foot_h / 2.0),
                     (p.foot_d, p.foot_w, p.foot_h), "dark", tag="foot"))

    if p.traverse and not upper:
        e.append(box((0.0, 0.0, 0.055),
                     (0.030, p.traverse_w, 0.050), "alu", tag="traverse"))
        for sy in (-1, 1):
            e.append(box((0.0, sy * p.traverse_w / 2.0, 0.0),
                         (p.foot_d, p.foot_w, p.foot_h), "dark", tag="foot"))

    if p.label and not upper:
        e.append(box((d / 2.0 + 0.001, y_stile, p.label_z),
                     (0.002, p.stile_w * 0.8, 0.55), "blue",
                     collide=False, tag="label"))
        e.append(box((d / 2.0 + 0.002, y_stile, p.label_z - 0.10),
                     (0.002, p.stile_w * 0.6, 0.30), "yellow",
                     collide=False, tag="label"))
    return e


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------


def pose_a_frame(p: Params):
    """Both sections hinged at the apex, feet on the ground, opening 2*spread."""
    th = p.spread_half
    L = p.section_len
    dx = L * math.sin(th)
    lo = transform(section(p, False), rot_y(th), (-dx, 0.0, p.foot_h))
    up = transform(section(p, True), rot_y(-th), (dx, 0.0, p.foot_h))

    apex = np.array([0.0, 0.0, p.foot_h + L * math.cos(th)])
    extra = []
    # apex hinge blocks
    for sy in (-1, 1):
        extra.append(box((0.0, sy * 0.17, apex[2] - 0.04),
                         (0.10, 0.05, 0.06), "alu_dark", tag="hinge"))
    # restraint straps
    zs = p.strap_z_frac * L
    for sy in (-1, 1):
        a = np.array([-dx + zs * math.sin(th), sy * 0.19,
                      p.foot_h + zs * math.cos(th)])
        b = np.array([dx - zs * math.sin(th), sy * 0.19,
                      p.foot_h + zs * math.cos(th)])
        extra.append(beam(a, b, (0.030, 0.004), "dark", False, "strap"))
    return lo, up, extra, apex


def pose_leaning(p: Params):
    """Extended to 6.10 m and leaning at lean_angle from the horizontal."""
    beta = math.pi / 2.0 - p.lean_angle          # tilt from vertical
    R = rot_y(beta)
    overlap_offset = p.extended_len - p.section_len   # = 2.49 m
    depth = (p.stile_d_lo + p.stile_d_up) / 2.0

    lo = transform(section(p, False), R, (0.0, 0.0, p.foot_h))
    up = transform(section(p, True), R,
                   R @ np.array([-depth, 0.0, overlap_offset])
                   + np.array([0.0, 0.0, p.foot_h]))
    # guide brackets clamping the upper section to the lower
    extra = []
    for z in (overlap_offset + 0.05, p.section_len - 0.10):
        for sy in (-1, 1):
            c = R @ np.array([-depth / 2.0, sy * 0.18, z]) \
                + np.array([0.0, 0.0, p.foot_h])
            extra.append(box(c, (depth + 0.03, 0.05, 0.04), "alu_dark",
                             rpy=(0.0, beta, 0.0), tag="bracket"))
    top = R @ np.array([0.0, 0.0, p.extended_len]) + np.array([0, 0, p.foot_h])
    return lo, up, extra, top


def pose_folded(p: Params):
    """Closed and lying flat on the ground: transport state / trip hazard."""
    R = rot_y(math.pi / 2.0)                     # +Z -> +X
    z_lo = p.stile_d_lo / 2.0
    lo = transform(section(p, False), R, (0.0, 0.0, z_lo))
    up = transform(section(p, True), R,
                   (0.0, 0.0, z_lo + (p.stile_d_lo + p.stile_d_up) / 2.0))
    return lo, up, [], np.array([p.section_len, 0.0, 0.0])


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

POSE_TEXT = {
    "a_frame": "Set up as an A-frame, apex at 3.59 m, feet 1.75 m apart.",
    "leaning": "Extended to 6.10 m and leaning at 70 deg; the top reaches "
               "z = 5.82 m at x = 2.09 m from the feet.",
    "folded":  "Closed and lying flat on the ground (transport state, or a "
               "trip-hazard prop for detection datasets).",
}

MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author><name>generated by zarges_gen.py</name><email>-</email></author>
  <description>
    Two-part 2x12-rung aluminium combination ladder approximating ZARGES
    Z600 Art.-No. 40284: 3.61 m per section, 6.10 m extended, 15.0 kg,
    EN 131, max. 150 kg. {pose}
    Dimensionally faithful to the rating plate and the ZARGES Z600 dimension
    table; extrusion profiles, hinges, guide brackets and hardware are
    approximated. Simulation prop only -- not a CAD model and not valid for
    any structural calculation.
  </description>
</model>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()

    p = Params()
    static = not args.dynamic

    poses = {"a_frame": pose_a_frame(p),
             "leaning": pose_leaning(p),
             "folded": pose_folded(p)}

    # Each pose is its own model package, because <include> cannot select an
    # alternate .sdf inside a package -- it always loads what model.config
    # points at.
    dirs = {"a_frame": p.name,
            "leaning": f"{p.name}_leaning",
            "folded": f"{p.name}_folded"}

    for pose_name, (lo, up, extra, ref) in poses.items():
        model_name = dirs[pose_name]
        root = os.path.join(args.outdir, model_name)
        os.makedirs(os.path.join(root, "meshes"), exist_ok=True)

        allels = lo + up + extra          # one rigid body for the static poses
        with open(os.path.join(root, "model.sdf"), "w") as f:
            f.write(emit_sdf(model_name, {"base_link": allels},
                             {"base_link": p.mass}, static=static, mu=0.65))

        obj, mtl, nv = emit_obj(model_name, allels)
        base = os.path.join(root, "meshes", model_name)
        with open(base + ".obj", "w") as f:
            f.write(obj)
        with open(base + ".mtl", "w") as f:
            f.write(mtl)

        with open(os.path.join(root, "model.config"), "w") as f:
            f.write(MODEL_CONFIG.format(name=model_name,
                                        pose=POSE_TEXT[pose_name]))

        com, _ = inertial(allels, p.mass)
        print(f"{pose_name:11s} dir={model_name:28s} elements={len(allels):3d} "
              f"verts={nv:5d} CoM={np.round(com, 3)} ref={np.round(ref, 3)}")

    # articulated variant lives alongside the A-frame package
    lo, up, extra, apex = poses["a_frame"]
    links = {"section_lower": lo + extra, "section_upper": up}
    masses = mass_split(links, p.mass)
    joints = [dict(name="apex_hinge", type="revolute",
                   parent="section_lower", child="section_upper",
                   pose=(*apex, 0, 0, 0), axis=(0, 1, 0),
                   limit=(-0.02, math.radians(35)))]
    with open(os.path.join(args.outdir, p.name,
                           "model_articulated.sdf"), "w") as f:
        f.write(emit_sdf(f"{p.name}_articulated", links, masses,
                         joints=joints, static=False, mu=0.65))
    print("articulated  masses=" +
          ", ".join(f"{k}={v:.2f} kg" for k, v in masses.items()))
    return p, poses


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
obj_to_dae.py -- convert the OBJ+MTL written by corda_gen.py / zarges_gen.py
into a COLLADA (.dae) file that carries its materials into BOTH Gazebo
Classic and gz-sim.

Why this exists: Gazebo Classic's mesh loader does not read .mtl files. It
will load an OBJ's geometry and render it default white. COLLADA is the only
mesh format whose materials both Classic (Ogre1) and gz-sim (Ogre2) read
reliably, so .dae is the safe choice for any mesh-based visual.

This also computes per-face normals, which the OBJ export omits -- without
normals the shading is flat and washed out even where the colours do load.

Usage:
    python3 obj_to_dae.py path/to/mesh.obj [-o path/to/mesh.dae]
    python3 obj_to_dae.py 'models/**/meshes/*.obj'      # glob, writes in place
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from xml.sax.saxutils import escape

import numpy as np

NS = "http://www.collada.org/2005/11/COLLADASchema"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_mtl(path: str) -> dict:
    mats, cur = {}, None
    if not os.path.exists(path):
        return mats
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        k, v = parts[0], parts[1:]
        if k == "newmtl":
            cur = v[0]
            mats[cur] = {"Ka": (0.2, 0.2, 0.2), "Kd": (0.8, 0.8, 0.8),
                         "Ks": (0.0, 0.0, 0.0), "Ns": 30.0, "d": 1.0}
        elif cur and k in ("Ka", "Kd", "Ks"):
            mats[cur][k] = tuple(float(x) for x in v[:3])
        elif cur and k in ("Ns", "d"):
            mats[cur][k] = float(v[0])
    return mats


def parse_obj(path: str):
    """Return (verts Nx3, {material: [(i,j,k), ...]}, mtl_filename)."""
    verts, faces, mtllib = [], {}, None
    cur = "default"
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        k, v = parts[0], parts[1:]
        if k == "v":
            verts.append([float(x) for x in v[:3]])
        elif k == "mtllib":
            mtllib = v[0]
        elif k == "usemtl":
            cur = v[0]
        elif k == "f":
            # only plain "f a b c" and "a/b/c" forms are produced by our export
            idx = [int(p.split("/")[0]) - 1 for p in v]
            for i in range(1, len(idx) - 1):        # fan-triangulate
                faces.setdefault(cur, []).append((idx[0], idx[i], idx[i + 1]))
    return np.array(verts), faces, mtllib


# --------------------------------------------------------------------------
# COLLADA emission
# --------------------------------------------------------------------------


def _src(sid: str, data: np.ndarray, params=("X", "Y", "Z")) -> str:
    flat = data.reshape(-1)
    arr = " ".join(f"{x:.6f}" for x in flat)
    p = "".join(f'<param name="{n}" type="float"/>' for n in params)
    return (f'      <source id="{sid}">\n'
            f'        <float_array id="{sid}-array" count="{flat.size}">'
            f'{arr}</float_array>\n'
            f'        <technique_common>\n'
            f'          <accessor source="#{sid}-array" '
            f'count="{data.shape[0]}" stride="{data.shape[1]}">{p}</accessor>\n'
            f'        </technique_common>\n      </source>\n')


def _color(c, alpha=1.0) -> str:
    return f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {alpha:.4f}"


def build_dae(verts: np.ndarray, faces: dict, mats: dict) -> str:
    used = [m for m in faces if faces[m]]

    effects, materials, geoms, nodes = [], [], [], []

    for m in used:
        p = mats.get(m, {"Ka": (0.2,) * 3, "Kd": (0.8,) * 3,
                         "Ks": (0.0,) * 3, "Ns": 30.0, "d": 1.0})
        sm = escape(m)
        effects.append(
            f'    <effect id="{sm}-fx">\n'
            f'      <profile_COMMON>\n        <technique sid="common">\n'
            f'          <phong>\n'
            f'            <emission><color>0 0 0 1</color></emission>\n'
            f'            <ambient><color>{_color(p["Ka"])}</color></ambient>\n'
            f'            <diffuse><color>{_color(p["Kd"], p["d"])}</color>'
            f'</diffuse>\n'
            f'            <specular><color>{_color(p["Ks"])}</color>'
            f'</specular>\n'
            f'            <shininess><float>{p["Ns"]:.1f}</float></shininess>\n'
            f'            <index_of_refraction><float>1.0</float>'
            f'</index_of_refraction>\n'
            f'          </phong>\n        </technique>\n'
            f'      </profile_COMMON>\n    </effect>\n')
        materials.append(f'    <material id="{sm}-mat" name="{sm}">'
                         f'<instance_effect url="#{sm}-fx"/></material>\n')

        tris = np.array(faces[m], dtype=int)
        # compact the vertex list per material group
        uniq, inv = np.unique(tris.reshape(-1), return_inverse=True)
        pos = verts[uniq]
        idx = inv.reshape(-1, 3)

        # per-face normals
        a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        n = n / ln

        p_list = []
        for f_i, (i0, i1, i2) in enumerate(idx):
            p_list += [i0, f_i, i1, f_i, i2, f_i]
        p_txt = " ".join(str(x) for x in p_list)

        geoms.append(
            f'    <geometry id="{sm}-geo" name="{sm}">\n      <mesh>\n'
            + _src(f"{sm}-pos", pos)
            + _src(f"{sm}-nrm", n)
            + f'      <vertices id="{sm}-vtx">\n'
              f'        <input semantic="POSITION" source="#{sm}-pos"/>\n'
              f'      </vertices>\n'
              f'      <triangles material="{sm}-sym" count="{len(idx)}">\n'
              f'        <input semantic="VERTEX" source="#{sm}-vtx" '
              f'offset="0"/>\n'
              f'        <input semantic="NORMAL" source="#{sm}-nrm" '
              f'offset="1"/>\n'
              f'        <p>{p_txt}</p>\n      </triangles>\n'
              f'      </mesh>\n    </geometry>\n')

        nodes.append(
            f'        <node id="{sm}-node" name="{sm}" type="NODE">\n'
            f'          <instance_geometry url="#{sm}-geo">\n'
            f'            <bind_material><technique_common>\n'
            f'              <instance_material symbol="{sm}-sym" '
            f'target="#{sm}-mat"/>\n'
            f'            </technique_common></bind_material>\n'
            f'          </instance_geometry>\n        </node>\n')

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<COLLADA xmlns="{NS}" version="1.4.1">\n'
        '  <asset>\n'
        '    <contributor><authoring_tool>obj_to_dae.py</authoring_tool>'
        '</contributor>\n'
        '    <unit name="meter" meter="1"/>\n'
        '    <up_axis>Z_UP</up_axis>\n'
        '  </asset>\n'
        '  <library_effects>\n' + "".join(effects) + '  </library_effects>\n'
        '  <library_materials>\n' + "".join(materials) +
        '  </library_materials>\n'
        '  <library_geometries>\n' + "".join(geoms) +
        '  </library_geometries>\n'
        '  <library_visual_scenes>\n'
        '    <visual_scene id="Scene" name="Scene">\n' + "".join(nodes) +
        '    </visual_scene>\n  </library_visual_scenes>\n'
        '  <scene><instance_visual_scene url="#Scene"/></scene>\n'
        '</COLLADA>\n')


def convert(obj_path: str, out_path: str | None = None) -> str:
    verts, faces, mtllib = parse_obj(obj_path)
    mtl_path = os.path.join(os.path.dirname(obj_path),
                            mtllib or os.path.splitext(
                                os.path.basename(obj_path))[0] + ".mtl")
    mats = parse_mtl(mtl_path)
    if not mats:
        print(f"  warning: no MTL found next to {obj_path}", file=sys.stderr)
    dae = build_dae(verts, faces, mats)
    out_path = out_path or os.path.splitext(obj_path)[0] + ".dae"
    with open(out_path, "w") as f:
        f.write(dae)
    ntri = sum(len(v) for v in faces.values())
    print(f"{obj_path} -> {out_path}  "
          f"({len(verts)} verts, {ntri} tris, {len(faces)} materials)")
    return out_path


def convert_split(obj_path: str, model_name: str) -> str:
    """
    Write ONE .dae per material group, plus the matching <visual> blocks with
    the colours declared in SDF rather than carried by the mesh.

    Use this when a mesh renders untextured/white: it removes any dependency
    on the simulator picking up mesh-embedded materials. Colours then come
    from the SDF <material>, which every Gazebo version honours, and each
    group is still a single draw call.
    """
    verts, faces, mtllib = parse_obj(obj_path)
    mtl_path = os.path.join(os.path.dirname(obj_path),
                            mtllib or os.path.splitext(
                                os.path.basename(obj_path))[0] + ".mtl")
    mats = parse_mtl(mtl_path)
    stem = os.path.splitext(obj_path)[0]
    meshdir = os.path.basename(os.path.dirname(obj_path))

    blocks = []
    for m, tris in faces.items():
        if not tris:
            continue
        out = f"{stem}_{m}.dae"
        with open(out, "w") as f:
            f.write(build_dae(verts, {m: tris}, mats))
        p = mats.get(m, {"Ka": (0.2,) * 3, "Kd": (0.8,) * 3,
                         "Ks": (0.0,) * 3, "d": 1.0})
        uri = (f"model://{model_name}/{meshdir}/"
               f"{os.path.basename(out)}")
        blocks.append(
            f'      <visual name="visual_{m}">\n'
            f'        <geometry><mesh><uri>{uri}</uri></mesh></geometry>\n'
            f'        <material>\n'
            f'          <ambient>{_color(p["Ka"])}</ambient>\n'
            f'          <diffuse>{_color(p["Kd"], p["d"])}</diffuse>\n'
            f'          <specular>{_color(p["Ks"])}</specular>\n'
            f'        </material>\n'
            f'      </visual>\n')
        print(f"  {out}  ({len(tris)} tris)")

    snippet = "".join(blocks)
    snip_path = f"{stem}_visuals.xml"
    with open(snip_path, "w") as f:
        f.write(snippet)
    print(f"  -> {snip_path}  ({len(blocks)} visuals)")
    return snippet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--split", metavar="MODEL_NAME", default=None,
                    help="write one .dae per material plus SDF <visual> "
                         "blocks that declare the colours in SDF")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat, recursive=True)) or [pat])
    if args.output and len(paths) > 1:
        sys.exit("-o only makes sense with a single input")
    for p in paths:
        if args.split:
            print(p)
            convert_split(p, args.split)
        else:
            convert(p, args.output)


if __name__ == "__main__":
    main()

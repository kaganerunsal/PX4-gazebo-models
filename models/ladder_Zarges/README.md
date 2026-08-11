# ZARGES Z600 combination ladder, Art. 40284 — Gazebo models

A simulation asset approximating the **ZARGES Z600 two-part combination
ladder, Art.-No. 40284**, identified from its rating plate. Dimensions come
from the plate and from the ZARGES Z600 rung-ladder dimension table;
everything below that level (extrusion cross-sections, hinge and guide-bracket
hardware, bördelung of the rung/stile joint) is approximated. Simulation prop
only — not a CAD model, not valid for any structural or load calculation.

## Identification and dimensions

| Quantity | Rating plate | Model |
|---|---|---|
| Article / EAN | 40284 / 4003866402847 | — |
| Series | ZARGES Z600 | — |
| Built | 12/2015 | — |
| Mass | 15.00 kg | 15.00 kg |
| Length as A-frame | ca. 3.60 m | 3.61 m per section |
| Length extended | ca. 6.10 m | 6.10 m |
| Standard / load | EN 131, max. 150 kg | not enforced |

The 3.60 / 6.10 pair fixes the rung count. ZARGES lists **12 rungs = 3.61 m**
ladder length for the Z600 rung series, and two 3.61 m sections extended with
a 1.12 m overlap (4 rungs, comfortably above the EN 131 minimum) give exactly
6.10 m. So: **2 × 12 rungs**, 280 mm rung pitch, 30 mm rung depth, 420 mm
outer width on the lower section, 360 mm on the upper.

Derived geometry, per configuration:

| Pose | Standing height | Footprint | Notes |
|---|---|---|---|
| `a_frame` | apex 3.59 m | feet 1.75 m apart | ±14° half-opening |
| `leaning` | head 5.82 m at x = 2.09 m | — | 70° lean (EN 131 says 65–75°) |
| `folded` | 0.15 m | 3.61 × 0.42 m | lying flat |

## Packages

```
zarges_z600_40284/            A-frame  ← default
├── model.sdf                 71 primitive visuals + collisions, one rigid body
├── model_articulated.sdf     two links + revolute apex hinge, dynamic
├── model.config
├── meshes/*.obj + .mtl
├── ladder_test.sdf           world with all three poses + a wall
├── README.md
├── zarges_gen.py             the generator
└── simgen_core.py            shared primitive/SDF/OBJ machinery

zarges_z600_40284_leaning/    extended, 70° against a wall
zarges_z600_40284_folded/     closed, lying on the ground
```

Three separate packages rather than one with three SDF files, because
`<include>` always loads whatever `model.config` points at — it cannot select
an alternate `.sdf` inside a package.

As with the Corda scaffold, geometry is primitives-only (no mesh dependency)
and materials carry both a Classic `<script>` name and modern
`<ambient>/<diffuse>/<specular>`, so it loads and looks right in Gazebo
Classic 11 and in gz-sim Garden/Harmonic/Ionic alike. The OBJ is there if you
want a single draw call or a Blender round-trip.

## Install

```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/models   # gz-sim
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models         # Classic
gz sim -r /path/to/models/zarges_z600_40284/ladder_test.sdf
```

## Articulated variant

`model_articulated.sdf` splits the ladder into `section_lower` (8.21 kg) and
`section_upper` (6.79 kg) joined by a revolute `apex_hinge` about the Y axis,
limited to −0.02 … +0.61 rad. Drive it to animate opening and closing, or let
it fall. Mass properties are integrated per link (per-element tensor, rotated,
parallel-axis summed), not bounding-box guesses.

The static poses are single rigid bodies with the full 15 kg and
CoM at (−0.03, 0, 1.80) for the A-frame.

Note the real ladder is a *sliding* pair, not a pure hinge: making the A-frame
and extending it are different kinematics. The revolute joint models the
A-frame spread only. If you need the push-up motion, add a prismatic joint
along the section axis instead — `pose_leaning()` shows the offset geometry.

## Retargeting

Everything is in the `Params` class:

```python
rungs = 10; section_len = 3.05     # Z600 table: 10 rungs = 3.05 m
lean_angle = math.radians(65)      # shallower, closer to the EN 131 limit
traverse = True                    # add the stabiliser bar + wider feet
```

`traverse` defaults to **off**: the mandatory stabiliser bar for leaning
ladders over 3 m came in with the 2018 EN 131-2 revision, and this unit is
dated 12/2015, so a bar is unlikely to be fitted. Turn it on if the physical
ladder you're matching has one.

## Known simplifications

- Stiles are plain box beams; the real Z600 profile is a C-section with
  punched holes (visible in your photo) and end reinforcements.
- Rungs are rectangular; the real ones are flanged into the stile.
- The rating plate is two flat colour patches (blue over yellow) on the lower
  stile at z ≈ 2.30 m, not a texture. If you want it legible for a
  vision pipeline, UV-map a photo of the plate onto that face in Blender —
  the OBJ preserves the `blue` / `yellow` material groups so it's easy to
  select.
- No hinge detail, no locking hooks, no rope.
- The `folded` pose rests the sections on their profile faces; a real closed
  ladder on the ground usually lies slightly twisted.

## Meshes: use the .dae, not the .obj

Gazebo Classic's mesh loader ignores `.mtl` files and renders OBJ geometry
default white. `obj_to_dae.py` (included) converts the exported OBJ into
COLLADA with per-material phong effects and computed face normals, which both
Classic and gz-sim read correctly:

```bash
python3 obj_to_dae.py '*/meshes/*.obj'
```

The `model.sdf` files here are primitives-only, so they need no mesh or
material files at all — that is the most robust path. Only reach for the mesh
if you need the single-draw-call version.

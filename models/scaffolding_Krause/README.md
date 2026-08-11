# Krause CORDA 4.2 m mobile scaffold — Gazebo model

A simulation asset approximating **KRAUSE CORDA MontageGerüst, Art.-No. 10091497**.
Dimensions come from the manufacturer datasheet; everything below that level of
detail (joint hardware, extrusion profiles, brand decals) is approximated. This
is a *simulation prop*, not a CAD model, and it is not valid for any structural
or load calculation.

## Datasheet vs. model

| Quantity | Datasheet | Model |
|---|---|---|
| Working height | ca. 5.00 m | — (reference only) |
| Platform height | ca. 3.00 m | 3.000 m (deck top surface) |
| Scaffold height | ca. 4.20 m | 4.200 m (top of guardrail) |
| Base footprint | ca. 1.65 × 1.35 m | 1.65 × 1.35 m (castor centres) |
| Platform | ca. 1.50 × 0.56 m | 1.50 × 0.56 m |
| Mass | 48.0 kg | 48.0 kg |
| Deck load | 150 kg/m² | not enforced (see below) |

Frame geometry: two ladder frames 1.50 m apart (X), 0.60 m wide (Y), rung pitch
0.28 m, X-bracing on both long faces, guardrail + mid-rail + toeboards around
the deck, four splayed stabiliser arms with Ø125 mm castors.

Origin is at **ground level, centred on the footprint**, +X along the long axis.

## Files

```
krause_corda_4200/
├── model.sdf            primitive geometry (71 visuals + 71 collisions)  ← default
├── model_mesh.sdf       mesh visual + 9 coarse box collisions            ← alternative
├── model.config
├── meshes/
│   ├── krause_corda_4200.obj   1756 verts, triangulated, material groups
│   └── krause_corda_4200.mtl
├── scaffold_test.sdf    minimal world with two instances
└── corda_gen.py         the generator — edit Params, re-run
```

`model.sdf` is primitives-only, so it has **no mesh dependency** and loads
identically in Gazebo Classic 11 and gz-sim (Garden / Harmonic / Ionic).
Materials carry both a Classic `<script>` name and modern
`<ambient>/<diffuse>/<specular>` values, so colours survive either renderer.

Swap to `model_mesh.sdf` (rename it over `model.sdf`, or point `model.config`
at it) if you would rather have one draw call and cheap box collisions — a
reasonable trade if you are spawning many instances or doing heavy lidar
raycasting.

## Install

```bash
# gz-sim
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/models
gz sim -r /path/to/models/krause_corda_4200/scaffold_test.sdf

# Gazebo Classic
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models
```

Then `<include><uri>model://krause_corda_4200</uri></include>` in any world.
For PX4 SITL, drop the folder into `PX4-Autopilot/Tools/simulation/gz/models/`.

## Static vs. dynamic

The model ships `<static>true</static>`. Keep it that way unless you actually
need the tower to fall over: a 4.2 m tower on four castors with its CoM at
2.36 m is a stiff, tippy body, and at the default 1 kHz step DART will jitter
it unless you also model castor joints and increase solver iterations.

Mass properties are integrated over all 71 elements (per-element tensor,
rotated, parallel-axis summed) rather than approximated by a bounding box —
tube walls are handled via apparent densities in `DENSITY`:

```
mass 48.0 kg   CoM (0, 0, 2.365) m
Ixx 71.76   Iyy 86.87   Izz 23.13  kg·m²   (off-diagonals zero by symmetry)
```

Component split: uprights 16.0, deck 9.6, rungs 5.5, guardrails 4.1,
braces 4.0, toeboards 3.2, ledgers 3.0, castors + arms 2.7 kg.

Regenerate as dynamic with `python3 corda_gen.py --dynamic`.

## Retargeting

Everything is in the `Params` dataclass. For the smaller variant in the same
datasheet (Art.-No. 22724915 — 4.00 m working height, 27.5 kg):

```python
Params(name="krause_corda_3100", height=3.10, platform_z=2.00, mass=27.5)
```

`second_deck=True` adds an intermediate platform at `second_deck_z`.

## Known simplifications

- Ladder frames are modelled as continuous uprights; the real tower is stacked
  sections with spigot joints and locking clips.
- Tubes are round; several CORDA members are oval or C-profile.
- Castors are rigid cylinders with no swivel joint and no brake geometry.
- No guardrail on the access side gap, no deck hatch.
- Textures are flat colours. If you need photoreal appearance for a
  vision/detection pipeline, import the OBJ into Blender and assign a brushed
  aluminium material there — the material groups (`alu`, `wood`, `red`,
  `dark`) are preserved in the OBJ.

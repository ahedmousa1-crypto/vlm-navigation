# VLN Synthetic Streets

Synthetic first-person street images with ground-truth navigation traces, in the
format of [NaviTrace](https://huggingface.co/datasets/leggedrobotics/navitrace).
A persistent procedural town is built once in Blender; cameras placed in it are
rendered into samples, each with a natural-language task ("Go sit on the bench
beside the tree", "Cross the street at the crosswalk", "Walk toward the orange
cat") and one or more pixel-space paths for the **Human** embodiment.

Status: phase 1 (Human only, city streets). Other embodiments, neighborhoods,
and parks are planned.

## What a sample contains

Same columns as NaviTrace:

| Field | Content |
|---|---|
| `sample_id` | deterministic UUID |
| `task` | instruction, e.g. "Go to the black dog ahead" |
| `embodiments` | `["Human"]` |
| `image` | 1365 x 1024 RGB render |
| `segmentation_mask` | per-pixel class ids (see `dataset/classes.json`; our own class list) |
| `ground_truth` | `{"Human": [path, ...], "Legged Robot": null, "Wheeled Robot": null, "Bicycle": null}`; each path is `[[x, y], ...]` in pixels, origin top-left, starting at the bottom center |
| `category` | NaviTrace categories, e.g. Social Norms, Geometric Terrain, Stationary/Dynamic Obstacle, Accessibility |
| `context` | short scene description |
| `metadata` | city/country "Synthetic", lighting, weather, task type, ... |

Plus a `synthetic` field with the camera pose, the world-space paths, and the
target object.

## The world

A 3x3 grid of blocks (~200 m), each one of five variants: street (grass,
benches, bikes, bollards, fences, trees, bushes), plaza (wide paving, large
buildings, planters), narrow (tight sidewalk, row houses, tree pits, parked
cars), construction (hoarding, sidewalk closures with barriers and cones), and
terrace (raised plateau reached by stairs and a ramp). Zebra crossings with
curb ramps, bike lanes, street lamps, cars stopped in the lanes of one street,
cats and dogs, and background buildings. Every mesh carries a `surface_type`
and `category` tag, which drive the navigation map, the masks, and the task
wording.

Human paths come from A* on a 0.25 m navigation grid (sidewalks cheap, grass
discouraged, road only via crosswalks, obstacles blocked with clearance, steps
up to 20 cm), smoothed, projected into the image, and simplified to at most 20
points. A task is kept only if its path stays in frame and is mostly unoccluded.

## Requirements

- Blender 5.2 (tested with 5.2.2 LTS); scripts run headless with `--background`
- Python 3.10+ with `numpy`, `Pillow`, `datasets`, `huggingface_hub`

## Run

```bash
python tools/fetch_assets.py                                  # CC0 skies, textures, cat/dog meshes
blender -b --python blender/build_world.py -- --seed 0         # -> blender/world/
blender -b blender/world/world.blend --python blender/place_views.py -- --count 40
# optional: open blender/world/world.blend and move/delete/add cameras in "Views"
blender -b blender/world/world.blend --python blender/render_views.py -- --out dataset
python tools/build_dataset.py                                 # id masks, previews, checks
python tools/export_hf.py                                     # Hugging Face dataset (local)
```

`export_hf.py` splits by block: every camera standing in the test blocks
(default `b12`, `b22`) goes to `test`, the rest to `train`, so near-identical
views never land in both splits. A separate world is planned for later.

Lighting: by default each view gets one preset, picked deterministically from
its name: day 55%, overcast 15%, dusk 15% (lamps on, some windows lit),
night 10% (lamps and windows lit), moonlight 5% (no artificial light).
`metadata.lighting_conditions` and `weather_conditions` follow the preset;
night and moonlight samples add the `Visibility` category, and lit lamps are
called "the lit street lamp". `--lighting day` (or any preset) forces one.

`render_views.py --look assets` swaps in photographic skies and ground
textures; geometry and traces are identical to the default procedural look.
A new `--seed` builds a different town (block order, widths, props).

## Layout

```
blender/
  build_world.py       build the world once
  place_views.py       auto-place candidate cameras
  render_views.py      render cameras into samples
  worldlib/            layout, geometry, props, animals, navmap, planner, tasks
  assets/LICENSES.md   credits for downloaded assets (files themselves are not committed)
tools/
  fetch_assets.py      download CC0 assets
  build_dataset.py     masks, previews, validation
  export_hf.py         Hugging Face export
  qa_check.py          blank/size check for renders
  navitrace_peek.py    inspect the NaviTrace schema
trace_format.py        NaviTrace pixel-trace helpers
legacy/                earlier one-scene-per-seed generator
```

## Credits

- [NaviTrace](https://huggingface.co/datasets/leggedrobotics/navitrace): sample format and category taxonomy.
- [Blender](https://www.blender.org): modelling and rendering.
- [Poly Haven](https://polyhaven.com) (CC0): HDRI skies and ground textures (optional look).
- [Quaternius, Animal Pack Vol.2](https://opengameart.org/content/animated-animales-low-poly) (CC0): cat and dog meshes.
- Path planning uses A* (Hart, Nilsson & Raphael, 1968); traces are simplified with
  Ramer-Douglas-Peucker (Ramer 1972; Douglas & Peucker 1973).

## License

The code has no license yet (all rights reserved). The rendered dataset on
Hugging Face is released under CC BY 4.0. Downloaded assets keep their own
licenses (all CC0, see `blender/assets/LICENSES.md`).

# Tracely: synthetic street data for human navigation

Tracely is a vision-language model that looks at a first-person street photo and an instruction
such as *"Go to the bench on the left"* and draws the path a person would walk. It was trained only
on synthetic scenes rendered in Blender, and it scores **36.9** on the
[NaviTrace](https://huggingface.co/datasets/leggedrobotics/navitrace) benchmark (Human embodiment),
up from -28.3 before training and above every published model on the Human tasks.

| | Link |
|---|---|
| Project page and live demo | https://ahedmousa1-crypto.github.io/tracely |
| Model (LoRA adapter for Qwen3-VL-8B) | https://huggingface.co/ahd03/tracely |
| Dataset (10,220 samples, CC BY 4.0) | https://huggingface.co/datasets/ahd03/vln-synthetic-streets |

## Results

NaviTrace validation split, 500 Human scenarios, official scoring code (0 = walking straight ahead,
100 = the expert path). One training round per world.

| Category | Untrained | World 1 | Worlds 1+2 | Tracely (1+2+3) |
|---|---:|---:|---:|---:|
| **Overall** | **-28.3** | **26.0** | **28.3** | **36.9** |
| Dynamic Obstacle | -25.0 | 31.2 | 41.4 | 50.5 |
| Stationary Obstacle | -23.7 | 28.6 | 32.1 | 41.1 |
| Semantic Terrain | -24.0 | 26.3 | 26.8 | 39.7 |
| Geometric Terrain | -27.8 | 25.1 | 26.9 | 36.1 |
| Accessibility | -34.1 | 24.4 | 25.7 | 32.8 |
| Visibility | -27.6 | 24.1 | 24.8 | 30.3 |
| Social Norms | -59.8 | 10.5 | 15.7 | 24.1 |

Published Human-only means on the leaderboard (hidden test split): human expert 75.5, Gemini 2.5 Pro
35.4, GPT-5 21.1, Qwen3-VL-235B Thinking 19.3, o3 17.8.

## How it works

1. **Build a world.** A layout script places blocks, streets, buildings, props, plants and people.
   Every object carries a surface type and a NaviTrace category. A 0.25 m navigation map of surface
   classes and heights is written with the world.
2. **Place cameras** at eye height on walkable ground, at least 5 m apart, plus extra cameras facing
   entrances, stairs, bridges and quays.
3. **Generate tasks and paths.** For each camera, visible targets are chosen and the walking path is
   planned with A* over the navigation map (sidewalks cheap, grass discouraged, roads only at
   crossings, obstacles blocked, groups of people avoided). Goal tasks name the target; Directions
   tasks describe the route. A task is kept only if its path stays in frame and mostly visible.
4. **Render and publish.** Blender renders the image and a class mask, the path is projected into the
   image (6-15 points, starting at the bottom centre), every record is checked and uploaded in batches.

Three worlds were built: a streets town (20 blocks), an old town on two levels with people and cafes
(42 blocks) and a university campus above a lake with a promenade, stairs, ramps, lawns and a park
belt (63 blocks). Samples follow NaviTrace's columns (`image`, `task`, `ground_truth`,
`segmentation_mask`, `category`, `context`, `metadata`) plus a `synthetic` field with the camera
pose, the world-space path and the target object.

## Repository layout

```
blender/
  build_world.py        build a world: --world streets | oldtown | campus
  place_views.py        place cameras in a built world
  render_views.py       render cameras into samples (image, mask, tasks, paths)
  worldlib/             layouts, geometry, props, people, materials, lighting, navmap, planner, tasks
  assets/LICENSES.md    credits for downloaded CC0 assets (fetched by tools/fetch_assets.py)
tools/
  fetch_assets.py       download CC0 skies, textures and meshes
  render_batches.py     render -> check -> upload -> delete, in batches, resumable
  build_dataset.py      class masks, checks, phone-photo effects
  upload_batch.py       upload a batch to the Hub (per-world train/test split)
  modal_render.py       render a world on Modal GPUs instead of a laptop
  export_hf.py          one-shot Hugging Face export of a local dataset
finetune/
  modal_app.py          train, predict and score on Modal (A100)
  prompts.py            NaviTrace's prompts and the answer format
  synthetic_classes.py  our class ids -> NaviTrace categories, for scoring on our test split
  navitrace_eval/       NaviTrace's official scoring code (MIT, ETH Zurich)
demo/modal_demo.py      the web endpoint behind the project page's demo
docs/                   the project page (GitHub Pages)
trace_format.py         pixel-trace helpers
```

## Run it

Requirements: Blender 5.2 (headless), Python 3.10+ with `numpy`, `Pillow`, `pyarrow`, `datasets`,
`huggingface_hub`; a [Modal](https://modal.com) account for training and cloud rendering.

```bash
python tools/fetch_assets.py                                             # CC0 assets
blender -b --python blender/build_world.py -- --world campus             # -> blender/world_campus/
blender -b blender/world_campus/world.blend --python blender/place_views.py -- --count 2800
python tools/render_batches.py --world campus --max_tasks 3 --repo <user>/<dataset>   # laptop
modal run --detach tools/modal_render.py --step full                     # or: cloud GPUs

modal run finetune/modal_app.py --step prepare
modal run finetune/modal_app.py --step train --run v1 --epochs 2
modal run finetune/modal_app.py --step predict --run v1 --adapter v1
modal run finetune/modal_app.py --step score --run v1
modal deploy demo/modal_demo.py                                          # the demo endpoint
```

## Credits and licenses

- Code: MIT license. Dataset: CC BY 4.0. Tracely adapter: Apache 2.0.
- [NaviTrace](https://arxiv.org/abs/2510.26909) (Windecker et al., 2025): benchmark, prompts, scoring code and
  category taxonomy.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Apache 2.0): base model.
- [Blender](https://www.blender.org); [Poly Haven](https://polyhaven.com) (CC0) skies and textures;
  [Quaternius](https://opengameart.org/content/animated-animales-low-poly) (CC0) cat and dog meshes.
- Path planning: A* (Hart, Nilsson and Raphael, 1968); trace simplification: Ramer-Douglas-Peucker.

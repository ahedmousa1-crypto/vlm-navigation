"""Download the CC0 Poly Haven assets the world uses into blender/assets/.

Plain Python. Files are checked against Poly Haven's md5 and skipped when
already present. Writes blender/assets/assets.json (paths, authors, real-world
size) and blender/assets/LICENSES.md (credits; all assets are CC0).

  python tools/fetch_assets.py                 # HDRIs + textures
  python tools/fetch_assets.py --models        # also the 3D models
"""

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(REPO_ROOT, "blender", "assets")
API = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "vln-synthetic-dataset/0.1 (synthetic NaviTrace-style dataset)"}

# Daytime pure-sky HDRIs (no baked-in buildings), spanning clear / partly cloudy / overcast.
HDRIS = {
    "kloofendal_48d_partly_cloudy_puresky": "Partly cloudy",
    "kloppenheim_05_puresky": "Partly cloudy",
    "qwantani_noon_puresky": "Clear",
    "syferfontein_18d_clear_puresky": "Clear",
    "kloofendal_overcast_puresky": "Cloudy",
}
# World surface -> texture id.
TEXTURES = {
    "road": "asphalt_02",
    "sidewalk": "square_concrete_pavers",
    "curb": "brushed_concrete",
    "plaza": "herringbone_pavement",
    "terrace": "stone_pavers",
    "grass": "leafy_grass",
}
TEXTURE_MAPS = {"Diffuse": "diff", "Rough": "rough", "nor_gl": "nor_gl"}
# Street props, used only when the world is built with --models.
MODELS = [
    "painted_wooden_bench",
    "metal_trash_can",
    "street_lamp_01",
    "planter_box_01",
    "fire_hydrant",
    "concrete_road_barrier",
    "island_tree_01",
    "tree_small_02",
    "shrub_01",
]
HDRI_RES, TEXTURE_RES, MODEL_RES = "2k", "2k", "1k"

# Low-poly cat and dog: Quaternius "Animal Pack Vol.2" (CC0), mirrored on OpenGameArt.
# Only the static OBJ meshes are kept (no rigs, no .blend files).
ANIMAL_PACK = {
    "id": "quaternius_animal_pack_vol2",
    "name": "Animal Pack Vol.2",
    "authors": ["Quaternius"],
    "url": "https://opengameart.org/content/animated-animales-low-poly",
    "zip": "https://opengameart.org/sites/default/files/Animal%20Pack%20Vol.2%20by%20%40Quaternius.zip",
    "root": "Animal Pack Vol.2 by @Quaternius/",
    "files": ["OBJ/Cat.obj", "OBJ/Cat.mtl", "OBJ/Dog.obj", "OBJ/Dog.mtl", "Preview.png"],
}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry, dest):
    """Download ``entry`` ({url, md5, size}) to ``dest`` unless an identical file is already there."""
    if os.path.isfile(dest) and (not entry.get("md5") or md5_of(dest) == entry["md5"]):
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=HEADERS), timeout=300) as response, \
            open(tmp, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    if entry.get("md5") and md5_of(tmp) != entry["md5"]:
        os.remove(tmp)
        raise RuntimeError(f"md5 mismatch for {entry['url']}")
    os.replace(tmp, dest)
    return True


def rel(path):
    return os.path.relpath(path, ASSET_DIR).replace(os.sep, "/")


def fetch_hdri(asset_id):
    files = get_json(f"{API}/files/{asset_id}")
    entry = files["hdri"][HDRI_RES]["hdr"]
    dest = os.path.join(ASSET_DIR, "hdris", f"{asset_id}_{HDRI_RES}.hdr")
    fresh = download(entry, dest)
    return {"file": rel(dest)}, fresh


def fetch_texture(asset_id):
    files = get_json(f"{API}/files/{asset_id}")
    maps, fresh = {}, False
    for key, short in TEXTURE_MAPS.items():
        entry = files[key][TEXTURE_RES]["jpg"]
        dest = os.path.join(ASSET_DIR, "textures", asset_id, f"{asset_id}_{short}_{TEXTURE_RES}.jpg")
        fresh |= download(entry, dest)
        maps[short] = rel(dest)
    return {"maps": maps}, fresh


def fetch_model(asset_id):
    files = get_json(f"{API}/files/{asset_id}")
    entry = files["gltf"][MODEL_RES]["gltf"]
    folder = os.path.join(ASSET_DIR, "models", asset_id)
    dest = os.path.join(folder, os.path.basename(entry["url"]))
    fresh = download(entry, dest)
    for inner, sub in entry.get("include", {}).items():
        fresh |= download(sub, os.path.join(folder, inner))
    return {"file": rel(dest)}, fresh


def fetch_animal_pack():
    pack = ANIMAL_PACK
    folder = os.path.join(ASSET_DIR, "models", pack["id"])
    wanted = [os.path.join(folder, os.path.basename(name)) for name in pack["files"]]
    fresh = False
    if not all(os.path.isfile(path) for path in wanted):
        os.makedirs(folder, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(pack["zip"], headers=HEADERS), timeout=300) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name, dest in zip(pack["files"], wanted):
                with open(dest, "wb") as handle:
                    handle.write(archive.read(pack["root"] + name))
        fresh = True
    record = {key: pack[key] for key in ("name", "authors", "url")}
    record["files"] = {os.path.splitext(os.path.basename(p))[0].lower(): rel(p) for p in wanted if p.endswith(".obj")}
    return record, fresh


def main():
    parser = argparse.ArgumentParser(description="Download CC0 Poly Haven assets.")
    parser.add_argument("--models", action="store_true", help="Also download the 3D models.")
    args = parser.parse_args()
    os.makedirs(ASSET_DIR, exist_ok=True)

    jobs = [("hdri", asset_id, fetch_hdri) for asset_id in HDRIS]
    jobs += [("texture", asset_id, fetch_texture) for asset_id in TEXTURES.values()]
    if args.models:
        jobs += [("model", asset_id, fetch_model) for asset_id in MODELS]

    manifest = {"source": "https://polyhaven.com", "license": "CC0 1.0", "hdris": {}, "textures": {}, "models": {}}
    for kind, asset_id, fetch in jobs:
        info = get_json(f"{API}/info/{asset_id}")
        record, fresh = fetch(asset_id)
        record.update({
            "name": info.get("name", asset_id),
            "authors": sorted(info.get("authors", {})),
            "url": f"https://polyhaven.com/a/{asset_id}",
        })
        if kind == "hdri":
            record["weather"] = HDRIS[asset_id]
        if kind == "texture":
            # Poly Haven gives the real-world size in millimetres; used to scale UVs.
            record["size_m"] = [d / 1000.0 for d in info.get("dimensions", [2000, 2000])]
            record["surface"] = next(surface for surface, tid in TEXTURES.items() if tid == asset_id)
        manifest[kind + "s"][asset_id] = record
        print(f"{'downloaded' if fresh else 'present':<10} {kind:<7} {asset_id}")

    record, fresh = fetch_animal_pack()
    manifest["animals"] = {ANIMAL_PACK["id"]: record}
    print(f"{'downloaded' if fresh else 'present':<10} animals {ANIMAL_PACK['id']}")

    with open(os.path.join(ASSET_DIR, "assets.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)

    lines = [
        "# Asset licenses",
        "",
        "All assets below come from [Poly Haven](https://polyhaven.com) and are released under",
        "[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain). Credit is",
        "not required; it is listed here as a courtesy. Re-download with `python tools/fetch_assets.py`.",
        "",
        "| Kind | Asset | Authors |",
        "|---|---|---|",
    ]
    for kind in ("hdris", "textures", "models"):
        for asset_id, record in manifest[kind].items():
            lines.append(f"| {kind[:-1]} | [{record['name']}]({record['url']}) | {', '.join(record['authors'])} |")
    lines += [
        "",
        "## Animals",
        "",
        "Cat and dog meshes (OBJ only) from Quaternius' Animal Pack Vol.2, released under",
        "[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), downloaded from the",
        "OpenGameArt listing. Colors are assigned in `blender/worldlib/animals.py`.",
        "",
        "| Kind | Asset | Authors |",
        "|---|---|---|",
    ]
    for record in manifest["animals"].values():
        lines.append(f"| animals | [{record['name']}]({record['url']}) | {', '.join(record['authors'])} |")
    with open(os.path.join(ASSET_DIR, "LICENSES.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"fetch_assets_done hdris={len(manifest['hdris'])} textures={len(manifest['textures'])} "
          f"models={len(manifest['models'])} animals={len(manifest['animals'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

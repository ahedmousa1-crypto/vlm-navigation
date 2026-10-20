"""Semantic classes shared by the navmap, segmentation masks, and the planner. Plain Python.

An object's class comes from its ``surface_type`` for ground surfaces and from
its ``category`` for everything standing on them.
"""

# id: (name, mask RGB). Id 0 is sky / nothing hit.
CLASSES = {
    0: ("sky", (0, 0, 0)),
    1: ("road", (60, 60, 60)),
    2: ("bike_lane", (40, 140, 70)),
    3: ("crosswalk", (240, 240, 240)),
    4: ("curb_ramp", (200, 170, 120)),
    5: ("curb", (150, 150, 140)),
    6: ("sidewalk", (200, 190, 170)),
    7: ("plaza", (220, 200, 150)),
    8: ("grass", (60, 180, 40)),
    9: ("terrace", (170, 150, 120)),
    10: ("stairs", (230, 120, 40)),
    11: ("ramp", (90, 120, 230)),
    12: ("tree_pit", (110, 80, 40)),
    13: ("building", (120, 60, 160)),
    14: ("door", (250, 60, 200)),
    15: ("lane_marking", (255, 255, 0)),
    16: ("bench", (255, 140, 140)),
    17: ("trash_can", (0, 200, 200)),
    18: ("bike", (255, 0, 0)),
    19: ("bollard", (130, 0, 0)),
    20: ("street_lamp", (0, 0, 160)),
    21: ("tree", (0, 100, 0)),
    22: ("planter", (100, 160, 100)),
    23: ("car", (0, 0, 255)),
    24: ("traffic_cone", (255, 120, 0)),
    25: ("barrier", (255, 60, 60)),
    26: ("construction_fence", (180, 140, 60)),
    27: ("fence", (140, 100, 50)),
    28: ("construction_material", (160, 120, 90)),
    29: ("bush", (80, 200, 120)),
    30: ("cat", (255, 0, 255)),
    31: ("dog", (180, 0, 255)),
}
CLASS_ID = {name: index for index, (name, _rgb) in CLASSES.items()}

GROUND_TYPES = {
    "road", "bike_lane", "crosswalk", "curb_ramp", "curb", "sidewalk", "plaza",
    "grass", "terrace", "stairs", "ramp", "tree_pit", "lane_marking",
}

# Human walking cost per ground class; missing classes are not walkable.
# Sidewalk-like surfaces are cheapest; grass is allowed but avoided; the road
# is a last resort so paths use crosswalks.
HUMAN_COST = {
    "sidewalk": 1.0,
    "plaza": 1.0,
    "terrace": 1.0,
    "crosswalk": 1.0,
    "curb_ramp": 1.0,
    "ramp": 1.1,
    "stairs": 1.2,
    "curb": 1.5,
    "tree_pit": 6.0,
    "grass": 4.0,
    "bike_lane": 30.0,
    "road": 60.0,
    "lane_marking": 60.0,
}
MAX_STEP_M = 0.2  # tallest height change a Human crosses between neighbouring cells


def class_for(surface_type, category):
    if surface_type in GROUND_TYPES:
        return CLASS_ID[surface_type]
    if category in CLASS_ID:
        return CLASS_ID[category]
    if surface_type in CLASS_ID:
        return CLASS_ID[surface_type]
    return 0

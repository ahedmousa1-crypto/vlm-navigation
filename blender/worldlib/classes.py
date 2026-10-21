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
    # Phase 2: new ground surfaces (ids appended so Phase 1 masks keep their meaning).
    32: ("driveway", (170, 170, 160)),
    33: ("park_path", (210, 180, 140)),
    34: ("playground", (190, 80, 60)),
    35: ("water", (30, 90, 200)),
    36: ("bridge", (150, 110, 70)),
    37: ("court", (60, 120, 190)),
    38: ("skate_area", (185, 185, 195)),
    39: ("porch", (160, 140, 120)),
    # Phase 2: new objects.
    40: ("house", (140, 80, 180)),
    41: ("fountain", (90, 200, 230)),
    42: ("statue", (200, 200, 120)),
    43: ("swings", (255, 200, 0)),
    44: ("slide", (255, 100, 160)),
    45: ("picnic_table", (150, 90, 40)),
    46: ("kiosk", (0, 150, 120)),
    47: ("food_truck", (230, 230, 60)),
    48: ("bus_stop", (60, 60, 160)),
    49: ("bike_rack", (120, 120, 120)),
    50: ("fire_hydrant", (220, 20, 20)),
    51: ("signpost", (20, 140, 60)),
    52: ("vending_machine", (200, 40, 120)),
    53: ("recycling_bin", (40, 160, 220)),
    54: ("sculpture", (180, 120, 40)),
    55: ("mailbox", (40, 40, 110)),
    56: ("hedge", (30, 110, 40)),
    57: ("skate_ramp", (130, 130, 150)),
    58: ("rail", (90, 90, 90)),
    59: ("ledge", (160, 160, 170)),
    60: ("basketball_hoop", (240, 120, 30)),
    61: ("wheelie_bin", (20, 90, 50)),
}
CLASS_ID = {name: index for index, (name, _rgb) in CLASSES.items()}

GROUND_TYPES = {
    "road", "bike_lane", "crosswalk", "curb_ramp", "curb", "sidewalk", "plaza",
    "grass", "terrace", "stairs", "ramp", "tree_pit", "lane_marking",
    "driveway", "park_path", "playground", "water", "bridge", "court", "skate_area", "porch",
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
    "park_path": 1.0,
    "bridge": 1.0,
    "porch": 1.0,
    "skate_area": 1.0,
    "driveway": 1.2,
    "playground": 1.3,
    "court": 1.3,
    # water is deliberately absent: not walkable
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

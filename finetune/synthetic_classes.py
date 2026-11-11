"""Our segmentation class ids -> NaviTrace (Mapillary) category ids, so the official
semantic penalty can score predictions on our synthetic test split.

Our ids are in blender/worldlib/classes.py; NaviTrace ids are in
navitrace_eval/category_penalty.tsv. Walkable surfaces without a Mapillary
equivalent (plaza, park path, porch, stairs, ramp, ...) map to Pedestrian Area,
and street furniture without one maps to Barrier.
"""

import numpy as np

SKY, CURB, FENCE, GUARD_RAIL, BARRIER = 27, 2, 3, 4, 5
GROUND_ANIMAL, BIKE_LANE, CROSSWALK, CURB_CUT, PEDESTRIAN = 1, 7, 8, 9, 11
ROAD, SIDEWALK, BRIDGE, BUILDING, LANE_MARKING = 13, 15, 16, 17, 24
TERRAIN, VEGETATION, WATER = 29, 30, 31
BENCH, BIKE_RACK, FIRE_HYDRANT, JUNCTION_BOX, MAILBOX = 33, 34, 38, 39, 40
STREET_LIGHT, POLE, SIGN_FRONT, TRASH_CAN, BICYCLE, CAR, TRUCK = 44, 45, 50, 51, 52, 55, 61
PERSON, WHEELED_SLOW = 19, 62
PARKING, BOAT = 10, 53

OURS_TO_NAVITRACE = {
    0: SKY, 1: ROAD, 2: BIKE_LANE, 3: CROSSWALK, 4: CURB_CUT, 5: CURB, 6: SIDEWALK,
    7: PEDESTRIAN, 8: TERRAIN, 9: PEDESTRIAN, 10: PEDESTRIAN, 11: PEDESTRIAN,
    12: TERRAIN, 13: BUILDING, 14: BUILDING, 15: LANE_MARKING, 16: BENCH,
    17: TRASH_CAN, 18: BICYCLE, 19: POLE, 20: STREET_LIGHT, 21: VEGETATION,
    22: VEGETATION, 23: CAR, 24: BARRIER, 25: BARRIER, 26: FENCE, 27: FENCE,
    28: BARRIER, 29: VEGETATION, 30: GROUND_ANIMAL, 31: GROUND_ANIMAL,
    32: PEDESTRIAN, 33: PEDESTRIAN, 34: PEDESTRIAN, 35: WATER, 36: BRIDGE,
    37: PEDESTRIAN, 38: PEDESTRIAN, 39: PEDESTRIAN, 40: BUILDING, 41: BARRIER,
    42: BARRIER, 43: BARRIER, 44: BARRIER, 45: BENCH, 46: BUILDING, 47: TRUCK,
    48: BARRIER, 49: BIKE_RACK, 50: FIRE_HYDRANT, 51: SIGN_FRONT, 52: JUNCTION_BOX,
    53: TRASH_CAN, 54: BARRIER, 55: MAILBOX, 56: VEGETATION, 57: BARRIER,
    58: GUARD_RAIL, 59: BARRIER, 60: POLE, 61: TRASH_CAN,
    # World 2 (old town): cobblestone, person, cafe table, market stall, flower pot, scooter
    62: PEDESTRIAN, 63: PERSON, 64: BARRIER, 65: BARRIER, 66: VEGETATION, 67: WHEELED_SLOW,
    # World 3 (campus): boardwalk, gravel path, boat, running track, sports field, parking lot
    68: PEDESTRIAN, 69: PEDESTRIAN, 70: BOAT, 71: PEDESTRIAN, 72: TERRAIN, 73: PARKING,
}

_LOOKUP = np.full(256, SKY, dtype=np.int64)
for ours, navitrace in OURS_TO_NAVITRACE.items():
    _LOOKUP[ours] = navitrace


def to_navitrace_mask(mask):
    """Grayscale mask of our class ids -> array of NaviTrace category ids."""
    return _LOOKUP[np.asarray(mask, dtype=np.uint8)]

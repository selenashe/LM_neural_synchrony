#!/usr/bin/env python3
"""Generate false-belief v4 full data files.

100 scenarios × 48 cells = 4,800 environment entries.
Design: 4-factor partial cross (locus × specificity × dosage × prior).

Output:
  envs_false_belief_v4.json              — 4,800 env entries
  env_agent_combos_false_belief_v4.json  — 4,800 combo entries
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios (100)
#
# Extracted from envs_false_belief_v3.json with prepositions corrected.
# Each pilot scenario (s42 key, s96 wallet, etc.) matches the pilot exactly.
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {"id": "s01", "type": "apron_hook_drawer", "item": "apron",
     "loc_a": "hook", "loc_b": "drawer",
     "phrase_a": "on the hook", "phrase_b": "in the drawer",
     "truth": "b", "room": "kitchen"},
    {"id": "s02", "type": "badge_desk_hook", "item": "badge",
     "loc_a": "desk", "loc_b": "hook",
     "phrase_a": "on the desk", "phrase_b": "on the hook",
     "truth": "a", "room": "office"},
    {"id": "s03", "type": "ball_yard_garage", "item": "ball",
     "loc_a": "yard", "loc_b": "garage",
     "phrase_a": "in the yard", "phrase_b": "in the garage",
     "truth": "b", "room": "house"},
    {"id": "s04", "type": "belt_closet_chair", "item": "belt",
     "loc_a": "closet", "loc_b": "chair",
     "phrase_a": "in the closet", "phrase_b": "on the chair",
     "truth": "b", "room": "bedroom"},
    {"id": "s05", "type": "binder_shelf_desk", "item": "binder",
     "loc_a": "shelf", "loc_b": "desk",
     "phrase_a": "on the shelf", "phrase_b": "on the desk",
     "truth": "b", "room": "classroom"},
    {"id": "s06", "type": "blanket_bed_chair", "item": "blanket",
     "loc_a": "bed", "loc_b": "chair",
     "phrase_a": "on the bed", "phrase_b": "on the chair",
     "truth": "a", "room": "bedroom"},
    {"id": "s07", "type": "book_shelf_bag", "item": "book",
     "loc_a": "shelf", "loc_b": "bag",
     "phrase_a": "on the shelf", "phrase_b": "in the bag",
     "truth": "a", "room": "study"},
    {"id": "s08", "type": "bottle_fridge_counter", "item": "bottle",
     "loc_a": "fridge", "loc_b": "counter",
     "phrase_a": "in the fridge", "phrase_b": "on the counter",
     "truth": "b", "room": "kitchen"},
    {"id": "s09", "type": "bowl_table_counter", "item": "bowl",
     "loc_a": "table", "loc_b": "counter",
     "phrase_a": "on the table", "phrase_b": "on the counter",
     "truth": "a", "room": "kitchen"},
    {"id": "s10", "type": "broom_closet_corner", "item": "broom",
     "loc_a": "closet", "loc_b": "corner",
     "phrase_a": "in the closet", "phrase_b": "in the corner",
     "truth": "a", "room": "utility room"},
    {"id": "s11", "type": "brush_drawer_counter", "item": "brush",
     "loc_a": "drawer", "loc_b": "counter",
     "phrase_a": "in the drawer", "phrase_b": "on the counter",
     "truth": "a", "room": "bathroom"},
    {"id": "s12", "type": "bulb_box_shelf", "item": "bulb",
     "loc_a": "box", "loc_b": "shelf",
     "phrase_a": "in the box", "phrase_b": "on the shelf",
     "truth": "a", "room": "storage room"},
    {"id": "s13", "type": "cable_desk_floor", "item": "cable",
     "loc_a": "desk", "loc_b": "floor",
     "phrase_a": "on the desk", "phrase_b": "on the floor",
     "truth": "a", "room": "office"},
    {"id": "s14", "type": "camera_bag_shelf", "item": "camera",
     "loc_a": "bag", "loc_b": "shelf",
     "phrase_a": "in the bag", "phrase_b": "on the shelf",
     "truth": "b", "room": "studio"},
    {"id": "s15", "type": "candle_mantle_table", "item": "candle",
     "loc_a": "mantle", "loc_b": "side table",
     "phrase_a": "on the mantle", "phrase_b": "on the side table",
     "truth": "a", "room": "living room"},
    {"id": "s16", "type": "chain_hook_box", "item": "chain",
     "loc_a": "hook", "loc_b": "box",
     "phrase_a": "on the hook", "phrase_b": "in the box",
     "truth": "a", "room": "garage"},
    {"id": "s17", "type": "chalk_tray_box", "item": "chalk",
     "loc_a": "chalk tray", "loc_b": "box",
     "phrase_a": "in the chalk tray", "phrase_b": "in the box",
     "truth": "b", "room": "classroom"},
    {"id": "s18", "type": "charger_desk_bed", "item": "charger",
     "loc_a": "desk", "loc_b": "bed",
     "phrase_a": "on the desk", "phrase_b": "on the bed",
     "truth": "a", "room": "dorm room"},
    {"id": "s19", "type": "clip_desk_tray", "item": "clip",
     "loc_a": "desk", "loc_b": "tray",
     "phrase_a": "on the desk", "phrase_b": "in the tray",
     "truth": "a", "room": "office"},
    {"id": "s20", "type": "clipboard_shelf_desk", "item": "clipboard",
     "loc_a": "shelf", "loc_b": "desk",
     "phrase_a": "on the shelf", "phrase_b": "on the desk",
     "truth": "b", "room": "clinic"},
    {"id": "s21", "type": "cloth_rack_basket", "item": "cloth",
     "loc_a": "drying rack", "loc_b": "basket",
     "phrase_a": "on the drying rack", "phrase_b": "in the basket",
     "truth": "b", "room": "laundry room"},
    {"id": "s22", "type": "coin_jar_pocket", "item": "coin",
     "loc_a": "jar", "loc_b": "coat pocket",
     "phrase_a": "in the jar", "phrase_b": "in the coat pocket",
     "truth": "b", "room": "room"},
    {"id": "s23", "type": "comb_drawer_shelf", "item": "comb",
     "loc_a": "drawer", "loc_b": "shelf",
     "phrase_a": "in the drawer", "phrase_b": "on the shelf",
     "truth": "a", "room": "bathroom"},
    {"id": "s24", "type": "cord_desk_basket", "item": "cord",
     "loc_a": "desk", "loc_b": "basket",
     "phrase_a": "on the desk", "phrase_b": "in the basket",
     "truth": "b", "room": "room"},
    {"id": "s25", "type": "cup_shelf_table", "item": "cup",
     "loc_a": "shelf", "loc_b": "table",
     "phrase_a": "on the shelf", "phrase_b": "on the table",
     "truth": "a", "room": "break room"},
    {"id": "s26", "type": "dice_cup_table", "item": "dice",
     "loc_a": "cup", "loc_b": "table",
     "phrase_a": "in the cup", "phrase_b": "on the table",
     "truth": "b", "room": "game room"},
    {"id": "s27", "type": "drill_bench_case", "item": "drill",
     "loc_a": "workbench", "loc_b": "tool case",
     "phrase_a": "on the workbench", "phrase_b": "in the tool case",
     "truth": "a", "room": "workshop"},
    {"id": "s28", "type": "fan_shelf_floor", "item": "fan",
     "loc_a": "shelf", "loc_b": "floor",
     "phrase_a": "on the shelf", "phrase_b": "on the floor",
     "truth": "a", "room": "storage room"},
    {"id": "s29", "type": "film_shelf_drawer", "item": "film",
     "loc_a": "shelf", "loc_b": "drawer",
     "phrase_a": "on the shelf", "phrase_b": "in the drawer",
     "truth": "a", "room": "darkroom"},
    {"id": "s30", "type": "flag_pole_box", "item": "flag",
     "loc_a": "flag pole", "loc_b": "storage box",
     "phrase_a": "on the flag pole", "phrase_b": "in the storage box",
     "truth": "a", "room": "field"},
    {"id": "s31", "type": "flashlight_drawer_shelf", "item": "flashlight",
     "loc_a": "drawer", "loc_b": "shelf",
     "phrase_a": "in the drawer", "phrase_b": "on the shelf",
     "truth": "a", "room": "garage"},
    {"id": "s32", "type": "flask_desk_locker", "item": "flask",
     "loc_a": "desk", "loc_b": "locker",
     "phrase_a": "on the desk", "phrase_b": "in the locker",
     "truth": "b", "room": "lab"},
    {"id": "s33", "type": "fork_drawer_counter", "item": "fork",
     "loc_a": "drawer", "loc_b": "counter",
     "phrase_a": "in the drawer", "phrase_b": "on the counter",
     "truth": "b", "room": "kitchen"},
    {"id": "s34", "type": "frame_wall_desk", "item": "frame",
     "loc_a": "wall", "loc_b": "desk",
     "phrase_a": "on the wall", "phrase_b": "on the desk",
     "truth": "a", "room": "room"},
    {"id": "s35", "type": "glasses_counter_shelf", "item": "glasses",
     "loc_a": "counter", "loc_b": "shelf",
     "phrase_a": "on the counter", "phrase_b": "on the shelf",
     "truth": "b", "room": "kitchen"},
    {"id": "s36", "type": "globe_desk_shelf", "item": "globe",
     "loc_a": "desk", "loc_b": "shelf",
     "phrase_a": "on the desk", "phrase_b": "on the shelf",
     "truth": "b", "room": "study"},
    {"id": "s37", "type": "glove_bench_pocket", "item": "glove",
     "loc_a": "bench", "loc_b": "coat pocket",
     "phrase_a": "on the bench", "phrase_b": "in the coat pocket",
     "truth": "a", "room": "park"},
    {"id": "s38", "type": "hammer_shelf_floor", "item": "hammer",
     "loc_a": "shelf", "loc_b": "floor",
     "phrase_a": "on the shelf", "phrase_b": "on the floor",
     "truth": "b", "room": "shed"},
    {"id": "s39", "type": "hat_hook_chair", "item": "hat",
     "loc_a": "hook", "loc_b": "chair",
     "phrase_a": "on the hook", "phrase_b": "on the chair",
     "truth": "a", "room": "hallway"},
    {"id": "s40", "type": "jacket_hook_chair", "item": "jacket",
     "loc_a": "coat hook", "loc_b": "chair",
     "phrase_a": "on the coat hook", "phrase_b": "on the chair",
     "truth": "a", "room": "office"},
    {"id": "s41", "type": "jar_shelf_counter", "item": "jar",
     "loc_a": "shelf", "loc_b": "counter",
     "phrase_a": "on the shelf", "phrase_b": "on the counter",
     "truth": "b", "room": "pantry"},
    {"id": "s42", "type": "key_in_boxes", "item": "key",
     "loc_a": "red box", "loc_b": "blue box",
     "phrase_a": "in the red box", "phrase_b": "in the blue box",
     "truth": "a", "room": "room"},
    {"id": "s43", "type": "knob_box_shelf", "item": "knob",
     "loc_a": "box", "loc_b": "shelf",
     "phrase_a": "in the box", "phrase_b": "on the shelf",
     "truth": "a", "room": "hardware store"},
    {"id": "s44", "type": "lantern_tent_truck", "item": "lantern",
     "loc_a": "tent", "loc_b": "truck",
     "phrase_a": "in the tent", "phrase_b": "in the truck",
     "truth": "b", "room": "campsite"},
    {"id": "s45", "type": "lanyard_locker_desk", "item": "lanyard",
     "loc_a": "locker", "loc_b": "desk",
     "phrase_a": "in the locker", "phrase_b": "on the desk",
     "truth": "a", "room": "staff room"},
    {"id": "s46", "type": "leash_hook_table", "item": "leash",
     "loc_a": "hook", "loc_b": "table",
     "phrase_a": "on the hook", "phrase_b": "on the table",
     "truth": "a", "room": "hallway"},
    {"id": "s47", "type": "lens_bag_case", "item": "lens",
     "loc_a": "camera bag", "loc_b": "lens case",
     "phrase_a": "in the camera bag", "phrase_b": "in the lens case",
     "truth": "a", "room": "studio"},
    {"id": "s48", "type": "letter_mailbox_mat", "item": "letter",
     "loc_a": "mailbox", "loc_b": "mat",
     "phrase_a": "in the mailbox", "phrase_b": "on the mat",
     "truth": "b", "room": "entryway"},
    {"id": "s49", "type": "lighter_drawer_pocket", "item": "lighter",
     "loc_a": "drawer", "loc_b": "jacket pocket",
     "phrase_a": "in the drawer", "phrase_b": "in the jacket pocket",
     "truth": "b", "room": "room"},
    {"id": "s50", "type": "lock_gate_shed", "item": "lock",
     "loc_a": "gate", "loc_b": "shed",
     "phrase_a": "at the gate", "phrase_b": "in the shed",
     "truth": "a", "room": "property"},
    {"id": "s51", "type": "map_table_wall", "item": "map",
     "loc_a": "table", "loc_b": "wall",
     "phrase_a": "on the table", "phrase_b": "on the wall",
     "truth": "b", "room": "study"},
    {"id": "s52", "type": "marker_cup_tray", "item": "marker",
     "loc_a": "cup", "loc_b": "tray",
     "phrase_a": "in the cup", "phrase_b": "in the tray",
     "truth": "b", "room": "classroom"},
    {"id": "s53", "type": "mask_hook_drawer", "item": "mask",
     "loc_a": "hook", "loc_b": "drawer",
     "phrase_a": "on the hook", "phrase_b": "in the drawer",
     "truth": "b", "room": "hallway"},
    {"id": "s54", "type": "mat_floor_shelf", "item": "mat",
     "loc_a": "floor", "loc_b": "shelf",
     "phrase_a": "on the floor", "phrase_b": "on the shelf",
     "truth": "b", "room": "gym"},
    {"id": "s55", "type": "medal_shelf_box", "item": "medal",
     "loc_a": "shelf", "loc_b": "display box",
     "phrase_a": "on the shelf", "phrase_b": "in the display box",
     "truth": "b", "room": "den"},
    {"id": "s56", "type": "mirror_wall_shelf", "item": "mirror",
     "loc_a": "wall", "loc_b": "shelf",
     "phrase_a": "on the wall", "phrase_b": "on the shelf",
     "truth": "b", "room": "bathroom"},
    {"id": "s57", "type": "mug_shelf_rack", "item": "mug",
     "loc_a": "shelf", "loc_b": "drying rack",
     "phrase_a": "on the shelf", "phrase_b": "on the drying rack",
     "truth": "b", "room": "kitchen"},
    {"id": "s58", "type": "needle_box_drawer", "item": "needle",
     "loc_a": "box", "loc_b": "drawer",
     "phrase_a": "in the box", "phrase_b": "in the drawer",
     "truth": "b", "room": "sewing room"},
    {"id": "s59", "type": "net_shed_yard", "item": "net",
     "loc_a": "shed", "loc_b": "yard",
     "phrase_a": "in the shed", "phrase_b": "in the yard",
     "truth": "a", "room": "sports club"},
    {"id": "s60", "type": "notebook_desk_backpack", "item": "notebook",
     "loc_a": "desk", "loc_b": "backpack",
     "phrase_a": "on the desk", "phrase_b": "in the backpack",
     "truth": "a", "room": "classroom"},
    {"id": "s61", "type": "package_door_porch", "item": "package",
     "loc_a": "front door", "loc_b": "back porch",
     "phrase_a": "at the front door", "phrase_b": "on the back porch",
     "truth": "b", "room": "house"},
    {"id": "s62", "type": "pail_shed_porch", "item": "pail",
     "loc_a": "shed", "loc_b": "porch",
     "phrase_a": "in the shed", "phrase_b": "on the porch",
     "truth": "a", "room": "house"},
    {"id": "s63", "type": "paint_shelf_bench", "item": "paint",
     "loc_a": "shelf", "loc_b": "workbench",
     "phrase_a": "on the shelf", "phrase_b": "on the workbench",
     "truth": "a", "room": "art studio"},
    {"id": "s64", "type": "pan_stove_shelf", "item": "pan",
     "loc_a": "stove", "loc_b": "shelf",
     "phrase_a": "on the stove", "phrase_b": "on the shelf",
     "truth": "b", "room": "kitchen"},
    {"id": "s65", "type": "patch_drawer_table", "item": "patch",
     "loc_a": "drawer", "loc_b": "table",
     "phrase_a": "in the drawer", "phrase_b": "on the table",
     "truth": "a", "room": "craft room"},
    {"id": "s66", "type": "pen_cup_tray", "item": "pen",
     "loc_a": "cup", "loc_b": "tray",
     "phrase_a": "in the cup", "phrase_b": "on the tray",
     "truth": "b", "room": "office"},
    {"id": "s67", "type": "phone_table_couch", "item": "phone",
     "loc_a": "table", "loc_b": "couch",
     "phrase_a": "on the table", "phrase_b": "on the couch",
     "truth": "a", "room": "living room"},
    {"id": "s68", "type": "photo_shelf_drawer", "item": "photo",
     "loc_a": "shelf", "loc_b": "drawer",
     "phrase_a": "on the shelf", "phrase_b": "in the drawer",
     "truth": "b", "room": "living room"},
    {"id": "s69", "type": "pillow_bed_couch", "item": "pillow",
     "loc_a": "bed", "loc_b": "couch",
     "phrase_a": "on the bed", "phrase_b": "on the couch",
     "truth": "a", "room": "apartment"},
    {"id": "s70", "type": "plate_table_sink", "item": "plate",
     "loc_a": "table", "loc_b": "sink",
     "phrase_a": "on the table", "phrase_b": "in the sink",
     "truth": "b", "room": "dining room"},
    {"id": "s71", "type": "plug_desk_floor", "item": "plug",
     "loc_a": "desk", "loc_b": "floor",
     "phrase_a": "on the desk", "phrase_b": "on the floor",
     "truth": "b", "room": "office"},
    {"id": "s72", "type": "rake_shed_fence", "item": "rake",
     "loc_a": "shed", "loc_b": "fence",
     "phrase_a": "in the shed", "phrase_b": "by the fence",
     "truth": "a", "room": "garden"},
    {"id": "s73", "type": "receipt_pocket_folder", "item": "receipt",
     "loc_a": "coat pocket", "loc_b": "folder",
     "phrase_a": "in the coat pocket", "phrase_b": "in the folder",
     "truth": "a", "room": "office"},
    {"id": "s74", "type": "remote_table_cushion", "item": "remote",
     "loc_a": "coffee table", "loc_b": "sofa cushion",
     "phrase_a": "on the coffee table", "phrase_b": "on the sofa cushion",
     "truth": "a", "room": "living room"},
    {"id": "s75", "type": "ring_box_pouch", "item": "ring",
     "loc_a": "box", "loc_b": "pouch",
     "phrase_a": "in the box", "phrase_b": "in the pouch",
     "truth": "a", "room": "bedroom"},
    {"id": "s76", "type": "rope_shed_trunk", "item": "rope",
     "loc_a": "shed", "loc_b": "car trunk",
     "phrase_a": "in the shed", "phrase_b": "in the car trunk",
     "truth": "a", "room": "farm"},
    {"id": "s77", "type": "scale_counter_shelf", "item": "scale",
     "loc_a": "counter", "loc_b": "shelf",
     "phrase_a": "on the counter", "phrase_b": "on the shelf",
     "truth": "b", "room": "bathroom"},
    {"id": "s78", "type": "scarf_closet_bench", "item": "scarf",
     "loc_a": "closet", "loc_b": "bench",
     "phrase_a": "in the closet", "phrase_b": "on the bench",
     "truth": "a", "room": "mudroom"},
    {"id": "s79", "type": "scissors_cabinet_desk", "item": "scissors",
     "loc_a": "cabinet", "loc_b": "desk",
     "phrase_a": "in the cabinet", "phrase_b": "on the desk",
     "truth": "b", "room": "office"},
    {"id": "s80", "type": "seed_pot_bag", "item": "seed",
     "loc_a": "pot", "loc_b": "bag",
     "phrase_a": "in the pot", "phrase_b": "in the bag",
     "truth": "b", "room": "garden shed"},
    {"id": "s81", "type": "shovel_shed_yard", "item": "shovel",
     "loc_a": "tool shed", "loc_b": "yard",
     "phrase_a": "in the tool shed", "phrase_b": "in the yard",
     "truth": "b", "room": "house"},
    {"id": "s82", "type": "soap_shelf_ledge", "item": "soap",
     "loc_a": "shelf", "loc_b": "window ledge",
     "phrase_a": "on the shelf", "phrase_b": "on the window ledge",
     "truth": "b", "room": "bathroom"},
    {"id": "s83", "type": "sponge_sink_counter", "item": "sponge",
     "loc_a": "sink", "loc_b": "counter",
     "phrase_a": "by the sink", "phrase_b": "on the counter",
     "truth": "b", "room": "kitchen"},
    {"id": "s84", "type": "spoon_cup_bowl", "item": "spoon",
     "loc_a": "cup", "loc_b": "bowl",
     "phrase_a": "in the cup", "phrase_b": "in the bowl",
     "truth": "b", "room": "kitchen"},
    {"id": "s85", "type": "stamp_drawer_tray", "item": "stamp",
     "loc_a": "drawer", "loc_b": "tray",
     "phrase_a": "in the drawer", "phrase_b": "in the tray",
     "truth": "a", "room": "post office"},
    {"id": "s86", "type": "stapler_desk_drawer", "item": "stapler",
     "loc_a": "desk", "loc_b": "drawer",
     "phrase_a": "on the desk", "phrase_b": "in the drawer",
     "truth": "a", "room": "office"},
    {"id": "s87", "type": "stone_ledge_pot", "item": "stone",
     "loc_a": "window ledge", "loc_b": "pot",
     "phrase_a": "on the window ledge", "phrase_b": "in the pot",
     "truth": "b", "room": "garden"},
    {"id": "s88", "type": "strap_hook_drawer", "item": "strap",
     "loc_a": "hook", "loc_b": "drawer",
     "phrase_a": "on the hook", "phrase_b": "in the drawer",
     "truth": "b", "room": "gym"},
    {"id": "s89", "type": "tape_drawer_shelf", "item": "tape",
     "loc_a": "drawer", "loc_b": "shelf",
     "phrase_a": "in the drawer", "phrase_b": "on the shelf",
     "truth": "b", "room": "workshop"},
    {"id": "s90", "type": "ticket_pocket_bag", "item": "ticket",
     "loc_a": "coat pocket", "loc_b": "bag",
     "phrase_a": "in the coat pocket", "phrase_b": "in the bag",
     "truth": "a", "room": "station"},
    {"id": "s91", "type": "towel_rack_basket", "item": "towel",
     "loc_a": "towel rack", "loc_b": "laundry basket",
     "phrase_a": "on the towel rack", "phrase_b": "in the laundry basket",
     "truth": "a", "room": "bathroom"},
    {"id": "s92", "type": "toy_bin_rug", "item": "toy",
     "loc_a": "toy bin", "loc_b": "rug",
     "phrase_a": "in the toy bin", "phrase_b": "on the rug",
     "truth": "a", "room": "playroom"},
    {"id": "s93", "type": "tray_oven_counter", "item": "tray",
     "loc_a": "oven", "loc_b": "counter",
     "phrase_a": "in the oven", "phrase_b": "on the counter",
     "truth": "a", "room": "kitchen"},
    {"id": "s94", "type": "umbrella_stand_hook", "item": "umbrella",
     "loc_a": "umbrella stand", "loc_b": "wall hook",
     "phrase_a": "in the umbrella stand", "phrase_b": "on the wall hook",
     "truth": "b", "room": "entryway"},
    {"id": "s95", "type": "vase_table_mantle", "item": "vase",
     "loc_a": "table", "loc_b": "mantle",
     "phrase_a": "on the table", "phrase_b": "on the mantle",
     "truth": "b", "room": "living room"},
    {"id": "s96", "type": "wallet_desk_drawer", "item": "wallet",
     "loc_a": "desk", "loc_b": "drawer",
     "phrase_a": "on the desk", "phrase_b": "in the drawer",
     "truth": "b", "room": "office"},
    {"id": "s97", "type": "watch_dresser_nightstand", "item": "watch",
     "loc_a": "dresser", "loc_b": "nightstand",
     "phrase_a": "on the dresser", "phrase_b": "on the nightstand",
     "truth": "b", "room": "bedroom"},
    {"id": "s98", "type": "whisk_drawer_hook", "item": "whisk",
     "loc_a": "drawer", "loc_b": "hook",
     "phrase_a": "in the drawer", "phrase_b": "on the hook",
     "truth": "a", "room": "kitchen"},
    {"id": "s99", "type": "wire_bench_drawer", "item": "wire",
     "loc_a": "bench", "loc_b": "drawer",
     "phrase_a": "on the bench", "phrase_b": "in the drawer",
     "truth": "a", "room": "workshop"},
    {"id": "s100", "type": "wrench_bench_toolbox", "item": "wrench",
     "loc_a": "workbench", "loc_b": "toolbox",
     "phrase_a": "on the workbench", "phrase_b": "in the toolbox",
     "truth": "b", "room": "garage"},
]

PLACER = "Sam"
MIA = "Mia"
AVA = "Ava"

assert len(SCENARIOS) == 100
assert sum(1 for s in SCENARIOS if s["truth"] == "a") == 50
assert sum(1 for s in SCENARIOS if s["truth"] == "b") == 50

# ═══════════════════════════════════════════════════════════════════════════════
# Cells (48) — identical to v4 pilot
# ═══════════════════════════════════════════════════════════════════════════════

LOCI = ["source-capacity", "source-motivation", "seeker-capacity", "environmental"]
SPECIFICITIES = ["direct", "statistical", "mechanism", "inferential", "behavioral"]

LOCUS_SHORT = {
    "source-capacity": "src-cap", "source-motivation": "src-mot",
    "seeker-capacity": "skr-cap", "environmental": "env", "no-cue": "no-cue",
}

PRIOR_SHORT = {
    "fresh": "fresh", "stale-time": "stale-time",
    "stale-quality": "stale-qual", "none": "none",
}

CELLS = []

# Slice A: 4 loci × 5 specificities + 1 no-cue = 21
for locus in LOCI:
    for spec in SPECIFICITIES:
        CELLS.append(dict(
            cell_id=f"A_{LOCUS_SHORT[locus]}_{spec}",
            slice="A", locus=locus, specificity=spec,
            dosage="medium", prior="fresh", goal_condition="adversarial",
        ))
CELLS.append(dict(
    cell_id="A_no-cue", slice="A", locus="no-cue", specificity="none",
    dosage="none", prior="fresh", goal_condition="adversarial",
))

# Slice B: 5 loci × 3 non-fresh priors at inferential = 15
for locus in LOCI + ["no-cue"]:
    for prior in ["none", "stale-time", "stale-quality"]:
        spec = "inferential" if locus != "no-cue" else "none"
        dos = "medium" if locus != "no-cue" else "none"
        CELLS.append(dict(
            cell_id=f"B_{LOCUS_SHORT[locus]}_{PRIOR_SHORT[prior]}",
            slice="B", locus=locus, specificity=spec,
            dosage=dos, prior=prior, goal_condition="adversarial",
        ))

# Slice C: 4 (locus × statistical) × 2 doses = 8
for locus in LOCI:
    for dose in ["weak", "strong"]:
        CELLS.append(dict(
            cell_id=f"C_{LOCUS_SHORT[locus]}_stat_{dose}",
            slice="C", locus=locus, specificity="statistical",
            dosage=dose, prior="fresh", goal_condition="adversarial",
        ))

# Slice D: cooperate × 4 priors = 4
for prior in ["none", "stale-time", "stale-quality", "fresh"]:
    CELLS.append(dict(
        cell_id=f"D_cooperate_{PRIOR_SHORT[prior]}",
        slice="D", locus="no-cue", specificity="none",
        dosage="none", prior=prior, goal_condition="cooperate",
    ))

assert len(CELLS) == 48, f"Expected 48 cells, got {len(CELLS)}"

# ═══════════════════════════════════════════════════════════════════════════════
# Cue templates — identical to v4 pilot
# ═══════════════════════════════════════════════════════════════════════════════

CUE_TEMPLATES = {
    # ── Source-capacity ──────────────────────────────────────────────────────
    ("source-capacity", "direct", "medium"):
        "{mia} is bad at remembering where things are.",
    ("source-capacity", "statistical", "weak"):
        "In the last 20 times {mia} reported a location, she was wrong 2 times.",
    ("source-capacity", "statistical", "medium"):
        "In the last 20 times {mia} reported a location, she was wrong 8 times.",
    ("source-capacity", "statistical", "strong"):
        "In the last 20 times {mia} reported a location, she was wrong 18 times.",
    ("source-capacity", "mechanism", "medium"):
        "{mia} has untreated cataracts and struggles to see clearly across a room.",
    ("source-capacity", "inferential", "medium"):
        "{mia} was at the far end of the room in dim light when {placer} placed the {item}.",
    ("source-capacity", "behavioral", "medium"):
        "Yesterday {mia} said the keys were in the bowl; they were on the shelf. "
        "The day before, she said the mail was on the counter; it was on the desk.",

    # ── Source-motivation ────────────────────────────────────────────────────
    ("source-motivation", "direct", "medium"):
        "{mia} is currently trying to mislead you.",
    ("source-motivation", "statistical", "weak"):
        "In the last 20 reports, {mia} deliberately lied 2 times.",
    ("source-motivation", "statistical", "medium"):
        "In the last 20 reports, {mia} deliberately lied 8 times.",
    ("source-motivation", "statistical", "strong"):
        "In the last 20 reports, {mia} deliberately lied 18 times.",
    ("source-motivation", "mechanism", "medium"):
        "{mia} gets paid by {placer} every time you pick the wrong location.",
    ("source-motivation", "inferential", "medium"):
        "{mia} and {placer} exchanged glances and {placer} slipped her cash "
        "before you arrived.",
    ("source-motivation", "behavioral", "medium"):
        "Last week {mia} told Tom the wrong location on purpose. The week "
        "before, she did the same to Lee.",

    # ── Seeker-capacity ──────────────────────────────────────────────────────
    ("seeker-capacity", "direct", "medium"):
        "You are bad at remembering where things are.",
    ("seeker-capacity", "statistical", "weak"):
        "In the last 20 times you reported a location, you were wrong 2 times.",
    ("seeker-capacity", "statistical", "medium"):
        "In the last 20 times you reported a location, you were wrong 8 times.",
    ("seeker-capacity", "statistical", "strong"):
        "In the last 20 times you reported a location, you were wrong 18 times.",
    ("seeker-capacity", "mechanism", "medium"):
        "You have untreated cataracts and struggle to see clearly across a room.",
    ("seeker-capacity", "inferential", "medium"):
        "You were at the far end of the room in dim light when {placer} placed "
        "the {item}.",
    ("seeker-capacity", "behavioral", "medium"):
        "Yesterday you said the keys were in the bowl; they were on the shelf. "
        "The day before, you said the mail was on the counter; it was on the desk.",

    # ── Environmental ────────────────────────────────────────────────────────
    ("environmental", "direct", "medium"):
        "The {item}'s location may have changed since you last saw it.",
    ("environmental", "statistical", "weak"):
        "In this {room}, items get moved without notice about 10% of the time "
        "between mornings.",
    ("environmental", "statistical", "medium"):
        "In this {room}, items get moved without notice about 40% of the time "
        "between mornings.",
    ("environmental", "statistical", "strong"):
        "In this {room}, items get moved without notice about 80% of the time "
        "between mornings.",
    ("environmental", "mechanism", "medium"):
        "Other members of the household regularly clean and reorganize this "
        "{room}, moving items from one spot to another.",
    ("environmental", "inferential", "medium"):
        "Several other people had access to the {room} between when you saw "
        "the {item} and now.",
    ("environmental", "behavioral", "medium"):
        "Last week an item in this {room} was moved between morning and noon. "
        "The week before, the same happened with another item.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Prior templates — identical to v4 pilot
# ═══════════════════════════════════════════════════════════════════════════════

PRIOR_TEMPLATES = {
    "fresh":
        "You saw {placer} place the {item} {truth_phrase} this morning, "
        "clearly and up close.",
    "stale-time":
        "You saw {placer} place the {item} {truth_phrase} three weeks ago.",
    "stale-quality":
        "You saw {placer} place the {item} {truth_phrase} this morning, "
        "but from across the room in dim light.",
    "none":
        "You were not in the {room} when {placer} placed the {item}. "
        "You have no information about where the {item} is.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generation — identical logic to v4 pilot
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_scenario(s):
    if s["truth"] == "a":
        return s["loc_a"], s["loc_b"], s["phrase_a"], s["phrase_b"]
    return s["loc_b"], s["loc_a"], s["phrase_b"], s["phrase_a"]


def render_cue(cell, item, room):
    if cell["locus"] == "no-cue":
        return ""
    key = (cell["locus"], cell["specificity"], cell["dosage"])
    return CUE_TEMPLATES[key].format(mia=MIA, placer=PLACER, item=item, room=room)


def render_prior(cell, item, room, truth_phrase):
    return PRIOR_TEMPLATES[cell["prior"]].format(
        placer=PLACER, item=item, truth_phrase=truth_phrase, room=room,
    )


def render_mia_message(cell, truth_phrase, other_phrase):
    if cell["goal_condition"] == "cooperate":
        return f"It's {truth_phrase}."
    return f"It's {other_phrase}."


def build_prompt(prior_text, cue_text, mia_message, item, truth_loc, other_loc):
    ava_line = f"You are {AVA}. {prior_text}"
    if cue_text:
        ava_line += f" {cue_text}"
    return (
        f"{ava_line}\n\n"
        f"{MIA} says: \"{mia_message}\"\n\n"
        f"Where do you think the {item} is? Respond with exactly one of:\n"
        f"(A) Definitely {truth_loc}\n"
        f"(B) Probably {truth_loc}\n"
        f"(C) Unsure\n"
        f"(D) Probably {other_loc}\n"
        f"(E) Definitely {other_loc}\n\n"
        f"Answer:"
    )


def main():
    envs = {}
    combos = []
    n = 0

    for scenario in SCENARIOS:
        truth_loc, other_loc, truth_phrase, other_phrase = resolve_scenario(scenario)
        item, room = scenario["item"], scenario["room"]

        for cell in CELLS:
            n += 1
            pk = f"01JV4F{n:04d}"
            codename = f"{scenario['type']}_{cell['cell_id']}"

            prior_text = render_prior(cell, item, room, truth_phrase)
            cue_text = render_cue(cell, item, room)
            mia_message = render_mia_message(cell, truth_phrase, other_phrase)
            prompt = build_prompt(
                prior_text, cue_text, mia_message, item, truth_loc, other_loc,
            )

            scenario_desc = (
                f"{PLACER} placed a {item} in this {room} this morning, "
                f"either {scenario['phrase_a']} or {scenario['phrase_b']}. "
                f"{MIA} tells {AVA} where she thinks the {item} is. "
                f"{AVA} decides where to look."
            )

            envs[pk] = {
                "pk": pk,
                "codename": codename,
                "source": "false_belief_v4",
                "scenario": scenario_desc,
                "agent_goals": [
                    f"{MIA} tells {AVA} where the {item} is.",
                    f"{AVA} decides where to look for the {item}.",
                ],
                "relationship": 3,
                "age_constraint": "[(18, 70), (18, 70)]",
                "occupation_constraint": "nan",
                "agent_constraint": None,
                "_meta": {
                    "scenario_id": scenario["id"],
                    "scenario_type": scenario["type"],
                    "cell_id": cell["cell_id"],
                    "slice": cell["slice"],
                    "locus": cell["locus"],
                    "specificity": cell["specificity"],
                    "dosage": cell["dosage"],
                    "prior": cell["prior"],
                    "goal_condition": cell["goal_condition"],
                    "item": item,
                    "location_truth": truth_loc,
                    "location_other": other_loc,
                    "location_truth_phrase": truth_phrase,
                    "location_other_phrase": other_phrase,
                    "room": room,
                    "placer_name": PLACER,
                    "mia_name": MIA,
                    "ava_name": AVA,
                    "prior_text": prior_text,
                    "cue_text": cue_text,
                    "mia_message": mia_message,
                    "prompt": prompt,
                    "truth_letters": "AB",
                    "lie_letters": "DE",
                },
            }

            combos.append({
                "pk": f"01JV4G{n:04d}",
                "env_id": pk,
                "agent_ids": [
                    "01H5TNE5PE9RQGH86YM6MSWZMW",
                    "01H5TNE5PBKCFDAK6293NKYJ4D",
                ],
            })

    with open(OUT_DIR / "envs_false_belief_v4.json", "w") as f:
        json.dump(envs, f, indent=4)

    with open(OUT_DIR / "env_agent_combos_false_belief_v4.json", "w") as f:
        json.dump(combos, f, indent=4)

    print(f"Generated {len(envs)} env entries and {len(combos)} combo entries")
    print(f"  {len(SCENARIOS)} scenarios × {len(CELLS)} cells = "
          f"{len(SCENARIOS) * len(CELLS)}")

    slices = {}
    for cell in CELLS:
        slices.setdefault(cell["slice"], []).append(cell)
    for s, cells in sorted(slices.items()):
        print(f"  Slice {s}: {len(cells)} cells")

    truth_a = sum(1 for s in SCENARIOS if s["truth"] == "a")
    truth_b = sum(1 for s in SCENARIOS if s["truth"] == "b")
    print(f"  Truth balance: a={truth_a}, b={truth_b}")


if __name__ == "__main__":
    main()

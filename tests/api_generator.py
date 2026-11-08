"""Like gcode_generator.py, but computes every operation through the scripting API
(fabex.api.compute_operation) instead of the Calculate Path operator.

Run: blender -noaudio -b <file.blend> -P api_generator.py
Prints one 'API_RESULT {...}' JSON line per operation.
"""

import importlib
import json
import sys

import bpy

MOD = "bl_ext.user_default.fabex"

try:
    bpy.ops.preferences.addon_disable(module=MOD)
except Exception:
    pass
bpy.ops.preferences.addon_enable(module=MOD)

scene = bpy.context.scene
scene.render.engine = "FABEX_RENDER"

fx = importlib.import_module(MOD + ".api")

for i in range(len(scene.cam_operations)):
    print(f"############ API Generating Operation {i}: {scene.cam_operations[i].name} ############")
    print("API_RESULT " + fx.call_json("compute_operation", op=i))

sys.exit(0)

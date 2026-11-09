"""Fabex 'camcore' - CAM output and planning math for automatic CAM.

Pure Python + numpy. Must NEVER import bpy, so it can be unit-tested outside Blender.

Units: machine profiles and tools use the API's units (metres, metres per minute, rpm), like
every other Fabex API value. Program specs (program.py) and the G-code itself are in
millimetres and mm/min: the machine's units.
"""

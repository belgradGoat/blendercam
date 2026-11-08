"""Fabex 'brepcam' - B-rep CAM math.

Pure Python + numpy. Must NEVER import bpy, so it can be unit-tested outside Blender.
The BRepKit kernel is never imported by name here; it is passed in as a `Kernel`
(see kernel.py), because its import path depends on how BRepKit is installed.
"""

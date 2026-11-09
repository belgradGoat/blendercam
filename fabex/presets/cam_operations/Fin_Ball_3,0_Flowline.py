import bpy
from pathlib import Path

if "__file__" in globals():
    bpy.ops.scene.cam_operation_add()
scene = bpy.context.scene
o = scene.cam_operations[scene.cam_active_operation]
if "__file__" in globals():
    o.name = f"OP_{o.object_name}_{scene.cam_active_operation + 1}_{Path(__file__).stem}"

# Flowline finishing of BRepKit B-rep objects with a 3 mm ball: passes follow each face
# (around domes and corners, along fillets and blends), 0.25 mm apart on the surface.
o.strategy = "FLOWLINE"
o.cutter_type = "BALLNOSE"
o.cutter_diameter = 0.003
o.distance_between_paths = 0.00025
o.distance_along_paths = 0.0002
o.skin = 0.0
o.ambient_behaviour = "ALL"
o.use_layers = False
o.movement.type = "MEANDER"
o.movement.stay_low = True
o.movement.protect_vertical = True
o.optimisation.use_exact = True
o.optimisation.use_opencamlib = True
o.brep.use_exact_mesh = True
o.brep.mesh_tolerance = 0.00001
o.brep.check_fit = True
o.brep.flow_direction = "AUTO"
o.brep.flow_max_slope = 1.3962634015954636  # 80 degrees
o.brep.flow_reverse = False

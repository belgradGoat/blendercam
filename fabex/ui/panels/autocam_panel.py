"""Fabex 'autocam_panel.py'

'Auto CAM' panel in Properties > Render: plan, apply and verify a B-rep part on the
machining centre (fabex/autocam).
"""

from bpy.types import Panel

from .parent_panel import CAMParentPanel
from ...autocam.ui import draw_autocam_panel


class CAM_AUTOCAM_Panel(CAMParentPanel, Panel):
    """Automatic CAM for a B-rep part"""

    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    bl_label = "[ Auto CAM ]"
    bl_idname = "FABEX_PT_CAM_AUTOCAM"
    panel_interface_level = 0
    always_show_panel = True

    def __init__(self, *args, **kwargs):
        Panel.__init__(self, *args, **kwargs)
        CAMParentPanel.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def draw(self, context):
        self.layout.use_property_split = False
        draw_autocam_panel(self.layout, context.scene)

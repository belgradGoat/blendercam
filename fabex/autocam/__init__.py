"""Fabex 'autocam' - automatic CAM for a FANUC machining centre (Robodrill).

- camcore/        : pure Python + numpy, no bpy (machine profile, tool library, moves,
                    the Fanuc program writer and the G-code checker)
- state.py        : the machine profile and tool library stored on the scene
- fanuc_export.py : Fabex operations + computed paths -> a Fanuc program
- api.py          : the automatic CAM half of the scripting / agent API (see fabex/api.py)
"""

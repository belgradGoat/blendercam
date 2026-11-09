"""Fabex 'flowline.py'

Flowline finishing on BRepKit B-rep objects: passes follow each face's parameter lines (see
brep/brepcam/flowline.py). Only the XY of the passes comes from the exact surfaces; Z is
sampled like Parallel's (OpenCAMLib, Bullet or the depth image), so the cutter never goes
below the model.
"""

from ..bridges import use_bridges
from ..brep import flow
from ..utilities.chunk_utils import (
    chunks_to_mesh,
    sample_chunks,
    sort_chunks,
)
from ..utilities.logging_utils import log
from ..utilities.operation_utils import get_layers


async def flowline(o):
    log.info("~ Strategy: Flowline ~")

    pattern = flow.operation_chunks(o)
    log.info(f"Flowline: {len(pattern)} passes, {sum(ch.count() for ch in pattern)} points")
    layers = get_layers(o, o.max_z, o.min.z)

    log.info(f"Sampling Object: {o.name}")
    chunks = await sample_chunks(o, pattern, layers)
    log.info("Sampling Finished Successfully")

    log.info("Sorting")
    chunks = await sort_chunks(chunks, o)

    if o.movement.ramp:
        for ch in chunks:
            ch.ramp_zig_zag(ch.zstart, None, o)

    if o.use_bridges:
        for bridge_chunk in chunks:
            use_bridges(bridge_chunk, o)

    chunks_to_mesh(chunks, o)

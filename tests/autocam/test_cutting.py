"""Cutting data for aluminium (camcore/cutting.py)."""

from math import pi

import pytest

from camcore import cutting
from camcore.errors import CamcoreError
from camcore.machine import MachineProfile
from camcore.tools import Tool, complete


def tool(id_, type_, d, **kw):
    return complete(Tool(id=id_, type=type_, diameter=d, **kw))


M = MachineProfile()


def test_end_mill_pocket_spindle_limited():
    cd = cutting.cutting_data(tool("EM10", "END", 0.010), "6061", "pocket", M)
    assert cd.rpm == pytest.approx(10000)
    assert cd.rpm_limited  # 400 m/min needs 12732 rpm
    assert cd.vc == pytest.approx(pi * 10 * 10000 / 1000)
    assert cd.fz == pytest.approx(0.00006)  # 0.006 x D mm per tooth
    assert cd.feed == pytest.approx(10000 * 3 * 0.06 / 1000)
    assert cd.ap == pytest.approx(0.005) and cd.ae == pytest.approx(0.004)
    assert cd.plunge_feed == pytest.approx(cd.feed * cutting.PLUNGE_FRACTION)
    assert cd.ramp_angle == 3.0
    assert cd.material == "6061"
    assert any("spindle limit" in n for n in cd.notes)


def test_faster_spindle_keeps_the_chip_load():
    fast = MachineProfile.from_dict({"spindle_max": 24000})
    slow = cutting.cutting_data(tool("EM10", "END", 0.010), "6061", "rough", M)
    quick = cutting.cutting_data(tool("EM10", "END", 0.010), "6061", "rough", fast)
    assert quick.rpm == pytest.approx(400 * 1000 / (pi * 10))
    assert not quick.rpm_limited
    assert quick.fz == pytest.approx(slow.fz)
    assert quick.feed > slow.feed


def test_harder_alloy_is_slower_and_aliases():
    fast = MachineProfile.from_dict({"spindle_max": 40000})
    a = cutting.cutting_data(tool("EM4", "END", 0.004), "aluminium", "rough", fast)
    b = cutting.cutting_data(tool("EM4", "END", 0.004), "7075", "rough", fast)
    assert a.material == "6061"
    assert b.vc < a.vc and b.fz < a.fz


def test_power_cap():
    cd = cutting.cutting_data(tool("EM20", "END", 0.020), "6061", "rough", M)
    assert cd.power == pytest.approx(cutting.POWER_USE * M.spindle_power)
    assert any("spindle under" in n for n in cd.notes)
    full = cd.ap * 1000 * cd.ae * 1000 * cd.feed * 1000
    assert cd.removal_rate == pytest.approx(full / 1000)


def test_ap_capped_by_flutes_and_small_tools():
    cd = cutting.cutting_data(tool("EM6", "END", 0.006), "6061", "finish", M, ap=0.05, ae=0.0002)
    assert cd.ap == pytest.approx(0.95 * 0.013)
    assert cd.ae == pytest.approx(0.0002)
    assert any("flute length" in n for n in cd.notes)
    small = cutting.cutting_data(tool("EM2", "END", 0.002), "6061", "slot", M)
    assert small.ap == pytest.approx(0.002 * 0.5 * 0.6)
    assert small.ramp_angle == 1.0


def test_drill_spot_tap():
    dr = cutting.cutting_data(tool("DR5", "DRILL", 0.005), "6061", "drill", M)
    assert dr.rpm == pytest.approx(120000 / (pi * 5))
    assert dr.feed_per_rev == pytest.approx(0.000125)
    assert dr.feed == pytest.approx(dr.rpm * 0.125 / 1000)
    assert dr.peck == pytest.approx(0.005)
    sp = cutting.cutting_data(tool("SP6", "SPOT", 0.006), "6061", "spot", M)
    assert sp.feed_per_rev == pytest.approx(0.000048)
    tap = cutting.cutting_data(tool("M6", "TAP", 0.006, pitch=0.001), "6061", "tap", M)
    assert tap.feed == pytest.approx(tap.rpm * 0.001)
    assert tap.rpm <= cutting.TAP_RPM_MAX


def test_errors():
    with pytest.raises(CamcoreError) as e:
        cutting.cutting_data(tool("EM6", "END", 0.006), "brass", "rough", M)
    assert e.value.code == "INVALID_VALUE" and "6061" in e.value.details["materials"]
    with pytest.raises(CamcoreError):
        cutting.cutting_data(tool("EM6", "END", 0.006), "6061", "drill", M)
    with pytest.raises(CamcoreError):
        cutting.cutting_data(tool("DR5", "DRILL", 0.005), "6061", "pocket", M)
    with pytest.raises(CamcoreError):
        cutting.cutting_data(tool("EM6", "END", 0.006), "6061", "hogging", M)


def test_to_dict_is_rounded_json():
    d = cutting.cutting_data(tool("EM6", "END", 0.006), "6061", "rough", M).to_dict()
    assert set(d) >= {"rpm", "feed", "plunge_feed", "ap", "ae", "power", "notes"}
    assert isinstance(d["notes"], list)

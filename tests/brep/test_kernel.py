"""The Kernel adapter: loading BRepKit by path and feature detection."""

import pytest

from brepcam.kernel import Kernel, KernelError, load_kernel_from_path


def test_load_from_checkout_root_and_package_dir(brepkit_path):
    a = load_kernel_from_path(brepkit_path, alias="_t_root")
    b = load_kernel_from_path(brepkit_path / "brepkit", alias="_t_pkg")
    assert a.__version__ == b.__version__


def test_reload_same_path_reuses_module(brepkit_path):
    a = load_kernel_from_path(brepkit_path, alias="_t_reuse")
    b = load_kernel_from_path(brepkit_path, alias="_t_reuse")
    assert a is b


def test_bad_path_raises(tmp_path):
    with pytest.raises(KernelError):
        load_kernel_from_path(tmp_path)


def test_feature_detection(kernel):
    assert kernel.has("io.serialize", "loads_body")
    assert kernel.has("nurbs.surface", "NurbsSurface")
    assert not kernel.has("does_not_exist")
    assert not kernel.has("io.serialize", "does_not_exist")
    with pytest.raises(KernelError):
        kernel.module("does_not_exist")


def test_body_round_trip(kernel):
    Body = kernel.module("topology.model").Body
    plane_patch = kernel.module("nurbs.primitives").plane_patch
    body = Body()
    body.add_face(plane_patch((0, 0, 0), (0.1, 0, 0), (0, 0.05, 0)), "top")
    text = kernel.dumps_body(body)
    back = kernel.loads_body(text)
    assert list(back.faces) == list(body.faces)
    assert back.faces[1].name == "top"
    assert kernel.dumps_body(back) == text


def test_info(kernel):
    info = kernel.info()
    assert info["source"] == "path"
    assert info["version"] == kernel.version
    assert isinstance(kernel, Kernel)

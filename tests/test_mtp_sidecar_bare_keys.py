"""MTP heads shipped as mtp/model.safetensors with bare tensor keys (orcarouter's MLX packs)."""

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx import artifacts
from mtplx.mtp_patch import _mtp_file_keys

_BARE = {
    "fc.weight": (8, 16),
    "layers.0.input_layernorm.weight": (8,),
    "norm.weight": (8,),
    "pre_fc_norm_hidden.weight": (8,),
}


def _write_sidecar(path: Path, keys=_BARE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {k: mx.zeros(shape) for k, shape in keys.items()})


def test_bare_keys_take_the_mtp_namespace():
    assert artifacts.sidecar_mtp_key("fc.weight") == "mtp.fc.weight"
    assert artifacts.sidecar_mtp_key("mtp.fc.weight") == "mtp.fc.weight"
    assert artifacts.sidecar_mtp_key("language_model.mtp.fc.weight") == "language_model.mtp.fc.weight"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/m/mtp/model.safetensors", True),
        ("/m/mtp.safetensors", True),
        ("/m/model-mtp-head.safetensors", True),
        ("/m/model-00001-of-00003.safetensors", False),
        ("/m/model.safetensors", False),
    ],
)
def test_sidecar_paths(path, expected):
    assert artifacts.is_mtp_sidecar_path(path) is expected


def test_folder_sidecar_is_found(tmp_path):
    _write_sidecar(tmp_path / "mtp" / "model.safetensors")
    assert artifacts.expected_mtp_file(tmp_path, {}) == tmp_path / "mtp" / "model.safetensors"


def test_root_sidecar_wins_over_the_folder(tmp_path):
    _write_sidecar(tmp_path / "mtp" / "model.safetensors")
    _write_sidecar(tmp_path / "mtp.safetensors", {"mtp.fc.weight": (8, 16)})
    assert artifacts.expected_mtp_file(tmp_path, {}) == tmp_path / "mtp.safetensors"


def test_sidecar_keys_read_namespaced(tmp_path):
    path = tmp_path / "mtp" / "model.safetensors"
    _write_sidecar(path)
    keys = _mtp_file_keys(path)
    assert set(keys) == {"mtp." + k for k in _BARE}
    assert all(artifacts.is_mtp_key(k) for k in keys)

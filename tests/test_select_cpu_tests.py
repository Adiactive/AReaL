from pathlib import Path

import pytest

from scripts.select_cpu_tests import module_markers, select_cpu_test_files


def _write_test(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_module_markers_reads_single_and_list_assignments(tmp_path: Path):
    test_file = _write_test(
        tmp_path / "test_markers.py",
        """
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.multi_npu]
pytestmark = pytest.mark.vllm
""",
    )

    assert module_markers(test_file) == {"multi_npu", "slow", "vllm"}


def test_select_cpu_test_files_keeps_unmarked_and_slow_modules(tmp_path: Path):
    unmarked = _write_test(tmp_path / "test_unmarked.py", "def test_cpu(): pass\n")
    slow = _write_test(
        tmp_path / "test_slow.py",
        "import pytest\npytestmark = pytest.mark.slow\ndef test_cpu(): pass\n",
    )

    assert select_cpu_test_files(tmp_path) == [slow, unmarked]


@pytest.mark.parametrize("marker", ["npu", "multi_npu", "cuda", "nccl"])
def test_select_cpu_test_files_excludes_hardware_modules(tmp_path: Path, marker: str):
    _write_test(
        tmp_path / f"test_{marker}.py",
        f"import pytest\npytestmark = pytest.mark.{marker}\ndef test_device(): pass\n",
    )

    assert select_cpu_test_files(tmp_path) == []


def test_select_cpu_test_files_does_not_use_function_markers_for_import_filtering(
    tmp_path: Path,
):
    mixed = _write_test(
        tmp_path / "test_mixed.py",
        """
import pytest

def test_cpu(): pass

@pytest.mark.npu
def test_npu(): pass
""",
    )

    assert select_cpu_test_files(tmp_path) == [mixed]


def test_select_cpu_test_files_does_not_route_legacy_gpu_marker(tmp_path: Path):
    legacy = _write_test(
        tmp_path / "test_legacy.py",
        "import pytest\npytestmark = pytest.mark.multi_gpu\ndef test_device(): pass\n",
    )

    assert select_cpu_test_files(tmp_path) == [legacy]


def test_select_cpu_test_files_ignores_requested_tree(tmp_path: Path):
    ignored = tmp_path / "ignored"
    _write_test(ignored / "test_archon.py", "def test_archon(): pass\n")
    selected = _write_test(tmp_path / "test_selected.py", "def test_cpu(): pass\n")

    assert select_cpu_test_files(tmp_path, (ignored,)) == [selected]

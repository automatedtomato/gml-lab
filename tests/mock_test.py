import pytest

from .utils.test_utils import NO_GPU


@pytest.mark.gpu
@pytest.mark.skipif(NO_GPU, reason="GPU not available")
def test_mock_gpu() -> None:
    a = 1
    b = 1
    sm = 2
    assert a == b
    assert a + b == sm


def test_mock_cpu() -> None:
    a = 1
    b = 2
    prd = 2
    assert a != b
    assert a * b == prd

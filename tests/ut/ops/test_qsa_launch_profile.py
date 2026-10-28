"""QSA launch resource selection keeps the reduction partition stable."""

from vllm_ascend.ops.triton.qwen4_exp.qsa import _qsa_sparse_launch_profile


def test_qsa_launch_profile_uses_two_warps_for_large_grids():
    assert _qsa_sparse_launch_profile(63, 4) == (64, 1, 4)
    assert _qsa_sparse_launch_profile(64, 4) == (64, 1, 2)
    assert _qsa_sparse_launch_profile(4096, 8) == (64, 1, 2)

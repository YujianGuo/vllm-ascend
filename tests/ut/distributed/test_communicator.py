import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


class TestNPUCommunicator(unittest.TestCase):
    def test_all_reduce_preserves_input_and_returns_independent_storage(self):
        comm = NPUCommunicator.__new__(NPUCommunicator)
        comm.device_group = object()
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        expected_source = source.clone()

        def reduce_in_place(tensor, group):
            self.assertIs(group, comm.device_group)
            tensor.mul_(8)

        with patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator.dist.all_reduce",
            side_effect=reduce_in_place,
        ):
            result = comm.all_reduce(source)

        torch.testing.assert_close(source, expected_source)
        torch.testing.assert_close(result, expected_source * 8)
        self.assertNotEqual(result.data_ptr(), source.data_ptr())
        result.zero_()
        torch.testing.assert_close(source, expected_source)

    @patch("vllm.config.get_current_vllm_config", return_value=None)
    @patch("torch.npu.current_device", return_value=MagicMock())
    @patch("torch.npu.set_device", return_value=MagicMock())
    @patch("torch.distributed.get_process_group_ranks", return_value={0: 0, 1: 1})
    @patch("torch.distributed.get_group_rank", return_value={0: 0, 1: 1})
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.npu.device")
    def test_all_to_all_with_sizes(self, *_):
        def patched_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
            output_tensor_list[:] = [torch.tensor([10, 20]), torch.tensor([50, 60])]

        torch.distributed.all_to_all = patched_all_to_all

        scatter_sizes = [2, 2]
        gather_sizes = [2, 2]
        input_ = torch.tensor([10, 20, 30, 40])

        with patch.dict(dist.distributed_c10d._world.pg_map, {dist.group.WORLD: MagicMock()}, clear=False):
            comm = NPUCommunicator(cpu_group=dist.group.WORLD)

        output = comm.all_to_all(input_, scatter_sizes=scatter_sizes, gather_sizes=gather_sizes)

        assert output.tolist() == [10, 20, 50, 60]

    @patch("vllm.config.get_current_vllm_config", return_value=None)
    @patch("torch.npu.current_device", return_value=MagicMock())
    @patch("torch.npu.set_device", return_value=MagicMock())
    @patch("torch.distributed.get_process_group_ranks", return_value={0: 0, 1: 1})
    @patch("torch.distributed.get_group_rank", return_value={0: 0, 1: 1})
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.npu.device")
    def test_all_to_all_without_sizes(self, *_):
        def patched_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
            output_tensor_list[:] = [torch.tensor([[10, 20]]), torch.tensor([[50, 60]])]

        torch.distributed.all_to_all = patched_all_to_all

        input_ = torch.tensor([[10, 20], [30, 40]])

        with patch.dict(dist.distributed_c10d._world.pg_map, {dist.group.WORLD: MagicMock()}, clear=False):
            comm = NPUCommunicator(cpu_group=dist.group.WORLD)
            output = comm.all_to_all(input_, scatter_dim=0, gather_dim=0)

        assert output.tolist() == [[10, 20], [50, 60]]

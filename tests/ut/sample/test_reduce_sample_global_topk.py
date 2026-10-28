import torch


def test_global_topk_must_be_selected_after_tp_candidate_gather():
    """A local top-k union is not necessarily the global top-k."""
    # TP shards, rows are one request. With local k=2, the global top-2 are
    # shard-0 token 0 and shard-1 token 0; the second local candidate on each
    # shard must not survive the global merge.
    shard_values = torch.tensor([[10.0, 1.0], [9.0, 8.0]])
    shard_ids = torch.tensor([[0, 1], [2, 3]])
    values, order = torch.topk(shard_values.reshape(1, -1), k=2, dim=-1)
    ids = shard_ids.reshape(1, -1).gather(-1, order)
    assert torch.equal(values, torch.tensor([[10.0, 9.0]]))
    assert torch.equal(ids, torch.tensor([[0, 2]]))


def test_global_logsoftmax_normalizes_across_tp_shards():
    shards = [torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, -1.0]])]
    full = torch.cat(shards, dim=-1)
    logprobs = full.log_softmax(dim=-1)
    torch.testing.assert_close(logprobs.exp().sum(dim=-1), torch.ones(1))
    assert float(logprobs[0, 0]) < 0.0

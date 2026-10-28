import numpy as np
import vllm.v1.sample.rejection_sampler as rs

from vllm_ascend.sample.rejection_sampler import apply_sampling_constraints, expand_batch_to_tokens, rejection_sample

# TODO: delete this patch after apply_sampling_constraints and rejection_sample
#   are extracted to as class func of RejectionSampler
rs.apply_sampling_constraints = apply_sampling_constraints
rs.rejection_sample = rejection_sample
rs.expand_batch_to_tokens = expand_batch_to_tokens


def _parse_output_contiguous_prefix(
    output_token_ids,
    vocab_size,
    discard_req_indices=(),
    logprobs_tensors=None,
):
    """Parse async rejection output while enforcing prefix semantics.

    The rejection kernels are expected to write accepted/recovered tokens as a
    contiguous prefix and leave the remainder at ``PLACEHOLDER_TOKEN_ID``.
    During graph-buffer reuse or a transient device-to-host race, a stale token
    can appear after the first placeholder.  Treat that suffix as unavailable
    instead of feeding a non-contiguous sequence into the scheduler; this is
    also the contract consumed by PLE committed-token history.
    """
    output_token_ids_np = output_token_ids.cpu().numpy()
    valid_mask = (
        (output_token_ids_np != rs.PLACEHOLDER_TOKEN_ID)
        & (output_token_ids_np >= 0)
        & (output_token_ids_np < vocab_size)
    )
    # ``logical_and.accumulate`` keeps only the prefix before the first
    # placeholder/invalid token in each request row.
    valid_prefix_mask = np.logical_and.accumulate(valid_mask, axis=1)

    output_logprobs = None
    if logprobs_tensors is not None:
        cu_num_tokens = [0] + valid_prefix_mask.sum(axis=1).cumsum().tolist()
        filtered_tensors = logprobs_tensors.filter(valid_prefix_mask.flatten())
        output_logprobs = filtered_tensors.tolists(cu_num_tokens)

    if len(discard_req_indices) > 0:
        valid_prefix_mask[list(discard_req_indices)] = False
    outputs = [
        row[valid_prefix_mask[i]].tolist()
        for i, row in enumerate(output_token_ids_np)
    ]
    return outputs, output_logprobs


# ``model_runner_v1`` calls the upstream class directly.  Install the narrow
# parser override alongside the existing function patches so both sync and
# async output paths share the same contiguous-prefix contract.
rs.RejectionSampler.parse_output = staticmethod(_parse_output_contiguous_prefix)

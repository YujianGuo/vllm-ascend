# SPDX-License-Identifier: Apache-2.0

import hashlib
import random
from copy import copy

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer, greedy_sample
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer
from vllm_ascend.spec_decode.qwen_mtp_probability import sample_probability


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)


class AscendQwen4ExpMTPProposer(Qwen4ExpMTPProposer, AscendSpecDecodeBaseProposer):
    """Ascend proposer for Qwen4Exp multi-stream MTP."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)

        # vLLM allocates the base proposer buffer from ModelConfig directly and
        # no longer calls this proposer's _get_hidden_size() hook. Qwen4Exp MTP
        # consumes the full HC stream, so replace a single-stream allocation.
        qwen_hidden_size = self._get_hidden_size()
        if self.hidden_size != qwen_hidden_size:
            self.hidden_size = qwen_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._step_metadata_builders: dict[tuple[str, int], AttentionMetadataBuilder] = {}
        self.compact_draft_steps = self.dcp_size == 1 and not self.parallel_drafting
        self.use_repeated_proposals = get_ascend_config().qwen_mtp_repeated_proposals and self.dcp_size == 1
        self._qwen_probabilistic = self._enable_probabilistic_draft_probs
        self._qwen_draft_rng = {}
        if self._qwen_probabilistic:
            if not self.compact_draft_steps or self.use_repeated_proposals:
                raise ValueError("Probabilistic Qwen MTP requires compact steps and repeat=false")
            batch_capacity = self.runner.max_num_reqs
            vocab_size = self.runner.model_config.get_vocab_size()
            self._qwen_prob_buffer = torch.zeros(
                (batch_capacity, self.num_speculative_tokens, vocab_size),
                dtype=torch.float32,
                device=device,
            )
            self._qwen_temperature = torch.zeros(batch_capacity, dtype=torch.float32, device=device)
            self._qwen_uniforms = torch.full(
                (batch_capacity, self.num_speculative_tokens),
                0.5,
                dtype=torch.float32,
                device=device,
            )

    def _run_merged_draft(self, *args, **kwargs):
        # This Python counter describes the statically unrolled draft graph.
        # Every captured graph writes the same persistent per-step q buffers.
        self._qwen_probability_step = 0
        return AscendSpecDecodeBaseProposer._run_merged_draft(self, *args, **kwargs)

    def compute_draft_token_ids(self, hidden_states):
        if not self._qwen_probabilistic:
            return AscendSpecDecodeBaseProposer.compute_draft_token_ids(self, hidden_states)
        step = self._qwen_probability_step
        self._qwen_probability_step += 1
        if step >= self.num_speculative_tokens:
            raise RuntimeError("More probability writes than configured draft steps")
        logits = self.model.compute_logits(hidden_states)
        if logits.shape[0] > self.runner.max_num_reqs:
            # An idle DP's eager companion can derive more placeholder rows
            # from the active rank's prefill token count than live requests.
            # These rows have no proposal distribution consumed by target.
            return greedy_sample(logits)
        if get_ascend_config().enable_reduce_sample:
            logits = get_tp_group().all_gather(logits, dim=-1)
        vocab_size = self.runner.model_config.get_vocab_size()
        logits = logits[:, :vocab_size]
        batch = logits.shape[0]
        token_ids, probabilities = sample_probability(
            logits,
            self._qwen_temperature[:batch],
            self._qwen_uniforms[:batch, step],
        )
        self._qwen_prob_buffer[:batch, step].copy_(probabilities)
        return token_ids

    def _propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        *args,
        **kwargs,
    ):
        if self._qwen_probabilistic:
            # Generate one uniform per request and draft step, outside graphs.
            # Dedicated CPU RNGs do not consume target sampling RNG state.
            # Seed/request identity is identical across TP ranks.
            self._qwen_draft_rng = {
                key: value for key, value in self._qwen_draft_rng.items() if key in self.runner.requests
            }
            request_ids = self.runner.input_batch.req_ids
            uniforms = []
            for request_id in request_ids:
                if request_id not in self._qwen_draft_rng:
                    seed = self.runner.requests[request_id].sampling_params.seed
                    if seed is None:
                        seed = int.from_bytes(hashlib.blake2b(request_id.encode(), digest_size=8).digest(), "little")
                    self._qwen_draft_rng[request_id] = random.Random(int(seed) ^ 0x514D5450)
                rng = self._qwen_draft_rng[request_id]
                uniforms.append([rng.random() for _ in range(self.num_speculative_tokens)])
            batch = len(request_ids)
            self._qwen_uniforms.fill_(0.5)
            # Constructing a device tensor from a Python list performs a
            # blocking H2D copy after the target forward. A fresh pinned
            # allocation lets the copy stay ordered on the device stream
            # without blocking the host; its allocator tracks copy lifetime.
            uniforms_cpu = torch.tensor(
                uniforms, dtype=torch.float32, pin_memory=self._qwen_uniforms.device.type == "npu"
            ).clamp_max_(1 - 2**-24)
            self._qwen_uniforms[:batch].copy_(
                uniforms_cpu, non_blocking=True
            )
            self._qwen_temperature.zero_()
            if sampling_metadata.temperature is not None:
                self._qwen_temperature[:batch].copy_(sampling_metadata.temperature[:batch])
            self._last_draft_probs = None
        use_repetition = self.use_repeated_proposals and sampling_metadata.all_greedy
        if use_repetition:
            indices = token_indices_to_sample
            if indices is None:
                indices = common_attn_metadata.query_start_loc[1:] - 1
            # These are adjacent committed target tokens, including after rejection.
            # Gather before the draft forward reuses persistent input buffers.
            previous_ids = target_token_ids[indices]
        proposals = AscendSpecDecodeBaseProposer._propose(
            self,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            target_model_batch_desc,
            sampling_metadata,
            *args,
            **kwargs,
        )
        if use_repetition:
            repeated = (previous_ids == next_token_ids) & (next_token_ids >= 0)
            # Keep all candidates subject to the normal target rejection sampler.
            proposals = torch.where(repeated[:, None], next_token_ids[:, None], proposals)
        if self._qwen_probabilistic and not sampling_metadata.all_greedy:
            # Dummy companion graphs reuse the capture output buffers. Keep
            # the last real requests' q immutable until their target verify.
            self._last_draft_probs = self._qwen_prob_buffer[: len(request_ids)].clone()
        return proposals

    def build_dummy_draft_attn_metadata(self, common, num_tokens, num_reqs):
        if not self.compact_draft_steps:
            return super().build_dummy_draft_attn_metadata(common, num_tokens, num_reqs)
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            self._per_group_block_tables[gid] = self.runner.input_batch.block_table[gid].get_device_tensor()[:num_reqs]
        common = copy(common)
        self.slot_mapping_group[0].fill_(-1)
        common.slot_mapping = self.slot_mapping_group[0]
        self.seq_lens_group[0][:num_reqs].copy_(common.seq_lens)
        common.seq_lens = self.seq_lens_group[0][:num_reqs]
        self.query_start_loc_group[0][: num_reqs + 1].copy_(common.query_start_loc)
        common.query_start_loc = self.query_start_loc_group[0][: num_reqs + 1]
        steps, _ = self.build_draft_attn_metadata(common, num_tokens, num_tokens, num_reqs)
        positions = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        steps.extend(
            self.build_later_draft_attn_metadata(
                common,
                num_reqs,
                num_tokens,
                positions,
                CUDAGraphMode.FULL,
            )
        )
        return steps

    def _build_qwen_draft_metadata(self, common, draft_index, actual_reqs):
        if not self._step_metadata_builders:
            for group in self.draft_attn_groups:
                original = group.get_metadata_builder()
                for index in range(self.num_speculative_tokens):
                    builder = copy(original)
                    if isinstance(builder, QSAMetadataBuilder):
                        # All steps are prepared before any draft forward.
                        # Sharing these buffers overwrites step 0 with step 2.
                        for name in (
                            "token_to_req_buffer",
                            "slot_mapping_buffer",
                            "logical_positions_buffer",
                            "k_work_metadata_buffer",
                        ):
                            setattr(builder, name, torch.empty_like(getattr(original, name)))
                    self._step_metadata_builders[group.layer_names[0], index] = builder
        layers = {}
        for group in self.draft_attn_groups:
            group_common = copy(common)
            # Common metadata may cache the target's multi-token row mapping.
            # Each draft step and owner must populate its own persistent buffer.
            group_common._token_to_req_indices_cache = None
            group_common.block_table_tensor = self._per_group_block_tables[group.kv_cache_group_id][: common.num_reqs]
            builder = self._step_metadata_builders[group.layer_names[0], draft_index]
            kwargs = {"num_reqs_actual": actual_reqs} if isinstance(builder, QSAMetadataBuilder) else {}
            metadata = builder.build_for_drafting(group_common, draft_index, **kwargs)
            for name in group.layer_names:
                layers[name] = metadata
        return layers

    def build_draft_attn_metadata(
        self, common_attn_metadata, num_input_tokens, num_actual_tokens, num_reqs_actual=None
    ):
        if self.use_cuda_graph and not self.compact_draft_steps:
            return super().build_draft_attn_metadata(
                common_attn_metadata, num_input_tokens, num_actual_tokens, num_reqs_actual
            )
        if self.use_cuda_graph:
            common_attn_metadata = copy(common_attn_metadata)
            common_attn_metadata.num_actual_tokens = num_input_tokens
        layers = self._build_qwen_draft_metadata(common_attn_metadata, 0, num_reqs_actual)
        return [layers], layers[self.draft_attn_groups[0].layer_names[0]]

    def build_later_draft_attn_metadata(
        self, common, batch_size, input_batch_size, used_update_positions, aclgraph_runtime_mode, **kwargs
    ):
        if self.use_cuda_graph and not self.compact_draft_steps:
            return super().build_later_draft_attn_metadata(
                common, batch_size, input_batch_size, used_update_positions, aclgraph_runtime_mode, **kwargs
            )
        capacity = (
            max(input_batch_size // (self.num_speculative_tokens + 1), 1)
            if aclgraph_runtime_mode == CUDAGraphMode.FULL
            else batch_size
        )
        steps = []
        for index in range(1, self.num_speculative_tokens):
            positions = used_update_positions[0] if used_update_positions.ndim > 1 else used_update_positions
            position = positions + index
            valid = position < self.max_model_len
            position = torch.where(valid, position, 0)
            step = copy(common)
            step.num_reqs = capacity
            step.num_actual_tokens = capacity
            step.num_input_tokens = input_batch_size
            step.max_query_len = 1
            step.decode_token_per_req = 1
            step.query_start_loc = self.query_start_loc_group[index][: capacity + 1]
            step.query_start_loc.copy_(self.arange[: capacity + 1])
            step.query_start_loc_cpu = torch.arange(capacity + 1, dtype=torch.int32)
            step.seq_lens = self.seq_lens_group[index][:capacity]
            # Sampling positions already exclude rejected target tokens.
            # Advance once per draft step, independently of the cache owners.
            step.seq_lens[:batch_size].copy_(position + 1)
            step.seq_lens[batch_size:].fill_(0)
            step.seq_lens_cpu = None
            step._seq_lens_cpu = None
            table = self._per_group_block_tables[self.kv_cache_gid][:batch_size]
            page = table.gather(1, (position // self.block_size).long().view(-1, 1)).view(-1)
            slot = page * self.block_size + position % self.block_size
            step.slot_mapping = self.slot_mapping_group[index]
            step.slot_mapping[:batch_size].copy_(torch.where(valid, slot, -1))
            step.slot_mapping[batch_size:].fill_(-1)
            steps.append(self._build_qwen_draft_metadata(step, index, batch_size))
        return steps

    def _get_hidden_size(self) -> int:
        # ModelConfig.get_hidden_size() resolves the multimodal outer width for
        # Qwen3.8-Flash-Next. The MTP feedback is the text backbone's HC stream.
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)

    def maybe_pad_and_reduce(self, hidden_states, positions):
        # Qwen4Exp target and draft share the same TP group and HC width.
        return hidden_states, positions

    def maybe_all_gather_and_unpad(self, last_hidden_states, positions, hidden_states=None):
        return last_hidden_states, positions, hidden_states

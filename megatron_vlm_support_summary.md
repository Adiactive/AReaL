# Summary of Changes: Megatron Engine Support for Qwen2.5-VL and Qwen3-VL

This document outlines the modifications made to the AReaL framework to support training Qwen2.5-VL and Qwen3-VL models using the Megatron engine.

## 1. Ported VLM Megatron Modules
- Replicated the `Qwen2_5VLModel` Megatron wrappers from the `verl` framework into `areal/models/mcore/qwen2_5_vl`.
- Adjusted the implementation to use AReaL's native context parallelism primitives (`preprocess_packed_seqs_context_parallel`) instead of relying on `verl`'s external utilities.

## 2. Added Model Registry & Configuration Converts
- Created `areal/models/mcore/qwen2_5_vl_config.py` to extract and correctly map Qwen-specific configuration parameters (such as `mrope_section` mapping) to `TransformerConfig`.
- Updated `areal/models/mcore/registry.py` to natively map `Qwen2_5_VLForConditionalGeneration` and `Qwen3VLForConditionalGeneration` to this newly created custom Megatron wrapper module.

## 3. Bypassed `mbridge` for Vision Models
- Modified `areal/engine/megatron_engine.py` to detect if the loaded model is a vision model.
- Conditionally bypassed the `mbridge` generation logic for vision models, allowing the engine to directly initialize the native AReaL Megatron VLM models.

## 4. Enabled Multi-modal Data Forwarding
- Modified `_prepare_mb_list` inside `areal/engine/megatron_engine.py` to extract and flatten multimodal properties (`pixel_values`, `image_grid_thw`, `video_grid_thw`, `pixel_values_videos`) directly into micro-batches, ensuring parity with how the FSDP engine unpacks vision items from datasets.
- Updated `packed_context_parallel_forward` in `areal/engine/megatron_utils/packed_context_parallel.py` to natively forward all multi-modal parameters down to the underlying transformer block.

## 5. Fixed Sequence Packing (Crucial)
- Modified `preprocess_packed_seqs_context_parallel` to support processing `n-dimensional` tensors, fixing an assumption that it strictly received `1D` tensors.
- Instructed the engine **not** to preemptively pack `input_ids` when running vision models, as sequence chunking splits image index layouts across GPUs. Instead, the raw padded arrays and `cu_seqlens` are forwarded so `Qwen2_5VLModel` can handle image projection first, and sequentially pack `combined_embeddings` just before passing them to the backend language model.

"""Driving-clip input preparation.

The upstream R1Adapter hardcodes a single CLIP_ID. Repetitions need a
different clip each, so inputs are built directly from the dataset
interface instead.
"""
from __future__ import annotations

from .paths import bootstrap

bootstrap()

import physical_ai_av  # noqa: E402
from alpamayo_memopt.models import r1 as r1_module  # noqa: E402
from alpamayo_r1 import helper  # noqa: E402
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402


def prepare_inputs_for_clip(loaded, clip_id: str, device: str):
    """Build model inputs for an arbitrary clip_id."""
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=r1_module.HF_CACHED_REVISION)
    data = load_physical_aiavdataset(clip_id, t0_us=r1_module.T0_US, avdi=avdi)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(loaded.model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return helper.to_device({
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }, device)

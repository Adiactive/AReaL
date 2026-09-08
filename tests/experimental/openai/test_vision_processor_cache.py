# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy

import pytest
import torch

from tests.experimental.openai.test_vision_export import _EchoEngine
from tests.experimental.openai.vision_stubs import (
    IMAGE_PAD_ID,
    PATCHES_PER_IMAGE,
    StubProcessor,
    StubTokenizer,
    image_data_uri,
    make_image,
    user_message_with_image,
)

from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.client import (
    _acached_vision_prompt,
    concat_vision_prompt_with_parent,
)
from areal.infra.processor_cache import ProcessorCallCache


class CountingProcessor(StubProcessor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return super().__call__(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hf", "concat"])
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_grouped_proxy_prompts_share_processing_and_keep_collapsed_tokens(
    mode, api
):
    tokenizer = StubTokenizer()
    processor = CountingProcessor()
    cache = ProcessorCallCache()
    clients = [
        ArealOpenAI(
            engine=_EchoEngine(tokenizer),
            tokenizer=tokenizer,
            processor=processor,
            chat_template_type=mode,
            api_key="test",
        )
        for _ in range(2)
    ]
    image = make_image(31)

    async def generate(client):
        if api == "chat":
            response = await client.chat.completions.create(
                messages=[user_message_with_image(image, "describe")],
                model="default",
                max_completion_tokens=8,
                processor_cache=cache,
            )
        else:
            response = await client.responses.create(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_data_uri(image)},
                            {"type": "input_text", "text": "describe"},
                        ],
                    }
                ],
                model="default",
                max_output_tokens=8,
                tools=[],
                processor_cache=cache,
            )
        return client.get_interaction(response.id)

    try:
        first, second = await asyncio.gather(*(generate(client) for client in clients))
        assert processor.calls == 1
        assert first.multi_modal_input is not second.multi_modal_input
        assert first.multi_modal_input[0] is not second.multi_modal_input[0]
        assert (
            first.multi_modal_input[0]["pixel_values"]
            is second.multi_modal_input[0]["pixel_values"]
        )
        for interaction in (first, second):
            assert (
                interaction.model_response.input_tokens.count(IMAGE_PAD_ID)
                == PATCHES_PER_IMAGE
            )
            if mode == "concat":
                assert interaction.collapsed_input_ids.count(IMAGE_PAD_ID) == 1
        first.multi_modal_input[0]["extra"] = torch.tensor([1])
        assert "extra" not in second.multi_modal_input[0]
    finally:
        await asyncio.gather(*(client.close() for client in clients))


@pytest.mark.asyncio
async def test_concat_cache_distinguishes_actual_parent_tokens():
    tokenizer = StubTokenizer()
    processor = CountingProcessor()
    client = ArealOpenAI(
        engine=_EchoEngine(tokenizer),
        tokenizer=tokenizer,
        processor=processor,
        chat_template_type="concat",
        api_key="test",
    )
    try:
        response = await client.chat.completions.create(
            messages=[user_message_with_image(make_image(7), "describe")],
            model="default",
            max_completion_tokens=8,
        )
        parent = client.get_interaction(response.id)
        equal_parent = deepcopy(parent)
        changed_parent = deepcopy(parent)
        changed_parent.model_response.output_tokens[0] = 77
        messages = [{"role": "user", "content": "continue"}]
        cache = ProcessorCallCache()
        before = processor.calls

        async def prepare(candidate):
            return await _acached_vision_prompt(
                cache,
                concat_vision_prompt_with_parent,
                messages,
                candidate,
                tokenizer,
                processor,
            )

        first, equal, changed = await asyncio.gather(
            prepare(parent), prepare(equal_parent), prepare(changed_parent)
        )
        assert processor.calls - before == 2
        assert first.input_ids == equal.input_ids
        assert first.input_ids != changed.input_ids
        assert (
            first.multi_modal_input[0]["pixel_values"]
            is equal.multi_modal_input[0]["pixel_values"]
        )
        expected = list(equal.collapsed_input_ids)
        first.collapsed_input_ids.append(999)
        assert equal.collapsed_input_ids == expected
    finally:
        await client.close()

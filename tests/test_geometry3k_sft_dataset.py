import pytest

from areal.dataset.geometry3k import (
    _format_sft_conversation,
    _make_sft_loss_mask,
)


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        prompt = "<user>image question</user><assistant>"
        if add_generation_prompt:
            return prompt
        return f"{prompt}{messages[-1]['content']}</assistant>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


def test_format_vlm_sft_text_chat_response_is_exact_suffix():
    """The supervised response includes the answer and assistant terminator."""
    tokenizer = _Tokenizer()

    full_text, response_length = _format_sft_conversation(
        tokenizer,
        problem="image question",
        answer="3",
        chat_template_kwargs={"enable_thinking": False},
    )

    response = "3</assistant>"
    assert full_text.endswith(response)
    assert response_length == len(response.encode())


def test_format_vlm_sft_text_raw_response_includes_eos():
    tokenizer = _Tokenizer()
    tokenizer.eos_token = "</s>"

    full_text, response_length = _format_sft_conversation(
        tokenizer,
        problem="image question",
        answer="3",
        chat_template_kwargs=None,
    )

    assert full_text == "image question3</s>"
    assert response_length == len(b"3</s>")


def test_make_sft_loss_mask_supervises_response_suffix():
    """The answer token and terminator are supervised, not the prompt."""
    loss_mask = _make_sft_loss_mask(input_length=6, response_length=2)

    assert loss_mask == [0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize("response_length", [0, 7])
def test_make_sft_loss_mask_rejects_invalid_response_length(response_length):
    """Invalid response boundaries fail before reaching model training."""
    with pytest.raises(ValueError, match="Invalid SFT response length"):
        _make_sft_loss_mask(input_length=6, response_length=response_length)

# SPDX-License-Identifier: Apache-2.0
"""Install beside Gym's pinned vllm_model/app.py as areal_app.py."""

from typing import Any

from app import VLLMModel
from fastapi import Request
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


class AReaLRoutedModel(VLLMModel):
    """Keep the opaque attempt model while retaining reference preprocessing."""

    def _preprocess_chat_completion_create_params(
        self, request: Request, body_dict: dict[str, Any]
    ) -> dict[str, Any]:
        route = body_dict["model"]
        result = super()._preprocess_chat_completion_create_params(request, body_dict)
        result["model"] = route
        return result


if __name__ == "__main__":
    AReaLRoutedModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AReaLRoutedModel.run_webserver()

# SPDX-License-Identifier: Apache-2.0

"""Core components for AREAL."""

import importlib

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "RolloutController",
    "TrainController",
    "workflow_context",
    "Platform",
    "current_platform",
    "is_npu_available",
    "LocalScheduler",
    "RayScheduler",
    "SlurmScheduler",
    "LocalLauncher",
    "RayLauncher",
    "SlurmLauncher",
    "SGLangServerWrapper",
    "vLLMServerWrapper",
]

_LAZY_IMPORTS = {
    "LocalLauncher": "areal.infra.launcher",
    "LocalScheduler": "areal.infra.scheduler",
    "Platform": "areal.infra.platforms",
    "RayLauncher": "areal.infra.launcher",
    "RayScheduler": "areal.infra.scheduler",
    "RemoteInfBackendProtocol": "areal.infra.remote_inf_engine",
    "RemoteInfEngine": "areal.infra.remote_inf_engine",
    "RolloutController": "areal.infra.controller",
    "SGLangServerWrapper": "areal.infra.launcher",
    "SlurmLauncher": "areal.infra.launcher",
    "SlurmScheduler": "areal.infra.scheduler",
    "StalenessManager": "areal.infra.staleness_manager",
    "TrainController": "areal.infra.controller",
    "WorkflowExecutor": "areal.infra.workflow_executor",
    "check_trajectory_format": "areal.infra.workflow_executor",
    "current_platform": "areal.infra.platforms",
    "is_npu_available": "areal.infra.platforms",
    "vLLMServerWrapper": "areal.infra.launcher",
}


def __getattr__(name: str):
    if name == "workflow_context":
        module = importlib.import_module("areal.infra.workflow_context")
        globals()[name] = module
        return module
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)

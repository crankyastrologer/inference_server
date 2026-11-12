"""Engine: generation loop (scheduler/batching to come)."""

from nanoserve.engine.sequence import GenerationResult, SamplingParams

__all__ = ["generate", "GenerationResult", "SamplingParams"]


def __getattr__(name):
    if name == "generate":
        from nanoserve.engine.generate import generate

        return generate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

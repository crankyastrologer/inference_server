"""Compare a candidate probe dict against golden activations.

Two ladders are compared probe-by-probe. A failure names the first stage that
diverges, which is the whole point: "logits are 0.3 off" is useless, "layer7
attention output is 0.3 off while layer7.q_rope is exact" points at the bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# (rtol, atol) presets. float32 is the exactness reference; bf16 is deployment.
TOLERANCES = {
    "float32": (1e-3, 1e-4),
    "bfloat16": (1e-2, 2e-2),
}


@dataclass
class ProbeDiff:
    name: str
    ok: bool
    shape_ok: bool = True
    candidate_shape: tuple = ()
    golden_shape: tuple = ()
    max_abs: float = float("nan")
    mean_abs: float = float("nan")
    max_rel: float = float("nan")
    note: str = ""


def _diff(name: str, cand: torch.Tensor, gold: torch.Tensor, rtol: float, atol: float) -> ProbeDiff:
    if tuple(cand.shape) != tuple(gold.shape):
        return ProbeDiff(
            name=name,
            ok=False,
            shape_ok=False,
            candidate_shape=tuple(cand.shape),
            golden_shape=tuple(gold.shape),
            note="shape mismatch",
        )
    a = cand.detach().to(device="cpu", dtype=torch.float32)
    b = gold.detach().to(device="cpu", dtype=torch.float32)
    abs_err = (a - b).abs()
    max_abs = float(abs_err.max()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean()) if abs_err.numel() else 0.0
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((abs_err / denom).max()) if abs_err.numel() else 0.0
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    return ProbeDiff(
        name=name,
        ok=ok,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
    )


def compare_activations(
    candidate: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    *,
    dtype: str = "float32",
    rtol: float | None = None,
    atol: float | None = None,
    per_probe: dict[str, tuple[float, float]] | None = None,
) -> list[ProbeDiff]:
    """Diff every golden probe against the candidate. Extras/missing fail."""
    base_rtol, base_atol = TOLERANCES.get(dtype, TOLERANCES["float32"])
    rtol = base_rtol if rtol is None else rtol
    atol = base_atol if atol is None else atol
    per_probe = per_probe or {}

    diffs: list[ProbeDiff] = []
    for name in golden:
        if name not in candidate:
            diffs.append(ProbeDiff(name=name, ok=False, note="missing in candidate"))
            continue
        p_rtol, p_atol = per_probe.get(name, (rtol, atol))
        diffs.append(_diff(name, candidate[name], golden[name], p_rtol, p_atol))
    for name in candidate:
        if name not in golden:
            diffs.append(ProbeDiff(name=name, ok=False, note="extra in candidate"))
    return diffs


def first_failure(diffs: list[ProbeDiff]) -> ProbeDiff | None:
    for d in diffs:
        if not d.ok:
            return d
    return None


def format_report(diffs: list[ProbeDiff]) -> str:
    width = max((len(d.name) for d in diffs), default=10)
    width = max(width, len("probe"))
    header = f"{'probe'.ljust(width)}  ok   max_abs    mean_abs   max_rel   note"
    lines = [header, "-" * len(header)]
    for d in diffs:
        lines.append(
            f"{d.name.ljust(width)}  {'Y' if d.ok else 'N':<3}  "
            f"{d.max_abs:9.3e}  {d.mean_abs:9.3e}  {d.max_rel:9.3e}  {d.note}"
        )
    n_bad = sum(1 for d in diffs if not d.ok)
    lines.append(f"\n{len(diffs) - n_bad}/{len(diffs)} probes within tolerance")
    return "\n".join(lines)


def assert_activations_close(
    candidate: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    *,
    dtype: str = "float32",
    rtol: float | None = None,
    atol: float | None = None,
    per_probe: dict[str, tuple[float, float]] | None = None,
) -> list[ProbeDiff]:
    diffs = compare_activations(
        candidate, golden, dtype=dtype, rtol=rtol, atol=atol, per_probe=per_probe
    )
    bad = first_failure(diffs)
    if bad is not None:
        raise AssertionError(
            f"golden mismatch at '{bad.name}': {bad.note or ''}\n\n"
            + format_report(diffs)
        )
    return diffs

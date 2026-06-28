from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitorSpec:
    name: str
    description: str
    uses_policy: bool = False
    uses_sampler: bool = False


SUPPORTED_COMPETITORS = {
    "baseline": CompetitorSpec(
        name="baseline",
        description="Standard random crop/flip plus optional cutout training baseline.",
    ),
    "metaaugment": CompetitorSpec(
        name="metaaugment",
        description="MetaAugment sample-aware augmentation policy learner.",
        uses_policy=True,
        uses_sampler=True,
    ),
    "standard": CompetitorSpec(
        name="baseline",
        description="Alias for the standard random crop/flip plus optional cutout baseline.",
    ),
}


def normalize_competitor_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def get_competitor(name: str) -> CompetitorSpec:
    normalized = normalize_competitor_name(name)
    if normalized not in SUPPORTED_COMPETITORS:
        supported = ", ".join(sorted(SUPPORTED_COMPETITORS))
        raise ValueError(f"Unsupported competitor {name!r}. Supported competitors: {supported}")
    return SUPPORTED_COMPETITORS[normalized]

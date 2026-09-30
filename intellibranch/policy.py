from dataclasses import dataclass


@dataclass
class DispatchPolicy:
    """Defines 3-tier confidence criteria, multi-intent threshold, and OOD entropy boundary."""
    high_threshold: float = 0.75      # Minimum confidence for definite execution
    low_threshold: float = 0.40       # Minimum confidence below which request is isolated to Fallback
    margin_cutoff: float = 0.15       # Minimum required gap between Top-1 and Top-2
    max_entropy: float = 2.0          # Maximum allowable prediction entropy before triggering OOD Fallback
    pipeline_threshold: float = 0.30  # Minimum secondary confidence to qualify for multi-intent pipeline
    min_log_sum_exp: float = 0.0      # Minimum log-sum-exp energy boundary before OOD isolation (0 disables)


def default_dispatch_policy() -> DispatchPolicy:
    """Creates standard production-ready 3-tier routing criteria."""
    return DispatchPolicy(
        high_threshold=0.75,
        low_threshold=0.40,
        margin_cutoff=0.15,
        max_entropy=2.0,
        pipeline_threshold=0.30,
        min_log_sum_exp=0.0,
    )

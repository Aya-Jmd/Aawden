"""Conservative launch authorization for a single bounded Beam experiment."""
import math


def reserve_job(available_cents, spec, policy):
    timeout = spec["timeout"]
    if not isinstance(available_cents, int) or available_cents < 0:
        raise ValueError("A verified nonnegative integer credit balance is required")
    if not 0 < timeout <= policy["maximum_job_runtime_seconds"]:
        raise ValueError("The requested runtime exceeds the compute budget policy")
    if spec["gpu"] not in {"RTX4090", "RTX5090", "A10G"}:
        raise ValueError("No conservative cost bound is configured for this GPU")
    cpu, ram = float(spec["cpu"]), float(spec["memory"].removesuffix("Gi"))
    if cpu <= 0 or ram <= 0:
        raise ValueError("Positive CPU and RAM allocations are required")
    floor = round(100 * policy["minimum_beam_balance_usd"])
    margin = round(100 * policy["safety_margin_usd"])
    if floor < 1800 or margin < 100:
        raise ValueError("The protected $18 floor and $1 margin may not be reduced")
    # Upper planning rates, deliberately above the rates used by these experiments.
    hourly_upper_bound = 1.20 + .40 * cpu + .025 * ram
    reservation = math.ceil(100 * hourly_upper_bound * timeout / 3600)
    record = {"available_cents": available_cents, "minimum_balance_cents": floor,
              "safety_margin_cents": margin, "maximum_job_reserve_cents": reservation,
              "passed": available_cents - reservation >= floor + margin}
    if not record["passed"]:
        raise RuntimeError("The bounded job would encroach on the credit floor and safety margin")
    return record

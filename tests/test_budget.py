import pytest
from security_wsp.budget import reserve_job

POLICY = {"minimum_beam_balance_usd": 18, "safety_margin_usd": 1, "maximum_job_runtime_seconds": 2100}
SPEC = {"gpu": "RTX4090", "timeout": 900, "cpu": 2, "memory": "32Gi"}


def test_budget_reserves_full_timeout_and_protects_exact_boundary():
    record = reserve_job(1970, SPEC, POLICY)
    assert record["maximum_job_reserve_cents"] == 70
    assert record["passed"]
    with pytest.raises(RuntimeError, match="floor"):
        reserve_job(1969, SPEC, POLICY)
    with pytest.raises(RuntimeError, match="floor"):
        reserve_job(1800, SPEC, POLICY)


def test_unknown_costs_long_jobs_and_weakened_floor_are_rejected():
    for change in ({"gpu": "A100-80"}, {"timeout": 2101}, {"cpu": 0}):
        with pytest.raises(ValueError):
            reserve_job(3000, dict(SPEC, **change), POLICY)
    with pytest.raises(ValueError, match="may not be reduced"):
        reserve_job(3000, SPEC, dict(POLICY, minimum_beam_balance_usd=17))


def test_larger_resource_reservations_can_block_launch():
    assert reserve_job(1970, SPEC, POLICY)["passed"]
    with pytest.raises(RuntimeError):
        reserve_job(1970, dict(SPEC, timeout=2100), POLICY)

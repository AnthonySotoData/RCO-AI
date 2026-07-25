from src.models.schemas import OperationalMetrics, RevenueCycleRecord


def _safe_percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def calculate_operational_metrics(
    record: RevenueCycleRecord,
) -> OperationalMetrics:
    clearance_rate = _safe_percentage(
        record.cleared_accounts,
        record.scheduled_accounts,
    )

    authorization_completion_rate = _safe_percentage(
        record.authorizations_completed,
        record.authorization_required,
    )

    denial_rate = _safe_percentage(
        record.denials,
        record.scheduled_accounts,
    )

    accounts_per_fte = round(
        record.scheduled_accounts / record.staff_fte,
        2,
    )

    return OperationalMetrics(
        clearance_rate=clearance_rate,
        authorization_completion_rate=authorization_completion_rate,
        denial_rate=denial_rate,
        revenue_at_risk=round(record.denied_charges, 2),
        accounts_per_fte=accounts_per_fte,
    )
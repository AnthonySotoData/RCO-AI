from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_calculate_metrics_returns_expected_values() -> None:
    response = client.post(
        "/metrics",
        json={
            "record": {
                "record_date": "2026-07-22",
                "specialty": "Neurology",
                "payer": "Aetna",
                "scheduled_accounts": 100,
                "cleared_accounts": 92,
                "authorization_required": 40,
                "authorizations_completed": 35,
                "denials": 5,
                "total_charges": 125000,
                "denied_charges": 8000,
                "average_turnaround_hours": 31.5,
                "work_queue_volume": 120,
                "staff_fte": 3,
            }
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["specialty"] == "Neurology"
    assert body["metrics"]["clearance_rate"] == 92.0
    assert body["metrics"]["authorization_completion_rate"] == 87.5
    assert body["metrics"]["denial_rate"] == 5.0
    assert body["metrics"]["revenue_at_risk"] == 8000.0
    assert body["metrics"]["accounts_per_fte"] == 33.33


def test_metrics_rejects_zero_staff_fte() -> None:
    response = client.post(
        "/metrics",
        json={
            "record": {
                "record_date": "2026-07-22",
                "specialty": "Neurology",
                "payer": "Aetna",
                "scheduled_accounts": 100,
                "cleared_accounts": 92,
                "authorization_required": 40,
                "authorizations_completed": 35,
                "denials": 5,
                "total_charges": 125000,
                "denied_charges": 8000,
                "average_turnaround_hours": 31.5,
                "work_queue_volume": 120,
                "staff_fte": 0,
            }
        },
    )

    assert response.status_code == 422
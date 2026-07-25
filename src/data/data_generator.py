from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pandas as pd


SPECIALTY_PROFILES = {
    "Neurology": {
        "base_volume": 115,
        "auth_rate": 0.52,
        "revenue_per_account": 1650,
        "base_queue": 155,
        "base_staff": 4.8,
        "clearance_modifier": -0.018,
    },
    "Pulmonology": {
        "base_volume": 100,
        "auth_rate": 0.62,
        "revenue_per_account": 1800,
        "base_queue": 135,
        "base_staff": 4.3,
        "clearance_modifier": -0.014,
    },
    "Sleep": {
        "base_volume": 72,
        "auth_rate": 0.48,
        "revenue_per_account": 2100,
        "base_queue": 90,
        "base_staff": 3.2,
        "clearance_modifier": -0.008,
    },
    "Audiology": {
        "base_volume": 58,
        "auth_rate": 0.30,
        "revenue_per_account": 950,
        "base_queue": 65,
        "base_staff": 2.8,
        "clearance_modifier": -0.006,
    },
    "Speech": {
        "base_volume": 88,
        "auth_rate": 0.36,
        "revenue_per_account": 700,
        "base_queue": 75,
        "base_staff": 3.6,
        "clearance_modifier": -0.010,
    },
    "Home Care": {
        "base_volume": 82,
        "auth_rate": 0.70,
        "revenue_per_account": 2450,
        "base_queue": 145,
        "base_staff": 4.6,
        "clearance_modifier": -0.012,
    },
}


PAYER_PROFILES = {
    "Aetna": {
        "turnaround_modifier": 8,
        "denial_modifier": 0.012,
        "auth_modifier": 0.08,
        "clearance_modifier": -0.010,
    },
    "Independence Blue Cross": {
        "turnaround_modifier": 3,
        "denial_modifier": 0.006,
        "auth_modifier": 0.03,
        "clearance_modifier": -0.004,
    },
    "AmeriHealth": {
        "turnaround_modifier": 7,
        "denial_modifier": 0.014,
        "auth_modifier": 0.06,
        "clearance_modifier": -0.009,
    },
    "Cigna": {
        "turnaround_modifier": -2,
        "denial_modifier": 0.004,
        "auth_modifier": 0.02,
        "clearance_modifier": 0.002,
    },
    "Medicare": {
        "turnaround_modifier": -4,
        "denial_modifier": -0.004,
        "auth_modifier": -0.06,
        "clearance_modifier": 0.005,
    },
    "Medicaid": {
        "turnaround_modifier": 12,
        "denial_modifier": 0.018,
        "auth_modifier": 0.10,
        "clearance_modifier": -0.016,
    },
}


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(value, maximum))


def generate_dataset(
    days: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)

    rows: list[dict] = []
    start_date = date.today() - timedelta(days=days)

    queue_state = {
        (specialty, payer): float(profile["base_queue"])
        for specialty, profile in SPECIALTY_PROFILES.items()
        for payer in PAYER_PROFILES
    }

    specialty_disruptions = {
        specialty: 0
        for specialty in SPECIALTY_PROFILES
    }

    payer_disruptions = {
        payer: 0
        for payer in PAYER_PROFILES
    }

    for day_index in range(days):
        current_date = start_date + timedelta(days=day_index)

        weekend_factor = (
            0.68
            if current_date.weekday() >= 5
            else 1.0
        )

        seasonal_factor = (
            1.0
            + 0.10
            * math.sin(
                2
                * math.pi
                * day_index
                / 365
            )
        )

        for specialty in specialty_disruptions:
            if specialty_disruptions[specialty] > 0:
                specialty_disruptions[specialty] -= 1
            elif random.random() < 0.006:
                specialty_disruptions[specialty] = random.randint(
                    3,
                    12,
                )

        for payer in payer_disruptions:
            if payer_disruptions[payer] > 0:
                payer_disruptions[payer] -= 1
            elif random.random() < 0.004:
                payer_disruptions[payer] = random.randint(
                    4,
                    15,
                )

        for specialty, specialty_profile in (
            SPECIALTY_PROFILES.items()
        ):
            for payer, payer_profile in (
                PAYER_PROFILES.items()
            ):
                staffing_shortage = (
                    specialty_disruptions[specialty] > 0
                )

                payer_delay = (
                    payer_disruptions[payer] > 0
                )

                volume_surge = (
                    random.random() < 0.035
                )

                volume_multiplier = (
                    random.uniform(1.18, 1.42)
                    if volume_surge
                    else 1.0
                )

                scheduled_accounts = max(
                    10,
                    round(
                        specialty_profile["base_volume"]
                        * weekend_factor
                        * seasonal_factor
                        * volume_multiplier
                        + random.gauss(0, 8)
                    ),
                )

                staff_reduction = (
                    random.uniform(0.65, 0.82)
                    if staffing_shortage
                    else 1.0
                )

                staff_fte = round(
                    max(
                        1.0,
                        (
                            specialty_profile["base_staff"]
                            + random.gauss(0, 0.35)
                        )
                        * staff_reduction,
                    ),
                    2,
                )

                auth_rate = _clamp(
                    specialty_profile["auth_rate"]
                    + payer_profile["auth_modifier"]
                    + random.gauss(0, 0.025),
                    0.08,
                    0.92,
                )

                authorization_required = round(
                    scheduled_accounts * auth_rate
                )

                current_queue = queue_state[
                    (specialty, payer)
                ]

                workload_per_fte = (
                    scheduled_accounts
                    + current_queue * 0.35
                ) / staff_fte

                delay_event_hours = (
                    random.uniform(12, 28)
                    if payer_delay
                    else 0
                )

                average_turnaround_hours = _clamp(
                    8
                    + workload_per_fte * 0.72
                    + payer_profile[
                        "turnaround_modifier"
                    ]
                    + delay_event_hours
                    + random.gauss(0, 4),
                    6,
                    144,
                )

                authorization_completion_rate = _clamp(
                    1.02
                    - average_turnaround_hours * 0.0025
                    - current_queue * 0.00045
                    + staff_fte * 0.008
                    - (
                        0.055
                        if staffing_shortage
                        else 0
                    )
                    - (
                        0.045
                        if payer_delay
                        else 0
                    )
                    + random.gauss(0, 0.018),
                    0.58,
                    0.99,
                )

                authorizations_completed = round(
                    authorization_required
                    * authorization_completion_rate
                )

                clearance_rate = _clamp(
                    1.015
                    + specialty_profile[
                        "clearance_modifier"
                    ]
                    + payer_profile[
                        "clearance_modifier"
                    ]
                    - average_turnaround_hours * 0.00065
                    - current_queue * 0.00010
                    - (
                        0.035
                        if staffing_shortage
                        else 0
                    )
                    - (
                        0.025
                        if payer_delay
                        else 0
                    )
                    - (
                        0.018
                        if volume_surge
                        else 0
                    )
                    + authorization_completion_rate
                    * 0.025
                    + random.gauss(0, 0.012),
                    0.80,
                    0.985,
                )

                cleared_accounts = min(
                    scheduled_accounts,
                    round(
                        scheduled_accounts
                        * clearance_rate
                    ),
                )

                denial_rate = _clamp(
                    0.018
                    + payer_profile[
                        "denial_modifier"
                    ]
                    + (1 - clearance_rate) * 0.30
                    + (
                        1
                        - authorization_completion_rate
                    )
                    * 0.16
                    + (
                        0.012
                        if payer_delay
                        else 0
                    )
                    + random.gauss(0, 0.005),
                    0.004,
                    0.18,
                )

                denials = min(
                    scheduled_accounts,
                    round(
                        scheduled_accounts
                        * denial_rate
                    ),
                )

                revenue_per_account = (
                    specialty_profile[
                        "revenue_per_account"
                    ]
                    * random.uniform(0.88, 1.12)
                )

                total_charges = round(
                    scheduled_accounts
                    * revenue_per_account,
                    2,
                )

                denied_charges = round(
                    denials
                    * revenue_per_account
                    * random.uniform(0.75, 1.10),
                    2,
                )

                unresolved_authorizations = max(
                    0,
                    authorization_required
                    - authorizations_completed,
                )

                base_queue = specialty_profile[
                    "base_queue"
                ]

                excess_queue = max(
                    0,
                    current_queue - base_queue,
                )

                backlog_resolution = (
                    staff_fte * random.uniform(0.8, 1.5)
                    + excess_queue * 0.035
                )

                next_queue = max(
                    5,
                    current_queue
                    + unresolved_authorizations
                    - backlog_resolution
                    + random.gauss(0, 3),
                )

                queue_state[
                    (specialty, payer)
                ] = next_queue

                rows.append(
                    {
                        "record_date": current_date,
                        "specialty": specialty,
                        "payer": payer,
                        "scheduled_accounts": (
                            scheduled_accounts
                        ),
                        "cleared_accounts": (
                            cleared_accounts
                        ),
                        "authorization_required": (
                            authorization_required
                        ),
                        "authorizations_completed": (
                            authorizations_completed
                        ),
                        "denials": denials,
                        "total_charges": total_charges,
                        "denied_charges": (
                            denied_charges
                        ),
                        "average_turnaround_hours": (
                            round(
                                average_turnaround_hours,
                                2,
                            )
                        ),
                        "work_queue_volume": round(
                            current_queue
                        ),
                        "staff_fte": staff_fte,
                    }
                )

    return pd.DataFrame(rows)
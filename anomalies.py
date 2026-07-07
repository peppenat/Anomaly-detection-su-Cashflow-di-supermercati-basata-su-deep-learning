# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:30:01 2026

@author: ciok4
"""

import numpy as np
import pandas as pd

def get_def_anomaly_config():
    return {
        "level_shift": {
            "scope": None,
            "fraction": 0.0,
            "guaranteed_events_per_type": 1,
            "seed_offset": 40_000,
            "min_gap_days": 14,
            "duration_range": (7, 14),
            "soft_increase_mult_range": (1.05, 1.12),
            "hard_increase_mult_range": (1.20, 1.30),
        
            "soft_decrease_mult_range": (0.88, 0.95),
            "hard_decrease_mult_range": (0.70, 0.80),
            "enabled_types": [
                "soft_increase",
                "hard_increase",
                "soft_decrease",
                "hard_decrease"
                ],
        },
        "pos_delay": {
            "scope": None,
            "fraction": 0.01,       
            "guaranteed_events_per_type": 1,
            "edge_margin_source_days": 10,
            "seed_offset": 20_000,   
            "min_event_distance": 10,
            "duration_range": (1, 1),
            "enabled_types": [
                "mild_delay",
                "moderate_delay",
                "strong_delay",
                "batch_backlog",
                "settlement_freeze"
            ],
        
            "type_probs": {
                "mild_delay": 0.20,
                "moderate_delay": 0.20,
                "strong_delay": 0.20,
                "batch_backlog": 0.20,
                "settlement_freeze": 0.20
            },
        
            # kernel Dirichlet:
            # [T+1, T+2, T+3, T+4, T+5]
            "delay_profiles": {
                # quasi normale:
                # leggero rallentamento fisiologico
                "mild_delay": {
                    "alpha": [17.5, 22.5, 9, 1, 0]
                },
        
                # ritardo chiaramente visibile
                "moderate_delay": {
                    "alpha": [10, 17.5, 15, 7.5, 0]
                },
        
                # settlement molto rallentato
                "strong_delay": {
                    "alpha": [2.5, 10, 17.5, 15, 5]
                },
        
                # backlog operativo:
                # molta massa su T+4
                "batch_backlog": {
                    "alpha": [1, 4, 25, 12.5, 7.5]
                },
        
                # quasi freeze:
                # la maggior parte arriva a T+5
                "settlement_freeze": {
                    "alpha": [0.25, 0.75, 1.5, 37.5, 10]
                }
            }
        }
    }


def inject_level_shift_anomalies(
    df,
    level_cfg,
    sales_col="daily_total_sales",
    seed=42,
    start_margin_days=35
):
    df = df.copy()

    anomaly_scope = level_cfg.get("scope", None)
    anomaly_fraction = level_cfg.get("fraction", 0.0)
    guaranteed_events_per_type = level_cfg.get("guaranteed_events_per_type", 1)
    duration_range = level_cfg.get("duration_range", (7, 14))
    min_gap_days = level_cfg.get("min_gap_days", 14)
    
    enabled_types = level_cfg.get(
        "enabled_types",
        [
            "soft_increase",
            "hard_increase",
            "soft_decrease",
            "hard_decrease"
        ]
    )

    soft_increase_range = level_cfg.get("soft_increase_mult_range", (1.05, 1.12))
    hard_increase_range = level_cfg.get("hard_increase_mult_range", (1.20, 1.30))
    soft_decrease_range = level_cfg.get("soft_decrease_mult_range", (0.88, 0.95))
    hard_decrease_range = level_cfg.get("hard_decrease_mult_range", (0.70, 0.80))


    df["is_level_shift_anomaly"] = 0
    df["lsa_type"] = "normal"
    df["lsa_severity"] = "normal"
    df["lsa_mult"] = 1.0
    df["lsa_event_id"] = -1
    df["lsa_day_in_event"] = -1
    df["lsa_duration"] = 0

    if anomaly_scope is None:
        return df

    n = len(df)

    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    train_pos = np.arange(start_margin_days, train_end)
    val_pos = np.arange(train_end + start_margin_days, val_end)
    test_pos = np.arange(val_end + start_margin_days, n)

    all_event_types = {
        "soft_increase": soft_increase_range,
        "hard_increase": hard_increase_range,
        "soft_decrease": soft_decrease_range,
        "hard_decrease": hard_decrease_range,
    }
    
    event_types = [
        (event_type, all_event_types[event_type])
        for event_type in enabled_types
    ]

    def inject_lsa_on_positions(
        df,
        candidate_pos,
        rng,
        start_event_id
    ):
        if len(candidate_pos) == 0:
            return df, start_event_id

        candidate_pos_set = set(candidate_pos)
        occupied_pos = set()

        event_id = start_event_id

        def try_inject_single_event(event_type, mult_range, event_id):
            max_attempts = 500

            for _ in range(max_attempts):
                duration = int(
                    rng.integers(
                        duration_range[0],
                        duration_range[1] + 1
                    )
                )

                start_pos = int(rng.choice(candidate_pos))
                end_pos = start_pos + duration

                if end_pos > n:
                    continue

                event_pos = np.arange(start_pos, end_pos)

                # evento interamente nello split corrente
                if not set(event_pos).issubset(candidate_pos_set):
                    continue

                # distanza minima dentro lo stesso split
                forbidden_pos = set(
                    np.arange(
                        max(0, start_pos - min_gap_days),
                        min(n, end_pos + min_gap_days)
                    )
                )

                if occupied_pos.intersection(forbidden_pos):
                    continue

                event_idx = df.index[event_pos]
                multiplier = rng.uniform(*mult_range)

                severity = "soft" if event_type.startswith("soft") else "hard"

                df.loc[event_idx, sales_col] *= multiplier
                df.loc[event_idx, "is_level_shift_anomaly"] = 1
                df.loc[event_idx, "lsa_type"] = event_type
                df.loc[event_idx, "lsa_severity"] = severity
                df.loc[event_idx, "lsa_mult"] = multiplier
                df.loc[event_idx, "lsa_event_id"] = event_id
                df.loc[event_idx, "lsa_duration"] = duration
                df.loc[event_idx, "lsa_day_in_event"] = np.arange(duration)

                occupied_pos.update(event_pos)

                return True

            return False

        # eventi garantiti per ogni tipo
        for event_type, mult_range in event_types:
            for _ in range(guaranteed_events_per_type):
                success = try_inject_single_event(
                    event_type,
                    mult_range,
                    event_id
                )

                if success:
                    event_id += 1

        # eventi random opzionali
        n_random_events = int(len(candidate_pos) * anomaly_fraction)

        for _ in range(n_random_events):
            event_type, mult_range = event_types[
                rng.integers(0, len(event_types))
            ]

            success = try_inject_single_event(
                event_type,
                mult_range,
                event_id
            )

            if success:
                event_id += 1

        return df, event_id

    rng_test = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 1)
    rng_train = np.random.default_rng(seed + 2)

    event_id = 0

    if anomaly_scope == "test":
        df, event_id = inject_lsa_on_positions(
            df,
            test_pos,
            rng_test,
            event_id
        )

    elif anomaly_scope == "val_test":
        df, event_id = inject_lsa_on_positions(
            df,
            val_pos,
            rng_val,
            event_id
        )

        df, event_id = inject_lsa_on_positions(
            df,
            test_pos,
            rng_test,
            event_id
        )

    elif anomaly_scope == "all":
        df, event_id = inject_lsa_on_positions(
            df,
            train_pos,
            rng_train,
            event_id
        )

        df, event_id = inject_lsa_on_positions(
            df,
            val_pos,
            rng_val,
            event_id
        )

        df, event_id = inject_lsa_on_positions(
            df,
            test_pos,
            rng_test,
            event_id
        )

    else:
        raise ValueError(
            "scope must be None, 'test', 'val_test' or 'all'"
        )

    return df




def inject_pos_delay(
    df,
    pos_cfg,
    anomaly_seed=42
):
    """
    Inietta anomalie POS delay marcando solo i giorni sorgente.

    Source day:
    giorno di vendita POS il cui settlement seguirà un profilo anomalo.

    Effect day:
    giorni in cui il cashflow POS risulterà alterato.
    Vengono marcati dopo, in add_pos_settlement_features(),
    perché dipendono dai business days e dai ritardi T+1...T+5.
    """

    df = df.copy()

    # =========================
    # CONFIG
    # =========================
    anomaly_scope = pos_cfg.get("scope", None)
    anomaly_fraction = pos_cfg.get("fraction", 0.01)

    duration_range = pos_cfg.get("duration_range", (1, 1))
    min_event_distance = pos_cfg.get("min_event_distance", 10)
    edge_margin_source_days = pos_cfg.get("edge_margin_source_days", 14)
    

    enabled_types = pos_cfg.get(
        "enabled_types",
        ["mild_delay", "moderate_delay", "strong_delay"]
    )

    type_probs_dict = pos_cfg.get(
        "type_probs",
        {t: 1 / len(enabled_types) for t in enabled_types}
    )

    guaranteed_events_per_type = pos_cfg.get(
        "guaranteed_events_per_type",
        0
    )

    # =========================
    # INIT GROUND TRUTH
    # =========================

    # source days
    df["is_pos_delay_source_day"] = 0
    df["pos_delay_source_day_in_event"] = -1
    df["pos_delay_source_duration"] = 0

    # effect days
    df["is_pos_delay_effect_day"] = 0
    df["pos_delay_effect_day_in_event"] = -1
    df["pos_delay_effect_duration"] = 0

    # event metadata
    df["pos_delay_event_id"] = -1
    df["pos_delay_type"] = "normal"

    if anomaly_scope is None:
        return df, set()

    # =========================
    # SPLIT TEMPORALE
    # =========================
    n = len(df)

    train_end = int(0.70 * n)
    val_end = int(0.80 * n)

    train_pos = np.arange(0, train_end)
    val_pos = np.arange(train_end, val_end)
    test_pos = np.arange(val_end, n)

    if anomaly_scope == "test":
        active_splits = [
            ("test", test_pos, anomaly_seed)
        ]

    elif anomaly_scope == "val_test":
        active_splits = [
            ("val", val_pos, anomaly_seed + 1),
            ("test", test_pos, anomaly_seed)
        ]

    elif anomaly_scope == "all":
        active_splits = [
            ("train", train_pos, anomaly_seed + 2),
            ("val", val_pos, anomaly_seed + 1),
            ("test", test_pos, anomaly_seed)
        ]

    else:
        raise ValueError(
            "pos_cfg['scope'] must be None, 'test', 'val_test' or 'all'"
        )

    # =========================
    # TYPE PROBABILITIES
    # =========================
    type_probs = np.array(
        [type_probs_dict.get(t, 0.0) for t in enabled_types],
        dtype=float
    )

    if type_probs.sum() == 0:
        type_probs = np.ones(len(enabled_types)) / len(enabled_types)
    else:
        type_probs = type_probs / type_probs.sum()

    pos_delay_idx = set()
    next_event_id = 0

    def inject_events_on_positions(
        df,
        candidate_pos,
        rng,
        start_event_id
    ):
        if len(candidate_pos) == 0:
            return df, set(), start_event_id

        # =========================
        # SOURCE DAY CANDIDATI
        # =========================
        candidate_df = df.iloc[candidate_pos].copy()
        candidate_df["_pos"] = candidate_pos

        # I source day POS possono essere anche weekend/festivi:
        # sono giorni di vendita, non giorni di settlement bancario.
        candidate_source = (
            candidate_df
            .sort_values("date")
            .copy()
        )

        if len(candidate_source) == 0:
            return df, set(), start_event_id

        # Posizione del candidato source all'interno dello split.
        # Serve solo per evitare eventi troppo vicini ai bordi.
        candidate_source["_source_pos_in_split"] = np.arange(
            len(candidate_source)
        )

        max_source_pos = candidate_source["_source_pos_in_split"].max()

        candidate_source = candidate_source[
            (
                candidate_source["_source_pos_in_split"]
                >= edge_margin_source_days
            ) &
            (
                candidate_source["_source_pos_in_split"]
                <= max_source_pos - edge_margin_source_days
            )
        ].copy()

        if candidate_source.empty:
            return df, set(), start_event_id

        candidate_pos = candidate_source["_pos"].to_numpy()

        candidate_pos_set = set(candidate_pos)
        occupied_pos = set()
        selected_source_idx = set()

        event_id = start_event_id

        n_random_events = int(len(candidate_pos) * anomaly_fraction)

        forced_types = []

        for t in enabled_types:
            forced_types.extend(
                [t] * guaranteed_events_per_type
            )

        event_types_to_create = forced_types.copy()

        for _ in range(n_random_events):
            event_types_to_create.append(
                rng.choice(enabled_types, p=type_probs)
            )

        max_attempts = max(
            1000,
            len(event_types_to_create) * 200
        )

        attempts = 0
        created_events = 0

        while (
            created_events < len(event_types_to_create)
            and attempts < max_attempts
        ):

            attempts += 1

            delay_type = event_types_to_create[created_events]

            duration = int(
                rng.integers(
                    duration_range[0],
                    duration_range[1] + 1
                )
            )

            start_pos = int(
                rng.choice(candidate_pos)
            )

            end_pos = start_pos + duration

            if end_pos > n:
                continue

            event_pos = np.arange(start_pos, end_pos)

            # l'evento deve restare nello split
            if not set(event_pos).issubset(candidate_pos_set):
                continue

            # evita eventi sovrapposti o troppo vicini
            protected_pos = np.arange(
                max(0, start_pos - min_event_distance),
                min(n, end_pos + min_event_distance)
            )

            if len(set(protected_pos).intersection(occupied_pos)) > 0:
                continue

            # =========================
            # SOURCE DAYS
            # =========================
            for day_in_event, pos in enumerate(event_pos):

                idx = df.index[pos]

                df.loc[idx, "is_pos_delay_source_day"] = 1
                df.loc[idx, "pos_delay_source_day_in_event"] = day_in_event
                df.loc[idx, "pos_delay_source_duration"] = len(event_pos)

                df.loc[idx, "pos_delay_event_id"] = event_id
                df.loc[idx, "pos_delay_type"] = delay_type

                selected_source_idx.add(idx)

            occupied_pos.update(protected_pos)

            event_id += 1
            created_events += 1

        return df, selected_source_idx, event_id

    # =========================
    # APPLY SPLITS
    # =========================
    for split_name, split_pos, split_seed in active_splits:

        rng = np.random.default_rng(split_seed)

        df, idx, next_event_id = inject_events_on_positions(
            df=df,
            candidate_pos=split_pos,
            rng=rng,
            start_event_id=next_event_id
        )

        pos_delay_idx.update(idx)

    return df, pos_delay_idx





def add_business_days(date, n_business_days, business_dates, end_date):
    future_business_dates = business_dates[business_dates > date]

    if len(future_business_dates) >= n_business_days:
        return future_business_dates.iloc[n_business_days - 1]
    else:
        return pd.Timestamp(end_date)


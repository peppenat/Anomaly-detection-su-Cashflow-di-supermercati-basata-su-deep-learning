# %% Imports
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies import (
    get_def_anomaly_config,
    inject_level_shift_anomalies,
    inject_pos_delay,
)
from supermarkets import all_params

from project_paths import (
    CLEAN_DATA_PATH,
    LEVEL_SHIFT_CONTAMINATION_DIR,
    LEVEL_SHIFT_SENSITIVITY_DIR,
    POS_DELAY_CONTAMINATION_DIR,
    POS_DELAY_SENSITIVITY_DIR,
)


def generate_level_shift_sensitivity_experiment_from_clean(
    clean_path=CLEAN_DATA_PATH,
    saving_path=LEVEL_SHIFT_SENSITIVITY_DIR,
    force_recompute=False
):
    """
    Genera i dataset di sensitivity per il caso level shift a partire
    dal dataset clean.

    Per ogni combinazione di:
    - direzione dello shift;
    - durata;
    - moltiplicatore;
    - seed;

    vengono iniettati eventi level shift esclusivamente nel test set.

    Se force_recompute=False, i dataset già presenti non vengono rigenerati.
    """

    clean_df = pd.read_csv(clean_path)
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    durations = [7, 10, 14, 21]

    upward_multipliers = [
        1.025, 1.05, 1.075, 1.10, 1.15, 1.20, 1.30
    ]

    downward_multipliers = [
        0.975, 0.95, 0.925, 0.90, 0.85, 0.80, 0.70
    ]

    seeds = [42, 43, 44, 45, 46]
    n_events = 4

    experiments = []

    for duration in durations:
        for mult in upward_multipliers:
            for seed in seeds:
                experiments.append(
                    ("increase", "soft_increase", duration, mult, seed)
                )

    for duration in durations:
        for mult in downward_multipliers:
            for seed in seeds:
                experiments.append(
                    ("decrease", "soft_decrease", duration, mult, seed)
                )

    for direction, event_type, duration, mult, seed in experiments:

        output_path = os.path.join(
            saving_path,
            direction,
            f"dur_{duration:02d}_mult_{mult:.3f}",
            f"seed_{seed}"
        )

        output_csv_path = os.path.join(
            output_path,
            "all_stores_cashflow.csv"
        )

        # Evita di rigenerare dataset già prodotti.
        if os.path.exists(output_csv_path) and not force_recompute:
            print(f"[=] Già esistente: {output_path}")
            continue

        df_out = []

        for store_id, store_df in clean_df.groupby("store_id"):

            temp = store_df.sort_values("date").copy()

            level_cfg = deepcopy(
                get_def_anomaly_config()["level_shift"]
            )

            # Gli shift vengono iniettati soltanto nello split test.
            level_cfg["scope"] = "test"
            level_cfg["fraction"] = 0.0
            level_cfg["guaranteed_events_per_type"] = n_events
            level_cfg["enabled_types"] = [event_type]
            level_cfg["duration_range"] = (duration, duration)
            level_cfg["min_gap_days"] = 14

            if direction == "increase":
                level_cfg["soft_increase_mult_range"] = (mult, mult)
            else:
                level_cfg["soft_decrease_mult_range"] = (mult, mult)

            temp = inject_level_shift_anomalies(
                temp,
                level_cfg,
                sales_col="daily_total_sales",
                seed=seed + int(store_id)
            )

            df_out.append(temp)

        df_out = (
            pd.concat(df_out, ignore_index=True)
              .sort_values(["store_id", "date"])
              .reset_index(drop=True)
        )

        os.makedirs(output_path, exist_ok=True)

        df_out.to_csv(
            output_csv_path,
            index=False
        )

        print(f"[+] Salvato: {output_path}")

    print("\n>>> LEVEL SHIFT SENSITIVITY DA CLEAN COMPLETATO <<<")



def generate_pos_delay_sensitivity_experiment_from_clean(
    clean_path=CLEAN_DATA_PATH,
    saving_path=POS_DELAY_SENSITIVITY_DIR,
    delay_types=None,
    source_durations=None,
    seeds=None,
    min_guaranteed_events=5,
    min_event_distance=18,
    edge_margin_source_days=14,
    normal_settlement_seed=123_456,
    force_recompute=False
):
    """
    Genera dataset di sensitivity per POS delay partendo dal dataset clean.

    Varia:
    - delay_type
    - source_duration
    - seed

    Tiene fissi:
    - scope = test
    - fraction = 0.0
    - guaranteed_events_per_type = min_guaranteed_events
    - min_event_distance = 18
    - edge_margin_source_days = 14

    Salva solo le colonne necessarie per tutta l'analisi POS:
    - inference modello POS;
    - costruzione results_df;
    - profile comparison;
    - detected windows;
    - ground truth event-level.
    """


    clean_df = pd.read_csv(clean_path)
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    if delay_types is None:
        delay_types = [
            "mild_delay",
            "moderate_delay",
            "strong_delay",
            "batch_backlog",
            "settlement_freeze"
        ]

    if source_durations is None:
        source_durations = [1, 2, 3]

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    saving_path = Path(saving_path)
    saving_path.mkdir(parents=True, exist_ok=True)

    # Colonne necessarie per rigenerare il cashflow POS.
    required_cols = [
        "date",
        "store_id",
        "week_day",
        "month",
        "holiday",
        "actual_holiday",
        "pos_card_sales",
        "pos_volume_ratio"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in clean_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Nel clean dataset mancano queste colonne: {missing_cols}"
        )

    # Colonne finali da salvare nei dataset sensitivity.
    pos_analysis_cols = [
        "date",
        "day",
        "store_id",
        "week_day",
        "month",
        "holiday",
        "actual_holiday",
        "pos_card_sales",
        "pos_net_cf",
        "pos_volume_ratio",

        "is_point_anomaly",
        "pa_type",
        "pa_mult",

        "is_pos_delay_source_day",
        "pos_delay_source_day_in_event",
        "pos_delay_source_duration",

        "is_pos_delay_effect_day",
        "pos_delay_effect_day_in_event",
        "pos_delay_effect_duration",

        "pos_delay_event_id",
        "pos_delay_type"
    ]

    # Se il clean non contiene point anomalies, le inizializziamo come normali
    # perché sono nella ground truth del modello POS salvato.
    if "is_point_anomaly" not in clean_df.columns:
        clean_df["is_point_anomaly"] = 0

    if "pa_type" not in clean_df.columns:
        clean_df["pa_type"] = "normal"

    if "pa_mult" not in clean_df.columns:
        clean_df["pa_mult"] = 1.0

    params_by_store = {
        store["store_id"]: store["params"]
        for store in all_params
    }

    def _store_seed_component(store_id):
        digits = "".join(
            ch for ch in str(store_id)
            if ch.isdigit()
        )

        if digits != "":
            return int(digits)

        return sum(
            ord(ch)
            for ch in str(store_id)
        )

    def _add_business_days(date, n_business_days, business_dates, end_date):
        future_business_dates = (
            business_dates[business_dates > date]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(future_business_dates) >= n_business_days:
            return future_business_dates.iloc[n_business_days - 1]

        return pd.Timestamp(end_date)

    def _apply_pos_delay_to_clean_store(
        store_df,
        pos_cfg,
        experiment_seed,
        store_id
    ):
        df = store_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").copy()

        store_params = params_by_store[store_id]
        pos_commission_rate = store_params["pos_commission_rate"]

        # Importo POS netto da redistribuire sui giorni di settlement.
        # Non viene salvato: serve solo internamente.
        df["pos_net_amount"] = (
            df["pos_card_sales"].astype(float)
            * (1 - pos_commission_rate)
        ).round(2)

        # =========================
        # RESET GT POS DELAY
        # =========================
        pos_gt_cols = [
            "is_pos_delay_source_day",
            "pos_delay_source_day_in_event",
            "pos_delay_source_duration",
            "is_pos_delay_effect_day",
            "pos_delay_effect_day_in_event",
            "pos_delay_effect_duration",
            "pos_delay_event_id",
            "pos_delay_type"
        ]

        for col in pos_gt_cols:
            if col in df.columns:
                df = df.drop(columns=col)

        # =========================
        # INIEZIONE SOURCE DAYS
        # =========================
        df, pos_delay_idx = inject_pos_delay(
            df=df,
            pos_cfg=pos_cfg,
            anomaly_seed=(
                experiment_seed
                + _store_seed_component(store_id)
                + pos_cfg.get("seed_offset", 0)
            )
        )

        # =========================
        # BUSINESS DAYS
        # =========================
        business_dates = (
            df.loc[df["holiday"].astype(int) == 0, "date"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        end_date = df["date"].max()
        max_effect_business_days = 5

        # =========================
        # EFFECT DAYS POS DELAY
        # =========================
        source_days = df[
            df["is_pos_delay_source_day"].astype(int) == 1
        ].copy()

        for event_id, g in source_days.groupby("pos_delay_event_id"):

            delay_type = g["pos_delay_type"].mode().iloc[0]
            effect_dates = []

            for _, source_row in g.iterrows():

                source_date = pd.to_datetime(source_row["date"])

                for k in range(1, max_effect_business_days + 1):

                    effect_dates.append(
                        _add_business_days(
                            date=source_date,
                            n_business_days=k,
                            business_dates=business_dates,
                            end_date=end_date
                        )
                    )

            effect_dates = (
                pd.Series(effect_dates, dtype="datetime64[ns]")
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            effect_duration = len(effect_dates)

            for day_in_effect, effect_date in enumerate(effect_dates):

                mask = df["date"] == effect_date

                df.loc[mask, "is_pos_delay_effect_day"] = 1
                df.loc[mask, "pos_delay_effect_day_in_event"] = day_in_effect
                df.loc[mask, "pos_delay_effect_duration"] = effect_duration

                df.loc[mask, "pos_delay_event_id"] = event_id
                df.loc[mask, "pos_delay_type"] = delay_type

        # =========================
        # RICALCOLO SOLO POS_NET_CF
        # =========================
        lambda_low = 3
        lambda_high = 5

        gamma_low = 3 / 4
        gamma_high = 3 / 2

        alpha_low_5 = np.array([40.0, 7.5, 2.5, 0.0, 0.0], dtype=float)
        alpha_base_5 = np.array([35.0, 12.5, 2.5, 0.0, 0.0], dtype=float)
        alpha_stress_5 = np.array([20.0, 17.5, 12.5, 0.0, 0.0], dtype=float)

        settlement_days = [1, 2, 3, 4, 5]

        pos_cf_by_date = pd.Series(
            0.0,
            index=pd.to_datetime(df["date"])
        )

        for row_pos, (idx, row) in enumerate(df.iterrows()):

            date = pd.to_datetime(row["date"])

            if idx in pos_delay_idx:

                delay_type = row["pos_delay_type"]

                alpha = np.array(
                    pos_cfg["delay_profiles"][delay_type]["alpha"],
                    dtype=float
                )

            else:

                volume_ratio = float(row["pos_volume_ratio"])

                low_weight = np.exp(
                    -lambda_low
                    * max(volume_ratio - 0.8, 0) ** gamma_low
                )

                stress_weight = 1 - np.exp(
                    -lambda_high
                    * max(volume_ratio - 1, 0) ** gamma_high
                )

                base_weight = 1 - low_weight - stress_weight
                base_weight = np.clip(base_weight, 0, 1)

                alpha = (
                    low_weight * alpha_low_5
                    + base_weight * alpha_base_5
                    + stress_weight * alpha_stress_5
                )

            # Dirichlet richiede alpha strettamente positivi.
            alpha = np.clip(alpha, 1e-6, None)

            rng_row = np.random.default_rng(
                normal_settlement_seed
                + _store_seed_component(store_id) * 1_000_000
                + row_pos
            )

            weights = rng_row.dirichlet(alpha)

            for delay, weight in zip(settlement_days, weights):

                settlement_date = _add_business_days(
                    date=date,
                    n_business_days=delay,
                    business_dates=business_dates,
                    end_date=end_date
                )

                if settlement_date in pos_cf_by_date.index:
                    pos_cf_by_date.loc[settlement_date] += (
                        float(row["pos_net_amount"])
                        * float(weight)
                    )

        df["pos_net_cf"] = (
            df["date"]
            .map(pos_cf_by_date)
            .fillna(0.0)
            .astype(float)
            .round(2)
        )

        # =========================
        # OUTPUT MINIMO PER POS
        # =========================
        df = df[pos_analysis_cols].copy()

        return df

    summary_rows = []

    total_experiments = (
        len(delay_types)
        * len(source_durations)
        * len(seeds)
    )

    exp_counter = 0

    for delay_type in delay_types:

        for source_duration in source_durations:

            for seed in seeds:

                exp_counter += 1
                
                output_path = (
                    saving_path
                    / delay_type
                    / f"srcdur_{source_duration}"
                    / f"seed_{seed}"
                )
                
                csv_path = output_path / "all_stores_cashflow.csv"
                config_path = output_path / "config.json"
                
                # Se il dataset esiste già, non rigenerarlo.
                # La riga viene comunque recuperata per mantenere completa la summary finale.
                if csv_path.exists() and not force_recompute:
                
                    if config_path.exists():
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                
                        summary_rows.append({
                            "delay_type": delay_type,
                            "source_duration": source_duration,
                            "seed": seed,
                            "csv_path": str(csv_path),
                            "config_path": str(config_path),
                            "n_rows": config["n_rows"],
                            "n_source_days": config["n_source_days"],
                            "n_effect_days": config["n_effect_days"],
                            "n_events": config["n_events"],
                        })
                
                    else:
                        # Fallback per dataset vecchi senza config.json.
                        existing_df = pd.read_csv(csv_path)
                
                        summary_rows.append({
                            "delay_type": delay_type,
                            "source_duration": source_duration,
                            "seed": seed,
                            "csv_path": str(csv_path),
                            "config_path": "",
                            "n_rows": len(existing_df),
                            "n_source_days": int(
                                existing_df["is_pos_delay_source_day"].astype(int).sum()
                            ),
                            "n_effect_days": int(
                                existing_df["is_pos_delay_effect_day"].astype(int).sum()
                            ),
                            "n_events": int(
                                existing_df.loc[
                                    existing_df["pos_delay_event_id"] != -1,
                                    "pos_delay_event_id",
                                ].nunique()
                            ),
                        })
                
                    print(
                        f"[=] Già esistente: "
                        f"type={delay_type} | srcdur={source_duration} | seed={seed}"
                    )
                    continue
                
                print(
                    f"[{exp_counter}/{total_experiments}] "
                    f"POS delay | type={delay_type} | "
                    f"srcdur={source_duration} | seed={seed}"
                )
                
                df_out_parts = []

                for store_id, store_df in clean_df.groupby("store_id"):

                    pos_cfg = deepcopy(
                        get_def_anomaly_config()["pos_delay"]
                    )

                    pos_cfg["scope"] = "test"
                    pos_cfg["fraction"] = 0.0

                    pos_cfg["guaranteed_events_per_type"] = (
                        min_guaranteed_events
                    )

                    pos_cfg["enabled_types"] = [delay_type]
                    pos_cfg["type_probs"] = {
                        delay_type: 1.0
                    }

                    pos_cfg["duration_range"] = (
                        source_duration,
                        source_duration
                    )

                    pos_cfg["min_event_distance"] = min_event_distance

                    pos_cfg["edge_margin_source_days"] = edge_margin_source_days

                    temp = _apply_pos_delay_to_clean_store(
                        store_df=store_df,
                        pos_cfg=pos_cfg,
                        experiment_seed=seed,
                        store_id=store_id
                    )

                    df_out_parts.append(temp)

                df_out = (
                    pd.concat(df_out_parts, ignore_index=True)
                    .sort_values(["store_id", "date"])
                    .reset_index(drop=True)
                )

                # Salva solo le colonne necessarie per l'analisi POS.
                df_out = df_out[pos_analysis_cols].copy()

                output_path.mkdir(parents=True, exist_ok=True)

                df_out.to_csv(csv_path, index=False)

                config = {
                    "delay_type": delay_type,
                    "source_duration": source_duration,
                    "seed": seed,
                    "scope": "test",
                    "fraction": 0.0,
                    "min_guaranteed_events": min_guaranteed_events,
                    "min_event_distance": min_event_distance,
                    "edge_margin_source_days": edge_margin_source_days,
                    "normal_settlement_seed": normal_settlement_seed,
                    "saved_columns": pos_analysis_cols,
                    "n_rows": len(df_out),
                    "n_source_days": int(
                        df_out["is_pos_delay_source_day"]
                        .astype(int)
                        .sum()
                    ),
                    "n_effect_days": int(
                        df_out["is_pos_delay_effect_day"]
                        .astype(int)
                        .sum()
                    ),
                    "n_events": int(
                        df_out.loc[
                            df_out["pos_delay_event_id"] != -1,
                            "pos_delay_event_id"
                        ].nunique()
                    )
                }

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)

                summary_rows.append({
                    "delay_type": delay_type,
                    "source_duration": source_duration,
                    "seed": seed,
                    "csv_path": str(csv_path),
                    "config_path": str(config_path),
                    "n_rows": config["n_rows"],
                    "n_source_days": config["n_source_days"],
                    "n_effect_days": config["n_effect_days"],
                    "n_events": config["n_events"]
                })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = saving_path / "pos_delay_sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n>>> POS DELAY SENSITIVITY COMPLETATA <<<")
    print(f"Dataset generati: {len(summary_df)}")
    print(f"Summary salvato in: {summary_path}")

    return summary_df



def generate_level_shift_contamination_experiment_from_clean(
    clean_path=CLEAN_DATA_PATH,
    saving_path=LEVEL_SHIFT_CONTAMINATION_DIR,
    contamination_levels=(0, 5, 10, 20, 30, 50),
    global_seed=42,
    force_recompute=False,
):
    """
    Genera i dataset per l'esperimento di robustezza rispetto alla contaminazione
    dell'assunzione normal-only.

    Versione nested/incrementale:
    - viene campionato una sola volta un piano massimo di eventi level shift;
    - ogni livello usa un prefisso dello stesso piano;
    - quindi un livello superiore contiene gli eventi dei livelli inferiori
      e aggiunge nuovi eventi fino alla quota target.

    I livelli sono codici interi in decimi di punto percentuale:

        0  -> 0.0%
        5  -> 0.5%
        10 -> 1.0%
        20 -> 2.0%
        30 -> 3.0%
        50 -> 5.0%

    La contaminazione è controllata come quota globale di store-day anomali,
    calcolata separatamente su train e validation.

    Il test resta pulito.

    Output invariati:
        saving_path/
            contamination_<level>/
                all_stores_cashflow.csv
                contamination_summary.csv
                contamination_type_summary.csv
                config.json
            contamination_experiment_summary.csv
            contamination_experiment_type_summary.csv
    """



    clean_path = Path(clean_path)
    saving_path = Path(saving_path)

    if not clean_path.exists():
        raise FileNotFoundError(f"Dataset clean non trovato: {clean_path}")

    saving_path.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # CONVERSIONE LIVELLI
    # =========================================================

    def level_code_to_percent(level_code):
        """
        Converte il codice intero nel livello percentuale effettivo.

        Esempi:
            5  -> 0.5
            10 -> 1.0
            50 -> 5.0
        """
        return int(level_code) / 10.0

    def level_code_to_fraction(level_code):
        """
        Converte il codice intero nella frazione effettiva.

        Esempi:
            5  -> 0.005
            10 -> 0.010
            50 -> 0.050
        """
        return level_code_to_percent(level_code) / 100.0

    # =========================================================
    # LOAD CLEAN DATASET
    # =========================================================

    clean_df = pd.read_csv(clean_path)
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    clean_df = (
        clean_df
        .sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )

    if "daily_total_sales" not in clean_df.columns:
        raise ValueError("Nel clean dataset manca la colonna daily_total_sales.")

    # =========================================================
    # CONFIG LEVEL SHIFT CONTAMINATION
    # =========================================================

    level_cfg_contamination = {
        "duration_range": (7, 14),
        "min_gap_days": 14,

        "enabled_types": [
            "soft_increase",
            "hard_increase",
            "soft_decrease",
            "hard_decrease",
        ],

        "soft_increase_mult_range": (1.025, 1.10),
        "hard_increase_mult_range": (1.15, 1.30),

        "soft_decrease_mult_range": (0.90, 0.975),
        "hard_decrease_mult_range": (0.70, 0.85),
    }

    # =========================================================
    # HELPERS BASE
    # =========================================================

    def initialize_level_shift_columns(df):
        """
        Riparte dal clean dataset e inizializza tutte le colonne level shift.
        """

        out = df.copy()

        out["is_level_shift_anomaly"] = 0
        out["lsa_type"] = "normal"
        out["lsa_severity"] = "normal"
        out["lsa_mult"] = 1.0
        out["lsa_event_id"] = -1
        out["lsa_day_in_event"] = -1
        out["lsa_duration"] = 0

        out["is_level_shift_contamination"] = 0
        out["lsa_contamination_split"] = "none"
        out["lsa_contamination_level"] = 0
        out["lsa_contamination_percent"] = 0.0
        out["lsa_contamination_target_fraction"] = 0.0

        return out

    def add_temporal_split(df, train_size=0.70, val_size=0.10):
        """
        Aggiunge split train/val/test per store, mantenendo l'ordine temporale.
        """

        out = df.copy()
        out["_split"] = ""

        for store_id, g in out.groupby("store_id", sort=False):
            idx = g.sort_values("date").index.to_numpy()

            n = len(idx)
            train_end = int(train_size * n)
            val_end = int((train_size + val_size) * n)

            out.loc[idx[:train_end], "_split"] = "train"
            out.loc[idx[train_end:val_end], "_split"] = "val"
            out.loc[idx[val_end:], "_split"] = "test"

        return out

    # =========================================================
    # CONFIG DERIVATA
    # =========================================================

    duration_range = level_cfg_contamination.get("duration_range", (7, 14))
    min_gap_days = int(level_cfg_contamination.get("min_gap_days", 14))

    enabled_types = level_cfg_contamination.get(
        "enabled_types",
        [
            "soft_increase",
            "hard_increase",
            "soft_decrease",
            "hard_decrease",
        ],
    )

    soft_increase_range = level_cfg_contamination.get(
        "soft_increase_mult_range",
        (1.025, 1.10),
    )
    hard_increase_range = level_cfg_contamination.get(
        "hard_increase_mult_range",
        (1.15, 1.30),
    )
    soft_decrease_range = level_cfg_contamination.get(
        "soft_decrease_mult_range",
        (0.90, 0.975),
    )
    hard_decrease_range = level_cfg_contamination.get(
        "hard_decrease_mult_range",
        (0.70, 0.85),
    )

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

    if len(event_types) == 0:
        raise ValueError("enabled_types non può essere vuoto.")

    # =========================================================
    # CAMPIONAMENTO PIANO NESTED
    # =========================================================

    def get_candidate_stores(df_plan, split_name, start_margin_days):
        """
        Restituisce gli store che hanno abbastanza giorni nello split.
        """

        stores = []

        for store_id, g in df_plan[df_plan["_split"] == split_name].groupby("store_id"):
            if len(g) >= start_margin_days + duration_range[0]:
                stores.append(store_id)

        return stores

    def choose_balanced_event_type(type_day_counts, rng):
        """
        Sceglie il tipo con meno giorni anomali già prodotti nello split.
        In caso di parità sceglie casualmente tra i tipi meno rappresentati.
        """

        min_count = min(type_day_counts.values())

        candidates = [
            (event_type, mult_range)
            for event_type, mult_range in event_types
            if type_day_counts[event_type] == min_count
        ]

        chosen_idx = int(rng.integers(0, len(candidates)))
        return candidates[chosen_idx]

    def try_sample_single_level_shift_event(
        df_plan,
        split_name,
        rng,
        event_id,
        type_day_counts,
        occupied_event_idx_by_store,
        start_margin_days,
        forced_max_duration=None,
    ):
        """
        Campiona un singolo evento level shift senza applicarlo al dataframe.

        Restituisce:
        - success: bool
        - event: dict oppure None

        Mantiene la stessa logica essenziale della vecchia injection:
        - evento dentro lo stesso split;
        - start dopo start_margin_days;
        - distanza minima da altri eventi nello stesso store;
        - tipo scelto bilanciando i giorni anomali per tipo.
        """

        candidate_stores = get_candidate_stores(
            df_plan=df_plan,
            split_name=split_name,
            start_margin_days=start_margin_days,
        )

        if len(candidate_stores) == 0:
            return False, None

        event_type, mult_range = choose_balanced_event_type(
            type_day_counts,
            rng,
        )

        min_duration, max_duration = duration_range

        if forced_max_duration is not None:
            max_duration = min(max_duration, int(forced_max_duration))

        if max_duration < min_duration:
            duration = min_duration
        else:
            duration = int(
                rng.integers(min_duration, max_duration + 1)
            )

        for _ in range(500):
            store_id = rng.choice(candidate_stores)

            store_split_idx = (
                df_plan[
                    (df_plan["store_id"] == store_id)
                    & (df_plan["_split"] == split_name)
                ]
                .sort_values("date")
                .index
                .to_numpy()
            )

            if len(store_split_idx) < start_margin_days + duration:
                continue

            # Come nella vecchia logica: si evita l'inizio dello split.
            # Il bordo finale è gestito dal controllo "evento nello stesso split".
            candidate_start_idx = store_split_idx[start_margin_days:]

            if len(candidate_start_idx) == 0:
                continue

            start_idx = int(rng.choice(candidate_start_idx))
            end_idx = start_idx + duration

            event_idx = np.arange(start_idx, end_idx)

            # L'evento deve restare nello stesso store e nello stesso split.
            if not set(event_idx).issubset(set(store_split_idx)):
                continue

            store_idx = (
                df_plan[df_plan["store_id"] == store_id]
                .sort_values("date")
                .index
                .to_numpy()
            )

            forbidden_idx = store_idx[
                (store_idx >= start_idx - min_gap_days)
                & (store_idx < end_idx + min_gap_days)
            ]

            occupied_event_idx = occupied_event_idx_by_store.get(store_id, set())

            if len(set(forbidden_idx).intersection(occupied_event_idx)) > 0:
                continue

            multiplier = float(rng.uniform(*mult_range))
            severity = "soft" if event_type.startswith("soft") else "hard"

            event = {
                "event_id": int(event_id),
                "split": split_name,
                "store_id": store_id,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "event_idx": event_idx.tolist(),
                "duration": int(duration),
                "lsa_type": event_type,
                "lsa_severity": severity,
                "lsa_mult": multiplier,
            }

            return True, event

        return False, None

    def sample_nested_level_shift_contamination_plan(
        df_base,
        max_contamination_fraction,
        splits=("train", "val"),
        seed=42,
        train_size=0.70,
        val_size=0.10,
        start_margin_days=35,
        max_attempts_per_split=20_000,
    ):
        """
        Campiona una sola volta il piano massimo di eventi level shift.

        I livelli inferiori e superiori useranno prefissi di questo piano.
        L'annidamento viene gestito separatamente per train e validation.
        """

        rng = np.random.default_rng(seed)

        df_plan = add_temporal_split(
            df_base,
            train_size=train_size,
            val_size=val_size,
        )

        split_sizes = (
            df_plan
            .groupby("_split")
            .size()
            .to_dict()
        )

        plan_rows = []
        next_event_id = 0

        for split_name in splits:
            n_days = int(split_sizes.get(split_name, 0))
            target_max_days = int(round(max_contamination_fraction * n_days))

            if target_max_days <= 0:
                continue

            type_day_counts = {
                event_type: 0
                for event_type, _ in event_types
            }

            occupied_event_idx_by_store = {}
            current_days = 0
            attempts = 0
            rank_in_split = 0

            while attempts < max_attempts_per_split:
                attempts += 1

                remaining_days = target_max_days - current_days

                if remaining_days <= 0:
                    break

                min_duration, max_duration = duration_range

                # Stessa logica della vecchia injection:
                # se il residuo è sotto la durata minima, si decide se fermarsi
                # oppure accettare un ultimo overshoot, scegliendo la distanza
                # minore dal target.
                if remaining_days < min_duration:
                    distance_if_stop = remaining_days
                    distance_if_add = min_duration - remaining_days

                    if distance_if_stop <= distance_if_add:
                        break

                    forced_max_duration = min_duration
                else:
                    forced_max_duration = min(max_duration, remaining_days)

                success, event = try_sample_single_level_shift_event(
                    df_plan=df_plan,
                    split_name=split_name,
                    rng=rng,
                    event_id=next_event_id,
                    type_day_counts=type_day_counts,
                    occupied_event_idx_by_store=occupied_event_idx_by_store,
                    start_margin_days=start_margin_days,
                    forced_max_duration=forced_max_duration,
                )

                if not success:
                    continue

                event["rank_in_split"] = rank_in_split
                event["cumulative_days_after_event"] = current_days + event["duration"]

                plan_rows.append(event)

                store_id = event["store_id"]
                occupied_event_idx_by_store.setdefault(store_id, set()).update(
                    event["event_idx"]
                )

                type_day_counts[event["lsa_type"]] += event["duration"]
                current_days += event["duration"]

                next_event_id += 1
                rank_in_split += 1

            if current_days < target_max_days:
                print(
                    f"ATTENZIONE: per split={split_name} generati "
                    f"{current_days}/{target_max_days} giorni anomali "
                    "nel piano massimo."
                )

        plan = pd.DataFrame(plan_rows)

        if not plan.empty:
            plan = (
                plan
                .sort_values(["split", "rank_in_split"])
                .reset_index(drop=True)
            )

        return plan, split_sizes

    def select_plan_for_level(
        plan,
        split_sizes,
        contamination_fraction,
        splits=("train", "val"),
    ):
        """
        Seleziona un prefisso del piano massimo per il livello richiesto.

        La selezione minimizza la distanza dal numero target di giorni anomali
        nello split, mantenendo l'annidamento.
        """

        if plan.empty or contamination_fraction <= 0:
            return plan.iloc[0:0].copy()

        selected_parts = []

        for split_name in splits:
            n_days = int(split_sizes.get(split_name, 0))
            target_days = int(round(contamination_fraction * n_days))

            split_plan = (
                plan[plan["split"] == split_name]
                .sort_values("rank_in_split")
                .copy()
            )

            if split_plan.empty or target_days <= 0:
                selected_parts.append(split_plan.iloc[0:0].copy())
                continue

            durations = split_plan["duration"].astype(int).to_numpy()
            cumulative = np.cumsum(durations)

            # Opzione 0 eventi.
            best_k = 0
            best_distance = abs(target_days - 0)

            for k, cum_days in enumerate(cumulative, start=1):
                distance = abs(target_days - int(cum_days))

                if distance < best_distance:
                    best_distance = distance
                    best_k = k

            selected_parts.append(split_plan.iloc[:best_k].copy())

        if len(selected_parts) == 0:
            return plan.iloc[0:0].copy()

        return pd.concat(selected_parts, ignore_index=True)

    # =========================================================
    # APPLICAZIONE DEL PIANO
    # =========================================================

    def apply_level_shift_contamination_plan(
        df_base,
        level_plan,
        contamination_level_code,
        contamination_percent,
        contamination_fraction,
        sales_col="daily_total_sales",
    ):
        """
        Applica al clean dataset un sottoinsieme del piano massimo.
        """

        df_level = initialize_level_shift_columns(df_base)

        if level_plan.empty:
            df_level["lsa_contamination_level"] = int(contamination_level_code)
            df_level["lsa_contamination_percent"] = float(contamination_percent)
            df_level["lsa_contamination_target_fraction"] = float(contamination_fraction)
            return df_level

        for _, event in level_plan.iterrows():
            event_idx = np.array(event["event_idx"], dtype=int)

            duration = int(event["duration"])
            event_id = int(event["event_id"])
            event_type = event["lsa_type"]
            severity = event["lsa_severity"]
            multiplier = float(event["lsa_mult"])
            split_name = event["split"]

            df_level.loc[event_idx, sales_col] *= multiplier

            df_level.loc[event_idx, "is_level_shift_anomaly"] = 1
            df_level.loc[event_idx, "lsa_type"] = event_type
            df_level.loc[event_idx, "lsa_severity"] = severity
            df_level.loc[event_idx, "lsa_mult"] = multiplier
            df_level.loc[event_idx, "lsa_event_id"] = event_id
            df_level.loc[event_idx, "lsa_duration"] = duration
            df_level.loc[event_idx, "lsa_day_in_event"] = np.arange(duration)

            df_level.loc[event_idx, "is_level_shift_contamination"] = 1
            df_level.loc[event_idx, "lsa_contamination_split"] = split_name

        df_level["lsa_contamination_level"] = int(contamination_level_code)
        df_level["lsa_contamination_percent"] = float(contamination_percent)
        df_level["lsa_contamination_target_fraction"] = float(contamination_fraction)

        return df_level

    # =========================================================
    # SUMMARY
    # =========================================================

    def build_level_summaries(
        df_level,
        split_sizes,
        contamination_level_code,
        contamination_percent,
        contamination_fraction,
        splits=("train", "val"),
        train_size=0.70,
        val_size=0.10,
    ):
        """
        Costruisce summary e type_summary con colonne compatibili
        con la versione precedente.
        """

        temp = add_temporal_split(
            df_level,
            train_size=train_size,
            val_size=val_size,
        )

        summary_rows = []
        type_summary_rows = []

        for split_name in splits:
            n_days = int(split_sizes.get(split_name, 0))
            target_days = int(round(contamination_fraction * n_days))

            split_mask = temp["_split"] == split_name

            split_anom_mask = (
                split_mask
                & (temp["is_level_shift_anomaly"].astype(int) == 1)
            )

            actual_days = int(split_anom_mask.sum())
            actual_fraction = actual_days / n_days if n_days > 0 else np.nan

            split_anom = temp[split_anom_mask].copy()

            n_events = int(
                split_anom["lsa_event_id"].nunique()
                if not split_anom.empty
                else 0
            )

            n_affected_stores = int(
                split_anom["store_id"].nunique()
                if not split_anom.empty
                else 0
            )

            summary_rows.append({
                "split": split_name,
                "contamination_level": int(contamination_level_code),
                "target_fraction": float(contamination_fraction),
                "actual_fraction": float(actual_fraction),
                "n_days": n_days,
                "target_anomaly_days": target_days,
                "actual_anomaly_days": actual_days,
                "n_events": n_events,
                "n_affected_stores": n_affected_stores,
                "reached_target": actual_days >= target_days,
            })

            for event_type, _ in event_types:
                type_mask = (
                    split_anom_mask
                    & (temp["lsa_type"] == event_type)
                )

                type_days = int(type_mask.sum())

                if actual_days > 0:
                    type_share = type_days / actual_days
                else:
                    type_share = 0.0

                type_events = int(
                    temp.loc[type_mask, "lsa_event_id"].nunique()
                    if type_days > 0
                    else 0
                )

                type_summary_rows.append({
                    "split": split_name,
                    "contamination_level": int(contamination_level_code),
                    "target_fraction": float(contamination_fraction),
                    "lsa_type": event_type,
                    "anomaly_days": type_days,
                    "anomaly_day_share": float(type_share),
                    "n_events": type_events,
                })

        summary = pd.DataFrame(summary_rows)
        type_summary = pd.DataFrame(type_summary_rows)

        return summary, type_summary

    # =========================================================
    # CAMPIONAMENTO UNICO DEL PIANO MASSIMO
    # =========================================================

    contamination_levels = tuple(int(level) for level in contamination_levels)

    max_level_code = max(contamination_levels)
    max_contamination_fraction = level_code_to_fraction(max_level_code)

    plan, split_sizes = sample_nested_level_shift_contamination_plan(
        df_base=clean_df,
        max_contamination_fraction=max_contamination_fraction,
        splits=("train", "val"),
        seed=global_seed,
        train_size=0.70,
        val_size=0.10,
        start_margin_days=35,
        max_attempts_per_split=20_000,
    )

    all_summary = []
    all_type_summary = []

    print("\n>>> GENERAZIONE ESPERIMENTO LEVEL SHIFT CONTAMINATION <<<")
    print(f"Dataset clean: {clean_path}")
    print(f"Output: {saving_path}")
    print(f"Livelli codice: {contamination_levels}")
    print(
        "Livelli percentuali:",
        [level_code_to_percent(level) for level in contamination_levels],
    )
    print(
        "Piano nested massimo:",
        0 if plan.empty else int(plan["duration"].sum()),
        "giorni anomali totali",
    )

    # =========================================================
    # LOOP LIVELLI DI CONTAMINAZIONE
    # =========================================================

    for raw_level_code in contamination_levels:

        contamination_level_code = int(raw_level_code)
        contamination_percent = level_code_to_percent(contamination_level_code)
        contamination_fraction = level_code_to_fraction(contamination_level_code)

        level_label = str(contamination_level_code)

        level_dir = saving_path / f"contamination_{level_label}"
        level_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = level_dir / "all_stores_cashflow.csv"
        summary_path = level_dir / "contamination_summary.csv"
        type_summary_path = level_dir / "contamination_type_summary.csv"
        config_path = level_dir / "config.json"

        if (
            dataset_path.exists()
            and summary_path.exists()
            and type_summary_path.exists()
            and config_path.exists()
            and not force_recompute
        ):
            print(
                f">>> Livello codice {contamination_level_code} "
                f"({contamination_percent:g}%) già esistente, "
                "carico summary salvate."
            )

            summary = pd.read_csv(summary_path)
            type_summary = pd.read_csv(type_summary_path)

        else:
            print(
                f"\n>>> Generazione contaminazione livello codice "
                f"{contamination_level_code} ({contamination_percent:g}%) <<<"
            )

            level_plan = select_plan_for_level(
                plan=plan,
                split_sizes=split_sizes,
                contamination_fraction=contamination_fraction,
                splits=("train", "val"),
            )

            df_cont = apply_level_shift_contamination_plan(
                df_base=clean_df,
                level_plan=level_plan,
                contamination_level_code=contamination_level_code,
                contamination_percent=contamination_percent,
                contamination_fraction=contamination_fraction,
                sales_col="daily_total_sales",
            )

            summary, type_summary = build_level_summaries(
                df_level=df_cont,
                split_sizes=split_sizes,
                contamination_level_code=contamination_level_code,
                contamination_percent=contamination_percent,
                contamination_fraction=contamination_fraction,
                splits=("train", "val"),
                train_size=0.70,
                val_size=0.10,
            )

            # =================================================
            # COLONNE UNIFORMI
            # =================================================

            df_cont["lsa_contamination_level"] = contamination_level_code
            df_cont["lsa_contamination_percent"] = contamination_percent
            df_cont["lsa_contamination_target_fraction"] = contamination_fraction

            summary = summary.copy()
            summary["contamination_level"] = contamination_level_code
            summary["contamination_percent"] = contamination_percent
            summary["target_fraction"] = contamination_fraction

            type_summary = type_summary.copy()
            type_summary["contamination_level"] = contamination_level_code
            type_summary["contamination_percent"] = contamination_percent
            type_summary["target_fraction"] = contamination_fraction

            # =================================================
            # SAVE DATASET + SUMMARY
            # =================================================

            df_cont.to_csv(dataset_path, index=False)
            summary.to_csv(summary_path, index=False)
            type_summary.to_csv(type_summary_path, index=False)

            config_payload = {
                "experiment": "level_shift_train_val_contamination",
                "clean_path": str(clean_path),
                "saving_path": str(level_dir),

                "contamination_level": contamination_level_code,
                "contamination_level_unit": "tenths_of_percentage_point",
                "contamination_percent": contamination_percent,
                "contamination_fraction": contamination_fraction,

                "splits_contaminated": ["train", "val"],
                "test_contaminated": False,

                "global_seed": global_seed,
                "plan_seed": global_seed,
                "nested_contamination": True,
                "max_contamination_level": max_level_code,
                "max_contamination_fraction": max_contamination_fraction,

                "level_cfg_contamination": level_cfg_contamination,
                "notes": (
                    "La contaminazione è controllata come quota globale di "
                    "store-day anomali separatamente su train e validation. "
                    "I livelli sono nested: ogni livello superiore contiene "
                    "gli eventi già presenti nei livelli inferiori e aggiunge "
                    "nuovi eventi dal medesimo piano campionato una sola volta. "
                    "Il test resta pulito. La colonna contamination_level usa "
                    "una codifica intera in decimi di punto percentuale: "
                    "5 indica 0.5%, 10 indica 1%, 50 indica 5%."
                ),
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=4)

        # =====================================================
        # METADATI PER SUMMARY AGGREGATE
        # =====================================================

        summary = summary.copy()
        summary["contamination_level"] = contamination_level_code
        summary["contamination_percent"] = contamination_percent
        summary["contamination_fraction"] = contamination_fraction
        summary["dataset_path"] = str(dataset_path)
        summary["level_dir"] = str(level_dir)

        type_summary = type_summary.copy()
        type_summary["contamination_level"] = contamination_level_code
        type_summary["contamination_percent"] = contamination_percent
        type_summary["contamination_fraction"] = contamination_fraction
        type_summary["dataset_path"] = str(dataset_path)
        type_summary["level_dir"] = str(level_dir)

        all_summary.append(summary)
        all_type_summary.append(type_summary)

    # =========================================================
    # SUMMARY AGGREGATE
    # =========================================================

    experiment_summary = (
        pd.concat(all_summary, ignore_index=True)
        if len(all_summary) > 0
        else pd.DataFrame()
    )

    experiment_type_summary = (
        pd.concat(all_type_summary, ignore_index=True)
        if len(all_type_summary) > 0
        else pd.DataFrame()
    )

    experiment_summary_path = (
        saving_path / "contamination_experiment_summary.csv"
    )

    experiment_type_summary_path = (
        saving_path / "contamination_experiment_type_summary.csv"
    )

    experiment_summary.to_csv(experiment_summary_path, index=False)
    experiment_type_summary.to_csv(experiment_type_summary_path, index=False)

    print("\n>>> ESPERIMENTO CONTAMINAZIONE LEVEL SHIFT COMPLETATO <<<")
    print(f"Summary: {experiment_summary_path}")
    print(f"Summary per tipo: {experiment_type_summary_path}")

    return experiment_summary, experiment_type_summary



def generate_pos_delay_contamination_experiment_from_clean(
    clean_path=CLEAN_DATA_PATH,
    saving_path=POS_DELAY_CONTAMINATION_DIR,
    contamination_levels=(0, 1, 2, 3, 5, 10),
    global_seed=42,
    force_recompute=False,
    normal_settlement_seed=123_456,
):
    """
    Genera i dataset POS delay per l'esperimento di robustezza alla
    contaminazione normal-only.

    I livelli sono nested/incrementali:
    - viene campionato una sola volta un piano massimo di source day contaminati;
    - ogni livello usa un prefisso del piano;
    - quindi ogni livello superiore contiene esattamente gli eventi
      dei livelli inferiori e aggiunge soltanto nuovi source day.

    I livelli sono espressi in decimi di punto percentuale:

        0  -> 0.0%
        1  -> 0.1%
        2  -> 0.2%
        3  -> 0.3%
        5  -> 0.5%
        10 -> 1.0%

    La contaminazione è una quota di source store-day POS delay,
    calcolata globalmente e separatamente su train e validation.
    Il test resta completamente pulito.

    Output invariati:

        saving_path/
            contamination_<level>/
                all_stores_cashflow.csv
                contamination_summary.csv
                contamination_type_summary.csv
                config.json

            contamination_experiment_summary.csv
            contamination_experiment_type_summary.csv
    """



    clean_path = Path(clean_path)
    saving_path = Path(saving_path)

    if not clean_path.exists():
        raise FileNotFoundError(f"Dataset clean non trovato: {clean_path}")

    if len(contamination_levels) == 0:
        raise ValueError("contamination_levels non può essere vuoto.")

    contamination_levels = tuple(int(level) for level in contamination_levels)

    if any(level < 0 for level in contamination_levels):
        raise ValueError("I livelli di contaminazione devono essere >= 0.")

    saving_path.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # HELPER LIVELLI
    # =========================================================

    def level_code_to_percent(level_code):
        return int(level_code) / 10.0

    def level_code_to_fraction(level_code):
        return level_code_to_percent(level_code) / 100.0

    def store_seed_component(store_id):
        digits = "".join(ch for ch in str(store_id) if ch.isdigit())

        if digits:
            return int(digits)

        return sum(ord(ch) for ch in str(store_id))

    def add_business_days_local(
        date,
        n_business_days,
        business_dates,
        end_date,
    ):
        """
        Restituisce l'n-esimo business day strettamente successivo a date.
        Se non disponibile, usa end_date come fallback.
        """

        future_business_dates = (
            business_dates[
                business_dates > pd.to_datetime(date)
            ]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(future_business_dates) >= n_business_days:
            return future_business_dates.iloc[n_business_days - 1]

        return pd.Timestamp(end_date)

    # =========================================================
    # LOAD CLEAN DATASET
    # =========================================================

    clean_df = pd.read_csv(clean_path)
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    clean_df = (
        clean_df
        .sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )

    if "day" not in clean_df.columns:
        clean_df["day"] = clean_df["date"].dt.day

    required_cols = [
        "date",
        "store_id",
        "day",
        "week_day",
        "month",
        "holiday",
        "actual_holiday",
        "pos_card_sales",
        "pos_volume_ratio",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in clean_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Nel clean dataset mancano queste colonne: {missing_cols}"
        )

    if "is_point_anomaly" not in clean_df.columns:
        clean_df["is_point_anomaly"] = 0

    if "pa_type" not in clean_df.columns:
        clean_df["pa_type"] = "normal"

    if "pa_mult" not in clean_df.columns:
        clean_df["pa_mult"] = 1.0

    # =========================================================
    # CONFIGURAZIONE CONTAMINAZIONE POS
    # =========================================================

    pos_cfg_contamination = deepcopy(
        get_def_anomaly_config()["pos_delay"]
    )

    # Eventi semplici: un solo source day.
    pos_cfg_contamination["duration_range"] = (1, 1)

    # Mantiene separati source day dello stesso store.
    pos_cfg_contamination["min_event_distance"] = 18

    # Evita source day troppo vicini ai bordi degli split.
    pos_cfg_contamination["edge_margin_source_days"] = 14

    pos_cfg_contamination["guaranteed_events_per_type"] = 0

    pos_cfg_contamination["enabled_types"] = [
        "mild_delay",
        "moderate_delay",
        "strong_delay",
        "batch_backlog",
        "settlement_freeze",
    ]

    pos_cfg_contamination["type_probs"] = {
        delay_type: 1.0 / len(
            pos_cfg_contamination["enabled_types"]
        )
        for delay_type in pos_cfg_contamination["enabled_types"]
    }

    params_by_store = {
        store["store_id"]: store["params"]
        for store in all_params
    }

    pos_analysis_cols = [
        "date",
        "day",
        "store_id",
        "week_day",
        "month",
        "holiday",
        "actual_holiday",
        "pos_card_sales",
        "pos_net_cf",
        "pos_volume_ratio",

        "is_point_anomaly",
        "pa_type",
        "pa_mult",

        "is_pos_delay_source_day",
        "pos_delay_source_day_in_event",
        "pos_delay_source_duration",

        "is_pos_delay_effect_day",
        "pos_delay_effect_day_in_event",
        "pos_delay_effect_duration",

        "pos_delay_event_id",
        "pos_delay_type",

        "is_pos_delay_contamination",
        "pos_delay_contamination_split",
        "pos_delay_contamination_level",
        "pos_delay_contamination_percent",
        "pos_delay_contamination_target_fraction",
    ]

    # =========================================================
    # HELPER DATAFRAME
    # =========================================================

    def initialize_pos_delay_columns(df):
        """
        Riparte dal clean e azzera tutte le colonne specifiche POS delay.
        """

        out = df.copy()

        out["is_pos_delay_source_day"] = 0
        out["pos_delay_source_day_in_event"] = -1
        out["pos_delay_source_duration"] = 0

        out["is_pos_delay_effect_day"] = 0
        out["pos_delay_effect_day_in_event"] = -1
        out["pos_delay_effect_duration"] = 0

        out["pos_delay_event_id"] = -1
        out["pos_delay_type"] = "normal"

        out["is_pos_delay_contamination"] = 0
        out["pos_delay_contamination_split"] = "none"
        out["pos_delay_contamination_level"] = 0
        out["pos_delay_contamination_percent"] = 0.0
        out["pos_delay_contamination_target_fraction"] = 0.0

        return out

    def add_temporal_split(
        df,
        train_size=0.70,
        val_size=0.10,
    ):
        """
        Crea split temporali separatamente per store.
        """

        out = df.copy()
        out["_split"] = ""

        for _, g in out.groupby("store_id", sort=False):
            idx = g.sort_values("date").index.to_numpy()

            n = len(idx)
            train_end = int(train_size * n)
            val_end = int((train_size + val_size) * n)

            out.loc[idx[:train_end], "_split"] = "train"
            out.loc[idx[train_end:val_end], "_split"] = "val"
            out.loc[idx[val_end:], "_split"] = "test"

        return out

    def get_business_dates_by_store(df):
        business_dates_by_store = {}

        for store_id, g in df.groupby("store_id", sort=False):
            business_dates_by_store[store_id] = (
                g.loc[
                    g["holiday"].astype(int) == 0,
                    "date",
                ]
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

        return business_dates_by_store

    def get_effect_dates_for_source(
        source_date,
        business_dates,
        end_date,
        max_effect_business_days=5,
    ):
        """
        Restituisce i T+1, ..., T+5 business day generati da un source day.
        """

        effect_dates = []

        for delay in range(1, max_effect_business_days + 1):
            effect_dates.append(
                add_business_days_local(
                    date=source_date,
                    n_business_days=delay,
                    business_dates=business_dates,
                    end_date=end_date,
                )
            )

        return (
            pd.Series(effect_dates, dtype="datetime64[ns]")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    # =========================================================
    # COSTRUZIONE PIANO MASSIMO NESTED
    # =========================================================

    def choose_balanced_delay_type(type_counts, rng):
        """
        Seleziona casualmente un tipo tra quelli al momento meno frequenti.
        """

        min_count = min(type_counts.values())

        candidates = [
            delay_type
            for delay_type, count in type_counts.items()
            if count == min_count
        ]

        return str(rng.choice(candidates))

    def source_candidate_is_valid(
        df_plan,
        source_idx,
        split_name,
        occupied_dates_by_store,
        business_dates_by_store,
        end_date_by_store,
        max_effect_business_days,
        min_event_distance,
    ):
        """
        Un source day è valido se:

        - appartiene allo split corretto;
        - dista più di min_event_distance dagli altri source day
          dello stesso store;
        - tutti i suoi effect day restano nello stesso split.
        """

        row = df_plan.loc[source_idx]

        store_id = row["store_id"]
        source_date = pd.to_datetime(row["date"])

        if row["_split"] != split_name:
            return False

        used_dates = occupied_dates_by_store.get(store_id, [])

        for used_date in used_dates:
            if abs((source_date - used_date).days) <= min_event_distance:
                return False

        effect_dates = get_effect_dates_for_source(
            source_date=source_date,
            business_dates=business_dates_by_store[store_id],
            end_date=end_date_by_store[store_id],
            max_effect_business_days=max_effect_business_days,
        )

        split_by_date = (
            df_plan[
                df_plan["store_id"] == store_id
            ]
            .set_index("date")["_split"]
            .to_dict()
        )

        # Garantisce che train e validation contaminati non propaghino
        # effect day al test o allo split adiacente.
        for effect_date in effect_dates:
            if split_by_date.get(pd.Timestamp(effect_date)) != split_name:
                return False

        return True

    def sample_nested_pos_delay_contamination_plan(
        df_base,
        max_contamination_fraction,
        splits=("train", "val"),
        seed=42,
        train_size=0.70,
        val_size=0.10,
        max_effect_business_days=5,
        max_attempts_per_split=20_000,
    ):
        """
        Campiona una sola volta il piano massimo di source day.

        La posizione nel piano, rank_in_split, è fondamentale:
        ogni livello di contaminazione userà il prefisso del piano
        fino al numero di source day richiesto.
        """

        rng = np.random.default_rng(seed)

        df_plan = add_temporal_split(
            df_base,
            train_size=train_size,
            val_size=val_size,
        )

        business_dates_by_store = get_business_dates_by_store(df_plan)

        end_date_by_store = {
            store_id: pd.to_datetime(g["date"]).max()
            for store_id, g in df_plan.groupby(
                "store_id",
                sort=False,
            )
        }

        min_event_distance = int(
            pos_cfg_contamination["min_event_distance"]
        )

        edge_margin = int(
            pos_cfg_contamination["edge_margin_source_days"]
        )

        enabled_types = list(
            pos_cfg_contamination["enabled_types"]
        )

        plan_rows = []
        event_id = 0

        for split_name in splits:
            split_df = df_plan[
                df_plan["_split"] == split_name
            ].copy()

            n_days = len(split_df)

            target_max_source_days = int(
                round(max_contamination_fraction * n_days)
            )

            if target_max_source_days <= 0:
                continue

            type_counts = {
                delay_type: 0
                for delay_type in enabled_types
            }

            occupied_dates_by_store = {}

            candidate_idx_parts = []

            for _, g in split_df.groupby("store_id", sort=False):
                g = g.sort_values("date").copy()

                if len(g) <= 2 * edge_margin:
                    continue

                candidate_idx_parts.append(
                    g.iloc[
                        edge_margin:len(g) - edge_margin
                    ].index.to_numpy()
                )

            if not candidate_idx_parts:
                continue

            candidate_idx = np.concatenate(candidate_idx_parts)

            attempts = 0

            while (
                len(
                    [
                        row
                        for row in plan_rows
                        if row["split"] == split_name
                    ]
                )
                < target_max_source_days
                and attempts < max_attempts_per_split
            ):
                attempts += 1

                source_idx = int(rng.choice(candidate_idx))

                is_valid = source_candidate_is_valid(
                    df_plan=df_plan,
                    source_idx=source_idx,
                    split_name=split_name,
                    occupied_dates_by_store=occupied_dates_by_store,
                    business_dates_by_store=business_dates_by_store,
                    end_date_by_store=end_date_by_store,
                    max_effect_business_days=max_effect_business_days,
                    min_event_distance=min_event_distance,
                )

                if not is_valid:
                    continue

                source_row = df_plan.loc[source_idx]

                store_id = source_row["store_id"]
                source_date = pd.to_datetime(source_row["date"])

                delay_type = choose_balanced_delay_type(
                    type_counts=type_counts,
                    rng=rng,
                )

                effect_dates = get_effect_dates_for_source(
                    source_date=source_date,
                    business_dates=business_dates_by_store[store_id],
                    end_date=end_date_by_store[store_id],
                    max_effect_business_days=max_effect_business_days,
                )

                rank_in_split = len(
                    [
                        row
                        for row in plan_rows
                        if row["split"] == split_name
                    ]
                )

                plan_rows.append({
                    "event_id": event_id,
                    "split": split_name,
                    "rank_in_split": rank_in_split,
                    "store_id": store_id,
                    "source_idx": source_idx,
                    "source_date": source_date,
                    "source_duration": 1,
                    "pos_delay_type": delay_type,
                    "effect_dates": effect_dates,
                    "effect_duration": len(effect_dates),
                })

                occupied_dates_by_store.setdefault(
                    store_id,
                    [],
                ).append(source_date)

                type_counts[delay_type] += 1
                event_id += 1

            actual_split_events = len(
                [
                    row
                    for row in plan_rows
                    if row["split"] == split_name
                ]
            )

            if actual_split_events < target_max_source_days:
                print(
                    f"ATTENZIONE: per split={split_name} generati "
                    f"{actual_split_events}/{target_max_source_days} "
                    "source day nel piano massimo."
                )

        plan = pd.DataFrame(plan_rows)

        if not plan.empty:
            plan = (
                plan
                .sort_values(["split", "rank_in_split"])
                .reset_index(drop=True)
            )

        split_sizes = (
            df_plan
            .groupby("_split")
            .size()
            .to_dict()
        )

        return plan, split_sizes

    def select_plan_for_level(
        plan,
        split_sizes,
        contamination_fraction,
        splits=("train", "val"),
    ):
        """
        Estrae il prefisso del piano massimo richiesto dal livello.
        """

        if plan.empty or contamination_fraction <= 0:
            return plan.iloc[0:0].copy()

        selected_parts = []

        for split_name in splits:
            n_days = int(split_sizes.get(split_name, 0))

            target_source_days = int(
                round(contamination_fraction * n_days)
            )

            split_plan = plan[
                (plan["split"] == split_name)
                & (
                    plan["rank_in_split"]
                    < target_source_days
                )
            ].copy()

            selected_parts.append(split_plan)

        if not selected_parts:
            return plan.iloc[0:0].copy()

        return pd.concat(
            selected_parts,
            ignore_index=True,
        )

    # =========================================================
    # APPLICAZIONE DI UN LIVELLO DEL PIANO
    # =========================================================

    def apply_pos_delay_contamination_plan(
        df_base,
        level_plan,
        contamination_level_code,
        contamination_percent,
        contamination_fraction,
    ):
        """
        Ricostruisce il dataset partendo sempre dal clean e applica
        il sottoinsieme nested richiesto.
        """

        df_level = initialize_pos_delay_columns(df_base)
        df_level["date"] = pd.to_datetime(df_level["date"])

        if level_plan.empty:
            return df_level

        for _, event in level_plan.iterrows():
            event_id = int(event["event_id"])
            source_idx = int(event["source_idx"])
            split_name = event["split"]
            store_id = event["store_id"]
            delay_type = event["pos_delay_type"]

            effect_dates = list(event["effect_dates"])
            effect_duration = int(event["effect_duration"])

            # Source day.
            df_level.loc[
                source_idx,
                "is_pos_delay_source_day",
            ] = 1

            df_level.loc[
                source_idx,
                "pos_delay_source_day_in_event",
            ] = 0

            df_level.loc[
                source_idx,
                "pos_delay_source_duration",
            ] = 1

            df_level.loc[
                source_idx,
                "pos_delay_event_id",
            ] = event_id

            df_level.loc[
                source_idx,
                "pos_delay_type",
            ] = delay_type

            df_level.loc[
                source_idx,
                "is_pos_delay_contamination",
            ] = 1

            df_level.loc[
                source_idx,
                "pos_delay_contamination_split",
            ] = split_name

            # Effect days.
            for day_in_effect, effect_date in enumerate(effect_dates):
                effect_date = pd.Timestamp(effect_date)

                mask = (
                    (df_level["store_id"] == store_id)
                    & (df_level["date"] == effect_date)
                )

                df_level.loc[
                    mask,
                    "is_pos_delay_effect_day",
                ] = 1

                df_level.loc[
                    mask,
                    "pos_delay_effect_day_in_event",
                ] = day_in_effect

                df_level.loc[
                    mask,
                    "pos_delay_effect_duration",
                ] = effect_duration

                df_level.loc[
                    mask,
                    "pos_delay_event_id",
                ] = event_id

                df_level.loc[
                    mask,
                    "pos_delay_type",
                ] = delay_type

                df_level.loc[
                    mask,
                    "is_pos_delay_contamination",
                ] = 1

                df_level.loc[
                    mask,
                    "pos_delay_contamination_split",
                ] = split_name

        df_level["pos_delay_contamination_level"] = int(
            contamination_level_code
        )

        df_level["pos_delay_contamination_percent"] = float(
            contamination_percent
        )

        df_level["pos_delay_contamination_target_fraction"] = float(
            contamination_fraction
        )

        return df_level

    # =========================================================
    # RICALCOLO POS NET CASHFLOW
    # =========================================================

    def recompute_pos_net_cf_from_delay_labels(
        df_level,
        seed_for_settlement,
    ):
        """
        Ricalcola pos_net_cf usando le label POS delay.

        Il seed è costante tra livelli: uno stesso source day ha quindi
        gli stessi pesi Dirichlet in ogni livello che lo contiene.
        """

        output_parts = []

        lambda_low = 3
        lambda_high = 5

        gamma_low = 3 / 4
        gamma_high = 3 / 2

        alpha_low_5 = np.array(
            [40.0, 7.5, 2.5, 0.0, 0.0],
            dtype=float,
        )

        alpha_base_5 = np.array(
            [35.0, 12.5, 2.5, 0.0, 0.0],
            dtype=float,
        )

        alpha_stress_5 = np.array(
            [20.0, 17.5, 12.5, 0.0, 0.0],
            dtype=float,
        )

        settlement_days = [1, 2, 3, 4, 5]

        for store_id, store_df in df_level.groupby(
            "store_id",
            sort=False,
        ):
            df_store = store_df.copy()

            df_store["date"] = pd.to_datetime(
                df_store["date"]
            )

            df_store = df_store.sort_values("date").copy()

            if store_id not in params_by_store:
                raise KeyError(
                    f"store_id non trovato in all_params: {store_id}"
                )

            pos_commission_rate = params_by_store[
                store_id
            ]["pos_commission_rate"]

            df_store["pos_net_amount"] = (
                df_store["pos_card_sales"].astype(float)
                * (1 - pos_commission_rate)
            ).round(2)

            business_dates = (
                df_store.loc[
                    df_store["holiday"].astype(int) == 0,
                    "date",
                ]
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

            end_date = df_store["date"].max()

            pos_cf_by_date = pd.Series(
                0.0,
                index=pd.to_datetime(df_store["date"]),
            )

            store_seed = store_seed_component(store_id)

            for row_pos, (_, row) in enumerate(
                df_store.iterrows()
            ):
                date = pd.to_datetime(row["date"])

                if int(row["is_pos_delay_source_day"]) == 1:
                    delay_type = row["pos_delay_type"]

                    alpha = np.array(
                        pos_cfg_contamination[
                            "delay_profiles"
                        ][delay_type]["alpha"],
                        dtype=float,
                    )

                else:
                    volume_ratio = float(row["pos_volume_ratio"])

                    low_weight = np.exp(
                        -lambda_low
                        * max(volume_ratio - 0.8, 0)
                        ** gamma_low
                    )

                    stress_weight = 1 - np.exp(
                        -lambda_high
                        * max(volume_ratio - 1, 0)
                        ** gamma_high
                    )

                    base_weight = 1 - low_weight - stress_weight
                    base_weight = np.clip(
                        base_weight,
                        0,
                        1,
                    )

                    alpha = (
                        low_weight * alpha_low_5
                        + base_weight * alpha_base_5
                        + stress_weight * alpha_stress_5
                    )

                alpha = np.clip(alpha, 1e-6, None)

                rng_row = np.random.default_rng(
                    normal_settlement_seed
                    + int(seed_for_settlement)
                    + store_seed * 1_000_000
                    + row_pos
                )

                weights = rng_row.dirichlet(alpha)

                for delay, weight in zip(
                    settlement_days,
                    weights,
                ):
                    settlement_date = add_business_days_local(
                        date=date,
                        n_business_days=delay,
                        business_dates=business_dates,
                        end_date=end_date,
                    )

                    if settlement_date in pos_cf_by_date.index:
                        pos_cf_by_date.loc[
                            settlement_date
                        ] += (
                            float(row["pos_net_amount"])
                            * float(weight)
                        )

            df_store["pos_net_cf"] = (
                df_store["date"]
                .map(pos_cf_by_date)
                .fillna(0.0)
                .astype(float)
                .round(2)
            )

            df_store = df_store.drop(
                columns=["pos_net_amount"]
            )

            output_parts.append(df_store)

        return (
            pd.concat(output_parts, ignore_index=True)
            .sort_values(["store_id", "date"])
            .reset_index(drop=True)
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    def build_level_summaries(
        level_plan,
        split_sizes,
        contamination_level_code,
        contamination_percent,
        contamination_fraction,
        splits=("train", "val"),
    ):
        summary_rows = []
        type_summary_rows = []

        enabled_types = list(
            pos_cfg_contamination["enabled_types"]
        )

        for split_name in splits:
            n_days = int(split_sizes.get(split_name, 0))

            target_source_days = int(
                round(contamination_fraction * n_days)
            )

            split_plan = (
                level_plan[
                    level_plan["split"] == split_name
                ].copy()
                if not level_plan.empty
                else level_plan.copy()
            )

            actual_source_days = int(len(split_plan))

            actual_fraction = (
                actual_source_days / n_days
                if n_days > 0
                else 0.0
            )

            if actual_source_days > 0:
                n_effect_days = int(
                    split_plan
                    .explode("effect_dates")[
                        ["store_id", "effect_dates"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )

                n_affected_stores = int(
                    split_plan["store_id"].nunique()
                )

            else:
                n_effect_days = 0
                n_affected_stores = 0

            summary_rows.append({
                "split": split_name,
                "contamination_level": int(
                    contamination_level_code
                ),
                "contamination_percent": float(
                    contamination_percent
                ),
                "target_fraction": float(
                    contamination_fraction
                ),
                "actual_fraction": float(
                    actual_fraction
                ),
                "n_days": n_days,
                "target_anomaly_days": target_source_days,
                "actual_anomaly_days": actual_source_days,
                "actual_source_days": actual_source_days,
                "actual_effect_days": n_effect_days,
                "n_events": actual_source_days,
                "n_affected_stores": n_affected_stores,
            })

            for delay_type in enabled_types:
                type_plan = split_plan[
                    split_plan["pos_delay_type"] == delay_type
                ].copy()

                n_events = int(len(type_plan))

                if n_events > 0:
                    n_effect_type_days = int(
                        type_plan
                        .explode("effect_dates")[
                            ["store_id", "effect_dates"]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    )

                    n_type_stores = int(
                        type_plan["store_id"].nunique()
                    )

                else:
                    n_effect_type_days = 0
                    n_type_stores = 0

                type_summary_rows.append({
                    "split": split_name,
                    "delay_type": delay_type,
                    "pos_delay_type": delay_type,
                    "contamination_level": int(
                        contamination_level_code
                    ),
                    "contamination_percent": float(
                        contamination_percent
                    ),
                    "target_fraction": float(
                        contamination_fraction
                    ),
                    "n_events": n_events,
                    "source_days": n_events,
                    "effect_days": n_effect_type_days,
                    "n_affected_stores": n_type_stores,
                })

        return (
            pd.DataFrame(summary_rows),
            pd.DataFrame(type_summary_rows),
        )

    # =========================================================
    # CAMPIONAMENTO DEL PIANO MASSIMO
    # =========================================================

    max_level_code = max(contamination_levels)

    max_contamination_fraction = level_code_to_fraction(
        max_level_code
    )

    plan, split_sizes = sample_nested_pos_delay_contamination_plan(
        df_base=clean_df,
        max_contamination_fraction=max_contamination_fraction,
        splits=("train", "val"),
        seed=global_seed,
        train_size=0.70,
        val_size=0.10,
        max_effect_business_days=5,
        max_attempts_per_split=20_000,
    )

    all_summary = []
    all_type_summary = []

    print("\n>>> GENERAZIONE ESPERIMENTO POS DELAY CONTAMINATION <<<")
    print(f"Dataset clean: {clean_path}")
    print(f"Output: {saving_path}")
    print(f"Livelli codice: {contamination_levels}")
    print(
        "Livelli percentuali:",
        [
            level_code_to_percent(level)
            for level in contamination_levels
        ],
    )
    print(
        "Piano nested massimo:",
        0 if plan.empty else len(plan),
        "source day totali",
    )

    # =========================================================
    # LOOP LIVELLI
    # =========================================================

    for raw_level_code in contamination_levels:
        contamination_level_code = int(raw_level_code)

        contamination_percent = level_code_to_percent(
            contamination_level_code
        )

        contamination_fraction = level_code_to_fraction(
            contamination_level_code
        )

        level_dir = (
            saving_path
            / f"contamination_{contamination_level_code}"
        )

        level_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = level_dir / "all_stores_cashflow.csv"
        summary_path = level_dir / "contamination_summary.csv"

        type_summary_path = (
            level_dir
            / "contamination_type_summary.csv"
        )

        config_path = level_dir / "config.json"

        already_done = (
            dataset_path.exists()
            and summary_path.exists()
            and type_summary_path.exists()
            and config_path.exists()
            and not force_recompute
        )

        if already_done:
            print(
                f">>> Livello codice {contamination_level_code} "
                f"({contamination_percent:g}%) già esistente, "
                "carico summary salvate."
            )

            summary = pd.read_csv(summary_path)
            type_summary = pd.read_csv(type_summary_path)

        else:
            print(
                f"\n>>> Generazione contaminazione POS livello "
                f"{contamination_level_code} "
                f"({contamination_percent:g}%) <<<"
            )

            level_plan = select_plan_for_level(
                plan=plan,
                split_sizes=split_sizes,
                contamination_fraction=contamination_fraction,
                splits=("train", "val"),
            )

            df_cont = apply_pos_delay_contamination_plan(
                df_base=clean_df,
                level_plan=level_plan,
                contamination_level_code=contamination_level_code,
                contamination_percent=contamination_percent,
                contamination_fraction=contamination_fraction,
            )

            # Costante fra livelli: preserva la comparabilità nested.
            df_cont = recompute_pos_net_cf_from_delay_labels(
                df_level=df_cont,
                seed_for_settlement=global_seed,
            )

            summary, type_summary = build_level_summaries(
                level_plan=level_plan,
                split_sizes=split_sizes,
                contamination_level_code=contamination_level_code,
                contamination_percent=contamination_percent,
                contamination_fraction=contamination_fraction,
                splits=("train", "val"),
            )

            df_cont = df_cont[pos_analysis_cols].copy()

            df_cont.to_csv(dataset_path, index=False)
            summary.to_csv(summary_path, index=False)
            type_summary.to_csv(type_summary_path, index=False)

            config_payload = {
                "experiment": "pos_delay_train_val_contamination",
                "clean_path": str(clean_path),
                "saving_path": str(level_dir),
                "contamination_level": contamination_level_code,
                "contamination_level_unit": (
                    "tenths_of_percentage_point"
                ),
                "contamination_percent": contamination_percent,
                "contamination_fraction": contamination_fraction,
                "splits_contaminated": ["train", "val"],
                "test_contaminated": False,
                "global_seed": global_seed,
                "plan_seed": global_seed,
                "settlement_seed": global_seed,
                "normal_settlement_seed": normal_settlement_seed,
                "nested_contamination": True,
                "max_contamination_level": max_level_code,
                "max_contamination_fraction": (
                    max_contamination_fraction
                ),
                "pos_cfg_contamination": pos_cfg_contamination,
                "notes": (
                    "La contaminazione è controllata come quota globale di "
                    "source store-day POS delay. I livelli sono nested: ogni "
                    "livello superiore contiene i source day dei livelli "
                    "inferiori e aggiunge nuovi source day dal medesimo piano "
                    "campionato una sola volta. Gli effect day sono marcati e "
                    "pos_net_cf viene ricalcolato dopo l'iniezione. Il seed di "
                    "settlement resta costante tra livelli per rendere il "
                    "confronto incrementale pulito."
                ),
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=4)

        summary = summary.copy()
        summary["contamination_level"] = contamination_level_code
        summary["contamination_percent"] = contamination_percent
        summary["contamination_fraction"] = contamination_fraction
        summary["dataset_path"] = str(dataset_path)
        summary["level_dir"] = str(level_dir)

        type_summary = type_summary.copy()
        type_summary["contamination_level"] = contamination_level_code
        type_summary["contamination_percent"] = contamination_percent
        type_summary["contamination_fraction"] = contamination_fraction
        type_summary["dataset_path"] = str(dataset_path)
        type_summary["level_dir"] = str(level_dir)

        all_summary.append(summary)
        all_type_summary.append(type_summary)

    experiment_summary = (
        pd.concat(all_summary, ignore_index=True)
        if all_summary
        else pd.DataFrame()
    )

    experiment_type_summary = (
        pd.concat(all_type_summary, ignore_index=True)
        if all_type_summary
        else pd.DataFrame()
    )

    experiment_summary_path = (
        saving_path
        / "contamination_experiment_summary.csv"
    )

    experiment_type_summary_path = (
        saving_path
        / "contamination_experiment_type_summary.csv"
    )

    experiment_summary.to_csv(
        experiment_summary_path,
        index=False,
    )

    experiment_type_summary.to_csv(
        experiment_type_summary_path,
        index=False,
    )

    print("\n>>> ESPERIMENTO CONTAMINAZIONE POS DELAY COMPLETATO <<<")
    print(f"Summary: {experiment_summary_path}")
    print(f"Summary per tipo: {experiment_type_summary_path}")

    return experiment_summary, experiment_type_summary


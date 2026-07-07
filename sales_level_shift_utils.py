# -*- coding: utf-8 -*-
"""
Utility per la rilevazione e la valutazione delle anomalie sales level shift.

Il modulo raccoglie le funzioni comuni usate negli esperimenti:
1. individuazione dei dataset di sensitivity;
2. calcolo dello score dai residui del modello;
3. applicazione di soglie specifiche per store;
4. costruzione delle finestre rilevate e delle finestre ground truth;
5. valutazione event-level delle detection;
6. dataset, modello e postprocessing LSTM Autoencoder per il confronto reconstruction-based;
7. preprocessing adattivo e soglie robuste per gli esperimenti di contaminazione.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Embedding,
    Flatten,
    Concatenate,
    Dropout,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model

from lstm_utils import (
    encode_categorical,
    create_sequences,
    append_sequence_parts,
    concatenate_parts,
    create_ae_windows,
    build_dataset_inference,
    build_model_inputs,
    build_sales_ae_inputs,
    make_results_df,
)


# =========================================================
# FEATURE LIST
# =========================================================

def get_feature_lists():

    # Schema legacy usato dal modello sales selezionato e dai relativi artifact.
    # =========================
    # FEATURE NUMERICHE SEQUENZIALI
    # =========================
    seq_num_features = [
        "daily_total_sales",
    ]

    # =========================
    # FEATURE BOOLEANE SEQUENZIALI
    # =========================
    seq_bool_features = [
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    # =========================
    # FEATURE CATEGORICHE SEQUENZIALI
    # =========================
    seq_cat_features = [
        "week_day",
        "month",
        "day",
    ]

    # =========================
    # FEATURE NUMERICHE FINALI
    # =========================
    final_num_features = [
        "time_idx",
        "oil_price",
        "consumer_confidence",
        "fao",
    ]

    # =========================
    # FEATURE BOOLEANE FINALI
    # =========================
    # Indicatori 0/1 riferiti al tempo di previsione.
    # Restano covariate booleane e non vengono passati a embedding.
    final_bool_features = [
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    # =========================
    # FEATURE CATEGORICHE FINALI
    # =========================
    cat_features = [
        "store_id",
        "week_day",
        "month",
        "day",
    ]

    # =========================
    # GROUND TRUTH
    # Usata solo per conservarla nelle sequenze, non nel training.
    # Compatibile con la denominazione aggiornata level_shift.
    # =========================
    ground_truth_features = [
        "is_level_shift_anomaly",
        "lsa_type",
        "lsa_mult",
        "lsa_severity",
        "lsa_event_id",
        "lsa_day_in_event",
        "lsa_duration",
    ]

    target = "daily_total_sales"

    # Queste colonne vengono create come nel notebook originale.
    # sales_rm_7 e sales_rm_30 restano escluse dagli input se non sono nelle liste sopra.
    log_transform_features = [
        "daily_total_sales",
        "oil_price",
        "fao",
    ]

    scale_features = list(dict.fromkeys(
        seq_num_features
        + final_num_features
        + [target]
    ))

    return {
        "seq_num": seq_num_features,
        "seq_bool": seq_bool_features,
        "seq_cat": seq_cat_features,
        "final_num": final_num_features,
        "final_bool": final_bool_features,
        "cat": cat_features,
        "ground_truth": ground_truth_features,
        "target": target,
        "log_transform": log_transform_features,
        "scale": scale_features,
    }


# =========================================================
# TRAIN / VALIDATION DATASET BUILDER
# =========================================================

def build_dataset_train_val(
    df,
    window_size,
    train_size=0.70,
    val_size=0.10,
):
    features = get_feature_lists()

    df_prepared = df.copy()
    df_prepared["date"] = pd.to_datetime(df_prepared["date"])
    df_prepared = df_prepared.sort_values(
        ["store_id", "date"]
    ).reset_index(drop=True)

    df_prepared = ensure_ground_truth_columns(
        df_prepared,
        features["ground_truth"],
    )

    df_prepared["pre_holiday"] = (
        df_prepared
        .groupby("store_id")["actual_holiday"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    df_prepared["time_idx"] = (
        df_prepared
        .groupby("store_id")
        .cumcount()
    )

    for col in features["log_transform"]:
        df_prepared[col] = np.log1p(
            df_prepared[col].astype(float)
        )

    df_prepared, mappings = encode_categorical(
        df_prepared,
        features,
    )

    train_parts = {
        "X_seq_num": [],
        "X_seq_bool": [],
        "X_seq_cat": [],
        "X_final_num": [],
        "X_final_bool": [],
        "X_cat": [],
        "y": [],
        "date": [],
        "ground_truth": [],
    }

    val_parts = {
        "X_seq_num": [],
        "X_seq_bool": [],
        "X_seq_cat": [],
        "X_final_num": [],
        "X_final_bool": [],
        "X_cat": [],
        "y": [],
        "date": [],
        "ground_truth": [],
    }

    feature_scalers = {}

    # Gli scaler vengono stimati solo sul train e separatamente per store.
    for store_id, store_df in df_prepared.groupby("store_id"):
        store_df = store_df.sort_values("date").copy()

        n_rows = len(store_df)
        train_end = int(train_size * n_rows)
        val_end = int((train_size + val_size) * n_rows)

        train_df = store_df.iloc[:train_end].copy()
        val_df = store_df.iloc[train_end:val_end].copy()

        scaler = StandardScaler()
        scale_cols = features["scale"]

        scaler.fit(
            train_df[scale_cols].astype(float)
        )

        train_df[scale_cols] = scaler.transform(
            train_df[scale_cols].astype(float)
        )

        val_df[scale_cols] = scaler.transform(
            val_df[scale_cols].astype(float)
        )

        feature_scalers[store_id] = scaler

        append_sequence_parts(
            train_parts,
            create_sequences(
                train_df,
                features,
                window_size,
            ),
        )

        append_sequence_parts(
            val_parts,
            create_sequences(
                val_df,
                features,
                window_size,
            ),
        )

    train = concatenate_parts(train_parts)
    val = concatenate_parts(val_parts)

    return train, val, feature_scalers, mappings, features

# =========================================================
# BUILD TRAIN / VAL / TEST USANDO GLI ARTIFATTI DEL MODELLO SCELTO
# =========================================================

def build_dataset_train_val_test_from_artifacts(
    df,
    feature_scalers,
    mappings,
    features,
    window_size,
    train_size=0.70,
    val_size=0.10,
):
    features = normalize_sales_feature_schema(features)

    df_eval = df.copy()
    df_eval["date"] = pd.to_datetime(df_eval["date"])
    df_eval = df_eval.sort_values(["store_id", "date"]).copy()

    df_eval = ensure_ground_truth_columns(
        df_eval,
        features["ground_truth"],
    )

    df_eval["pre_holiday"] = (
        df_eval.groupby("store_id")["actual_holiday"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    df_eval["time_idx"] = df_eval.groupby("store_id").cumcount()

    df_eval["sales_rm_30"] = (
        df_eval.groupby("store_id")["daily_total_sales"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )
    df_eval["sales_rm_30"] = df_eval.groupby("store_id")["sales_rm_30"].bfill()

    df_eval["sales_rm_7"] = (
        df_eval.groupby("store_id")["daily_total_sales"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df_eval["sales_rm_7"] = df_eval.groupby("store_id")["sales_rm_7"].bfill()

    df_eval["days_to_month_end"] = (
        df_eval["date"].dt.days_in_month - df_eval["date"].dt.day
    )

    # Replica le trasformazioni applicate prima del salvataggio degli artifact.
    for col in features["log_transform"]:
        df_eval[col] = np.log1p(df_eval[col])

    # Usa gli stessi mapping del modello scelto
    for col, mapping in mappings.items():
        df_eval[col] = pd.Categorical(
            df_eval[col],
            categories=mapping,
        ).codes

        if (df_eval[col] < 0).any():
            raise ValueError(
                f"La colonna {col} contiene categorie non viste nel training."
            )

    split_parts = {
        split: {
            "X_seq_num": [],
            "X_seq_bool": [],
            "X_seq_cat": [],
            "X_final_num": [],
            "X_final_bool": [],
            "X_cat": [],
            "y": [],
            "date": [],
            "ground_truth": [],
        }
        for split in ["train", "val", "test"]
    }

    for store_id in df_eval["store_id"].unique():
        temp = df_eval[df_eval["store_id"] == store_id].sort_values("date").copy()
        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        # Riusa gli scaler del modello senza effettuare un nuovo fit.
        scaler = feature_scalers[store_id]
        num_to_scale = features["scale"]

        train_df[num_to_scale] = scaler.transform(train_df[num_to_scale].astype(float))
        val_df[num_to_scale] = scaler.transform(val_df[num_to_scale].astype(float))
        test_df[num_to_scale] = scaler.transform(test_df[num_to_scale].astype(float))

        append_sequence_parts(
            split_parts["train"],
            create_sequences(train_df, features, window_size),
        )
        append_sequence_parts(
            split_parts["val"],
            create_sequences(val_df, features, window_size),
        )
        append_sequence_parts(
            split_parts["test"],
            create_sequences(test_df, features, window_size),
        )

    train = concatenate_parts(split_parts["train"])
    val = concatenate_parts(split_parts["val"])
    test = concatenate_parts(split_parts["test"])

    return train, val, test


# =========================================================
# MODEL BUILDER
# =========================================================

def _cardinality(mappings, col):
    if col not in mappings:
        raise KeyError(f"Colonna categorica non trovata nei mappings: {col}")
    return len(mappings[col])


def build_lstm_sales_model(
    train,
    mappings,
    architecture_config,
    dropout_rate=0.05,
    learning_rate=1e-3,
):
    """
    Modello LSTM multi-input per daily_total_sales.
    L'architettura varia secondo ARCHITECTURE_CONFIGS.
    """

    X_seq_num_train = train["X_seq_num"]
    X_seq_bool_train = train["X_seq_bool"]
    X_final_num_train = train["X_final_num"]
    X_final_bool_train = train["X_final_bool"]

    window_size = X_seq_num_train.shape[1]
    n_num_features = X_seq_num_train.shape[2]
    n_bool_features = X_seq_bool_train.shape[2]
    n_final_num = X_final_num_train.shape[1]
    n_final_bool = X_final_bool_train.shape[1]

    n_seq_weekday = _cardinality(mappings, "week_day")
    n_seq_month = _cardinality(mappings, "month")
    n_seq_day = _cardinality(mappings, "day")

    n_store = _cardinality(mappings, "store_id")
    n_weekday = _cardinality(mappings, "week_day")
    n_month = _cardinality(mappings, "month")
    n_day = _cardinality(mappings, "day")

    # =========================
    # INPUT SEQUENZIALI
    # =========================
    seq_num_input = Input(
        shape=(window_size, n_num_features),
        name="seq_num_input",
    )

    seq_bool_input = Input(
        shape=(window_size, n_bool_features),
        name="seq_bool_input",
    )

    seq_weekday_input = Input(
        shape=(window_size,),
        name="seq_weekday_input",
    )

    seq_month_input = Input(
        shape=(window_size,),
        name="seq_month_input",
    )

    seq_day_input = Input(
        shape=(window_size,),
        name="seq_day_input",
    )

    # =========================
    # EMBEDDING SEQUENZIALI
    # =========================
    seq_week_emb = Embedding(
        input_dim=n_seq_weekday,
        output_dim=3,
        name="seq_weekday_embedding",
    )(seq_weekday_input)

    seq_month_emb = Embedding(
        input_dim=n_seq_month,
        output_dim=3,
        name="seq_month_embedding",
    )(seq_month_input)

    seq_day_emb = Embedding(
        input_dim=n_seq_day,
        output_dim=3,
        name="seq_day_embedding",
    )(seq_day_input)

    seq_input = Concatenate(axis=-1, name="seq_input_concat")([
        seq_num_input,
        seq_bool_input,
        seq_week_emb,
        seq_month_emb,
        seq_day_emb,
    ])

    # =========================
    # BLOCCO LSTM
    # =========================
    x_seq = LSTM(
        architecture_config["lstm_units"],
        name="lstm_block",
    )(seq_input)

    x_seq = Dense(
        architecture_config["seq_dense_units"],
        activation="relu",
        name="seq_dense",
    )(x_seq)

    # =========================
    # INPUT NUMERICO FINALE
    # =========================
    final_num_input = Input(
        shape=(n_final_num,),
        name="final_num_input",
    )

    final_bool_input = Input(
        shape=(n_final_bool,),
        name="final_bool_input",
    )

    # =========================
    # INPUT CATEGORICI FINALI
    # =========================
    store_input = Input(shape=(1,), name="store_id")
    week_input = Input(shape=(1,), name="week_day")
    month_input = Input(shape=(1,), name="month")
    day_input = Input(shape=(1,), name="day")

    # =========================
    # EMBEDDING FINALI
    # =========================
    store_emb = Flatten(name="store_embedding_flatten")(
        Embedding(n_store, 4, name="store_embedding")(store_input)
    )

    week_emb = Flatten(name="weekday_embedding_flatten")(
        Embedding(n_weekday, 3, name="weekday_embedding")(week_input)
    )

    month_emb = Flatten(name="month_embedding_flatten")(
        Embedding(n_month, 3, name="month_embedding")(month_input)
    )

    day_emb = Flatten(name="day_embedding_flatten")(
        Embedding(n_day, 3, name="day_embedding")(day_input)
    )


    # =========================
    # CONCATENAZIONE FINALE
    # =========================
    x = Concatenate(name="final_concat")([
        x_seq,
        final_num_input,
        final_bool_input,
        store_emb,
        week_emb,
        month_emb,
        day_emb,
    ])

    # =========================
    # DENSE FINALI
    # =========================
    x = Dense(
        architecture_config["dense_1_units"],
        activation="relu",
        name="dense_1",
    )(x)

    if dropout_rate > 0:
        x = Dropout(
            dropout_rate,
            name="dropout",
        )(x)

    x = Dense(
        architecture_config["dense_2_units"],
        activation="relu",
        name="dense_2",
    )(x)

    output = Dense(
        1,
        name="daily_total_sales_output",
    )(x)

    model = Model(
        inputs=[
            seq_num_input,
            seq_bool_input,
            seq_weekday_input,
            seq_month_input,
            seq_day_input,
            final_num_input,
            final_bool_input,
            store_input,
            week_input,
            month_input,
            day_input,
        ],
        outputs=output,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"],
    )

    return model


# =========================================================
# GROUND TRUTH SAFETY
# =========================================================

def ensure_ground_truth_columns(df, ground_truth_features):
    """
    Garantisce che le colonne ground truth esistano.
    Serve anche se il dataset è clean e non contiene colonne anomalia.
    """

    df = df.copy()

    defaults = {
        "is_level_shift_anomaly": 0,
        "lsa_type": "normal",
        "lsa_mult": 1.0,
        "lsa_severity": "normal",
        "lsa_event_id": -1,
        "lsa_day_in_event": 0,
        "lsa_duration": 0,
    }

    for col in ground_truth_features:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)

    return df


# =========================================================
# DATASET DEGLI ESPERIMENTI
# =========================================================

def list_sensitivity_datasets(base_path):
    """
    Cerca tutti i dataset generati per la sensitivity analysis.

    La struttura attesa delle cartelle è:

        base_path /
            direction /
                dur_<duration>_mult_<multiplier> /
                    seed_<seed> /
                        all_stores_cashflow.csv

    Restituisce un dataframe con il path del dataset e i metadati
    dell'esperimento: direzione, durata, moltiplicatore e seed.
    """

    base_path = Path(base_path)
    rows = []

    pattern = re.compile(
        r"dur_(\d+)_mult_([0-9.]+)"
    )

    for csv_path in base_path.glob("*/*/seed_*/all_stores_cashflow.csv"):

        direction = csv_path.parents[2].name
        exp_name = csv_path.parents[1].name
        seed_name = csv_path.parents[0].name

        match = pattern.match(exp_name)

        if match is None:
            continue

        rows.append({
            "path": csv_path,
            "direction": direction,
            "duration": int(match.group(1)),
            "multiplier": float(match.group(2)),
            "seed": int(seed_name.replace("seed_", ""))
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["direction", "duration", "multiplier", "seed"])
        .reset_index(drop=True)
    )


# =========================================================
# SALES CLASSICAL BASELINES
# =========================================================

def add_temporal_split(
    df,
    train_size=0.70,
    val_size=0.10,
    store_col="store_id",
    date_col="date",
    split_col="split",
):
    """Assegna train, validation e test mantenendo l'ordine temporale per store."""

    df = df.copy()
    df[split_col] = ""

    for _, group in df.groupby(store_col):
        idx = group.sort_values(date_col).index

        n_rows = len(idx)
        train_end = int(train_size * n_rows)
        val_end = int((train_size + val_size) * n_rows)

        df.loc[idx[:train_end], split_col] = "train"
        df.loc[idx[train_end:val_end], split_col] = "val"
        df.loc[idx[val_end:], split_col] = "test"

    return df


def add_pre_holiday_if_missing(
    df,
    store_col="store_id",
    actual_holiday_col="actual_holiday",
    pre_holiday_col="pre_holiday",
):
    """Ricostruisce l'indicatore pre_holiday quando non è presente."""

    df = df.copy()

    if pre_holiday_col in df.columns:
        return df

    df[pre_holiday_col] = (
        df.groupby(store_col)[actual_holiday_col]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    return df


def add_baseline_features(
    df,
    sales_lags=None,
    store_col="store_id",
    date_col="date",
    sales_col="daily_total_sales",
):
    """Costruisce lag e medie mobili usati dalle baseline sales."""

    if sales_lags is None:
        sales_lags = list(range(1, 8)) + [14, 21, 28]

    df = df.copy()
    df = df.sort_values([store_col, date_col]).copy()
    df = add_pre_holiday_if_missing(
        df,
        store_col=store_col,
    )

    df["time_idx"] = df.groupby(store_col).cumcount()

    for lag in sales_lags:
        df[f"sales_lag_{lag}"] = (
            df.groupby(store_col)[sales_col]
            .shift(lag)
        )

    df["sales_rm_7"] = (
        df.groupby(store_col)[sales_col]
        .transform(
            lambda values: values.shift(1).rolling(
                7,
                min_periods=1,
            ).mean()
        )
    )

    df["sales_rm_30"] = (
        df.groupby(store_col)[sales_col]
        .transform(
            lambda values: values.shift(1).rolling(
                30,
                min_periods=7,
            ).mean()
        )
    )

    fill_cols = (
        [f"sales_lag_{lag}" for lag in sales_lags]
        + ["sales_rm_7", "sales_rm_30"]
    )

    for col in fill_cols:
        df[col] = (
            df.groupby(store_col)[col]
            .bfill()
            .ffill()
        )

    return df


def build_one_hot_encoder():
    """Crea un OneHotEncoder compatibile con versioni diverse di scikit-learn."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def add_baseline_predictions(
    df,
    ridge_num_features,
    ridge_cat_features,
    ridge_alpha=1.0,
    target_col="daily_total_sales",
    split_col="split",
):
    """Aggiunge previsioni Seasonal Naive T-7 e Ridge a un dataframe sales."""

    df = df.copy()

    df["y_pred_seasonal_naive_t7"] = df["sales_lag_7"]

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ridge_num_features),
            ("cat", build_one_hot_encoder(), ridge_cat_features),
        ],
        remainder="drop",
    )

    ridge_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=ridge_alpha)),
        ]
    )

    train_features = ridge_num_features + ridge_cat_features

    ridge_model.fit(
        train_df[train_features],
        train_df[target_col],
    )

    df.loc[val_df.index, "y_pred_ridge"] = ridge_model.predict(
        val_df[train_features]
    )

    df.loc[test_df.index, "y_pred_ridge"] = ridge_model.predict(
        test_df[train_features]
    )

    return df


def make_baseline_results_df(
    df,
    split,
    pred_col,
    target_col="daily_total_sales",
    gt_col_candidates=None,
    split_col="split",
):
    """Converte le predizioni baseline nel formato atteso dal detector sales."""

    if gt_col_candidates is None:
        gt_col_candidates = [
            "is_level_shift_anomaly",
            "lsa_type",
            "lsa_mult",
            "lsa_event_id",
            "lsa_day_in_event",
            "lsa_duration",
            "lsa_severity",
            "severity_class",
        ]

    temp = df[df[split_col] == split].copy()

    gt_cols = [
        col
        for col in gt_col_candidates
        if col in temp.columns
    ]

    results = temp[
        ["date", "store_id", target_col, pred_col] + gt_cols
    ].copy()

    results = results.rename(
        columns={
            target_col: "y_true_original",
            pred_col: "y_pred_original",
        }
    )

    results["y_true"] = results["y_true_original"]
    results["y_pred"] = results["y_pred_original"]

    results["residual"] = (
        results["y_true_original"] - results["y_pred_original"]
    )

    results["abs_error"] = results["residual"].abs()
    results["squared_error"] = results["residual"] ** 2

    return results.reset_index(drop=True)


def run_level_shift_detector(
    val_results,
    test_results,
    model_name,
    score_window=7,
    n_std=3.5,
    min_consecutive=3,
    gap_tolerance=1,
    iou_threshold=0.20,
):
    """Applica il detector level shift ai residui di un predittore sales."""

    detector_output = run_level_shift_std_detector_details(
        val_results=val_results,
        test_results=test_results,
        score_window=score_window,
        n_std=n_std,
        min_consecutive=min_consecutive,
        gap_tolerance=gap_tolerance,
        iou_threshold=iou_threshold,
    )

    summary = detector_output["summary"].copy()
    summary.update({
        "model": model_name,
        "score_window": score_window,
        "n_std": n_std,
        "min_consecutive": min_consecutive,
        "gap_tolerance": gap_tolerance,
        "iou_threshold": iou_threshold,
    })

    return summary

# =========================================================
# CALCOLO DELLO SCORE
# =========================================================

def add_level_shift_score(
    df,
    residual_col="residual",
    store_col="store_id",
    window=7,
    min_periods=3
):
    """
    Calcola lo score di level shift come somma rolling centrata dei residui.

    Residui positivi persistenti producono uno score positivo.
    Residui negativi persistenti producono uno score negativo.
    Residui con segni alternati tendono invece a compensarsi.
    """

    df = df.copy()

    df["level_shift_score"] = (
        df.groupby(store_col)[residual_col]
          .transform(
              lambda s: s.rolling(
                  window=window,
                  min_periods=min_periods,
                  center=True
              ).sum()
          )
    )

    return df


# =========================================================
# SOGLIE DI DETECTION
# =========================================================

def compute_level_shift_std_thresholds_by_store(
    val_results,
    score_col="level_shift_score",
    store_col="store_id",
    n_std=3.0
):
    """
    Calcola soglie specifiche per store usando gli score del validation set.

    Per ogni store vengono calcolate due soglie:

        lower = mean - n_std * std
        upper = mean + n_std * std

    dove mean e std sono stimati sul validation set dello store.
    """

    thresholds = {}

    for store_id, g in val_results.groupby(store_col):

        mean = g[score_col].mean()
        std = g[score_col].std()

        thresholds[store_id] = {
            "lower": mean - n_std * std,
            "upper": mean + n_std * std,
            "mean": mean,
            "std": std
        }

    return thresholds


def detect_level_shift_from_score_by_store(
    results_df,
    thresholds,
    score_col="level_shift_score",
    store_col="store_id"
):
    """
    Applica la detection puntuale usando le soglie specifiche per store.

    Un punto viene marcato come anomalo se lo score supera la soglia
    superiore o scende sotto la soglia inferiore. Le finestre evento
    vengono costruite in un passaggio successivo.
    """

    df = results_df.copy()

    df["lower_threshold"] = df[store_col].map(
        lambda s: thresholds[s]["lower"]
    )

    df["upper_threshold"] = df[store_col].map(
        lambda s: thresholds[s]["upper"]
    )

    df["is_level_shift_detected_raw"] = (
        (df[score_col] > df["upper_threshold"]) |
        (df[score_col] < df["lower_threshold"])
    ).astype(int)

    df["level_shift_direction"] = np.where(
        df[score_col] > df["upper_threshold"],
        "increase",
        np.where(
            df[score_col] < df["lower_threshold"],
            "decrease",
            "normal"
        )
    )

    return df


# =========================================================
# COSTRUZIONE DELLE FINESTRE
# =========================================================

def build_gt_level_shift_windows(
    df,
    store_col="store_id",
    date_col="date",
    event_col="lsa_event_id",
    type_col="lsa_type",
    severity_col="lsa_severity",
    mult_col="lsa_mult",
    duration_col="lsa_duration"
):
    """
    Converte le label ground truth giornaliere in finestre evento.

    Ogni evento ground truth è identificato dalla coppia:
    store_id, lsa_event_id.
    """

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    gt = temp[
        (temp[event_col] != -1) &
        (temp["is_level_shift_anomaly"] == 1)
    ].copy()

    out = []

    for (store_id, event_id), g in gt.groupby([store_col, event_col]):

        start = g[date_col].min()
        end = g[date_col].max()

        out.append({
            "store_id": store_id,
            "gt_event_id": event_id,
            "gt_start": start,
            "gt_end": end,
            "gt_duration": (end - start).days + 1,
            "gt_type": g[type_col].mode().iloc[0],
            "gt_severity": g[severity_col].mode().iloc[0],
            "gt_mult_mean": g[mult_col].astype(float).mean(),
            "gt_duration_original": g[duration_col].max()
        })

    return pd.DataFrame(out)


def build_detected_windows_from_center_points(
    df,
    detected_col="is_level_shift_detected_raw",
    date_col="date",
    store_col="store_id",
    window_size=7,
    min_consecutive=3,
    gap_tolerance=0
):
    """
    Costruisce finestre di detection a partire dai punti marcati come anomali.

    La funzione:
    - considera i punti in cui detected_col = 1;
    - raggruppa punti vicini tollerando eventuali gap brevi;
    - scarta blocchi con meno di min_consecutive rilevazioni;
    - trasforma ogni punto rilevato in una finestra centrata;
    - unisce finestre sovrapposte o separate da un gap tollerabile.

    gap_tolerance indica quanti giorni non rilevati possono essere presenti
    tra due punti/blocchi senza interrompere la stessa finestra candidata.
    """

    out = []

    # Estensione della finestra centrata intorno al punto rilevato.
    half_left = window_size // 2
    half_right = window_size - half_left - 1

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    # Le finestre vengono costruite separatamente per ciascuno store.
    for store_id, g in temp.groupby(store_col):

        g = g.sort_values(date_col).copy()

        # Mantiene solo i punti che hanno superato la soglia di detection.
        detected_points = g[g[detected_col].astype(int) == 1].copy()

        if detected_points.empty:
            continue

        detected_points = (
            detected_points
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        # =========================================================
        # COSTRUZIONE DEI BLOCCHI DI PUNTI RILEVATI
        # =========================================================
        # I punti rilevati vengono raggruppati se sono consecutivi
        # oppure separati da un numero di giorni non rilevati <= gap_tolerance.
        #
        # Esempio con gap_tolerance = 1:
        # 1 1 0 1 viene trattato come un unico blocco.
        # =========================================================

        blocks = []
        current_block = [detected_points.iloc[0]]

        for i in range(1, len(detected_points)):

            prev_date = pd.to_datetime(detected_points.iloc[i - 1][date_col])
            curr_date = pd.to_datetime(detected_points.iloc[i][date_col])

            gap_days = (curr_date - prev_date).days - 1

            if gap_days <= gap_tolerance:
                current_block.append(detected_points.iloc[i])
            else:
                blocks.append(current_block)
                current_block = [detected_points.iloc[i]]

        blocks.append(current_block)

        # =========================================================
        # FILTRO DI PERSISTENZA + CONVERSIONE IN FINESTRE
        # =========================================================
        # Un blocco viene mantenuto solo se contiene almeno
        # min_consecutive punti rilevati.
        # =========================================================

        for block in blocks:

            if len(block) < min_consecutive:
                continue

            intervals = []

            # Ogni punto rilevato diventa il centro di una finestra temporale.
            for row in block:

                center = pd.to_datetime(row[date_col])

                start = center - pd.Timedelta(days=half_left)
                end = center + pd.Timedelta(days=half_right)

                intervals.append((start, end))

            intervals = sorted(intervals)

            # =====================================================
            # MERGE DELLE FINESTRE
            # =====================================================
            # Le finestre sovrapposte, adiacenti o separate da un gap
            # tollerabile vengono fuse in un'unica detection window.
            # =====================================================

            cur_start, cur_end = intervals[0]

            for start, end in intervals[1:]:

                if start <= cur_end + pd.Timedelta(days=gap_tolerance + 1):
                    cur_end = max(cur_end, end)

                else:
                    w = g[
                        (g[date_col] >= cur_start) &
                        (g[date_col] <= cur_end)
                    ].copy()

                    out.append({
                        "store_id": store_id,
                        "detected_start": cur_start,
                        "detected_end": cur_end,
                        "detected_duration": (cur_end - cur_start).days + 1,
                        "max_abs_score": w["level_shift_score"].abs().max(),
                        "mean_abs_score": w["level_shift_score"].abs().mean(),
                        "mean_residual": w["residual"].mean(),
                        "sum_residual": w["residual"].sum(),
                        "direction": (
                            "increase"
                            if w["level_shift_score"].mean() > 0
                            else "decrease"
                        )
                    })

                    cur_start, cur_end = start, end

            # Salva l'ultima finestra del blocco corrente.
            w = g[
                (g[date_col] >= cur_start) &
                (g[date_col] <= cur_end)
            ].copy()

            out.append({
                "store_id": store_id,
                "detected_start": cur_start,
                "detected_end": cur_end,
                "detected_duration": (cur_end - cur_start).days + 1,
                "max_abs_score": w["level_shift_score"].abs().max(),
                "mean_abs_score": w["level_shift_score"].abs().mean(),
                "mean_residual": w["residual"].mean(),
                "sum_residual": w["residual"].sum(),
                "direction": (
                    "increase"
                    if w["level_shift_score"].mean() > 0
                    else "decrease"
                )
            })

    return pd.DataFrame(out)


# =========================================================
# CONFRONTO TRA INTERVALLI
# =========================================================

def interval_iou(start_a, end_a, start_b, end_b):
    """
    Calcola la Intersection over Union tra due intervalli temporali.

    Gli estremi degli intervalli sono considerati inclusivi.
    """

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)

    inter = max((inter_end - inter_start).days + 1, 0)

    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)

    union = (union_end - union_start).days + 1

    return inter / union if union > 0 else 0.0


# =========================================================
# VALUTAZIONE EVENT-LEVEL
# =========================================================

def evaluate_detected_windows_event_level(
    gt_windows,
    detected_windows,
    store_col="store_id",
    iou_threshold=0.10
):
    """
    Valuta le finestre rilevate rispetto alle finestre ground truth.

    Un evento ground truth è considerato rilevato se la sua migliore IoU
    con una finestra detected dello stesso store è almeno pari a
    `iou_threshold`.

    Restituisce:
    - gt_eval: eventi ground truth con informazioni di matching;
    - det_eval: finestre rilevate con informazioni di matching;
    - summary: metriche event-level.
    """

    gt = gt_windows.copy()
    det = detected_windows.copy()

    if gt.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "n_gt_events": 0,
            "n_detected_events": len(det),
            "tp": 0,
            "fp": len(det),
            "fn": 0,
            "precision": 0.0,
            "recall": np.nan,
            "f1": np.nan,
            "mean_iou": np.nan,
            "mean_detection_det_offset_start": np.nan,
            "mean_detection_det_offset_end": np.nan
        }

    if det.empty:
        gt_eval = gt.copy()
        gt_eval["matched"] = 0
        gt_eval["best_iou"] = 0.0
        gt_eval["matched_detected_id"] = -1
        gt_eval["detection_det_offset_start_days"] = np.nan
        gt_eval["detection_det_offset_end_days"] = np.nan

        return gt_eval, pd.DataFrame(), {
            "n_gt_events": len(gt_eval),
            "n_detected_events": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_eval),
            "precision": np.nan,
            "recall": 0.0,
            "f1": np.nan,
            "mean_iou": 0.0,
            "mean_detection_det_offset_start": np.nan,
            "mean_detection_det_offset_end": np.nan
        }

    gt["gt_start"] = pd.to_datetime(gt["gt_start"])
    gt["gt_end"] = pd.to_datetime(gt["gt_end"])

    det["detected_start"] = pd.to_datetime(det["detected_start"])
    det["detected_end"] = pd.to_datetime(det["detected_end"])
    det = det.reset_index(drop=True)
    det["detected_id"] = det.index

    gt_eval = gt.copy()
    gt_eval["matched"] = 0
    gt_eval["best_iou"] = 0.0
    gt_eval["matched_detected_id"] = -1
    gt_eval["detection_det_offset_start_days"] = np.nan
    gt_eval["detection_det_offset_end_days"] = np.nan

    det_eval = det.copy()
    det_eval["matched"] = 0
    det_eval["best_iou"] = 0.0
    det_eval["matched_gt_event_id"] = -1

    for gt_idx, gt_row in gt_eval.iterrows():

        same_store_det = det_eval[
            det_eval[store_col] == gt_row[store_col]
        ]

        best_iou = 0.0
        best_det_id = -1
        best_det_offset_start = np.nan
        best_det_offset_end = np.nan

        for _, det_row in same_store_det.iterrows():

            iou = interval_iou(
                gt_row["gt_start"],
                gt_row["gt_end"],
                det_row["detected_start"],
                det_row["detected_end"]
            )

            if iou > best_iou:
                best_iou = iou
                best_det_id = det_row["detected_id"]
                best_det_offset_start = (
                    det_row["detected_start"] - gt_row["gt_start"]
                ).days
                best_det_offset_end = (
                    det_row["detected_end"] - gt_row["gt_end"]
                ).days

        if best_iou >= iou_threshold:
            gt_eval.loc[gt_idx, "matched"] = 1
            gt_eval.loc[gt_idx, "best_iou"] = best_iou
            gt_eval.loc[gt_idx, "matched_detected_id"] = best_det_id
            gt_eval.loc[gt_idx, "detection_det_offset_start_days"] = best_det_offset_start
            gt_eval.loc[gt_idx, "detection_det_offset_end_days"] = best_det_offset_end

    matched_det_ids = set(
        gt_eval.loc[
            gt_eval["matched"] == 1,
            "matched_detected_id"
        ].astype(int)
    )

    for det_idx, det_row in det_eval.iterrows():

        if det_row["detected_id"] in matched_det_ids:

            matched_gt = gt_eval[
                gt_eval["matched_detected_id"] == det_row["detected_id"]
            ]

            det_eval.loc[det_idx, "matched"] = 1
            det_eval.loc[det_idx, "best_iou"] = matched_gt["best_iou"].max()
            det_eval.loc[det_idx, "matched_gt_event_id"] = (
                matched_gt
                .sort_values("best_iou", ascending=False)
                ["gt_event_id"]
                .iloc[0]
            )

    tp = int(gt_eval["matched"].sum())
    fn = int((gt_eval["matched"] == 0).sum())
    fp = int((det_eval["matched"] == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision)
        and pd.notna(recall)
        and (precision + recall) > 0
        else np.nan
    )

    summary = {
        "n_gt_events": len(gt_eval),
        "n_detected_events": len(det_eval),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (
            gt_eval.loc[gt_eval["matched"] == 1, "best_iou"].mean()
            if tp > 0
            else np.nan
        ),
        "mean_detection_det_offset_start": (
            gt_eval.loc[
                gt_eval["matched"] == 1,
                "detection_det_offset_start_days"
            ].mean()
            if tp > 0
            else np.nan
        ),
        "mean_detection_det_offset_end": (
            gt_eval.loc[
                gt_eval["matched"] == 1,
                "detection_det_offset_end_days"
            ].mean()
            if tp > 0
            else np.nan
        )
    }

    return gt_eval, det_eval, summary

# =========================================================
# INFERENCE LSTM SALES E DETECTOR STD
# =========================================================

def run_sales_lstm_inference_for_dataset(
    csv_path,
    model,
    feature_scalers,
    mappings,
    features,
    window_size,
):
    """
    Esegue l'inference LSTM sales su validation e test di un dataset.

    Il preprocessing riusa gli artifact del modello selezionato, senza
    modificare gli split temporali o le trasformazioni già definite.
    """

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    # Gli artifact legacy possono ancora elencare colonne WCA non più usate.
    features = normalize_sales_feature_schema(features)

    _, val, test = build_dataset_inference(
        df,
        feature_scalers,
        mappings,
        features,
        window_size=window_size,
    )

    val_pred = model.predict(
        build_model_inputs(val),
        verbose=0,
    )

    test_pred = model.predict(
        build_model_inputs(test),
        verbose=0,
    )

    val_results = make_results_df(
        val,
        val_pred,
        features,
        feature_scalers,
    )

    test_results = make_results_df(
        test,
        test_pred,
        features,
        feature_scalers,
    )

    return {
        "val_results": val_results,
        "test_results": test_results,
    }


def run_level_shift_std_detector_details(
    val_results,
    test_results,
    score_window=7,
    n_std=3.5,
    min_consecutive=3,
    gap_tolerance=1,
    iou_threshold=0.20,
    residual_col="residual",
    score_col="level_shift_score",
    store_col="store_id",
):
    """
    Applica il detector level shift con soglie STD e restituisce tutti gli step.

    Oltre alla sintesi event-level, conserva score, soglie e finestre per le
    analisi che richiedono i dettagli della detection.
    """

    val_scored = add_level_shift_score(
        val_results,
        residual_col=residual_col,
        store_col=store_col,
        window=score_window,
    )

    test_scored = add_level_shift_score(
        test_results,
        residual_col=residual_col,
        store_col=store_col,
        window=score_window,
    )

    thresholds = compute_level_shift_std_thresholds_by_store(
        val_scored,
        score_col=score_col,
        store_col=store_col,
        n_std=n_std,
    )

    test_detected = detect_level_shift_from_score_by_store(
        test_scored,
        thresholds,
        score_col=score_col,
        store_col=store_col,
    )

    detected_windows = build_detected_windows_from_center_points(
        test_detected,
        detected_col="is_level_shift_detected_raw",
        store_col=store_col,
        window_size=score_window,
        min_consecutive=min_consecutive,
        gap_tolerance=gap_tolerance,
    )

    gt_windows = build_gt_level_shift_windows(
        test_detected,
        store_col=store_col,
    )

    gt_eval, det_eval, summary = evaluate_detected_windows_event_level(
        gt_windows,
        detected_windows,
        store_col=store_col,
        iou_threshold=iou_threshold,
    )

    return {
        "val_scored": val_scored,
        "test_scored": test_scored,
        "thresholds": thresholds,
        "test_detected": test_detected,
        "detected_windows": detected_windows,
        "gt_windows": gt_windows,
        "gt_eval": gt_eval,
        "det_eval": det_eval,
        "summary": summary,
    }


# =========================================================
# SALES AUTOENCODER
# =========================================================

def get_sales_ae_feature_lists():
    # Schema specifico dell'AE: le feature sequenziali descrivono la finestra da ricostruire.
    seq_num_features = [
        "daily_total_sales",
        "oil_price",
        "consumer_confidence",
        "fao",
        "time_idx",
    ]

    seq_bool_features = [
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    seq_cat_features = [
        "week_day",
        "month",
        "day",
        "store_id",
    ]

    ground_truth_features = [
        "is_level_shift_anomaly",
        "lsa_type",
        "lsa_mult",
        "lsa_event_id",
        "lsa_day_in_event",
        "lsa_duration",
        "lsa_severity",
    ]

    log_cols = [
        "daily_total_sales",
        "oil_price",
        "fao",
    ]

    scale_cols = list(
        dict.fromkeys(seq_num_features + ["daily_total_sales"])
    )

    return {
        "seq_num": seq_num_features,
        "seq_bool": seq_bool_features,
        "seq_cat": seq_cat_features,
        "target": "daily_total_sales",
        "ground_truth": ground_truth_features,
        "log": log_cols,
        "scale": scale_cols,
    }


def build_sales_ae_dataset(
    df,
    window_size=14,
    train_size=0.70,
    val_size=0.10,
):
    """Costruisce train, validation e test per il Sales Autoencoder."""

    features = get_sales_ae_feature_lists()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"]).copy()

    df = ensure_ground_truth_columns(
        df,
        features["ground_truth"],
    )

    df["time_idx"] = df.groupby("store_id").cumcount()

    df["pre_holiday"] = (
        df.groupby("store_id")["actual_holiday"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    for col in features["log"]:
        df[col] = np.log1p(df[col])

    mappings = {}

    for col in features["seq_cat"]:
        df[col], mapping = pd.factorize(df[col])
        mappings[col] = mapping

    train_parts = []
    val_parts = []
    test_parts = []
    feature_scalers = {}

    for store_id in df["store_id"].unique():
        temp = df[df["store_id"] == store_id].sort_values("date").copy()

        n_rows = len(temp)
        train_end = int(train_size * n_rows)
        val_end = int((train_size + val_size) * n_rows)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        scaler = StandardScaler()
        scaler.fit(train_df[features["scale"]].astype(float))

        feature_scalers[store_id] = scaler

        train_df[features["scale"]] = scaler.transform(
            train_df[features["scale"]].astype(float)
        )
        val_df[features["scale"]] = scaler.transform(
            val_df[features["scale"]].astype(float)
        )
        test_df[features["scale"]] = scaler.transform(
            test_df[features["scale"]].astype(float)
        )

        train_parts.append(
            create_ae_windows(
                train_df,
                features,
                window_size,
            )
        )
        val_parts.append(
            create_ae_windows(
                val_df,
                features,
                window_size,
            )
        )
        test_parts.append(
            create_ae_windows(
                test_df,
                features,
                window_size,
            )
        )

    def concatenate_ae_parts(parts):
        return {
            key: np.concatenate(
                [part[key] for part in parts],
                axis=0,
            )
            for key in parts[0]
        }

    return (
        concatenate_ae_parts(train_parts),
        concatenate_ae_parts(val_parts),
        concatenate_ae_parts(test_parts),
        feature_scalers,
        mappings,
        features,
    )


def build_sales_ae_dataset_inference(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=14,
    train_size=0.70,
    val_size=0.10,
):
    """Costruisce train, validation e test per un Sales Autoencoder salvato."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"]).copy()

    df = ensure_ground_truth_columns(
        df,
        features["ground_truth"],
    )

    df["time_idx"] = df.groupby("store_id").cumcount()

    df["pre_holiday"] = (
        df.groupby("store_id")["actual_holiday"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    for col in features["log"]:
        df[col] = np.log1p(df[col])

    for col, mapping in mappings.items():
        df[col] = pd.Categorical(
            df[col],
            categories=mapping,
        ).codes

    train_parts = []
    val_parts = []
    test_parts = []

    for store_id in df["store_id"].unique():
        temp = df[df["store_id"] == store_id].sort_values("date").copy()

        n_rows = len(temp)
        train_end = int(train_size * n_rows)
        val_end = int((train_size + val_size) * n_rows)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        scaler = feature_scalers[store_id]

        train_df[features["scale"]] = scaler.transform(
            train_df[features["scale"]].astype(float)
        )
        val_df[features["scale"]] = scaler.transform(
            val_df[features["scale"]].astype(float)
        )
        test_df[features["scale"]] = scaler.transform(
            test_df[features["scale"]].astype(float)
        )

        train_parts.append(
            create_ae_windows(
                train_df,
                features,
                window_size,
            )
        )
        val_parts.append(
            create_ae_windows(
                val_df,
                features,
                window_size,
            )
        )
        test_parts.append(
            create_ae_windows(
                test_df,
                features,
                window_size,
            )
        )

    def concatenate_ae_parts(parts):
        return {
            key: np.concatenate(
                [part[key] for part in parts],
                axis=0,
            )
            for key in parts[0]
        }

    return (
        concatenate_ae_parts(train_parts),
        concatenate_ae_parts(val_parts),
        concatenate_ae_parts(test_parts),
    )


def build_sales_lstm_autoencoder(
    train,
    latent_dim=8,
    learning_rate=1e-3,
):
    """Costruisce l'Autoencoder LSTM per finestre di vendite."""

    X_seq_num = train["X_seq_num"]
    X_seq_bool = train["X_seq_bool"]
    X_seq_cat = train["X_seq_cat"]

    window_size = X_seq_num.shape[1]
    n_num = X_seq_num.shape[2]
    n_bool = X_seq_bool.shape[2]

    n_weekday = int(X_seq_cat[:, :, 0].max()) + 1
    n_month = int(X_seq_cat[:, :, 1].max()) + 1
    n_day = int(X_seq_cat[:, :, 2].max()) + 1
    n_store = int(X_seq_cat[:, :, 3].max()) + 1

    seq_num_input = Input(
        shape=(window_size, n_num),
        name="seq_num_input",
    )
    seq_bool_input = Input(
        shape=(window_size, n_bool),
        name="seq_bool_input",
    )
    weekday_input = Input(
        shape=(window_size,),
        name="weekday_input",
    )
    month_input = Input(
        shape=(window_size,),
        name="month_input",
    )
    day_input = Input(
        shape=(window_size,),
        name="day_input",
    )
    store_input = Input(
        shape=(window_size,),
        name="store_input",
    )

    weekday_emb = Embedding(
        input_dim=n_weekday,
        output_dim=3,
    )(weekday_input)

    month_emb = Embedding(
        input_dim=n_month,
        output_dim=3,
    )(month_input)

    day_emb = Embedding(
        input_dim=n_day,
        output_dim=4,
    )(day_input)

    store_emb = Embedding(
        input_dim=n_store,
        output_dim=4,
    )(store_input)

    x = Concatenate(axis=-1)([
        seq_num_input,
        seq_bool_input,
        weekday_emb,
        month_emb,
        day_emb,
        store_emb,
    ])

    x = LSTM(64, return_sequences=True)(x)

    latent = LSTM(
        latent_dim,
        return_sequences=False,
        name="latent_vector",
    )(x)

    x = RepeatVector(window_size)(latent)
    x = LSTM(latent_dim, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)

    output = TimeDistributed(
        Dense(1),
        name="sales_reconstruction",
    )(x)

    model = Model(
        inputs=[
            seq_num_input,
            seq_bool_input,
            weekday_input,
            month_input,
            day_input,
            store_input,
        ],
        outputs=output,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="mae",
        metrics=["mae"],
    )

    return model


# =========================================================
# SALES AUTOENCODER: SCORE E FINESTRE EVENT-LEVEL
# =========================================================

def make_ae_base_results_df(data, y_pred):
    """Costruisce gli score giornalieri a partire dalle ricostruzioni AE."""

    y_true = data["y"]
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred

    window_size = y_true.shape[1]
    center_pos = window_size // 2

    return pd.DataFrame({
        "store_id": data["store_id"],
        "center_date": pd.to_datetime(data["date"]),
        "window_start": (
            pd.to_datetime(data["date"])
            - pd.to_timedelta(center_pos, unit="D")
        ),
        "window_end": (
            pd.to_datetime(data["date"])
            + pd.to_timedelta(
                window_size - center_pos - 1,
                unit="D",
            )
        ),
        "ae_mse_score": np.mean(errors ** 2, axis=(1, 2)),
        "ae_mae_score": np.mean(np.abs(errors), axis=(1, 2)),
        "ae_max_abs_error": np.max(np.abs(errors), axis=(1, 2)),
        "ae_signed_mean_error": np.mean(errors, axis=(1, 2)),
        "y_true_center": y_true[:, center_pos, 0],
        "y_pred_center": y_pred[:, center_pos, 0],
        "center_error": errors[:, center_pos, 0],
        "center_abs_error": np.abs(errors[:, center_pos, 0]),
    })


def add_level_shift_window_ground_truth(results, data):
    """Aggiunge la ground truth level shift alle finestre AE."""

    results = results.copy()
    ground_truth = data["ground_truth"]

    is_anomaly = []
    anomaly_ratio = []
    lsa_type = []
    lsa_mult = []
    lsa_event_id = []
    lsa_duration = []

    for window_ground_truth in ground_truth:
        flags = window_ground_truth[:, 0].astype(int)
        ratio = flags.mean()

        is_anomaly.append(int(ratio > 0))
        anomaly_ratio.append(ratio)

        if flags.sum() > 0:
            anomaly_rows = window_ground_truth[flags == 1]

            lsa_type.append(
                pd.Series(anomaly_rows[:, 1]).mode().iloc[0]
            )
            lsa_mult.append(
                float(np.mean(anomaly_rows[:, 2].astype(float)))
            )
            lsa_event_id.append(
                pd.Series(anomaly_rows[:, 3]).mode().iloc[0]
            )
            lsa_duration.append(
                int(np.max(anomaly_rows[:, 5].astype(float)))
            )
        else:
            lsa_type.append("normal")
            lsa_mult.append(1.0)
            lsa_event_id.append(-1)
            lsa_duration.append(0)

    results["is_level_shift_anomaly_window"] = is_anomaly
    results["level_shift_ratio"] = anomaly_ratio
    results["lsa_type_window"] = lsa_type
    results["lsa_mult_window"] = lsa_mult
    results["lsa_event_id_window"] = lsa_event_id
    results["lsa_duration_window"] = lsa_duration

    return results


def detect_ae_anomalies_zscore(
    results_df,
    score_col="ae_mae_score",
    threshold_dict=None,
    n_std=3.5,
):
    """Applica soglie z-score specifiche per store agli score AE."""

    df = results_df.copy()

    if threshold_dict is None:
        threshold_dict = {}

        for store_id, group in df.groupby("store_id"):
            mean = group[score_col].mean()
            std = group[score_col].std()

            threshold_dict[store_id] = {
                "mean": mean,
                "std": std,
                "upper": mean + n_std * std,
            }

    df["ae_threshold"] = df["store_id"].map(
        lambda store_id: threshold_dict[store_id]["upper"]
    )

    df["ae_zscore"] = (
        df[score_col]
        - df["store_id"].map(
            lambda store_id: threshold_dict[store_id]["mean"]
        )
    ) / (
        df["store_id"].map(
            lambda store_id: threshold_dict[store_id]["std"]
        )
        + 1e-8
    )

    df["is_ae_detected_window"] = (
        df["ae_zscore"] > n_std
    ).astype(int)

    df["ae_detected_direction"] = np.where(
        df["is_ae_detected_window"] == 0,
        "normal",
        np.where(
            df["ae_signed_mean_error"] > 0,
            "increase",
            "decrease",
        ),
    )

    return df, threshold_dict


def build_ae_detected_windows(
    df,
    detected_col="is_ae_detected_window",
    store_col="store_id",
    center_col="center_date",
    start_col="window_start",
    end_col="window_end",
    direction_col="ae_detected_direction",
    score_col="ae_zscore",
    raw_score_col="ae_mae_score",
    min_consecutive=3,
    gap_tolerance=1,
):
    """Costruisce finestre evento dalle finestre AE sopra soglia."""

    out = []

    temp = df.copy()
    temp[center_col] = pd.to_datetime(temp[center_col])
    temp[start_col] = pd.to_datetime(temp[start_col])
    temp[end_col] = pd.to_datetime(temp[end_col])

    for store_id, group in temp.groupby(store_col):
        group = group.sort_values(center_col).copy()

        detected = group[
            group[detected_col].astype(int) == 1
        ].copy()

        if detected.empty:
            continue

        detected = detected.sort_values(center_col).reset_index(drop=True)

        blocks = []
        current_block = [detected.iloc[0]]

        for i in range(1, len(detected)):
            previous_center = pd.to_datetime(
                detected.iloc[i - 1][center_col]
            )
            current_center = pd.to_datetime(
                detected.iloc[i][center_col]
            )

            gap_days = (
                current_center - previous_center
            ).days - 1

            if gap_days <= gap_tolerance:
                current_block.append(detected.iloc[i])
            else:
                blocks.append(current_block)
                current_block = [detected.iloc[i]]

        blocks.append(current_block)

        for block in blocks:
            if len(block) < min_consecutive:
                continue

            block_df = pd.DataFrame(block)

            detected_start = block_df[start_col].min()
            detected_end = block_df[end_col].max()

            out.append({
                "store_id": store_id,
                "detected_start": detected_start,
                "detected_end": detected_end,
                "detected_duration": (
                    detected_end - detected_start
                ).days + 1,
                "detected_direction": (
                    block_df[direction_col].mode().iloc[0]
                ),
                "max_ae_zscore": block_df[score_col].max(),
                "mean_ae_zscore": block_df[score_col].mean(),
                "max_ae_score": block_df[raw_score_col].max(),
                "mean_ae_score": block_df[raw_score_col].mean(),
                "n_detected_windows": len(block_df),
            })

    return pd.DataFrame(out)


def build_gt_windows_from_ae_results(test_results):
    """Ricostruisce le finestre ground truth osservabili dall'AE."""

    gt_rows = []

    anomalous_windows = test_results[
        test_results["is_level_shift_anomaly_window"].astype(int) == 1
    ].copy()

    for _, row in anomalous_windows.iterrows():
        gt_rows.append({
            "store_id": row["store_id"],
            "gt_event_id": row["lsa_event_id_window"],
            "gt_start": row["window_start"],
            "gt_end": row["window_end"],
            "gt_type": row["lsa_type_window"],
            "gt_mult_mean": row["lsa_mult_window"],
            "gt_duration_original": row["lsa_duration_window"],
        })

    if not gt_rows:
        return pd.DataFrame(
            columns=[
                "store_id",
                "gt_event_id",
                "gt_start",
                "gt_end",
                "gt_duration",
                "gt_type",
                "gt_mult_mean",
                "gt_duration_original",
            ]
        )

    gt_windows = (
        pd.DataFrame(gt_rows)
        .groupby(
            ["store_id", "gt_event_id"],
            as_index=False,
        )
        .agg(
            gt_start=("gt_start", "min"),
            gt_end=("gt_end", "max"),
            gt_type=("gt_type", lambda values: values.mode().iloc[0]),
            gt_mult_mean=("gt_mult_mean", "mean"),
            gt_duration_original=("gt_duration_original", "max"),
        )
    )

    gt_windows["gt_start"] = pd.to_datetime(
        gt_windows["gt_start"]
    )
    gt_windows["gt_end"] = pd.to_datetime(
        gt_windows["gt_end"]
    )

    gt_windows["gt_duration"] = (
        gt_windows["gt_end"] - gt_windows["gt_start"]
    ).dt.days + 1

    return gt_windows


def run_sales_ae_detector_on_dataset(
    csv_path,
    ae_pack,
    window_size,
    score_col="ae_mae_score",
    n_std=3.5,
    min_consecutive=3,
    gap_tolerance=1,
    iou_threshold=0.20,
    train_size=0.70,
    val_size=0.10,
):
    """
    Esegue la pipeline AE su validation e test di un dataset sales.

    Il preprocessing e gli artifact del modello vengono riutilizzati senza
    modificare split, score, soglie o valutazione event-level.
    """

    df_exp = pd.read_csv(csv_path)
    df_exp["date"] = pd.to_datetime(df_exp["date"])
    df_exp = df_exp.sort_values(["store_id", "date"]).reset_index(drop=True)

    model = ae_pack["model"]
    feature_scalers = ae_pack["feature_scalers"]
    mappings = ae_pack["mappings"]
    features = ae_pack["features"]

    _, val, test = build_sales_ae_dataset_inference(
        df_exp,
        feature_scalers,
        mappings,
        features,
        window_size=window_size,
        train_size=train_size,
        val_size=val_size,
    )

    val_pred = model.predict(build_sales_ae_inputs(val), verbose=0)
    test_pred = model.predict(build_sales_ae_inputs(test), verbose=0)

    val_results = make_ae_base_results_df(val, val_pred)
    test_results = make_ae_base_results_df(test, test_pred)

    val_results = add_level_shift_window_ground_truth(val_results, val)
    test_results = add_level_shift_window_ground_truth(test_results, test)

    # Le soglie sono calibrate sulla validation e riutilizzate sul test.
    val_results, ae_thresholds = detect_ae_anomalies_zscore(
        val_results,
        score_col=score_col,
        n_std=n_std,
    )

    test_results, _ = detect_ae_anomalies_zscore(
        test_results,
        score_col=score_col,
        threshold_dict=ae_thresholds,
        n_std=n_std,
    )

    detected_windows = build_ae_detected_windows(
        test_results,
        detected_col="is_ae_detected_window",
        min_consecutive=min_consecutive,
        gap_tolerance=gap_tolerance,
        score_col="ae_zscore",
        raw_score_col=score_col,
    )

    gt_windows = build_gt_windows_from_ae_results(test_results)

    _, _, summary = evaluate_detected_windows_event_level(
        gt_windows=gt_windows,
        detected_windows=detected_windows,
        iou_threshold=iou_threshold,
    )

    return summary


# =========================================================
# EVENT-LEVEL POOLED SUMMARY
# =========================================================

def safe_div(num, den):
    return np.nan if den == 0 else num / den


def pooled_f1(precision, recall):
    if pd.isna(precision) or pd.isna(recall):
        return np.nan
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def weighted_mean(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights)

    mask = values.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return np.nan

    return np.average(values[mask], weights=weights[mask])


def build_pooled_summary(df, group_cols):
    """Aggrega metriche event-level con conteggi pooled."""

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = pooled_f1(precision, recall)

        row.update({
            "n_runs": len(g),
            "n_gt_events": int(g["n_gt_events"].sum()),
            "n_detected_events": int(g["n_detected_events"].sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # La IoU media è definita sui soli true positive.
            "mean_iou": weighted_mean(g["mean_iou"], g["tp"]),
            "f1_run_mean": g["f1"].mean(),
            "f1_run_std": g["f1"].std(),
        })

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# CONTAMINATION-ROBUSTNESS SUPPORT
# =========================================================


def _default_contamination_sales_feature_lists():
    """Fallback coerente con lo schema legacy del tuning sales."""
    seq_num_features = [
        "daily_total_sales",
    ]

    seq_bool_features = [
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    seq_cat_features = [
        "week_day",
        "month",
        "day",
    ]

    final_num_features = [
        "time_idx",
        "oil_price",
        "consumer_confidence",
        "fao",
    ]

    # Schema legacy del tuning sales: anche le booleane finali sono categoriche.
    cat_features = [
        "store_id",
        "week_day",
        "month",
        "day",
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    ground_truth_features = [
        "is_level_shift_anomaly",
        "lsa_type",
        "lsa_mult",
        "lsa_severity",
        "lsa_event_id",
        "lsa_day_in_event",
        "lsa_duration",

        "is_level_shift_contamination",
        "lsa_contamination_split",
        "lsa_contamination_level",
        "lsa_contamination_target_fraction",
    ]

    target = "daily_total_sales"

    log_transform_features = [
        "daily_total_sales",
        "sales_rm_30",
        "sales_rm_7",
        "oil_price",
        "fao",
    ]

    scale_features = list(dict.fromkeys(
        seq_num_features
        + final_num_features
        + [target]
    ))

    return {
        "seq_num": seq_num_features,
        "seq_bool": seq_bool_features,
        "seq_cat": seq_cat_features,
        "final_num": final_num_features,
        "final_bool": [],
        "cat": cat_features,
        "ground_truth": ground_truth_features,
        "target": target,
        "log_transform": log_transform_features,
        "scale": scale_features,
    }


def normalize_sales_feature_schema(features):
    """Completa uno schema feature salvato dal tuning e rimuove metadata WCA legacy."""
    features = dict(features)
    features.setdefault("seq_num", [])
    features.setdefault("seq_bool", [])
    features.setdefault("seq_cat", [])
    features.setdefault("final_num", [])
    features.setdefault("final_bool", [])
    features.setdefault("cat", [])
    features.setdefault("ground_truth", [])
    features.setdefault("target", "daily_total_sales")
    features.setdefault("log_transform", [])

    legacy_wca_columns = {
        "is_weekday_contextual_anomaly",
        "wca_type",
        "wca_ratio",
        "wca_event_id",
        "wca_day_in_event",
        "wca_duration",
    }
    features["ground_truth"] = [
        col
        for col in features["ground_truth"]
        if col not in legacy_wca_columns
    ]

    if "scale" not in features:
        features["scale"] = list(dict.fromkeys(
            features.get("seq_num", [])
            + features.get("final_num", [])
            + [features["target"]]
        ))

    return features


def get_contamination_sales_feature_lists(feature_template=None):
    """Restituisce lo schema salvato dal tuning o il fallback legacy."""
    if feature_template is not None:
        return normalize_sales_feature_schema(feature_template)

    return normalize_sales_feature_schema(
        _default_contamination_sales_feature_lists()
    )


def ensure_contamination_ground_truth_columns(df, ground_truth_features):
    """Garantisce la presenza delle colonne GT richieste dallo schema feature."""
    df = df.copy()

    defaults = {
        "is_level_shift_anomaly": 0,
        "lsa_type": "normal",
        "lsa_mult": 1.0,
        "lsa_severity": "normal",
        "lsa_event_id": -1,
        "lsa_day_in_event": -1,
        "lsa_duration": 0,

        "is_level_shift_contamination": 0,
        "lsa_contamination_split": "none",
        "lsa_contamination_level": 0,
        "lsa_contamination_target_fraction": 0.0,
    }

    for col in ground_truth_features:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)

    return df


def encode_sales_categorical_fit(df, features):
    """Codifica le categoriche e conserva i mapping del livello corrente."""
    df = df.copy()
    mappings = {}

    all_cat_cols = list(dict.fromkeys(
        features.get("cat", []) + features.get("seq_cat", [])
    ))

    for col in all_cat_cols:
        if col not in df.columns:
            raise ValueError(f"Colonna categorica mancante: {col}")
        df[col], mapping = pd.factorize(df[col])
        mappings[col] = mapping

    return df, mappings


def encode_sales_categorical_transform(df, mappings):
    """Applica mapping categorici già stimati o salvati."""
    df = df.copy()

    for col, mapping in mappings.items():
        if col not in df.columns:
            raise ValueError(f"Colonna categorica mancante in inference: {col}")

        df[col] = pd.Categorical(df[col], categories=mapping).codes

        if (df[col] < 0).any():
            bad_values = df.loc[df[col] < 0, col].unique()
            raise ValueError(
                f"Valori non presenti nel mapping per colonna {col}: {bad_values}"
            )

    return df


def create_sales_sequences_from_features(df, features, window):
    """
    Costruisce sequenze LSTM per daily_total_sales.

    Per ogni target t:
    - le feature sequenziali usano [t-window, ..., t-1];
    - le feature finali descrivono il giorno t;
    - le ground truth sono conservate solo per valutazione.
    """
    features = normalize_sales_feature_schema(features)

    seq_num_features = features["seq_num"]
    seq_bool_features = features["seq_bool"]
    seq_cat_features = features["seq_cat"]
    final_num_features = features.get("final_num", [])
    final_bool_features = features.get("final_bool", [])
    cat_features = features["cat"]
    ground_truth_features = features["ground_truth"]
    target = features["target"]

    X_seq_num, X_seq_bool, X_seq_cat = [], [], []
    X_final_num, X_final_bool, X_cat = [], [], []
    y, dates, ground_truth = [], [], []

    seq_num_vals = (
        df[seq_num_features].values
        if seq_num_features else np.empty((len(df), 0))
    )
    seq_bool_vals = (
        df[seq_bool_features].values
        if seq_bool_features else np.empty((len(df), 0))
    )
    seq_cat_vals = (
        df[seq_cat_features].values
        if seq_cat_features else np.empty((len(df), 0))
    )
    final_num_vals = (
        df[final_num_features].values
        if final_num_features else np.empty((len(df), 0))
    )
    final_bool_vals = (
        df[final_bool_features].values
        if final_bool_features else np.empty((len(df), 0))
    )
    cat_vals = (
        df[cat_features].values
        if cat_features else np.empty((len(df), 0))
    )
    gt_vals = (
        df[ground_truth_features].values
        if ground_truth_features else np.empty((len(df), 0))
    )
    y_vals = df[target].values
    date_vals = df["date"].values

    for i in range(len(df) - window):
        target_pos = i + window

        X_seq_num.append(seq_num_vals[i:target_pos])
        X_seq_bool.append(seq_bool_vals[i:target_pos])
        X_seq_cat.append(seq_cat_vals[i:target_pos])

        X_final_num.append(final_num_vals[target_pos])
        X_final_bool.append(final_bool_vals[target_pos])
        X_cat.append(cat_vals[target_pos])

        y.append(y_vals[target_pos])
        dates.append(date_vals[target_pos])
        ground_truth.append(gt_vals[target_pos])

    return {
        "X_seq_num": np.array(X_seq_num, dtype=np.float32),
        "X_seq_bool": np.array(X_seq_bool, dtype=np.float32),
        "X_seq_cat": np.array(X_seq_cat, dtype=np.int32),
        "X_final_num": np.array(X_final_num, dtype=np.float32),
        "X_final_bool": np.array(X_final_bool, dtype=np.float32),
        "X_cat": np.array(X_cat, dtype=np.int32),
        "y": np.array(y, dtype=np.float32),
        "date": np.array(dates),
        "ground_truth": np.array(ground_truth, dtype=object),
    }


def _make_sales_parts_container():
    # Contenitore comune per accumulare le sequenze generate store per store.
    return {
        "X_seq_num": [],
        "X_seq_bool": [],
        "X_seq_cat": [],
        "X_final_num": [],
        "X_final_bool": [],
        "X_cat": [],
        "y": [],
        "date": [],
        "ground_truth": [],
    }


def _append_sales_parts(container, seq):
    for key in container:
        container[key].append(seq[key])


def _concat_sales_parts(container):
    return {
        key: np.concatenate(values, axis=0)
        for key, values in container.items()
    }


def _attach_sales_features(split, features):
    split = dict(split)
    split["_features"] = normalize_sales_feature_schema(features)
    return split


def build_sales_model_inputs_from_features(split, features=None):
    """Costruisce gli input Keras nell'ordine dello schema feature salvato."""
    if features is None:
        features = split.get("_features")

    if features is None:
        raise ValueError("Lo schema feature deve essere fornito o allegato allo split.")

    features = normalize_sales_feature_schema(features)

    X_seq_num = split["X_seq_num"]
    X_seq_bool = split["X_seq_bool"]
    X_seq_cat = split["X_seq_cat"]
    X_final_num = split["X_final_num"]
    X_final_bool = split.get("X_final_bool")
    X_cat = split["X_cat"]

    inputs = [
        X_seq_num,
        X_seq_bool,
    ]

    for i, _ in enumerate(features.get("seq_cat", [])):
        inputs.append(X_seq_cat[:, :, i])

    inputs.append(X_final_num)

    if len(features.get("final_bool", [])) > 0:
        inputs.append(X_final_bool)

    for i, _ in enumerate(features.get("cat", [])):
        inputs.append(X_cat[:, i])

    return inputs


def prepare_sales_dataframe_from_features(df, features):
    """Applica le feature deterministiche comuni a training e inference."""
    features = normalize_sales_feature_schema(features)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"]).copy()

    df = ensure_contamination_ground_truth_columns(
        df,
        features["ground_truth"],
    )

    if "pre_holiday" not in df.columns:
        df["pre_holiday"] = (
            df.groupby("store_id")["actual_holiday"]
              .shift(-1)
              .fillna(0)
              .astype(int)
        )
    else:
        df["pre_holiday"] = df["pre_holiday"].fillna(0).astype(int)

    df["time_idx"] = df.groupby("store_id").cumcount()

    # Create per coerenza con sales_build_LSTM_tuning.ipynb.
    if "sales_rm_30" not in df.columns:
        df["sales_rm_30"] = (
            df.groupby("store_id")["daily_total_sales"]
              .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        )
        df["sales_rm_30"] = df.groupby("store_id")["sales_rm_30"].bfill()

    if "sales_rm_7" not in df.columns:
        df["sales_rm_7"] = (
            df.groupby("store_id")["daily_total_sales"]
              .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
        df["sales_rm_7"] = df.groupby("store_id")["sales_rm_7"].bfill()

    df["days_to_month_end"] = (
        df["date"].dt.days_in_month - df["date"].dt.day
    )

    for col in features.get("log_transform", []):
        if col not in df.columns:
            raise ValueError(f"Colonna richiesta per log transform mancante: {col}")
        df[col] = np.log1p(df[col])

    return df


def build_sales_dataset_train_val_from_template(
    df,
    feature_template=None,
    window_size=28,
    train_size=0.70,
    val_size=0.10,
):
    """
    Costruisce train e validation con lo schema del modello sales selezionato.

    Lo scaler viene stimato sul train del livello corrente.
    """
    features = get_contamination_sales_feature_lists(feature_template)
    df = prepare_sales_dataframe_from_features(df, features)
    df, mappings = encode_sales_categorical_fit(df, features)

    feature_scalers = {}
    train_parts = _make_sales_parts_container()
    val_parts = _make_sales_parts_container()

    for store_id in df["store_id"].unique():
        temp = df[df["store_id"] == store_id].sort_values("date").copy()

        n = len(temp)
        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()

        num_to_scale = [
            col for col in features["scale"]
            if col in train_df.columns
        ]

        scaler = StandardScaler()
        scaler.fit(train_df[num_to_scale].astype(float))
        feature_scalers[store_id] = scaler

        train_df[num_to_scale] = scaler.transform(
            train_df[num_to_scale].astype(float)
        )
        val_df[num_to_scale] = scaler.transform(
            val_df[num_to_scale].astype(float)
        )

        _append_sales_parts(
            train_parts,
            create_sales_sequences_from_features(train_df, features, window_size),
        )
        _append_sales_parts(
            val_parts,
            create_sales_sequences_from_features(val_df, features, window_size),
        )

    train = _attach_sales_features(_concat_sales_parts(train_parts), features)
    val = _attach_sales_features(_concat_sales_parts(val_parts), features)

    return train, val, feature_scalers, mappings, features


def build_sales_dataset_inference_from_template(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=28,
    train_size=0.70,
    val_size=0.10,
):
    """Costruisce train, validation e test con preprocessing già appreso."""
    features = normalize_sales_feature_schema(features)

    df = prepare_sales_dataframe_from_features(df, features)
    df = encode_sales_categorical_transform(df, mappings)

    train_parts = _make_sales_parts_container()
    val_parts = _make_sales_parts_container()
    test_parts = _make_sales_parts_container()

    for store_id in df["store_id"].unique():
        temp = df[df["store_id"] == store_id].sort_values("date").copy()

        n = len(temp)
        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        # Riusa preprocessing e scala appresi dal livello di contaminazione corrente.
        scaler = feature_scalers[store_id]
        num_to_scale = [
            col for col in features["scale"]
            if col in train_df.columns
        ]

        train_df[num_to_scale] = scaler.transform(
            train_df[num_to_scale].astype(float)
        )
        val_df[num_to_scale] = scaler.transform(
            val_df[num_to_scale].astype(float)
        )
        test_df[num_to_scale] = scaler.transform(
            test_df[num_to_scale].astype(float)
        )

        _append_sales_parts(
            train_parts,
            create_sales_sequences_from_features(train_df, features, window_size),
        )
        _append_sales_parts(
            val_parts,
            create_sales_sequences_from_features(val_df, features, window_size),
        )
        _append_sales_parts(
            test_parts,
            create_sales_sequences_from_features(test_df, features, window_size),
        )

    train = _attach_sales_features(_concat_sales_parts(train_parts), features)
    val = _attach_sales_features(_concat_sales_parts(val_parts), features)
    test = _attach_sales_features(_concat_sales_parts(test_parts), features)

    return train, val, test


def make_sales_results_from_features(
    split,
    y_pred,
    features,
    feature_scalers=None,
):
    """Crea il dataframe dei risultati rispettando l'ordine delle categoriche."""
    features = normalize_sales_feature_schema(features)

    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(split["y"]).reshape(-1)

    store_idx = features.get("cat", []).index("store_id")

    results = pd.DataFrame({
        "date": pd.to_datetime(split["date"]),
        "store_id": split["X_cat"][:, store_idx].astype(int),
        "y_true": y_true,
        "y_pred": y_pred,
    })

    results["residual"] = results["y_true"] - results["y_pred"]
    results["abs_error"] = results["residual"].abs()
    results["squared_error"] = results["residual"] ** 2

    if feature_scalers is not None:
        target_col = features["target"]
        target_is_log = target_col in features.get("log_transform", [])

        results["y_true_original"] = np.nan
        results["y_pred_original"] = np.nan

        for store_id, idx in results.groupby("store_id").groups.items():
            scaler = feature_scalers[store_id]

            target_idx = list(scaler.feature_names_in_).index(target_col)
            target_mean = scaler.mean_[target_idx]
            target_std = np.sqrt(scaler.var_[target_idx])

            y_true_unscaled = (
                results.loc[idx, "y_true"] * target_std + target_mean
            )
            y_pred_unscaled = (
                results.loc[idx, "y_pred"] * target_std + target_mean
            )

            if target_is_log:
                results.loc[idx, "y_true_original"] = np.expm1(y_true_unscaled)
                results.loc[idx, "y_pred_original"] = np.expm1(y_pred_unscaled)
            else:
                results.loc[idx, "y_true_original"] = y_true_unscaled
                results.loc[idx, "y_pred_original"] = y_pred_unscaled

        results["residual_original"] = (
            results["y_true_original"] - results["y_pred_original"]
        )
        results["abs_error_original"] = results["residual_original"].abs()
        results["squared_error_original"] = results["residual_original"] ** 2

    if "ground_truth" in split and len(split["ground_truth"]) > 0:
        gt_cols = features.get("ground_truth", [])
        gt = pd.DataFrame(split["ground_truth"], columns=gt_cols)
        results = pd.concat(
            [results.reset_index(drop=True), gt.reset_index(drop=True)],
            axis=1,
        )

    return results


def _sales_cardinality(mappings, col):
    if col not in mappings:
        raise KeyError(f"Colonna categorica non trovata nei mappings: {col}")
    return len(mappings[col])


def _sales_embedding_dim_for_col(col):
    if col == "store_id":
        return 4
    if col in ["holiday", "actual_holiday", "pre_holiday"]:
        return 2
    return 3


def _sales_seq_input_name(col):
    if col == "week_day":
        return "seq_weekday_input"
    return f"seq_{col}_input"


def _sales_seq_embedding_name(col):
    if col == "week_day":
        return "seq_weekday_embedding"
    return f"seq_{col}_embedding"


def _sales_final_embedding_name(col):
    names = {
        "store_id": "store_embedding",
        "week_day": "weekday_embedding",
        "month": "month_embedding",
        "day": "day_embedding",
        "holiday": "holiday_embedding",
        "actual_holiday": "actual_holiday_embedding",
        "pre_holiday": "pre_holiday_embedding",
    }
    return names.get(col, f"{col}_embedding")


def build_lstm_sales_model_from_features(
    train,
    mappings,
    lstm_units=32,
    seq_dense_units=16,
    dense_1_units=32,
    dense_2_units=16,
    dropout_rate=0.025,
    learning_rate=5e-4,
):
    """
    Costruisce il modello LSTM sales usando lo schema feature allegato allo split.

    L'ordine degli input coincide con build_sales_model_inputs_from_features().
    """
    features = normalize_sales_feature_schema(train.get("_features"))

    X_seq_num_train = train["X_seq_num"]
    X_seq_bool_train = train["X_seq_bool"]
    X_final_num_train = train["X_final_num"]
    X_final_bool_train = train.get("X_final_bool")

    window_size = X_seq_num_train.shape[1]
    n_num_features = X_seq_num_train.shape[2]
    n_bool_features = X_seq_bool_train.shape[2]
    n_final_num = X_final_num_train.shape[1]
    n_final_bool = (
        X_final_bool_train.shape[1]
        if X_final_bool_train is not None
        and len(features.get("final_bool", [])) > 0
        else 0
    )

    seq_num_input = Input(
        shape=(window_size, n_num_features),
        name="seq_num_input",
    )

    seq_bool_input = Input(
        shape=(window_size, n_bool_features),
        name="seq_bool_input",
    )

    model_inputs = [seq_num_input, seq_bool_input]
    seq_concat_parts = [seq_num_input, seq_bool_input]

    for col in features.get("seq_cat", []):
        inp = Input(shape=(window_size,), name=_sales_seq_input_name(col))
        emb = Embedding(
            input_dim=_sales_cardinality(mappings, col),
            output_dim=_sales_embedding_dim_for_col(col),
            name=_sales_seq_embedding_name(col),
        )(inp)
        model_inputs.append(inp)
        seq_concat_parts.append(emb)

    seq_input = Concatenate(axis=-1, name="seq_input_concat")(seq_concat_parts)

    x_seq = LSTM(
        lstm_units,
        name="lstm_block",
    )(seq_input)

    x_seq = Dense(
        seq_dense_units,
        activation="relu",
        name="seq_dense",
    )(x_seq)

    final_num_input = Input(
        shape=(n_final_num,),
        name="final_num_input",
    )
    model_inputs.append(final_num_input)

    final_concat_parts = [
        x_seq,
        final_num_input,
    ]

    if n_final_bool > 0:
        final_bool_input = Input(
            shape=(n_final_bool,),
            name="final_bool_input",
        )
        model_inputs.append(final_bool_input)
        final_concat_parts.append(final_bool_input)

    for col in features.get("cat", []):
        inp = Input(shape=(1,), name=col)
        emb_name = _sales_final_embedding_name(col)
        emb = Flatten(name=f"{emb_name}_flatten")(
            Embedding(
                input_dim=_sales_cardinality(mappings, col),
                output_dim=_sales_embedding_dim_for_col(col),
                name=emb_name,
            )(inp)
        )
        model_inputs.append(inp)
        final_concat_parts.append(emb)

    x = Concatenate(name="final_concat")(final_concat_parts)

    x = Dense(
        dense_1_units,
        activation="relu",
        name="dense_1",
    )(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate, name="dropout")(x)

    x = Dense(
        dense_2_units,
        activation="relu",
        name="dense_2",
    )(x)

    output = Dense(1, name="daily_total_sales_output")(x)

    model = Model(
        inputs=model_inputs,
        outputs=output,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"],
    )

    return model


def infer_model_window_size(model, fallback=28):
    """Inferisce la lunghezza della sequenza dal primo input del modello."""
    try:
        input_shape = model.input_shape[0]
        window_size = input_shape[1]
        if window_size is not None:
            return int(window_size)
    except Exception:
        pass
    return int(fallback)


# =========================================================
# DOUBLE-MAD THRESHOLDS
# =========================================================


def compute_level_shift_double_mad_thresholds_by_store(
    val_results,
    score_col="level_shift_score",
    store_col="store_id",
    threshold_multiplier=3.5,
    mad_factor=1.4826,
    eps=1e-8,
    fallback_to_global_mad=True,
):
    """
    Calcola soglie store-specific usando MAD asimmetrica.

    Per ogni store:
        center = median(score)

        lower_mad = median(center - score), usando solo score <= center
        upper_mad = median(score - center), usando solo score >= center

        lower_scale = mad_factor * lower_mad
        upper_scale = mad_factor * upper_mad

    Soglie:
        lower = center - threshold_multiplier * lower_scale
        upper = center + threshold_multiplier * upper_scale
    """
    thresholds = {}

    for store_id, g in val_results.groupby(store_col):
        scores = g[score_col].dropna().astype(float)

        if scores.empty:
            raise ValueError(f"Nessuno score valido per store_id={store_id}")

        center = scores.median()

        lower_scores = scores[scores <= center]
        upper_scores = scores[scores >= center]

        lower_mad = (center - lower_scores).median()
        upper_mad = (upper_scores - center).median()

        global_mad = (scores - center).abs().median()

        lower_scale = mad_factor * lower_mad
        upper_scale = mad_factor * upper_mad
        global_scale = mad_factor * global_mad

        used_lower_fallback = False
        used_upper_fallback = False

        if pd.isna(lower_scale) or lower_scale < eps:
            if (
                fallback_to_global_mad
                and not pd.isna(global_scale)
                and global_scale >= eps
            ):
                lower_scale = global_scale
            else:
                lower_scale = eps

            used_lower_fallback = True

        if pd.isna(upper_scale) or upper_scale < eps:
            if (
                fallback_to_global_mad
                and not pd.isna(global_scale)
                and global_scale >= eps
            ):
                upper_scale = global_scale
            else:
                upper_scale = eps

            used_upper_fallback = True

        if pd.isna(lower_scale) or lower_scale < eps:
            lower_scale = eps
            used_lower_fallback = True

        if pd.isna(upper_scale) or upper_scale < eps:
            upper_scale = eps
            used_upper_fallback = True

        thresholds[store_id] = {
            "lower": center - threshold_multiplier * lower_scale,
            "upper": center + threshold_multiplier * upper_scale,
            "median": center,
            "lower_mad": lower_mad,
            "upper_mad": upper_mad,
            "global_mad": global_mad,
            "lower_scale": lower_scale,
            "upper_scale": upper_scale,
            "global_scale": global_scale,
            "threshold_method": "double_mad",
            "threshold_multiplier": threshold_multiplier,
            "mad_factor": mad_factor,
            "used_lower_fallback": used_lower_fallback,
            "used_upper_fallback": used_upper_fallback,
        }

    return thresholds


def double_mad_thresholds_to_dataframe(
    thresholds,
    contamination_level,
    threshold_method="double_mad",
    threshold_multiplier=3.5,
    mad_factor=1.4826,
):
    """Converte le soglie double-MAD in una tabella per store e livello."""
    rows = []

    for store_id, values in thresholds.items():
        row = {
            "contamination_level": int(contamination_level),
            "contamination_percent": int(contamination_level) / 10.0,
            "contamination_fraction": int(contamination_level) / 1000.0,
            "store_id": store_id,
            "threshold_method": values.get(
                "threshold_method",
                threshold_method,
            ),
            "threshold_multiplier": values.get(
                "threshold_multiplier",
                threshold_multiplier,
            ),
            "score_median": values.get("median", np.nan),
            "lower_mad": values.get("lower_mad", np.nan),
            "upper_mad": values.get("upper_mad", np.nan),
            "global_mad": values.get("global_mad", np.nan),
            "lower_scale": values.get("lower_scale", np.nan),
            "upper_scale": values.get("upper_scale", np.nan),
            "global_scale": values.get("global_scale", np.nan),
            "mad_factor": values.get("mad_factor", mad_factor),
            "used_lower_fallback": values.get("used_lower_fallback", False),
            "used_upper_fallback": values.get("used_upper_fallback", False),
            "lower_threshold": values.get("lower", np.nan),
            "upper_threshold": values.get("upper", np.nan),
        }
        row["threshold_half_width"] = (
            row["upper_threshold"] - row["lower_threshold"]
        ) / 2
        row["lower_threshold_distance"] = (
            row["score_median"] - row["lower_threshold"]
        )
        row["upper_threshold_distance"] = (
            row["upper_threshold"] - row["score_median"]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def classify_level_shift_severity(
    multiplier,
    soft_threshold=0.05,
    medium_threshold=0.15,
):
    """Classifica la severità dalla distanza del moltiplicatore da 1."""
    deviation = round(abs(float(multiplier) - 1.0), 3)

    if deviation <= soft_threshold:
        return "soft"
    if deviation <= medium_threshold:
        return "medium"
    return "hard"

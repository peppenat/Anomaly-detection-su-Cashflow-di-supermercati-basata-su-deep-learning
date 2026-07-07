# -*- coding: utf-8 -*-
"""Utility comuni per la pipeline POS delay.

Il modulo raccoglie:
1. preprocessing, costruzione dei dataset e modello LSTM POS;
2. inference con gli artifact del modello selezionato;
3. costruzione degli score a profili su business days e detector event-level;
4. helper condivisi dagli esperimenti di sensitivity, refinement, baseline,
   Autoencoder e contaminazione.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model

from lstm_utils import (
    build_pos_model_inputs,
    concatenate_parts,
    encode_categorical,
    evaluate_detected_windows_event_level,
    log_transform,
    make_results_df,
)


# =========================================================
# CONFIGURAZIONE BASE DEL DETECTOR
# =========================================================

POS_DELAY_DETECTOR_CONFIG = {
    "score_col": "pos_cos",
    "profile_window_size": 7,
    "z_threshold": 3.5,
    "min_consecutive": 2,
    "gap_tolerance": 1,
    "detected_window_mode": "profile_windows_union",
    "iou_threshold": 0.20,
}


# =========================================================
# PREPARAZIONE DEL DATAFRAME
# =========================================================

def prepare_pos_dataframe(df):
    """
    Rende il dataframe compatibile con la pipeline POS.

    I dataset di sensitivity POS sono volutamente minimi, quindi qui vengono
    ricostruite eventuali colonne derivate non salvate.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values(["store_id", "date"])
          .reset_index(drop=True)
    )

    if "day" not in df.columns:
        df["day"] = df["date"].dt.day

    if "weekend" not in df.columns:
        df["weekend"] = (df["week_day"].astype(int) >= 5).astype(int)

    if "actual_holiday" not in df.columns:
        raise ValueError(
            "Manca 'actual_holiday', necessaria per ricostruire "
            "pre_holiday."
        )

    if "is_point_anomaly" not in df.columns:
        df["is_point_anomaly"] = 0

    if "pa_type" not in df.columns:
        df["pa_type"] = "normal"

    if "pa_mult" not in df.columns:
        df["pa_mult"] = 1.0

    required_cols = [
        "date",
        "store_id",
        "week_day",
        "month",
        "holiday",
        "actual_holiday",
        "pos_card_sales",
        "pos_net_cf",

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
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Mancano colonne richieste per POS: {missing}")

    return df




# =========================================================
# SCHEMA DELLE FEATURE E SEQUENZE
# =========================================================

def get_feature_lists():
    """Restituisce lo schema delle feature del modello LSTM POS."""
    seq_num_features = [
        "pos_card_sales",
    ]

    seq_bool_features = [
        "holiday",
        "pre_holiday",
    ]

    seq_cat_features = [
        "week_day",
        "month",
    ]

    final_num_features = []

    final_bool_features = [
        "holiday",
        "actual_holiday",
        "pre_holiday",
    ]

    cat_features = [
        "store_id",
        "week_day",
        "month",
    ]

    ground_truth_features = [
        "is_pos_delay_source_day",
        "pos_delay_source_day_in_event",
        "pos_delay_source_duration",

        "is_pos_delay_effect_day",
        "pos_delay_effect_day_in_event",
        "pos_delay_effect_duration",

        "pos_delay_event_id",
        "pos_delay_type",
    ]

    return {
        "seq_num": seq_num_features,
        "seq_bool": seq_bool_features,
        "seq_cat": seq_cat_features,
        "final_num": final_num_features,
        "final_bool": final_bool_features,
        "cat": cat_features,
        "ground_truth": ground_truth_features,
        "target": "pos_net_cf",
        "log_transform": [
            "pos_card_sales",
            "pos_net_cf",
        ],
    }


def create_pos_sequences(df, features, window):
    """
    Converte una serie temporale giornaliera in sequenze per il modello LSTM POS.

    Per ogni istante target t:
    - le feature sequenziali usano la finestra [t-window, ..., t-1];
    - le feature finali descrivono il giorno target t;
    - il target y è pos_net_cf al giorno t.

    Le colonne di ground truth vengono conservate solo per analisi successive
    e non sono usate come input del modello.
    """

    seq_num_features = features["seq_num"]
    seq_bool_features = features["seq_bool"]
    seq_cat_features = features["seq_cat"]
    final_num_features = features.get("final_num", [])
    final_bool_features = features.get("final_bool", [])
    cat_features = features["cat"]
    ground_truth_features = features["ground_truth"]
    target = features["target"]

    X_seq_num = []
    X_seq_bool = []
    X_seq_cat = []
    X_final_num = []
    X_final_bool = []
    X_cat = []
    y = []
    dates = []
    ground_truth = []

    # Conversione preliminare in array NumPy per evitare accessi ripetuti al dataframe.
    seq_num_vals = df[seq_num_features].values
    seq_bool_vals = df[seq_bool_features].values
    seq_cat_vals = df[seq_cat_features].values

    # Nel caso POS non ci sono feature numeriche finali, ma il blocco resta generale.
    final_num_vals = (
        df[final_num_features].values
        if len(final_num_features) > 0
        else np.empty((len(df), 0))
    )

    final_bool_vals = df[final_bool_features].values
    cat_vals = df[cat_features].values
    ground_truth_vals = df[ground_truth_features].values
    y_vals = df[target].values
    d_vals = df["date"].values

    for i in range(len(df) - window):
        target_pos = i + window

        # Finestra storica precedente al giorno target.
        X_seq_num.append(seq_num_vals[i:target_pos])
        X_seq_bool.append(seq_bool_vals[i:target_pos])
        X_seq_cat.append(seq_cat_vals[i:target_pos])

        # Feature note o calendariali riferite al giorno target.
        X_final_num.append(final_num_vals[target_pos])
        X_final_bool.append(final_bool_vals[target_pos])
        X_cat.append(cat_vals[target_pos])

        # Target e metadati associati al giorno predetto.
        y.append(y_vals[target_pos])
        dates.append(d_vals[target_pos])
        ground_truth.append(ground_truth_vals[target_pos])

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


def make_pos_parts_container():
    """
    Inizializza il container usato per raccogliere le sequenze dei diversi store.

    Ogni lista verrà poi concatenata lungo l'asse delle osservazioni.
    """

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


def append_pos_sequence_parts(container, seq):
    """
    Accumula le sequenze generate per uno store dentro il container
    dello split corrispondente.
    """

    for key in container.keys():
        container[key].append(seq[key])




# =========================================================
# COSTRUZIONE DEI DATASET
# =========================================================

def build_dataset_train_val(df, window_size=7, train_size=0.70, val_size=0.10):
    """
    Costruisce train e validation set per il tuning del modello POS.

    La suddivisione è temporale e viene fatta separatamente per ogni store.
    Le trasformazioni di scala sono stimate solo sul train dello store, così da
    evitare leakage dal validation set.

    Il test set non viene costruito in questa funzione perché non è usato
    durante il tuning.
    """

    features = get_feature_lists()

    seq_num_features = features["seq_num"]
    final_num_features = features.get("final_num", [])
    target = features["target"]
    log_cols = features["log_transform"]

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["store_id", "date"]).copy()

    # Feature calendariale nota al giorno t:
    # vale 1 se il giorno successivo è una festività effettiva.
    data["pre_holiday"] = (
        data.groupby("store_id")["actual_holiday"]
            .shift(-1)
            .fillna(0)
            .astype(int)
    )

    # Trasformazione logaritmica sulle variabili monetarie selezionate.
    data = log_transform(data, log_cols)

    # Encoding intero delle variabili categoriche usate negli embedding.
    data, mappings = encode_categorical(data, features)

    feature_scalers = {}

    train_parts = make_pos_parts_container()
    val_parts = make_pos_parts_container()

    # Variabili numeriche da standardizzare per store.
    # Include il target perché il modello viene addestrato su target standardizzato.
    num_to_scale = list(dict.fromkeys(
        seq_num_features + final_num_features + [target]
    ))

    for store_id in data["store_id"].unique():
        temp = data[data["store_id"] == store_id].sort_values("date").copy()
        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()

        # Lo scaler è stimato solo sul train dello store.
        scaler = StandardScaler()
        scaler.fit(train_df[num_to_scale].astype(float))
        feature_scalers[store_id] = scaler

        train_df[num_to_scale] = scaler.transform(
            train_df[num_to_scale].astype(float)
        )
        val_df[num_to_scale] = scaler.transform(
            val_df[num_to_scale].astype(float)
        )

        # Generazione delle finestre LSTM per lo store corrente.
        append_pos_sequence_parts(
            train_parts,
            create_pos_sequences(train_df, features, window_size),
        )

        append_pos_sequence_parts(
            val_parts,
            create_pos_sequences(val_df, features, window_size),
        )

    train = concatenate_parts(train_parts)
    val = concatenate_parts(val_parts)

    return train, val, feature_scalers, mappings, features


def build_dataset_train_val_test_from_artifacts(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=7,
    train_size=0.70,
    val_size=0.10,
):
    """
    Ricostruisce gli split per il modello POS usando gli artifact salvati.

    Gli scaler e i mapping categorici sono quelli associati al modello promosso.
    Non viene effettuato alcun nuovo fit durante la valutazione finale.
    """

    seq_num_features = features["seq_num"]
    final_num_features = features.get("final_num", [])
    target = features["target"]
    log_cols = features["log_transform"]

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["store_id", "date"]).copy()

    data["pre_holiday"] = (
        data.groupby("store_id")["actual_holiday"]
            .shift(-1)
            .fillna(0)
            .astype(int)
    )

    data = log_transform(data, log_cols)

    # Riusa i mapping del modello promosso senza ricodificare le categorie.
    for col, mapping in mappings.items():
        data[col] = pd.Categorical(
            data[col],
            categories=mapping,
        ).codes

        if (data[col] < 0).any():
            raise ValueError(
                f"La colonna {col} contiene categorie non viste nel training."
            )

    num_to_scale = list(dict.fromkeys(
        seq_num_features + final_num_features + [target]
    ))

    train_parts = make_pos_parts_container()
    val_parts = make_pos_parts_container()
    test_parts = make_pos_parts_container()

    for store_id in data["store_id"].unique():
        temp = data[data["store_id"] == store_id].sort_values("date").copy()
        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        # Riusa lo scaler stimato sul train durante il tuning.
        scaler = feature_scalers[store_id]

        train_df[num_to_scale] = scaler.transform(
            train_df[num_to_scale].astype(float)
        )
        val_df[num_to_scale] = scaler.transform(
            val_df[num_to_scale].astype(float)
        )
        test_df[num_to_scale] = scaler.transform(
            test_df[num_to_scale].astype(float)
        )

        append_pos_sequence_parts(
            train_parts,
            create_pos_sequences(train_df, features, window_size),
        )

        append_pos_sequence_parts(
            val_parts,
            create_pos_sequences(val_df, features, window_size),
        )

        append_pos_sequence_parts(
            test_parts,
            create_pos_sequences(test_df, features, window_size),
        )

    train = concatenate_parts(train_parts)
    val = concatenate_parts(val_parts)
    test = concatenate_parts(test_parts)

    return train, val, test




# =========================================================
# MODELLO LSTM POS
# =========================================================

def build_lstm_pos_model(
    train,
    architecture_config,
    dropout_rate=0.05,
    learning_rate=1e-3,
):
    """
    Costruisce il modello LSTM per la previsione di pos_net_cf.

    Il modello combina:
    - un ramo sequenziale, basato sulla finestra storica di 7 giorni;
    - un ramo finale, basato sulle feature note al giorno target t.

    Le variabili categoriche vengono rappresentate tramite embedding, mentre
    le booleane entrano direttamente come valori binari.
    """

    X_seq_num = train["X_seq_num"]
    X_seq_bool = train["X_seq_bool"]
    X_seq_cat = train["X_seq_cat"]
    X_final_num = train["X_final_num"]
    X_final_bool = train["X_final_bool"]
    X_cat = train["X_cat"]

    lstm_units = architecture_config["lstm_units"]
    seq_dense_units = architecture_config["seq_dense_units"]
    dense_1_units = architecture_config["dense_1_units"]
    dense_2_units = architecture_config["dense_2_units"]

    # Dimensioni degli input ricavate direttamente dal training set.
    window_size = X_seq_num.shape[1]
    n_num_features = X_seq_num.shape[2]
    n_bool_features = X_seq_bool.shape[2]
    n_final_num = X_final_num.shape[1]
    n_final_bool = X_final_bool.shape[1]

    # Cardinalità delle variabili categoriche embedded.
    n_seq_weekday = int(X_seq_cat[:, :, 0].max()) + 1
    n_seq_month = int(X_seq_cat[:, :, 1].max()) + 1

    n_store = int(X_cat[:, 0].max()) + 1
    n_final_weekday = int(X_cat[:, 1].max()) + 1
    n_final_month = int(X_cat[:, 2].max()) + 1

    # -------------------------
    # Ramo sequenziale
    # -------------------------

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

    # La finestra storica combina variabili numeriche, booleane e categoriche embedded.
    seq_input = Concatenate(axis=-1, name="seq_input_concat")([
        seq_num_input,
        seq_bool_input,
        seq_week_emb,
        seq_month_emb,
    ])

    x_seq = LSTM(
        lstm_units,
        name="lstm_block",
    )(seq_input)

    x_seq = Dense(
        seq_dense_units,
        activation="relu",
        name="seq_dense",
    )(x_seq)

    # -------------------------
    # Ramo del giorno target
    # -------------------------

    final_num_input = Input(
        shape=(n_final_num,),
        name="final_num_input",
    )

    final_bool_input = Input(
        shape=(n_final_bool,),
        name="final_bool_input",
    )

    store_input = Input(
        shape=(1,),
        name="store_id_input",
    )

    final_weekday_input = Input(
        shape=(1,),
        name="final_weekday_input",
    )

    final_month_input = Input(
        shape=(1,),
        name="final_month_input",
    )

    store_emb = Flatten(name="store_embedding_flatten")(
        Embedding(
            input_dim=n_store,
            output_dim=5,
            name="store_embedding",
        )(store_input)
    )

    final_weekday_emb = Flatten(name="final_weekday_embedding_flatten")(
        Embedding(
            input_dim=n_final_weekday,
            output_dim=3,
            name="final_weekday_embedding",
        )(final_weekday_input)
    )

    final_month_emb = Flatten(name="final_month_embedding_flatten")(
        Embedding(
            input_dim=n_final_month,
            output_dim=3,
            name="final_month_embedding",
        )(final_month_input)
    )

    # Fusione tra rappresentazione storica e informazioni del giorno target.
    x = Concatenate(name="final_concat")([
        x_seq,
        final_num_input,
        final_bool_input,
        store_emb,
        final_weekday_emb,
        final_month_emb,
    ])

    x = Dense(
        dense_1_units,
        activation="relu",
        name="dense_1",
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    x = Dense(
        dense_2_units,
        activation="relu",
        name="dense_2",
    )(x)

    output = Dense(
        1,
        name="pos_net_cf_output",
    )(x)

    model = Model(
        inputs=[
            seq_num_input,
            seq_bool_input,
            seq_weekday_input,
            seq_month_input,
            final_num_input,
            final_bool_input,
            store_input,
            final_weekday_input,
            final_month_input,
        ],
        outputs=output,
    )

    # Loss MAE coerente con l'obiettivo di forecasting su scala trasformata e standardizzata.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"],
    )

    return model




# =========================================================
# INDICE DEI DATASET E CACHE
# =========================================================

def list_pos_delay_sensitivity_datasets(
    base_path,
    source_duration_filter=None,
    delay_types_filter=None,
):
    """
    Lista i dataset POS delay generati per sensitivity/tuning.

    Struttura attesa:

        base_path /
            delay_type /
                srcdur_<source_duration> /
                    seed_<seed> /
                        all_stores_cashflow.csv
    """

    base_path = Path(base_path)
    rows = []

    pattern = "* /srcdur_* /seed_* /all_stores_cashflow.csv".replace(" ", "")

    for csv_path in base_path.glob(pattern):
        delay_type = csv_path.parents[2].name
        srcdur_name = csv_path.parents[1].name
        seed_name = csv_path.parents[0].name

        srcdur_match = re.match(r"srcdur_(\d+)", srcdur_name)
        seed_match = re.match(r"seed_(\d+)", seed_name)

        if srcdur_match is None or seed_match is None:
            continue

        source_duration = int(srcdur_match.group(1))
        seed = int(seed_match.group(1))

        if source_duration_filter is not None:
            if source_duration not in source_duration_filter:
                continue

        if delay_types_filter is not None:
            if delay_type not in delay_types_filter:
                continue

        rows.append({
            "path": csv_path,
            "delay_type": delay_type,
            "source_duration": source_duration,
            "seed": seed,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["delay_type", "source_duration", "seed"])
        .reset_index(drop=True)
    )


def make_pos_delay_cache_path(dataset_row, cache_dir):
    """
    Costruisce il path della cache per un dataset identificato da
    delay_type, source_duration e seed.
    """

    delay_type = dataset_row["delay_type"]
    source_duration = int(dataset_row["source_duration"])
    seed = int(dataset_row["seed"])

    name = (
        f"{delay_type}"
        f"_srcdur_{source_duration}"
        f"_seed_{seed}"
        ".pkl"
    )

    return Path(cache_dir) / name




# =========================================================
# INFERENCE LSTM POS
# =========================================================

def add_columns_from_original(results_df, original_df, cols):
    """
    Aggiunge colonne giornaliere a results_df tramite merge su store_id/date.
    Serve perché make_results_df non conserva tutte le colonne calendariali.
    """

    results = results_df.copy()
    original = original_df.copy()

    results["date"] = pd.to_datetime(results["date"])
    original["date"] = pd.to_datetime(original["date"])

    merge_cols = ["store_id", "date"] + cols

    extra = (
        original[merge_cols]
        .drop_duplicates(["store_id", "date"])
    )

    return results.merge(
        extra,
        on=["store_id", "date"],
        how="left",
    )


def run_pos_model_inference(
    df,
    model,
    feature_scalers,
    mappings,
    features,
    window_size=7,
):
    """
    Esegue inference POS con un modello già addestrato.

    Restituisce i risultati di train, validation e test usando gli artifact
    di preprocessing associati al modello.
    """

    df = prepare_pos_dataframe(df)

    train, val, test = build_dataset_train_val_test_from_artifacts(
        df,
        feature_scalers=feature_scalers,
        mappings=mappings,
        features=features,
        window_size=window_size,
    )

    split_results = {}

    for split_name, split_data in {
        "train": train,
        "val": val,
        "test": test,
    }.items():
        prediction = model.predict(
            build_pos_model_inputs(split_data),
            verbose=0,
        ).reshape(-1)

        results = make_results_df(
            split_data,
            prediction,
            features,
            feature_scalers,
        )

        split_results[split_name] = add_columns_from_original(
            results,
            df,
            ["holiday", "week_day", "month", "actual_holiday"],
        )

    return (
        split_results["train"],
        split_results["val"],
        split_results["test"],
    )


def compute_results_for_dataset(
    csv_path,
    model,
    feature_scalers,
    mappings,
    features,
    window_size=7,
):
    """
    Carica un dataset POS e calcola train/val/test results.
    """

    df = pd.read_csv(csv_path)
    df = prepare_pos_dataframe(df)

    train_results, val_results, test_results = run_pos_model_inference(
        df=df,
        model=model,
        feature_scalers=feature_scalers,
        mappings=mappings,
        features=features,
        window_size=window_size,
    )

    return {
        "df": df,
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
    }




# =========================================================
# DETECTOR POS DELAY BASATO SU PROFILI
# =========================================================

def build_sliding_pos_profile_comparison(
    df,
    true_col="y_true_original",
    pred_col="y_pred_original",
    store_col="store_id",
    date_col="date",
    holiday_col="holiday",
    window_size=5,
    eps=1e-12,
):
    """
    Costruisce profile windows POS su business days.

    Ogni finestra confronta:
    - q_true: distribuzione normalizzata del POS reale nella finestra;
    - q_pred: distribuzione normalizzata del POS atteso nella finestra.

    Score:
    - pos_cos
    - pos_l1
    - pos_wasserstein
    """

    out = []

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    temp = temp.sort_values([store_col, date_col])

    for store_id, g in temp.groupby(store_col):
        g = (
            g[g[holiday_col].astype(int) == 0]
            .sort_values(date_col)
            .reset_index(drop=True)
            .copy()
        )

        if len(g) < window_size:
            continue

        for start in range(len(g) - window_size + 1):
            end = start + window_size
            w = g.iloc[start:end].copy()

            q_true = w[true_col].astype(float).values
            q_pred = w[pred_col].astype(float).values

            q_true = np.clip(q_true, eps, None)
            q_pred = np.clip(q_pred, eps, None)

            q_true = q_true / q_true.sum()
            q_pred = q_pred / q_pred.sum()

            pos_l1 = np.abs(q_true - q_pred).sum()

            pos_cos = cosine(
                q_true,
                q_pred,
            )

            pos_wasserstein = wasserstein_distance(
                np.arange(window_size),
                np.arange(window_size),
                u_weights=q_true,
                v_weights=q_pred,
            )

            center_idx = window_size // 2

            source_window = int(
                w["is_pos_delay_source_day"].astype(int).sum() > 0
            )

            effect_window = int(
                w["is_pos_delay_effect_day"].astype(int).sum() > 0
            )

            source_days = w[
                w["is_pos_delay_source_day"].astype(int) == 1
            ]

            effect_days = w[
                w["is_pos_delay_effect_day"].astype(int) == 1
            ]

            event_candidates = pd.concat([
                source_days[["pos_delay_event_id", "pos_delay_type"]],
                effect_days[["pos_delay_event_id", "pos_delay_type"]],
            ])

            event_candidates = event_candidates[
                event_candidates["pos_delay_event_id"] != -1
            ]

            if event_candidates.empty:
                event_id_window = -1
                type_window = "normal"
            else:
                event_id_window = event_candidates["pos_delay_event_id"].mode().iloc[0]
                type_window = event_candidates["pos_delay_type"].mode().iloc[0]

            row = {
                store_col: store_id,
                "window_start": w[date_col].min(),
                "window_end": w[date_col].max(),
                "center_date": w.iloc[center_idx][date_col],

                "pos_cos": pos_cos,
                "pos_l1": pos_l1,
                "pos_wasserstein": pos_wasserstein,

                "is_pos_delay_source_window": source_window,
                "is_pos_delay_effect_window": effect_window,
                "pos_delay_event_id_window": event_id_window,
                "pos_delay_type_window": type_window,
            }

            for i in range(window_size):
                row[f"q_true_{i}"] = q_true[i]
                row[f"q_pred_{i}"] = q_pred[i]

            out.append(row)

    return pd.DataFrame(out)


def compute_pos_profile_zscore(
    val_profiles_cmp,
    test_profiles_cmp,
    score_col="pos_cos",
    store_col="store_id",
):
    """
    Calcola z-score store-specific usando il validation set.
    """

    val_df = val_profiles_cmp.copy()
    test_df = test_profiles_cmp.copy()

    stats = (
        val_df
        .groupby(store_col)[score_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "score_mean",
            "std": "score_std",
        })
    )

    val_df = val_df.merge(
        stats,
        on=store_col,
        how="left",
    )

    test_df = test_df.merge(
        stats,
        on=store_col,
        how="left",
    )

    val_df["pos_profile_zscore"] = (
        (val_df[score_col] - val_df["score_mean"]) /
        (val_df["score_std"] + 1e-12)
    )

    test_df["pos_profile_zscore"] = (
        (test_df[score_col] - test_df["score_mean"]) /
        (test_df["score_std"] + 1e-12)
    )

    return val_df, test_df


def detect_pos_delay_windows(
    profiles_cmp,
    z_col="pos_profile_zscore",
    z_threshold=3.0,
):
    """
    Marca come rilevate le profile windows con z-score sopra soglia.
    """

    df = profiles_cmp.copy()

    df["is_pos_delay_detected_window"] = (
        df[z_col] > z_threshold
    ).astype(int)

    return df


def build_pos_detected_windows_from_profile_windows(
    profiles_cmp,
    detected_col="is_pos_delay_detected_window",
    store_col="store_id",
    window_start_col="window_start",
    window_end_col="window_end",
    center_col="center_date",
    score_col="pos_cos",
    min_consecutive=2,
    gap_tolerance=1,
    detected_half_window=2,
):
    """
    Costruisce detected windows POS delay partendo direttamente
    dalle profile windows rilevate.

    gap_tolerance:
    - 0: nessuna tolleranza;
    - 1: riempie pattern tipo 1-0-1;
    - 2: riempie buchi fino a due profile windows.
    """

    temp = profiles_cmp.copy()

    required_cols = [
        store_col,
        detected_col,
        window_start_col,
        window_end_col,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in temp.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Mancano queste colonne in profiles_cmp: {missing_cols}"
        )

    temp[window_start_col] = pd.to_datetime(temp[window_start_col])
    temp[window_end_col] = pd.to_datetime(temp[window_end_col])

    if center_col in temp.columns:
        temp[center_col] = pd.to_datetime(temp[center_col])
        sort_col = center_col
    else:
        temp["_center_tmp"] = (
            temp[window_start_col]
            + (temp[window_end_col] - temp[window_start_col]) / 2
        )
        sort_col = "_center_tmp"

    out = []

    for store_id, g in temp.groupby(store_col):
        g = (
            g.sort_values(sort_col)
             .reset_index(drop=True)
             .copy()
        )

        g["_detected_raw"] = g[detected_col].astype(int)
        g["_detected_smooth"] = g["_detected_raw"].copy()

        if gap_tolerance > 0:
            n = len(g)
            detected = g["_detected_raw"].values.copy()

            i = 0

            while i < n:
                if detected[i] == 1:
                    i += 1
                    continue

                gap_start = i

                while i < n and detected[i] == 0:
                    i += 1

                gap_end = i - 1
                gap_len = gap_end - gap_start + 1

                has_left_detection = (
                    gap_start > 0
                    and detected[gap_start - 1] == 1
                )

                has_right_detection = (
                    i < n
                    and detected[i] == 1
                )

                if (
                    has_left_detection
                    and has_right_detection
                    and gap_len <= gap_tolerance
                ):
                    detected[gap_start:gap_end + 1] = 1

            g["_detected_smooth"] = detected

        g["_block"] = (
            g["_detected_smooth"]
            .ne(g["_detected_smooth"].shift())
            .cumsum()
        )

        detected_runs = (
            g[g["_detected_smooth"] == 1]
            .groupby("_block")
        )

        for _, run in detected_runs:
            if len(run) < min_consecutive:
                continue

            detected_start = run[sort_col].min() - pd.Timedelta(days=detected_half_window)
            detected_end = run[sort_col].max() + pd.Timedelta(days=detected_half_window)

            row = {
                store_col: store_id,
                "detected_start": detected_start,
                "detected_end": detected_end,
                "detected_duration_calendar": (
                    detected_end - detected_start
                ).days + 1,

                "n_profile_windows_in_event": len(run),
                "n_raw_detected_profile_windows": int(
                    run["_detected_raw"].sum()
                ),
                "n_filled_gap_profile_windows": int(
                    run["_detected_smooth"].sum()
                    - run["_detected_raw"].sum()
                ),

                "detected_center_start": run[sort_col].min(),
                "detected_center_end": run[sort_col].max(),
            }

            if score_col in run.columns:
                row[f"{score_col}_mean"] = run[score_col].mean()
                row[f"{score_col}_max"] = run[score_col].max()

            out.append(row)

    return pd.DataFrame(out)


def build_gt_pos_delay_windows(
    df,
    store_col="store_id",
    date_col="date",
    source_col="is_pos_delay_source_day",
    effect_col="is_pos_delay_effect_day",
    event_col="pos_delay_event_id",
    type_col="pos_delay_type",
):
    """
    Costruisce finestre ground truth POS delay a livello evento.

    source_start/source_end:
        giorni di vendita POS affetti dal delay.

    effect_start/effect_end:
        giorni in cui l'effetto è visibile su pos_net_cf.

    gt_start/gt_end:
        alias dell'intervallo effect, usati per la valutazione.
    """

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    temp = (
        temp.sort_values([store_col, date_col])
            .reset_index(drop=True)
    )

    source = temp[
        (temp[source_col].astype(int) == 1) &
        (temp[event_col] != -1)
    ].copy()

    effect = temp[
        (temp[effect_col].astype(int) == 1) &
        (temp[event_col] != -1)
    ].copy()

    out = []

    if source.empty and effect.empty:
        return pd.DataFrame(columns=[
            store_col,
            "gt_event_id",
            "source_start",
            "source_end",
            "source_duration",
            "effect_start",
            "effect_end",
            "effect_duration",
            "gt_start",
            "gt_end",
            "gt_duration",
            "gt_type",
        ])

    event_keys = (
        pd.concat([
            source[[store_col, event_col]],
            effect[[store_col, event_col]],
        ])
        .drop_duplicates()
    )

    for _, key in event_keys.iterrows():
        store_id = key[store_col]
        event_id = key[event_col]

        g_src = source[
            (source[store_col] == store_id) &
            (source[event_col] == event_id)
        ].copy()

        g_eff = effect[
            (effect[store_col] == store_id) &
            (effect[event_col] == event_id)
        ].copy()

        if g_eff.empty:
            continue

        source_start = (
            g_src[date_col].min()
            if not g_src.empty
            else pd.NaT
        )

        source_end = (
            g_src[date_col].max()
            if not g_src.empty
            else pd.NaT
        )

        effect_start = g_eff[date_col].min()
        effect_end = g_eff[date_col].max()

        gt_type = (
            g_src[type_col].mode().iloc[0]
            if not g_src.empty
            else g_eff[type_col].mode().iloc[0]
        )

        out.append({
            store_col: store_id,
            "gt_event_id": event_id,

            "source_start": source_start,
            "source_end": source_end,
            "source_duration": len(g_src),

            "effect_start": effect_start,
            "effect_end": effect_end,
            "effect_duration": len(g_eff),

            "gt_start": effect_start,
            "gt_end": effect_end,
            "gt_duration": len(g_eff),

            "gt_type": gt_type,
        })

    return pd.DataFrame(out)


def run_detector_config_on_results(
    val_results,
    test_results,
    profile_window_size,
    score_col,
    z_threshold,
    min_consecutive,
    gap_tolerance,
    iou_threshold=0.20,
):
    """
    Esegue una configurazione del detector su uno specifico dataset.
    """

    val_profiles_cmp = build_sliding_pos_profile_comparison(
        val_results,
        window_size=profile_window_size,
    )

    test_profiles_cmp = build_sliding_pos_profile_comparison(
        test_results,
        window_size=profile_window_size,
    )

    if val_profiles_cmp.empty or test_profiles_cmp.empty:
        return None

    val_profiles_cmp, test_profiles_cmp = compute_pos_profile_zscore(
        val_profiles_cmp,
        test_profiles_cmp,
        score_col=score_col,
    )

    test_profiles_cmp = detect_pos_delay_windows(
        test_profiles_cmp,
        z_threshold=z_threshold,
    )

    detected_windows = build_pos_detected_windows_from_profile_windows(
        profiles_cmp=test_profiles_cmp,
        detected_col="is_pos_delay_detected_window",
        score_col=score_col,
        min_consecutive=min_consecutive,
        gap_tolerance=gap_tolerance,
    )

    gt_windows = build_gt_pos_delay_windows(
        test_results,
    )

    gt_eval, det_eval, summary = evaluate_detected_windows_event_level(
        gt_windows=gt_windows,
        detected_windows=detected_windows,
        iou_threshold=iou_threshold,
    )

    return {
        "summary": summary,
        "gt_eval": gt_eval,
        "det_eval": det_eval,
        "val_profiles_cmp": val_profiles_cmp,
        "test_profiles_cmp": test_profiles_cmp,
        "detected_windows": detected_windows,
        "gt_windows": gt_windows,
    }




# =========================================================
# HELPER PER LE ANALISI
# =========================================================

def pooled_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan


def pooled_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def pooled_f1(precision, recall):
    if pd.isna(precision) or pd.isna(recall):
        return np.nan
    if precision + recall == 0:
        return np.nan
    return 2 * precision * recall / (precision + recall)


def make_empty_event_summary():
    """Restituisce una summary event-level vuota per run non valutabili."""
    return {
        "n_gt_events": np.nan,
        "n_detected_events": np.nan,
        "tp": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "mean_iou": np.nan,
        "mean_detection_delay": np.nan,
        "mean_det_offset_start": np.nan,
        "mean_det_offset_end": np.nan,
    }


def add_detection_offsets(gt_eval, det_eval):
    """
    Aggiunge offset tra finestra rilevata e ground truth:
    - det_offset_start = detected_start - gt_start;
    - det_offset_end = detected_end - gt_end.

    Valori negativi indicano detection anticipata, valori positivi detection ritardata.
    """

    gt = gt_eval.copy()

    if gt.empty:
        return gt

    gt["gt_start"] = pd.to_datetime(gt["gt_start"])
    gt["gt_end"] = pd.to_datetime(gt["gt_end"])
    gt["det_offset_start"] = np.nan
    gt["det_offset_end"] = np.nan

    if det_eval.empty:
        return gt

    det = det_eval.copy()

    if "detected_id" not in det.columns:
        det = det.reset_index(drop=True)
        det["detected_id"] = det.index

    det["detected_start"] = pd.to_datetime(det["detected_start"])
    det["detected_end"] = pd.to_datetime(det["detected_end"])

    det_small = det[[
        "detected_id",
        "detected_start",
        "detected_end",
    ]].copy()

    gt = gt.merge(
        det_small,
        left_on="matched_detected_id",
        right_on="detected_id",
        how="left",
    )

    matched_mask = gt["matched"].astype(int) == 1

    gt.loc[matched_mask, "det_offset_start"] = (
        gt.loc[matched_mask, "detected_start"]
        - gt.loc[matched_mask, "gt_start"]
    ).dt.days

    gt.loc[matched_mask, "det_offset_end"] = (
        gt.loc[matched_mask, "detected_end"]
        - gt.loc[matched_mask, "gt_end"]
    ).dt.days

    return gt

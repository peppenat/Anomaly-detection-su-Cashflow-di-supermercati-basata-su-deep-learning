# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CATEGORICAL ENCODING
# =========================================================
def encode_categorical(df, features):
    """
    Codifica tutte le feature categoriche (sia sequenziali che finali)
    in interi usando pandas.factorize().

    Restituisce:
    - df trasformato
    - mappings: dizionario con mapping originale (utile per interpretabilità)
    """

    df = df.copy()
    mappings = {}

    # Unione delle colonne categoriche (senza duplicati)
    all_cat_cols = list(dict.fromkeys(
        features["cat"] + features["seq_cat"]
    ))

    for col in all_cat_cols:
        df[col], mapping = pd.factorize(df[col])
        mappings[col] = mapping

    return df, mappings


def create_sequences(df, features, window):
    """
    Trasforma un dataframe temporale in sequenze per LSTM.

    Per ogni istante t:
    - prende finestra [t-window, ..., t-1]
    - predice il valore a t
    """

    seq_num_features      = features["seq_num"]
    seq_bool_features     = features["seq_bool"]
    seq_cat_features      = features["seq_cat"]
    final_num_features    = features["final_num"]
    final_bool_features   = features.get("final_bool", [])
    cat_features          = features["cat"]
    ground_truth_features = features["ground_truth"]
    target                = features["target"]

    X_seq_num, X_seq_bool, X_seq_cat = [], [], []
    X_final_num, X_final_bool, X_cat = [], [], []
    y, dates = [], []
    ground_truth = []

    seq_num_vals      = df[seq_num_features].values
    seq_bool_vals     = df[seq_bool_features].values
    seq_cat_vals      = df[seq_cat_features].values
    final_num_vals    = df[final_num_features].values
    final_bool_vals   = df[final_bool_features].values
    cat_vals          = df[cat_features].values
    ground_truth_vals = df[ground_truth_features].values
    y_vals            = df[target].values
    d_vals            = df["date"].values

    for i in range(len(df) - window):
        X_seq_num.append(seq_num_vals[i:i + window])
        X_seq_bool.append(seq_bool_vals[i:i + window])
        X_seq_cat.append(seq_cat_vals[i:i + window])

        X_final_num.append(final_num_vals[i + window])
        X_final_bool.append(final_bool_vals[i + window])
        X_cat.append(cat_vals[i + window])

        y.append(y_vals[i + window])
        dates.append(d_vals[i + window])
        ground_truth.append(ground_truth_vals[i + window])

    return (
        np.array(X_seq_num, dtype=np.float32),
        np.array(X_seq_bool, dtype=np.float32),
        np.array(X_seq_cat, dtype=np.int32),
        np.array(X_final_num, dtype=np.float32),
        np.array(X_final_bool, dtype=np.float32),
        np.array(X_cat, dtype=np.int32),
        np.array(y, dtype=np.float32),
        np.array(dates),
        np.array(ground_truth, dtype=object),
    )


def create_ae_windows(df, features, window_size):
    """
    Crea finestre per Autoencoder.

    Input:
    - X_seq_num: numeriche sequenziali
    - X_seq_bool: booleane sequenziali
    - X_seq_cat: categoriche sequenziali

    Target:
    - y: sequenza di daily_total_sales da ricostruire
    """

    seq_num_features = features["seq_num"]
    seq_bool_features = features["seq_bool"]
    seq_cat_features = features["seq_cat"]
    target = features["target"]
    ground_truth_features = features["ground_truth"]

    X_seq_num = []
    X_seq_bool = []
    X_seq_cat = []
    y = []
    dates = []
    store_ids = []
    ground_truth = []

    seq_num_vals = df[seq_num_features].values
    seq_bool_vals = df[seq_bool_features].values
    seq_cat_vals = df[seq_cat_features].values
    target_vals = df[target].values
    date_vals = df["date"].values
    store_vals = df["store_id"].values
    gt_vals = df[ground_truth_features].values

    for i in range(len(df) - window_size + 1):

        end = i + window_size

        X_seq_num.append(seq_num_vals[i:end])
        X_seq_bool.append(seq_bool_vals[i:end])
        X_seq_cat.append(seq_cat_vals[i:end])

        # target = sales della stessa finestra
        y.append(target_vals[i:end].reshape(-1, 1))

        # data centrale per plotting/valutazione
        dates.append(date_vals[i + window_size // 2])

        # store id della finestra
        store_ids.append(store_vals[i + window_size // 2])

        # ground truth della finestra
        ground_truth.append(gt_vals[i:end])

    return {
        "X_seq_num": np.array(X_seq_num, dtype=np.float32),
        "X_seq_bool": np.array(X_seq_bool, dtype=np.float32),
        "X_seq_cat": np.array(X_seq_cat, dtype=np.int32),
        "y": np.array(y, dtype=np.float32),
        "date": np.array(dates),
        "store_id": np.array(store_ids),
        "ground_truth": np.array(ground_truth, dtype=object)
    }

def append_sequence_parts(container, seq):
    """
    Appende le sequenze generate in un container (train/val/test).
    """

    (
        X_seq_num,
        X_seq_bool,
        X_seq_cat,
        X_final_num,
        X_final_bool,
        X_cat,
        y,
        dates,
        ground_truth,
    ) = seq

    container["X_seq_num"].append(X_seq_num)
    container["X_seq_bool"].append(X_seq_bool)
    container["X_seq_cat"].append(X_seq_cat)
    container["X_final_num"].append(X_final_num)
    container["X_final_bool"].append(X_final_bool)
    container["X_cat"].append(X_cat)
    container["y"].append(y)
    container["date"].append(dates)
    container["ground_truth"].append(ground_truth)


def concatenate_parts(parts):
    """
    Concatena tutte le sequenze raccolte nei container
    """

    return {
        key: np.concatenate(values, axis=0)
        for key, values in parts.items()
    }


# =========================================================
# INPUT SPLITTING (PER MODELLO KERAS)
# =========================================================

def split_seq_categorical(X_seq_cat):
    """
    Divide le feature categoriche sequenziali
    in liste separate, rispettando l'ordine del modello:
    - week_day
    - month
    - day
    """

    return [
        X_seq_cat[:, :, 0],  # week_day
        X_seq_cat[:, :, 1],  # month
        X_seq_cat[:, :, 2],  # day
    ]


def split_categorical(X_cat):
    """
    Divide le feature categoriche finali
    in liste separate, rispettando l'ordine del modello:
    - store_id
    - week_day
    - month
    - day
    """

    return [
        X_cat[:, 0],  # store_id
        X_cat[:, 1],  # week_day
        X_cat[:, 2],  # month
        X_cat[:, 3],  # day
    ]


def build_model_inputs(split):
    """
    Costruisce la lista di input per il modello Sales LSTM aggiornato.

    Le variabili booleane finali sono passate tramite X_final_bool,
    non come categoriche con embedding.
    """

    X_seq_num    = split["X_seq_num"]
    X_seq_bool   = split["X_seq_bool"]
    X_seq_cat    = split["X_seq_cat"]
    X_final_num  = split["X_final_num"]
    X_final_bool = split["X_final_bool"]
    X_cat        = split["X_cat"]

    return (
        [X_seq_num] +
        [X_seq_bool] +
        split_seq_categorical(X_seq_cat) +
        [X_final_num] +
        [X_final_bool] +
        split_categorical(X_cat)
    )

def build_pos_model_inputs(split):
    X_seq_num    = split["X_seq_num"]
    X_seq_bool   = split["X_seq_bool"]
    X_seq_cat    = split["X_seq_cat"]
    X_final_num  = split["X_final_num"]
    X_final_bool = split["X_final_bool"]
    X_cat        = split["X_cat"]

    return [
        X_seq_num,
        X_seq_bool,
        X_seq_cat[:, :, 0],   # week_day sequenziale
        X_seq_cat[:, :, 1],   # month sequenziale
        X_final_num,
        X_final_bool,
        X_cat[:, 0],          # store_id finale
        X_cat[:, 1],          # week_day finale
        X_cat[:, 2],          # month finale
    ]

def build_mlp_electricity_inputs(split):
    
    X_num = split["X_num"]
    X_cat = split["X_cat"]

    return [
        X_num,
        X_cat[:, 0],  # store_id
        X_cat[:, 1],  # pre_holiday
        X_cat[:, 2]   # weekend
    ]

def build_mlp_logistics_inputs(data):
    X_num = data["X_num"]
    X_cat = data["X_cat"]

    return [
        X_num,
        X_cat[:, 0],  # store_id
        X_cat[:, 1],  # week_day
        X_cat[:, 2],  # month
        X_cat[:, 3],  # pre_holiday
        X_cat[:, 4],  # actual_holiday
        X_cat[:, 5],  # weekend
    ]

def build_mlp_waste_inputs(data):

    X_num = data["X_num"]
    X_cat = data["X_cat"]

    return [
        X_num,
        X_cat[:, 0],  # store_id
        X_cat[:, 1],  # week_day
        X_cat[:, 2],  # month
        X_cat[:, 3],  # pre_holiday
        X_cat[:, 4],  # actual_holiday
        X_cat[:, 5],  # weekend
    ]

def build_sales_ae_inputs(data):
    """
    Costruisce la lista input per il modello AE.
    Ordine:
    - numeriche sequenziali
    - booleane sequenziali
    - categoriche sequenziali separate
    """

    X_seq_num = data["X_seq_num"]
    X_seq_bool = data["X_seq_bool"]
    X_seq_cat = data["X_seq_cat"]

    return [
        X_seq_num,
        X_seq_bool,
        X_seq_cat[:, :, 0],  # week_day
        X_seq_cat[:, :, 1],  # month
        X_seq_cat[:, :, 2],  # day
        X_seq_cat[:, :, 3],  # store_id
    ]

def make_results_df(split, y_pred, features, feature_scalers=None):

    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(split["y"]).reshape(-1)

    results = pd.DataFrame({
        "date": pd.to_datetime(split["date"]),
        "store_id": split["X_cat"][:, 0].astype(int),
        "y_true": y_true,
        "y_pred": y_pred
    })

    # errori su scala modello
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
                results.loc[idx, "y_true"] * target_std
                + target_mean
            )

            y_pred_unscaled = (
                results.loc[idx, "y_pred"] * target_std
                + target_mean
            )

            if target_is_log:
                results.loc[idx, "y_true_original"] = np.expm1(y_true_unscaled)
                results.loc[idx, "y_pred_original"] = np.expm1(y_pred_unscaled)
            else:
                results.loc[idx, "y_true_original"] = y_true_unscaled
                results.loc[idx, "y_pred_original"] = y_pred_unscaled

        # errori su scala originale
        results["residual_original"] = (
            results["y_true_original"] - results["y_pred_original"]
        )
        results["abs_error_original"] = results["residual_original"].abs()
        results["squared_error_original"] = results["residual_original"] ** 2

    # ground truth alla fine
    if "ground_truth" in split and len(split["ground_truth"]) > 0:

        gt_cols = features.get("ground_truth", [])

        gt = pd.DataFrame(
            split["ground_truth"],
            columns=gt_cols
        )

        results = pd.concat(
            [
                results.reset_index(drop=True),
                gt.reset_index(drop=True)
            ],
            axis=1
        )

    return results



def log_transform(df, cols):
    df = df.copy()
    
    for col in cols:
        df[col] = np.log1p(df[col])
        
    return df


def plot_pred_vs_true_splits(results_dict, sample_size=None):
    """
    Plot y_pred vs y_true per Train / Validation / Test.
    Le anomalie ground truth sono evidenziate in rosso.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

    for ax, (split_name, res) in zip(axes, results_dict.items()):

        plot_df = res.copy()

        if sample_size is not None and len(plot_df) > sample_size:
            plot_df = plot_df.sample(sample_size, random_state=42)

        normal_df = plot_df[plot_df["is_point_anomaly"] == 0]
        anomaly_df = plot_df[plot_df["is_point_anomaly"] == 1]

        ax.scatter(
            normal_df["y_true"],
            normal_df["y_pred"],
            alpha=0.25,
            s=3,
            label="Normal"
        )

        ax.scatter(
            anomaly_df["y_true"],
            anomaly_df["y_pred"],
            color="red",
            alpha=0.8,
            s=12,
            label="Ground truth anomaly"
        )

        mn = min(plot_df["y_true"].min(), plot_df["y_pred"].min())
        mx = max(plot_df["y_true"].max(), plot_df["y_pred"].max())

        ax.plot([mn, mx], [mn, mx], "--")

        ax.set_title(split_name)
        ax.set_xlabel("Valore reale")
        ax.set_ylabel("Valore predetto")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Predetto vs Reale - Train / Validation / Test")
    plt.tight_layout()
    plt.show()
    

def build_dataset_inference(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=7,
    train_size=0.70,
    val_size=0.10
):
    """
    Costruisce train/val/test per inference usando:
    - stesse proporzioni di sales_build_lstm
    - scaler già salvati
    - mappings già salvati

    """

    # =========================
    # PREPROCESSING BASE
    # =========================
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # =========================
    # PRE-HOLIDAY
    # =========================
    df["pre_holiday"] = (
        df.groupby("store_id")["actual_holiday"]
          .shift(-1)
          .fillna(0)
          .astype(int)
    )

    # =========================
    # ORDINAMENTO + TIME INDEX
    # =========================
    df = df.sort_values(["store_id", "date"]).copy()
    df["time_idx"] = df.groupby("store_id").cumcount()

    # =========================
    # FEATURE DERIVATE
    # =========================
    df["days_to_month_end"] = (
        df["date"].dt.days_in_month - df["date"].dt.day
    )

    df["sales_rm_30"] = (
        df.groupby("store_id")["daily_total_sales"]
          .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    ).fillna(0)

    df["sales_rm_30"] = (
        df.groupby("store_id")["sales_rm_30"]
          .bfill()
    )

    # =========================
    # LOG TRANSFORM
    # deve essere identica al training
    # =========================
    log_cols = features.get("log_transform", [])
    
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # =========================
    # ENCODING CATEGORICHE
    # =========================
    for col, mapping in mappings.items():
        df[col] = pd.Categorical(
            df[col],
            categories=mapping
        ).codes

    # =========================
    # CONTAINER
    # =========================
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
    
    test_parts = {
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

    # =========================
    # LOOP PER STORE
    # =========================
    for store_id in df["store_id"].unique():

        temp = (
            df[df["store_id"] == store_id]
            .sort_values("date")
            .copy()
        )

        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df = temp.iloc[train_end:val_end].copy()
        test_df = temp.iloc[val_end:].copy()

        # =========================
        # SCALING FEATURE NUMERICHE
        # =========================
        # Usa lo scaler salvato dal training clean.
        # Non viene fatto fit.
        num_to_scale = features["scale"]

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

        # =========================
        # CREAZIONE SEQUENZE
        # =========================
        train_seq = create_sequences(train_df, features, window_size)
        val_seq = create_sequences(val_df, features, window_size)
        test_seq = create_sequences(test_df, features, window_size)

        # =========================
        # APPEND
        # =========================
        append_sequence_parts(train_parts, train_seq)
        append_sequence_parts(val_parts, val_seq)
        append_sequence_parts(test_parts, test_seq)

    # =========================
    # CONCATENAZIONE
    # =========================
    train = concatenate_parts(train_parts)
    val = concatenate_parts(val_parts)
    test = concatenate_parts(test_parts)

    return train, val, test


def build_sales_ae_dataset_inference(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=14,
    train_size=0.70,
    val_size=0.10
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["store_id", "date"]).copy()
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
            categories=mapping
        ).codes

    train_parts = []
    val_parts = []
    test_parts = []

    for store_id in df["store_id"].unique():

        temp = (
            df[df["store_id"] == store_id]
            .sort_values("date")
            .copy()
        )

        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

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
            create_ae_windows(train_df, features, window_size)
        )

        val_parts.append(
            create_ae_windows(val_df, features, window_size)
        )

        test_parts.append(
            create_ae_windows(test_df, features, window_size)
        )

    def concat_ae_parts(parts):
        return {
            key: np.concatenate(
                [p[key] for p in parts],
                axis=0
            )
            for key in parts[0].keys()
        }

    train = concat_ae_parts(train_parts)
    val = concat_ae_parts(val_parts)
    test = concat_ae_parts(test_parts)

    return train, val, test

def make_ae_base_results_df(data, y_pred):
    """
    Results dataframe generico per Autoencoder.
    Una riga = una finestra ricostruita.
    """

    y_true = data["y"]
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred

    window_size = y_true.shape[1]
    center_pos = window_size // 2

    results = pd.DataFrame({
        "store_id": data["store_id"],
        "center_date": pd.to_datetime(data["date"]),

        "window_start": (
            pd.to_datetime(data["date"])
            - pd.to_timedelta(center_pos, unit="D")
        ),

        "window_end": (
            pd.to_datetime(data["date"])
            + pd.to_timedelta(window_size - center_pos - 1, unit="D")
        ),

        "ae_mse_score": np.mean(errors ** 2, axis=(1, 2)),
        "ae_mae_score": np.mean(np.abs(errors), axis=(1, 2)),
        "ae_max_abs_error": np.max(np.abs(errors), axis=(1, 2)),
        "ae_signed_mean_error": np.mean(errors, axis=(1, 2)),

        "y_true_center": y_true[:, center_pos, 0],
        "y_pred_center": y_pred[:, center_pos, 0],
        "center_error": errors[:, center_pos, 0],
        "center_abs_error": np.abs(errors[:, center_pos, 0])
    })

    return results

def build_dataset_pos_inference(
    df,
    feature_scalers,
    mappings,
    features,
    window_size=7,
    train_size=0.70,
    val_size=0.10
):
    """
    Costruisce train/val/test per inference del modello POS.

    Replica il preprocessing usato in POS_build_lstm:
    - pre_holiday
    - pos_card_sales_rm_30
    - log_transform su pos_card_sales, pos_net_cf, pos_card_sales_rm_30
    - encoding categoriche con mappings salvati
    - scaling con feature_scalers salvati

    Non fa fit di nulla.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # =========================
    # ORDINAMENTO + TIME INDEX
    # =========================
    df = df.sort_values(["store_id", "date"]).copy()
    df["time_idx"] = df.groupby("store_id").cumcount()

    # =========================
    # PRE-HOLIDAY
    # =========================
    df["pre_holiday"] = (
        df.groupby("store_id")["actual_holiday"]
          .shift(-1)
          .fillna(0)
          .astype(int)
    )

    # =========================
    # ROLLING MEAN POS
    # =========================
    df["pos_card_sales_rm_30"] = (
        df.groupby("store_id")["pos_card_sales"]
          .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    ).fillna(0)

    df["pos_card_sales_rm_30"] = (
        df.groupby("store_id")["pos_card_sales_rm_30"]
          .bfill()
    )

    # =========================
    # LOG TRANSFORM
    # =========================
    df = log_transform(
        df,
        [
            "pos_card_sales",
            "pos_net_cf",
            "pos_card_sales_rm_30"
        ]
    )

    # =========================
    # ENCODING CATEGORICHE
    # =========================
    for col, mapping in mappings.items():
        df[col] = pd.Categorical(
            df[col],
            categories=mapping
        ).codes

    # =========================
    # CONTAINER
    # =========================
    train_parts = {
        "X_seq_num": [], "X_seq_bool": [], "X_seq_cat": [],
        "X_final_num": [], "X_final_bool": [], "X_cat": [],
        "y": [], "date": [], "ground_truth": []
    }

    val_parts = {
        "X_seq_num": [], "X_seq_bool": [], "X_seq_cat": [],
        "X_final_num": [], "X_final_bool": [], "X_cat": [],
        "y": [], "date": [], "ground_truth": []
    }

    test_parts = {
        "X_seq_num": [], "X_seq_bool": [], "X_seq_cat": [],
        "X_final_num": [], "X_final_bool": [], "X_cat": [],
        "y": [], "date": [], "ground_truth": []
    }

    # =========================
    # COLONNE DA SCALARE
    # =========================
    num_to_scale = list(dict.fromkeys(
        features["seq_num"] +
        features["final_num"] +
        [features["target"]]
    ))

    # =========================
    # LOOP PER STORE
    # =========================
    for store_id in df["store_id"].unique():

        temp = df[df["store_id"] == store_id].sort_values("date").copy()
        n = len(temp)

        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)

        train_df = temp.iloc[:train_end].copy()
        val_df   = temp.iloc[train_end:val_end].copy()
        test_df  = temp.iloc[val_end:].copy()

        # =========================
        # SCALING CON SCALER SALVATO
        # =========================
        scaler = feature_scalers[store_id]

        train_df[num_to_scale] = train_df[num_to_scale].astype(float)
        val_df[num_to_scale]   = val_df[num_to_scale].astype(float)
        test_df[num_to_scale]  = test_df[num_to_scale].astype(float)

        train_df[num_to_scale] = scaler.transform(train_df[num_to_scale])
        val_df[num_to_scale]   = scaler.transform(val_df[num_to_scale])
        test_df[num_to_scale]  = scaler.transform(test_df[num_to_scale])

        # =========================
        # CREAZIONE SEQUENZE
        # =========================
        train_seq = create_sequences(train_df, features, window_size)
        val_seq   = create_sequences(val_df, features, window_size)
        test_seq  = create_sequences(test_df, features, window_size)

        append_sequence_parts(train_parts, train_seq)
        append_sequence_parts(val_parts, val_seq)
        append_sequence_parts(test_parts, test_seq)

    # =========================
    # CONCATENAZIONE
    # =========================
    train = concatenate_parts(train_parts)
    val   = concatenate_parts(val_parts)
    test  = concatenate_parts(test_parts)

    return train, val, test

def build_weekday_contextual_detected_windows_from_center_points(
    df,
    detected_col="is_weekday_contextual_detected_raw",
    date_col="center_date",
    store_col="store_id",
    window_size=7,
    min_consecutive=2
):
    """
    Costruisce finestre finali partendo dai punti-centro rilevati.

    Pipeline:
    - punti oltre threshold
    - filtro consecutive points
    - espansione a finestre centrate
    - merge delle finestre sovrapposte
    """

    out = []

    half_left = window_size // 2
    half_right = window_size - half_left - 1

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    for store_id, g in temp.groupby(store_col):

        g = g.sort_values(date_col).reset_index(drop=True)

        detected = g[g[detected_col] == 1].copy()

        if detected.empty:
            continue

        # =========================
        # FILTRO CONSECUTIVI
        # =========================
        detected["block"] = (
            detected[date_col]
            .diff()
            .dt.days
            .ne(1)
            .cumsum()
        )

        valid_detected = []

        for _, block in detected.groupby("block"):

            if len(block) >= min_consecutive:
                valid_detected.append(block)

        if len(valid_detected) == 0:
            continue

        detected = pd.concat(valid_detected)

        # =========================
        # ESPANSIONE CENTRI
        # =========================
        intervals = []

        for _, row in detected.iterrows():

            center = row[date_col]

            start = center - pd.Timedelta(days=half_left)
            end = center + pd.Timedelta(days=half_right)

            intervals.append((start, end))

        intervals = sorted(intervals)

        # =========================
        # MERGE INTERVALLI
        # =========================
        merged = []

        cur_start, cur_end = intervals[0]

        for start, end in intervals[1:]:

            if start <= cur_end + pd.Timedelta(days=1):

                cur_end = max(cur_end, end)

            else:

                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end

        merged.append((cur_start, cur_end))

        # =========================
        # OUTPUT
        # =========================
        for window_id, (start, end) in enumerate(merged):

            out.append({
                "store_id": store_id,
                "detected_window_id": window_id,
                "detected_start": start,
                "detected_end": end,
                "detected_duration": (end - start).days + 1
            })

    return pd.DataFrame(out)

def interval_iou(start_a, end_a, start_b, end_b):
    """
    IoU temporale tra due intervalli chiusi [start, end].
    """

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)

    inter = max((inter_end - inter_start).days + 1, 0)

    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)

    union = (union_end - union_start).days + 1

    return inter / union if union > 0 else 0.0

def evaluate_detected_windows_event_level(
    gt_windows,
    detected_windows,
    store_col="store_id",
    iou_threshold=0.10
):
    """
    Valutazione event-level:
    - un evento GT è rilevato se almeno una finestra detected
      dello stesso store ha IoU >= iou_threshold.
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
            "mean_detection_delay": np.nan
        }

    if det.empty:
        gt_eval = gt.copy()
        gt_eval["matched"] = 0
        gt_eval["best_iou"] = 0.0
        gt_eval["matched_detected_id"] = -1
        gt_eval["detection_delay_days"] = np.nan

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
            "mean_detection_delay": np.nan
        }

    gt["gt_start"] = pd.to_datetime(gt["gt_start"])
    gt["gt_end"] = pd.to_datetime(gt["gt_end"])

    det["detected_start"] = pd.to_datetime(det["detected_start"])
    det["detected_end"] = pd.to_datetime(det["detected_end"])

    det = det.reset_index(drop=True)
    det["detected_id"] = det.index
    det["matched"] = 0
    det["matched_gt_event_id"] = -1
    det["best_iou"] = 0.0

    gt_rows = []

    for _, gt_row in gt.iterrows():

        same_store_det = det[
            det[store_col] == gt_row[store_col]
        ]

        best_iou = 0.0
        best_det_id = -1
        best_delay = np.nan

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
                best_delay = (
                    det_row["detected_start"] - gt_row["gt_start"]
                ).days

        matched = int(best_iou >= iou_threshold)

        if matched:
            det.loc[
                det["detected_id"] == best_det_id,
                "matched"
            ] = 1

            det.loc[
                det["detected_id"] == best_det_id,
                "matched_gt_event_id"
            ] = gt_row["gt_event_id"]

            det.loc[
                det["detected_id"] == best_det_id,
                "best_iou"
            ] = best_iou

        row = gt_row.to_dict()
        row["matched"] = matched
        row["best_iou"] = best_iou
        row["matched_detected_id"] = best_det_id
        row["detection_delay_days"] = best_delay if matched else np.nan

        gt_rows.append(row)

    gt_eval = pd.DataFrame(gt_rows)
    det_eval = det.copy()

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
        "mean_iou": gt_eval.loc[
            gt_eval["matched"] == 1,
            "best_iou"
        ].mean(),
        "mean_detection_delay": gt_eval.loc[
            gt_eval["matched"] == 1,
            "detection_delay_days"
        ].mean()
    }

    return gt_eval, det_eval, summary

def summarize_gt_eval_by_type(
    gt_eval,
    type_col="gt_type"
):
    df = gt_eval.copy()

    summary = (
        df
        .groupby(type_col, dropna=False)
        .agg(
            n_events=("matched", "count"),
            detected_events=("matched", "sum"),
            recall=("matched", "mean"),
            mean_iou=(
                "best_iou",
                lambda x: x[df.loc[x.index, "matched"] == 1].mean()
            ),
            mean_detection_delay=(
                "detection_delay_days",
                lambda x: x[df.loc[x.index, "matched"] == 1].mean()
            )
        )
        .reset_index()
    )

    summary["missed_events"] = (
        summary["n_events"] - summary["detected_events"]
    )

    return summary

# %% AE windows

def create_pos_ae_windows(df, features, window_size):

    X_seq_num = []
    X_seq_bool = []
    X_seq_cat = []
    y = []
    dates = []
    store_ids = []
    ground_truth = []

    seq_num_vals = df[features["seq_num"]].values
    seq_bool_vals = df[features["seq_bool"]].values
    seq_cat_vals = df[features["seq_cat"]].values
    target_vals = df[[features["target"]]].values
    gt_vals = df[features["ground_truth"]].values
    date_vals = df["date"].values
    store_vals = df["store_id"].values

    center_pos = window_size // 2

    for i in range(len(df) - window_size + 1):

        start = i
        end = i + window_size
        center = i + center_pos

        X_seq_num.append(seq_num_vals[start:end])
        X_seq_bool.append(seq_bool_vals[start:end])
        X_seq_cat.append(seq_cat_vals[start:end])

        y.append(target_vals[start:end])
        ground_truth.append(gt_vals[start:end])

        dates.append(date_vals[center])
        store_ids.append(store_vals[center])

    return {
        "X_seq_num": np.array(X_seq_num, dtype=np.float32),
        "X_seq_bool": np.array(X_seq_bool, dtype=np.float32),
        "X_seq_cat": np.array(X_seq_cat, dtype=np.int32),
        "y": np.array(y, dtype=np.float32),
        "date": np.array(dates),
        "store_id": np.array(store_ids, dtype=np.int32),
        "ground_truth": np.array(ground_truth, dtype=object)
    }

# %% Input builder AE POS

def build_pos_ae_inputs(data):

    X_seq_num = data["X_seq_num"]
    X_seq_bool = data["X_seq_bool"]
    X_seq_cat = data["X_seq_cat"]

    return [
        X_seq_num,
        X_seq_bool,
        X_seq_cat[:, :, 0],  # week_day
        X_seq_cat[:, :, 1],  # month
        X_seq_cat[:, :, 2]   # store_id
    ]

# =========================================================
# DETECTOR SENSITIVITY
# =========================================================

def build_one_at_a_time_config_df(base_config, parameter_values):
    """
    Costruisce configurazioni one-at-a-time.

    Ogni riga modifica un solo parametro rispetto alla configurazione base.
    La configurazione base è inclusa anche come riga separata.
    """

    rows = []

    rows.append({
        "config_id": 0,
        "variation_group": "base_config",
        "varied_parameter": "base_config",
        "varied_value": "reference",
        **base_config,
    })

    config_id = 1

    for parameter, values in parameter_values.items():
        for value in values:
            config = base_config.copy()
            config[parameter] = value

            rows.append({
                "config_id": config_id,
                "variation_group": parameter,
                "varied_parameter": parameter,
                "varied_value": value,
                **config,
            })
            config_id += 1

    return pd.DataFrame(rows)
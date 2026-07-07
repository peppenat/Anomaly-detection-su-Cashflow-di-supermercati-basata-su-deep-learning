from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from workalendar.europe import Italy

from anomalies import (
    add_business_days,
    get_def_anomaly_config,
    inject_level_shift_anomalies,
    inject_pos_delay,
)
from supermarkets import all_params

from project_paths import (
    CLEAN_DATA_PATH,
    CLEAN_DIR,
    EXTERNAL_INDICES_CACHE_PATH,
    LEVEL_SHIFT_SENSITIVITY_DIR,
    LEVEL_SHIFT_CONTAMINATION_DIR,
    POS_DELAY_SENSITIVITY_DIR,
    POS_DELAY_CONTAMINATION_DIR,
    ensure_artifact_directories,
)
from experiment_dataset_generation import (
    generate_level_shift_contamination_experiment_from_clean,
    generate_level_shift_sensitivity_experiment_from_clean,
    generate_pos_delay_contamination_experiment_from_clean,
    generate_pos_delay_sensitivity_experiment_from_clean,
)


def get_external_indices(
    cache_path=EXTERNAL_INDICES_CACHE_PATH,
    force_download=False,
):
    cache_path = Path(cache_path)

    # ==================================
    # LOAD CACHE (evita download multipli)
    # ==================================
    if cache_path.exists() and not force_download:
        print(f">>> Carico indici esterni da cache: {cache_path}")
        
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        
        return df

    print(">>> Scarico indici esterni da FRED / yfinance...")

    API_KEY = '3d0439b03f06c9faf751f4be1697f26a'
    fred = Fred(api_key=API_KEY) 

    # ==================================
    # Storico della serie
    # ==================================

    # Periodo storico della serie.
    start_date = '2018-01-01'
    end_date = '2024-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Calendario italiano per le festività
    cal = Italy()
    is_holiday = [1 if cal.is_working_day(d) is False else 0 for d in dates]

    # Costruiamo il dataframe base
    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'week_day': dates.dayofweek,  # 0=lunedì, 6=domenica
        'holiday': is_holiday,
        'actual_holiday': [1 if cal.is_holiday(d) else 0 for d in dates]
    })

    # =========================
    # PRE-HOLIDAY
    # =========================
    # 1 se il giorno successivo è una festività
    df["pre_holiday"] = (
        df["actual_holiday"]
        .shift(-1)          # guarda il giorno dopo
        .fillna(0)          # ultimo giorno → 0
        .astype(int)
    )

    # Aggiungiamo una colonna 'weekend' (1 se sabato o domenica)
    df['weekend'] = (df['week_day'] >= 5).astype(int)

    # ==================================================================
    # Variabili macroeconomiche: Petrolio, EURIBOR, 
    # ==================================================================

    # ------------------------------------------------------------------
    # Variabile - Prezzo carburanti (Brent)
    # ------------------------------------------------------------------

    brent = yf.download(
        'BZ=F',
        start=start_date,
        end=end_date,
        progress=False
    )

    # La colonna 'Close' rappresenta il prezzo del Brent
    brent_prices = brent['Close']

    # Allinea al calendario giornaliero (riempi weekend con ultimo valore disponibile)
    brent_aligned = brent_prices.reindex(dates, method='ffill').bfill()

    # Aggiungi al dataframe principale
    df['oil_price'] = brent_aligned.values

    # ------------------------------------------------------------------
    # Variabile - Tasso Euribor 3 mesi (FRED)
    # ------------------------------------------------------------------
    # Influenza il costo del credito e quindi la capacità di spesa
    euribor_series = fred.get_series(
        'IR3TIB01EZM156N',
        observation_start=start_date,
        observation_end=end_date
    )
    euribor_daily = euribor_series.reindex(dates)
    euribor_smooth = euribor_daily.interpolate(method='linear').bfill().ffill()
    df['euribor'] = euribor_smooth.values

    # ------------------------------------------------------------------
    # Variabile - Consumer Confidence (FRED)
    # ------------------------------------------------------------------
    # Misura l’ottimismo delle famiglie → impatta la domanda
    confidence = fred.get_series(
        'CSCICP03ITM665S',
        observation_start=start_date,
        observation_end=end_date
    )
    confidence_daily = confidence.reindex(dates)
    confidence_smooth = confidence_daily.interpolate(method='linear').bfill().ffill()
    df['consumer_confidence'] = confidence_smooth.values.flatten()

    # ------------------------------------------------------------------
    # Variabile - Inflazione IPCA Italia (FRED)
    # ------------------------------------------------------------------
    inflation = fred.get_series(
        'ITACPIALLMINMEI',
        observation_start=start_date,
        observation_end=end_date
    )
    inflation_daily = inflation.reindex(dates)
    inflation_smooth = inflation_daily.interpolate(method='linear').bfill().ffill()
    df['inflation_index'] = inflation_smooth.values.flatten()

    # ------------------------------------------------------------------
    # Variabile - Prezzi energia (HICP)
    # ------------------------------------------------------------------
    consumer_prices = fred.get_series(
        'CP0450ITM086NEST',
        observation_start=start_date,
        observation_end=end_date
    )
    consumer_prices = consumer_prices.reindex(dates)
    consumer_prices_smooth = consumer_prices.interpolate(method='linear').bfill().ffill()
    df['consumer_prices'] = consumer_prices_smooth.values.flatten()

    # ------------------------------------------------------------------
    # Variabile - Food Price Index (FAO)
    # ------------------------------------------------------------------
    fao = fred.get_series(
        'PFOODINDEXM',
        observation_start=start_date,
        observation_end=end_date
    )
    fao = fao.reindex(dates)
    fao_smooth = fao.interpolate(method='linear').bfill().ffill()
    df['fao'] = fao_smooth.values.flatten()

    # ------------------------------------------------------------------
    # Variabile - Pandemic Uncertainty Index
    # ------------------------------------------------------------------
    wupi = fred.get_series(
        'WUPI',
        observation_start=start_date,
        observation_end=end_date
    )
    wupi = wupi.reindex(dates)
    wupi_smooth = wupi.interpolate(method='linear').bfill().ffill()
    df['pandemic_uncertainty'] = wupi_smooth.values.flatten()

    # ==================================
    # SAVE CACHE
    # ==================================
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    print(f">>> Indici esterni salvati in cache: {cache_path}")

    return df

def add_daily_temperature_feature(df, global_seed):
    """
    Genera temperatura giornaliera sintetica:
    - temperature

    Usa:
    - componente stagionale annuale
    - rumore autocorrelato AR(1)
    """

    rng = np.random.default_rng(global_seed + 20_036)


    # =========================
    # DAILY TEMPERATURE
    # =========================
    day_of_year = df["date"].dt.dayofyear

    seasonal_temp = (
        15
        + 10 * np.sin(
            2 * np.pi * (day_of_year - 170) / 365
        )
    )

    temp_noise = np.zeros(len(df))

    for t in range(1, len(df)):
        temp_noise[t] = (
            0.92 * temp_noise[t - 1]
            + rng.normal(0, 1.5)
        )

    df["temperature"] = np.clip(
        seasonal_temp + temp_noise,
        -5,
        42
    )

    return df

def add_sales_rolling_features(df):
    """
    Genera feature rolling legate alle vendite.

    Assume:
    - df contiene un singolo store
    - righe già ordinate per date

    Feature create:
    - sales_rm_7
    - sales_rm_30
    - sales_std_30
    - sales_pressure
    """

    # =========================
    # ROLLING MEAN 7
    # =========================
    df["sales_rm_7"] = (
        df["daily_total_sales"]
        .shift(1)  # NO leakage
        .rolling(7, min_periods=1)
        .mean()
        .bfill()
    )

    # =========================
    # ROLLING MEAN 30
    # =========================
    df["sales_rm_30"] = (
        df["daily_total_sales"]
        .shift(1)  # NO leakage
        .rolling(30, min_periods=7)
        .mean()
        .bfill()
    )

    # =========================
    # ROLLING STD 30
    # =========================
    df["sales_std_30"] = (
        df["daily_total_sales"]
        .shift(1)  # NO leakage
        .rolling(30, min_periods=7)
        .std()
        .bfill()
        .fillna(0)
    )

    # =========================
    # SALES PRESSURE
    # =========================
    # Misura il livello di vendite
    # rispetto al comportamento recente
    df["sales_pressure"] = (
        df["daily_total_sales"]
        / df["sales_rm_30"]
    )

    # =========================
    # SAFETY FILL
    # =========================
    df["sales_pressure"] = (
        df["sales_pressure"]
        .replace([np.inf, -np.inf], 1)
        .fillna(1)
    )

    return df


def add_daily_electricity_features(df, params, global_seed):
    """
    Genera componenti giornaliere di consumo elettrico:
    - refrigeration_load
    - hvac_load
    - lighting_load
    - operational_load
    - daily_electricity_consumption
    - daily_electricity_cost

    Nota:
    i valori sono sintetici e interpretabili come kWh/giorno e €/giorno.
    """
    
    rng = np.random.default_rng(global_seed + 20_010)
    

    # =========================
    # PARAMETRI BASE STORE
    # =========================
    base_monthly_kwh_need = params.get("base_monthly_kwh_need", 0)
    
    # Prezzo unitario sintetico: scala il consumo in costo.
    electricity_unit_cost_2014 = 0.20  # €/kWh indicativo sintetico

    avg_daily_kwh_need = base_monthly_kwh_need / 30

    # Ripartizione carichi
    base_shares = np.array([0.45, 0.25, 0.20, 0.10])
    
    # casualità leggera intorno alle quote medie
    shares = rng.dirichlet(base_shares * 120)
    
    ref_share, hvac_share, lighting_share, operational_share = shares
    
    base_refrigeration_load = avg_daily_kwh_need * ref_share
    base_hvac_load          = avg_daily_kwh_need * hvac_share
    base_lighting_load      = avg_daily_kwh_need * lighting_share
    base_operational_load   = avg_daily_kwh_need * operational_share


    # =========================
    # PREZZO ENERGIA
    # =========================
    df["energy_price_index_norm"] = (
        df["consumer_prices"] / df["consumer_prices"].iloc[0]
    )

    # =========================
    # TRAFFIC / AFFLUENZA PROXY
    # =========================
    # Usiamo daily_total_sales come proxy dell'affluenza.
    # L'effetto è volutamente moderato:
    # più clienti => più aperture banchi frigo, porte automatiche,
    # calore interno, attività operativa.
    
    sales_activity_ratio = np.clip(
        df["sales_pressure"],
        0.70,
        1.40
    )
    
    traffic_effect_ref = 1 + 0.06 * (sales_activity_ratio - 1)
    traffic_effect_hvac = 1 + 0.08 * (sales_activity_ratio - 1)
    traffic_effect_operational = 1 + 0.04 * (sales_activity_ratio - 1)

    # =========================
    # REFRIGERATION LOAD
    # =========================
    
    # effetto temperatura: sopra i 18°C il carico aumenta
    temp_above_ref = np.maximum(df["temperature"] - 18, 0)
    
    # effetto non lineare leggero:
    # giornate molto calde aumentano il carico più che proporzionalmente
    temperature_effect_ref = 1 + 0.020 * temp_above_ref + 0.0015 * temp_above_ref**2
    
    # weekend: più aperture dei banchi frigo e maggiore attività store
    weekend_effect_ref = 1 + 0.025 * df["weekend"]
    
    # festività/pre-festività: maggiore attività alimentare
    pre_holiday_effect_ref = 1 + 0.03 * df["pre_holiday"]
    
    # rumore giornaliero moderato
    refrigeration_noise = np.clip(
        rng.normal(1, 0.025, len(df)),
        0.90,
        1.10
    )
    
    df["refrigeration_load"] = (
        base_refrigeration_load
        * temperature_effect_ref
        * weekend_effect_ref
        * pre_holiday_effect_ref
        * traffic_effect_ref
        * refrigeration_noise
    )
    

    # =========================
    # HVAC LOAD
    # =========================
    
    # gradi sopra/sotto la zona neutra
    cooling_degree = np.maximum(df["temperature"] - 21, 0)
    heating_degree = np.maximum(14 - df["temperature"], 0)
    
    # effetto non lineare leggero
    cooling_effect = 1 + 0.040 * cooling_degree + 0.0015 * cooling_degree**2
    heating_effect = 1 + 0.035 * heating_degree + 0.0010 * heating_degree**2
    
    # combino caldo e freddo: normalmente uno dei due è 1
    temperature_effect_hvac = cooling_effect * heating_effect
    
    # weekend: più affluenza/aperture porte
    weekend_effect_hvac = 1 + 0.015 * df["weekend"]
    
    # pre-festività: store più attivo
    pre_holiday_effect_hvac = (
        1 + 0.02 * df["pre_holiday"]
    )
    
    # rumore moderato
    hvac_noise = np.clip(
        rng.normal(1, 0.035, len(df)),
        0.85,
        1.15
    )
    
    df["hvac_load"] = (
        base_hvac_load
        * temperature_effect_hvac
        * weekend_effect_hvac
        * pre_holiday_effect_hvac
        * traffic_effect_hvac
        * hvac_noise
    )

    # =========================
    # LIGHTING LOAD
    # =========================
    df["lighting_load"] = (
        base_lighting_load
        * rng.normal(1, 0.01, len(df))
    )
    
    
    # =========================
    # OPERATIONAL LOAD QUASI FISSO
    # =========================
    df["operational_load"] = (
        base_operational_load
        * traffic_effect_operational
        * rng.normal(1, 0.01, len(df))
    )

    # =========================
    # TOTALE CONSUMI E COSTO GIORNALIERO
    # =========================
    df["daily_electricity_consumption"] = (
        df["refrigeration_load"]
        + df["hvac_load"]
        + df["lighting_load"]
        + df["operational_load"]
    )

    df["daily_electricity_cost"] = (
        df["daily_electricity_consumption"]
        * df["energy_price_index_norm"]
        * electricity_unit_cost_2014
    )

    return df

def add_daily_logistics_features(df, params, global_seed):
    """
    Genera componenti giornaliere di costo logistico:
    - daily_logistics_cost

    I valori sono costi positivi giornalieri.
    Nel cashflow andranno poi registrati come uscita negativa.
    """

    rng = np.random.default_rng(global_seed + 30_010)

    base_logistics = params.get("base_logistics", 0)

    # =========================
    # BASE GIORNALIERA
    # =========================
    base_daily_logistics = base_logistics / 30

    sales_pressure = np.clip(
        df["sales_pressure"],
        0.70,
        1.50
    )
    
    volume_effect = np.power(
        sales_pressure,
        0.30
    )

    df["logistics_volume_effect"] = volume_effect

    # =========================
    # FOOD SHARE / COLD CHAIN
    # =========================
    food_share = (
        df["daily_food_sales"]
        / df["daily_total_sales"]
    )
    
    avg_food_share = (
        food_share
        .shift(1)
        .rolling(30, min_periods=7)
        .mean()
        .bfill()
    )
    
    food_pressure = (
        food_share / avg_food_share
    )
    
    food_pressure = np.clip(
        food_pressure,
        0.70,
        1.30
    )
    
    food_effect = np.power(
        food_pressure,
        0.20
    )

    df["logistics_food_effect"] = food_effect

    # =========================
    # TEMPERATURE / COLD CHAIN
    # =========================

    temp_above_ref = np.maximum(
        df["temperature"] - 18,
        0
    )
    
    
    cold_chain_intensity = (
        temp_above_ref
        * food_share
    )
    
    cold_chain_effect = (
        1
        + 0.010 * cold_chain_intensity
        + 0.0003 * cold_chain_intensity**2
    )
    
    cold_chain_effect = np.clip(
        cold_chain_effect,
        1.00,
        1.25
    )

    df["logistics_cold_chain_effect"] = cold_chain_effect

    # =========================
    # FUEL / OIL PRICE
    # =========================
    oil_rm_14 = (
        df["oil_price"]
        .rolling(14, min_periods=1)
        .mean()
    )
    
    oil_norm = (
        oil_rm_14
        / oil_rm_14.median()
    )
    
    fuel_effect = (
        1
        + 0.25 * (oil_norm - 1)
    )
    
    df["logistics_fuel_effect"] = fuel_effect
    
    # =========================
    # CALENDAR EFFECT
    # =========================
    weekday_factor = df["week_day"].map({
        0: 1.18,  # lunedì: replenish dopo weekend
        1: 1.00,
        2: 0.96,
        3: 1.00,
        4: 1.12,  # venerdì: pre-weekend
        5: 1.10,
        6: 0.72   # domenica: poche consegne
    }).astype(float)

    pre_holiday_effect = 1 + 0.22 * df["pre_holiday"]
    actual_holiday_effect = 1 - 0.35 * df["actual_holiday"]

    calendar_effect = (
        weekday_factor
        * pre_holiday_effect
        * actual_holiday_effect
    )

    df["logistics_calendar_effect"] = calendar_effect

    # =========================
    # MONTH SEASONALITY
    # =========================
    month_factor = df["month"].map({
        1: 1.02,
        2: 0.98,
        3: 1.00,
        4: 1.02,
        5: 1.03,
        6: 1.06,
        7: 1.08,
        8: 0.95,
        9: 1.03,
        10: 1.00,
        11: 1.04,
        12: 1.18
    }).astype(float)

    # =========================
    # RUMORE AUTOCORRELATO AR(1)
    # =========================
    logistics_noise = np.zeros(len(df))

    for t in range(1, len(df)):
        logistics_noise[t] = (
            0.85 * logistics_noise[t - 1]
            + rng.normal(0, 0.025)
        )

    noise_effect = np.clip(1 + logistics_noise, 0.88, 1.12)

    df["logistics_noise_effect"] = noise_effect

    # =========================
    # COSTO FINALE
    # =========================
    df["daily_logistics_cost"] = (
        base_daily_logistics
        * volume_effect
        * food_effect
        * cold_chain_effect
        * fuel_effect
        * calendar_effect
        * month_factor
        * noise_effect
    )


    df["daily_logistics_cost"] = df["daily_logistics_cost"].round(2)

    return df

def add_pos_settlement_features(
    df,
    params,
    anomaly_config,
    seed
):
    df = df.copy()
    rng = np.random.default_rng(seed + 10_000)

    pos_commission_rate = params["pos_commission_rate"]

    # 1. POS card share
    base_card_share = rng.uniform(0.55, 0.75)

    weekend_card_effect  = np.where(df["weekend"] == 1, 0.03, 0.00)
    holiday_card_effect  = np.where(df["holiday"] == 1, 0.02, 0.00)
    december_card_effect = np.where(df["month"] == 12, 0.04, 0.00)

    card_share_noise = rng.normal(0, 0.025, len(df))

    df["pos_card_share"] = np.clip(
        base_card_share
        + weekend_card_effect
        + holiday_card_effect
        + december_card_effect
        + card_share_noise,
        0.35,
        0.90
    )

    # 2. Importi POS / cash
    df["pos_card_sales"] = (
        df["daily_total_sales"] * df["pos_card_share"]
    ).round(2)

    df["cash_sales_amount"] = (
        df["daily_total_sales"] * (1 - df["pos_card_share"])
    ).round(2)

    df["pos_net_amount"] = (
        df["pos_card_sales"] * (1 - pos_commission_rate)
    ).round(2)

    # 3. Rolling volume POS no leakage
    df["pos_card_sales_rm_30"] = (
        df["pos_card_sales"]
        .shift(1)
        .rolling(30, min_periods=7)
        .mean()
        .bfill()
    )

    df["pos_volume_ratio"] = (
        df["pos_card_sales"] / df["pos_card_sales_rm_30"]
    ).replace([np.inf, -np.inf], 1).fillna(1)

    # 4. Iniezione anomalie POS delay
    pos_cfg = anomaly_config["pos_delay"]

    df, pos_delay_idx = inject_pos_delay(
        df=df,
        pos_cfg=pos_cfg,
        anomaly_seed=seed + pos_cfg.get("seed_offset", 0)
    )

    # 5. Business days
    business_dates = (
        df.loc[df["holiday"] == 0, "date"]
        .sort_values()
        .reset_index(drop=True)
    )

    end_date = df["date"].max()
    
    # =========================
    # EFFECT DAYS POS DELAY
    # =========================

    def get_future_business_dates(
        date,
        n_business_days,
        business_dates
    ):
        """
        Restituisce i primi n business days successivi a `date`.
    
        T+1 = primo business day strettamente successivo alla data sorgente.
        """
    
        future_dates = (
            business_dates[business_dates > date]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
    
        return list(
            future_dates.iloc[:n_business_days]
        )


    max_effect_business_days = 5
    
    source_days = df[
        df["is_pos_delay_source_day"].astype(int) == 1
    ].copy()
    
    for event_id, g in source_days.groupby("pos_delay_event_id"):
    
        delay_type = g["pos_delay_type"].mode().iloc[0]
    
        effect_dates = []
    
        for _, source_row in g.iterrows():
    
            source_date = pd.to_datetime(source_row["date"])
    
            effect_dates.extend(
                get_future_business_dates(
                    date=source_date,
                    n_business_days=max_effect_business_days,
                    business_dates=business_dates
                )
            )
    
        effect_dates = sorted(set(effect_dates))
    
        effect_duration = len(effect_dates)
    
        for day_in_effect, effect_date in enumerate(effect_dates):
    
            mask = df["date"] == effect_date
    
            df.loc[mask, "is_pos_delay_effect_day"] = 1
            df.loc[mask, "pos_delay_effect_day_in_event"] = day_in_effect
            df.loc[mask, "pos_delay_effect_duration"] = effect_duration
    
            df.loc[mask, "pos_delay_event_id"] = event_id
            df.loc[mask, "pos_delay_type"] = delay_type
    

    lambda_low = 3
    lambda_high = 5

    gamma_low = 3 / 4
    gamma_high = 3 / 2

    transactions = []

    for idx, row in df.iterrows():

        date = row["date"]

        # cash entra subito
        transactions.append({
            "date": date,
            "type": "cash_sales_cf",
            "amount": row["cash_sales_amount"],
            "due_date": date
        })

        # POS settlement normale o anomalo
        settlement_days = [1, 2, 3, 4, 5]
        
        if idx in pos_delay_idx:
        
            delay_type = row["pos_delay_type"]
        
            alpha = np.array(
                pos_cfg["delay_profiles"][delay_type]["alpha"],
                dtype=float
            )
        
        else:
        
            volume_ratio = row["pos_volume_ratio"]
        
            low_weight = np.exp(
                -lambda_low * max(volume_ratio - 0.8, 0) ** gamma_low
            )
        
            stress_weight = 1 - np.exp(
                -lambda_high * max(volume_ratio - 1, 0) ** gamma_high
            )
        
            base_weight = 1 - low_weight - stress_weight
            base_weight = np.clip(base_weight, 0, 1)
        
            alpha_low_5 = np.array([40.0, 7.5, 2.5, 0.0, 0.0], dtype=float)
            alpha_base_5 = np.array([35.0, 12.5, 2.5, 0.0, 0.0], dtype=float)
            alpha_stress_5 = np.array([20.0, 17.5, 12.5, 0.0, 0.0], dtype=float)
        
            alpha = (
                low_weight * alpha_low_5
                + base_weight * alpha_base_5
                + stress_weight * alpha_stress_5
            )

        weights = rng.dirichlet(alpha)

        for delay, weight in zip(settlement_days, weights):

            settlement_date = add_business_days(
                date,
                delay,
                business_dates,
                end_date
            )

            transactions.append({
                "date": date,
                "type": "pos_net_cf",
                "amount": row["pos_net_amount"] * weight,
                "due_date": settlement_date
            })

    return df, transactions

def add_daily_waste_features(df, params, seed):
    """
    Genera waste giornaliero in modo più realistico.

    Idea:
    - prima stimiamo expected_food_sales usando solo info passata/nota
    - poi generiamo daily_food_replenishment
    - poi waste nasce da:
        1. spreco fisiologico sulla supply
        2. extra waste da eccesso di supply rispetto alle vendite reali
    """

    rng = np.random.default_rng(seed + 40_010)


    base_waste_rate = params["waste_rate"]

    # =====================================================
    # 1. EXPECTED FOOD SALES — solo passato + calendario
    # =====================================================

    df["food_sales_ewm_30"] = (
        df["daily_food_sales"]
        .shift(2)
        .ewm(span=21, adjust=False)
        .mean()
        .bfill()
    )
    
    df["food_sales_ewm_7"] = (
        df["daily_food_sales"]
        .shift(1)
        .ewm(span=7, adjust=False)
        .mean()
        .bfill()
    )
    
    base_expected_food_sales = (
        0.45 * df["food_sales_ewm_30"]
        + 0.55 * df["food_sales_ewm_7"]
    )

    # profilo settimanale già coerente con la generazione vendite
    weekday_factor = df["week_day"].map({
        0: 0.95,
        1: 0.93,
        2: 0.98,
        3: 1.00,
        4: 1.03,
        5: 1.06,
        6: 1.04
    }).astype(float)

    avg_weekday_factor = weekday_factor.mean()
    weekday_effect = weekday_factor / avg_weekday_factor
    
    month_factor = df["month"].map({
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        5: 1.00,
        6: 1.01,
        7: 1.02,
        8: 0.95,
        9: 1.00,
        10: 1.00,
        11: 1.01,
        12: 1.06
    }).astype(float)
    
    month_effect = month_factor / month_factor.mean()

    df["expected_food_sales"] = (
        base_expected_food_sales
        * weekday_effect
        * month_effect
        * (1 + 0.25 * df["pre_holiday"])
        * (1 - 0.15 * df["actual_holiday"])
    )

    # =====================================================
    # 2. SUPPLY BUFFER
    # =====================================================

    # incertezza domanda: più sales_std_30 è alta, più buffer
    sales_volatility = df["sales_std_30"]/ df["sales_rm_30"]

    sales_volatility = np.clip(sales_volatility, 0, 0.40)

    # caldo: aumenta prudenza e deperibilità
    temp_above_ref = np.maximum(df["temperature"] - 18, 0)

    temperature_supply_effect = np.clip(
        1 + 0.004 * temp_above_ref,
        1.00,
        1.08
    )

    volatility_effect = np.clip(
        1 + 0.50 * sales_volatility,
        1.00,
        1.20
    )

    calendar_supply_effect = (
        1
        + 0.10 * df["pre_holiday"]
    )

    supply_noise = np.zeros(len(df))
    
    for t in range(1, len(df)):
        supply_noise[t] = (
            0.80 * supply_noise[t - 1]
            + rng.normal(0, 0.015)
        )
    
    noise_effect = np.clip(
        1 + supply_noise,
        0.94,
        1.08
    )

    df["supply_buffer"] = (
        1.03
        * temperature_supply_effect
        * volatility_effect
        * calendar_supply_effect
        * noise_effect
    )

    df["supply_buffer"] = np.clip(
        df["supply_buffer"],
        1.00,
        1.45
    )

    # daily_food_replenishment rappresenta la nuova merce food
    # resa disponibile nel giorno corrente (replenishment operativo),
    # non l'intero inventario fisico del supermercato.    

    df["daily_food_replenishment"] = (
        df["expected_food_sales"]
        * df["supply_buffer"]
    )

    # =====================================================
    # 3. DYNAMIC WASTE RATES
    # =====================================================
    
    # =====================================================
    # FOOD COMPOSITION
    # =====================================================
    
    base_fresh_share = params["fresh_food_share"]
    
    food_ratio = df["food_ratio"]
    
    food_ratio_rm30 = (
        food_ratio
        .shift(1)
        .rolling(30, min_periods=7)
        .mean()
        .bfill()
    )
    
    food_pressure = food_ratio / food_ratio_rm30
    
    
    fresh_share_noise = np.zeros(len(df))
    
    for t in range(1, len(df)):
        fresh_share_noise[t] = (
            0.85 * fresh_share_noise[t - 1]
            + rng.normal(0, 0.010)
        )
    
    fresh_share_noise_effect = np.clip(
        1 + fresh_share_noise,
        0.95,
        1.05
    )
    
    # supponiamo che se quota food sul totale aumenta, aumenta anche la componente fresh (debolmente)
    
    fresh_share = np.clip(
        base_fresh_share
        * np.power(food_pressure, 0.15)
        * fresh_share_noise_effect,
        0.20,
        0.80
    )
    
    stable_share = 1 - fresh_share
    
    df["fresh_share"] = fresh_share
    
    # =====================================================
    # BASE WASTE RATES
    # =====================================================
    
    base_fresh_waste_rate = base_waste_rate * 0.60
    base_stable_waste_rate = base_waste_rate * 0.05
    
    
    # =====================================================
    # TEMPERATURE EFFECT
    # =====================================================
    
    temperature_effect = np.clip(
        1 + 0.008 * temp_above_ref + 0.0004 * temp_above_ref**2,
        1.00,
        1.25
    )
    
    df["waste_temperature_effect"] = temperature_effect
    
    # =====================================================
    # COLD CHAIN EFFECT
    # =====================================================
    
    cold_chain_effect = np.clip(
        df["logistics_cold_chain_effect"],
        1.00,
        1.25
    )
    
    
    df["waste_cold_chain_effect"] = cold_chain_effect
    
    # =====================================================
    # OVERSTOCK
    # =====================================================
    
    df["excess_food_supply"] = np.maximum(
        df["daily_food_replenishment"] - df["daily_food_sales"],
        0
    )
        
    # =====================================================
    # CALENDAR EFFECT
    # =====================================================
    
    post_holiday = (
        df["actual_holiday"]
        .shift(1)
        .fillna(0)
    )
    
    fresh_calendar_effect = (
        1
        + 0.05 * df["actual_holiday"]
        + 0.08 * post_holiday
        + 0.02 * df["pre_holiday"]
    )
    
    
    stable_calendar_effect = (
        1
        + 0.02 * df["actual_holiday"]
        + 0.03 * post_holiday
    )
    
    
    # =====================================================
    # DAILY NOISE
    # =====================================================
    
    fresh_noise = np.zeros(len(df))
    stable_noise = np.zeros(len(df))
    
    for t in range(1, len(df)):
    
        fresh_noise[t] = rng.normal(0, 0.05)
    
        stable_noise[t] = rng.normal(0, 0.025)
    
    fresh_noise_effect = np.clip(
        1 + fresh_noise,
        0.85,
        1.20
    )
    
    stable_noise_effect = np.clip(
        1 + stable_noise,
        0.92,
        1.10
    )
    
    
    # =====================================================
    # FINAL DYNAMIC RATES
    # =====================================================
    
    df["fresh_spoilage_rate"] = (
        base_fresh_waste_rate
        * temperature_effect
        * cold_chain_effect
        * fresh_calendar_effect
        * fresh_noise_effect
    )
    
    df["stable_spoilage_rate"] = (
        base_stable_waste_rate
        * stable_calendar_effect
        * stable_noise_effect
    )
    
    
    df["fresh_spoilage_rate"] = np.clip(
        df["fresh_spoilage_rate"],
        base_fresh_waste_rate * 0.50,
        base_fresh_waste_rate * 3.00
    )
    
    df["stable_spoilage_rate"] = np.clip(
        df["stable_spoilage_rate"],
        base_stable_waste_rate * 0.50,
        base_stable_waste_rate * 2.00
    )
    
    
    # =====================================================
    # FINAL WASTE
    # =====================================================
    # baseline_*_waste:
    # waste fisiologico/operativo normale
    # (deterioramento, handling, shelf-life, cold chain, ecc.)
    #
    # *_overstock_waste:
    # waste aggiuntivo causato da replenishment
    # eccessivo rispetto alla domanda reale
    # =====================================================
    
    fresh_replenishment = df["daily_food_replenishment"] * fresh_share
    stable_replenishment = df["daily_food_replenishment"] * stable_share
    
    fresh_excess_supply = df["excess_food_supply"] * fresh_share
    stable_excess_supply = df["excess_food_supply"] * stable_share
    
    baseline_fresh_spoilage = (
        fresh_replenishment
        * df["fresh_spoilage_rate"]
    )
    
    baseline_stable_spoilage = (
        stable_replenishment
        * df["stable_spoilage_rate"]
    )
    
    # percentuali di merce invenduta che diventa waste
    
    fresh_conv_noise = np.zeros(len(df))
    stable_conv_noise = np.zeros(len(df))
    
    for t in range(1, len(df)):
    
        fresh_conv_noise[t] = (
            0.35 * fresh_conv_noise[t - 1]
            + rng.normal(0, 0.12)
        )
    
        stable_conv_noise[t] = (
            0.25 * stable_conv_noise[t - 1]
            + rng.normal(0, 0.05)
        )
    
    fresh_conversion_factor = np.clip(
        1 + fresh_conv_noise,
        0.65,
        1.45
    )
    
    stable_conversion_factor = np.clip(
        1 + stable_conv_noise,
        0.80,
        1.25
    )
    
    fresh_overstock_conversion = np.clip(
        0.50
        * temperature_effect
        * cold_chain_effect
        * fresh_conversion_factor,
        0.25,
        0.90
    )
    
    stable_overstock_conversion = np.clip(
        0.01
        * (1 + 0.10 * (temperature_effect - 1))
        * stable_conversion_factor,
        0.003,
        0.035
    )
    
    fresh_overstock_waste = (
        fresh_excess_supply
        * fresh_overstock_conversion
    )
    
    stable_overstock_waste = (
        stable_excess_supply
        * stable_overstock_conversion
    )
    
    df["fresh_waste"] = (
        baseline_fresh_spoilage
        + fresh_overstock_waste
    )
    
    df["stable_waste"] = (
        baseline_stable_spoilage
        + stable_overstock_waste
    )
    
    df["waste"] = (
        df["fresh_waste"]
        + df["stable_waste"]
    ).round(2)
    
    df["waste_rate"] = (
        df["waste"]
        / df["daily_food_replenishment"]
    )

    return df

# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ==============================GENERAZIONE DEI DATI DEL SUPERMERCATO================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================

# 1 - ================= PARAMETRI DAILY SALES =================

def generate_store_data(df, params, store_id, seed, anomaly_config=None):
    
    
    if anomaly_config is None:
        anomaly_config = get_def_anomaly_config()
    
    rng = np.random.default_rng(seed)
    
    # =========================
    # TEMPERATURE
    # =========================
    df = add_daily_temperature_feature(
        df=df,
        global_seed=seed
    )
    
    # =========================================================
    # 1. ESTRAZIONE PARAMETRI
    # =========================================================
    base_food_sales          = params.get("base_food_sales", 0)
    base_nonfood_sales       = params.get("base_nonfood_sales", 0)
    base_sales               = base_food_sales + base_nonfood_sales 
    
    annual_trend             = params.get("annual_trend", 0)
    supplier_revenue_monthly = params.get("supplier_revenue_monthly", 0)
    
    cogs_ratio               = params.get("cogs_ratio", 0)
    
    base_salary              = params.get("base_salary", 0)
    base_rent                = params.get("base_rent", 0)
    base_marketing           = params.get("base_marketing", 0)
    base_it                  = params.get("base_it", 0)
    base_admin               = params.get("base_admin", 0)
    base_other               = params.get("base_other", 0)
    
    insurance_annual         = params.get("insurance_annual", 0)
    base_fixed_tax           = params.get("base_fixed_tax", 0)
    tax_rate                 = params.get("tax_rate", 0)

    start_date = df['date'].min()
    end_date   = df['date'].max()

    # =========================================================
    # 2. CALCOLO VENDITE GIORNALIERE (TOTALE, FOOD E NON-FOOD)
    # =========================================================


    days_since_start   = (df['date'] - df['date'].min()).dt.days
    trend_factor       = (1 + annual_trend) ** (days_since_start / 365)
    weekly_factor      = {0:0.90, 1:0.88, 2:0.95, 3:1.00, 4:1.10, 5:1.20, 6:1.15}
    weekly_effect      = df['week_day'].map(weekly_factor)
    month_end_effect   = np.where(df['day'] >= 25, 1.15, 1.00)
    month_effect       = df['month'].map({1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0.90, 9:1, 10:1, 11:1, 12:1.20})
    holiday_effect     = 1 - (0.2 * df['holiday'])
    pre_holiday_effect = np.where(df['actual_holiday'].shift(-1) == 1, 1.3, 1.00)

    # =========================
    # NORMALIZZAZIONI
    # =========================

    fao_norm = df["fao"] / df["fao"].iloc[0]
    confidence_norm = df["consumer_confidence"] / 100.0
    oil_norm = df["oil_price"] / df["oil_price"].iloc[0]

    # =========================
    # EFFETTO MACROECONOMICO
    # =========================

    coeff_fao = 0.15
    coeff_conf = 1.5
    coeff_oil = -0.05

    fao_mult = 1 + coeff_fao * (fao_norm - 1)
    conf_mult = 1 + coeff_conf * (confidence_norm - 1)
    oil_mult = 1 + coeff_oil * (oil_norm - 1)

    macro_effect = (
        fao_mult
        * conf_mult
        * oil_mult
    )

    noise_ind = rng.normal(0, 0.020, len(df))
    
    noise_ar = np.zeros(len(df))
    
    for t in range(1, len(df)):
        noise_ar[t] = (
            0.85 * noise_ar[t - 1]
            + rng.normal(0, 0.010)
        )
    
    total_noise = 1 + noise_ind + noise_ar
    
    total_noise = np.clip(
        total_noise,
        0.85,
        1.15
    )     

    # Vendite Totali + Rumore
    daily_sales = (
        base_sales
        * trend_factor
        * weekly_effect
        * month_end_effect
        * month_effect
        * holiday_effect
        * pre_holiday_effect
        * macro_effect
        * total_noise
        )   

    
    
    df['daily_total_sales'] = daily_sales.round(2)
    
    
    # =========================
    # LEVEL SHIFT ANOMALIES
    # =========================
    level_cfg = anomaly_config["level_shift"]
    
    df = inject_level_shift_anomalies(
        df,
        level_cfg=level_cfg,
        sales_col="daily_total_sales",
        seed=seed + level_cfg.get("seed_offset", 40_000)
    )

    base_food_ratio = (
        base_food_sales / base_sales
    )
    
    weekday_food_effect = df["week_day"].map({
        0: -0.005,  # lun
        1: -0.010,  # mar
        2: -0.010,  # mer
        3:  0.000,  # gio
        4:  0.015,  # ven
        5:  0.030,  # sab
        6:  0.015   # dom
    }).astype(float)
    
    pre_holiday_food_effect = (
        0.025 * df["pre_holiday"]
    )
    
    food_ratio_noise = rng.normal(
        0,
        0.01,
        len(df)
    )
    
    food_ratio = (
        base_food_ratio
        + weekday_food_effect
        + pre_holiday_food_effect
        + food_ratio_noise
    )
    
    food_ratio = np.clip(
        food_ratio,
        0.40,
        0.97
    )

    df["daily_food_sales"] = (
        df["daily_total_sales"] * food_ratio
    ).round(2)
    
    df["daily_nonfood_sales"] = (
        df["daily_total_sales"] - df["daily_food_sales"]
    ).round(2)
    
    df["food_ratio"] = food_ratio

    
    # =========================
    # SALES ROLLING FEATURES:
    # - sales_rm_7
    # - sales_rm_30
    # - sales_std_30
    # - sales_pressure
    # =========================
    df = add_sales_rolling_features(df)
    
    
    # =========================================================
    # DAILY ELECTRICITY / UTILITIES GENERATION
    # =========================================================
    df = add_daily_electricity_features(
        df=df,
        params=params,
        global_seed=seed
    )
    
    
    
    # =========================
    # DAILY LOGISTICS FEATURES ADDED:
    # - daily_logistics_cost
    # =========================
    df = add_daily_logistics_features(
        df=df,
        params=params,
        global_seed=seed
    )
    
    
    
    df = add_daily_waste_features(
        df,
        params,
        seed
    )
    
    
    df, transactions = add_pos_settlement_features(
        df=df,
        params=params,
        anomaly_config=anomaly_config,
        seed=seed
    )
        
    # Pagamenti fornitori (COGS)
    payment_dates = set()
    for month_period in df['date'].dt.to_period('M').unique():
        days_in_month  = df[df['date'].dt.to_period('M') == month_period]['date'].tolist()
        num_payments = rng.choice([3, 4])
        selected_dates = rng.choice(days_in_month, size=num_payments, replace=False)
        payment_dates.update(selected_dates)
    payment_dates.add(df['date'].iloc[-1])
    

    accumulated_order_value = base_sales * cogs_ratio * 21
    for _, row in df.iterrows():
        sales = row['daily_total_sales']
        waste = row['waste']
        daily_order = sales * cogs_ratio
        waste_cost = waste * cogs_ratio
        
        accumulated_order_value += daily_order + waste_cost
        
        if row['date'] in payment_dates:
            delay = rng.choice([30, 60, 90], p=[0.15, 0.7, 0.15])
            due_date = row['date'] + pd.Timedelta(days=delay)
            transactions.append({'date': row['date'], 'type': 'cogs_payment', 'amount': -accumulated_order_value, 'due_date': due_date})
            accumulated_order_value = 0

    # Costi Fissi Mensili
    for month in df['date'].dt.to_period('M').unique():
        last_day  = pd.Timestamp(month.end_time).normalize()
        first_day = pd.Timestamp(month.start_time).normalize()
        
        if last_day not in df['date'].values or first_day not in df['date'].values: continue
            
        inflation_factor = df.loc[df['date'] == last_day, 'inflation_index'].iloc[0] / df['inflation_index'].iloc[0]
        
        # Stipendi (daily_salary come richiesto, anche se pagati a fine mese)
        transactions.append({'date': last_day, 'type': 'daily_salary', 'amount': -(base_salary * inflation_factor), 'due_date': last_day})
        # Entrate Fornitori
        transactions.append({'date': last_day, 'type': 'supplier_revenue_monthly', 'amount': supplier_revenue_monthly, 'due_date': last_day})
        
        # Affitto, IT, Admin (Giorno 5)
        pay_date_early = first_day + pd.Timedelta(days=4) 
        if pay_date_early <= pd.Timestamp(end_date):
            transactions.append({'date': pay_date_early, 'type': 'rent',  'amount': -base_rent,  'due_date': pay_date_early})
            transactions.append({'date': pay_date_early, 'type': 'it',    'amount': -base_it,    'due_date': pay_date_early})
            transactions.append({'date': pay_date_early, 'type': 'admin', 'amount': -base_admin, 'due_date': pay_date_early})

        # Logistica, Marketing, Utenze, Altro (Giorno 15)
        pay_date_mid = first_day + pd.Timedelta(days=14) 
        if pay_date_mid <= pd.Timestamp(end_date):
            # Bolletta elettrica mensile generata dai consumi giornalieri
            month_mask = df["date"].dt.to_period("M") == month
            
            electricity_bill = df.loc[
                month_mask,
                "daily_electricity_cost"
            ].sum()
            
            transactions.append({
                "date": pay_date_mid,
                "type": "electricity_bill",
                "amount": -round(electricity_bill, 2),
                "due_date": pay_date_mid
            })
            monthly_logistics_cost = df.loc[
                month_mask,
                "daily_logistics_cost"
            ].sum()
            
            transactions.append({
                "date": pay_date_mid,
                "type": "logistics",
                "amount": -round(monthly_logistics_cost, 2),
                "due_date": pay_date_mid
            })
            transactions.append({'date': pay_date_mid, 'type': 'marketing', 'amount': -base_marketing, 'due_date': pay_date_mid})
            transactions.append({'date': pay_date_mid, 'type': 'other',     'amount': -(base_other * inflation_factor), 'due_date': pay_date_mid})

    # Assicurazione Annuale
    for date in pd.date_range(start_date, end_date, freq='YS'):
        pay_date = date + pd.Timedelta(days=9) 
        if pay_date <= pd.Timestamp(end_date):
            transactions.append({'date': pay_date, 'type': 'insurance', 'amount': -insurance_annual, 'due_date': pay_date})

    # Tasse Trimestrali
    temp_trans_df = pd.DataFrame(transactions)
    temp_trans_df['due_date'] = pd.to_datetime(temp_trans_df['due_date'])
    for q_end in pd.date_range(start_date, end_date, freq='QE'):
        pay_date = q_end + pd.Timedelta(days=20)
        if pay_date <= pd.Timestamp(end_date):
            mask = (temp_trans_df['due_date'].dt.year == q_end.year) & (temp_trans_df['due_date'].dt.quarter == q_end.quarter)
            quarterly_net = temp_trans_df.loc[mask, 'amount'].sum()
            variable_tax = quarterly_net * tax_rate if quarterly_net > 0 else 0
            inflation_factor = df.loc[df['date'] == q_end, 'inflation_index'].iloc[0] / df['inflation_index'].iloc[0]
            transactions.append({'date': q_end, 'type': 'taxes', 'amount': -(base_fixed_tax * inflation_factor + variable_tax), 'due_date': pay_date})


    # =========================================================
    # 4. CREAZIONE DATASET E FILTRO COLONNE FINALI
    # =========================================================
    trans_df = pd.DataFrame(transactions)

    # Raggruppamento Cashflow totale
    daily_net = trans_df.groupby('due_date')['amount'].sum().reset_index()
    daily_net.columns = ['date', 'net_inflow']

    # Pivot delle transazioni in colonne separate
    pivot_trans = trans_df.pivot_table(index='due_date', columns='type', values='amount', aggfunc='sum').reset_index()
    pivot_trans.rename(columns={'due_date': 'date'}, inplace=True)

    # Merge
    final_df = df.merge(daily_net, on='date', how='left')
    final_df = final_df.merge(pivot_trans, on='date', how='left') 

    final_df.fillna(0, inplace=True)
    
    # Saldo di cassa
    final_df['cash_balance'] = 100000 + final_df['net_inflow'].cumsum()

    final_df['store_id'] = store_id

    # ---------------------------------------------------------
    # LA LISTA EXACTA DELLE COLONNE RICHIESTE
    # ---------------------------------------------------------
    expected_cols = [
        'date', 'store_id',
        
        # calendario / macro
        'year', 'month', 'day', 'week_day',
        'holiday', 'actual_holiday', 'pre_holiday', 'weekend',
        'oil_price', 'euribor', 'consumer_confidence',
        'inflation_index', 'consumer_prices', 'fao',
        'pandemic_uncertainty',
        
        # vendite
        'daily_nonfood_sales', 'daily_food_sales', 'daily_total_sales',
        
        # cashflow principali
        'supplier_revenue_monthly', 'cogs_payment',
        
        # logistica
        'daily_logistics_cost', 'logistics',
        
        # waste
        'waste',
        
        # POS / vendite cash
        'pos_card_share', 'pos_card_sales', 'cash_sales_cf', 'pos_net_cf', 
        
        # bolletta elettricità
        'temperature', 'daily_electricity_consumption', 'daily_electricity_cost', 'electricity_bill',
        
        # costi operativi
        'daily_salary', 'marketing',
        'it', 'admin', 'other', 'insurance', 'taxes', 'rent',
        
        # stato finanziario
        'cash_balance', 'net_inflow',
        
        # Ground truth Pos delay
        "is_pos_delay_source_day", "pos_delay_source_day_in_event", "pos_delay_source_duration", "is_pos_delay_effect_day", "pos_delay_effect_day_in_event", "pos_delay_effect_duration", "pos_delay_type", "pos_delay_event_id", "pos_volume_ratio",
        
        # Ground truth level shift
        'is_level_shift_anomaly', 'lsa_type', 'lsa_severity', 'lsa_mult', 'lsa_event_id', 'lsa_day_in_event', 'lsa_duration'
    ]

    # Assicuriamoci che tutte le colonne esistano (se in un negozio una spesa non è mai scattata, la crea a 0)
    for col in expected_cols:
        if col not in final_df.columns:
            print(f"La colonna {col} non è stata trovata per {store_id}")
            final_df[col] = 0.0

    # FILTRO FINALE: Conserva solo le colonne richieste
    final_df = final_df[expected_cols]
    
    # Restituiamo il dataframe filtrato (così anche il file aggregato avrà questa struttura)
    return final_df

def generate_all_stores_data(saving_path, global_seed=42, anomaly_config=None):
    saving_path = Path(saving_path)
    saving_path.mkdir(parents=True, exist_ok=True)

    print("Scaricando indici economici da FRED (richiede qualche secondo)...")
    df = get_external_indices()
    print("Indici scaricati. Inizio la generazione per i singoli supermercati...")
    
    global_rng = np.random.default_rng(global_seed)

    all_data = pd.DataFrame()
    for store in all_params:
        store_seed = int(global_rng.integers(0, 1_000_000))
        ds = generate_store_data(
                 df,
                 store["params"],
                 store["store_id"],
                 seed=store_seed,
                 anomaly_config=anomaly_config
                 )
        all_data = pd.concat([all_data, ds], ignore_index=True)
        
    all_data.to_csv(f'{saving_path}/all_stores_cashflow.csv', index=False)
    print("--> Master CSV 'all_stores_cashflow.csv' completato!")



def main():
    global_seed = 42
    force_recompute = False

    ensure_artifact_directories()

    # =========================================================
    # 1. DATASET CLEAN
    # =========================================================

    clean_config = get_def_anomaly_config()

    clean_config["level_shift"]["scope"] = None
    clean_config["pos_delay"]["scope"] = None

    if force_recompute or not CLEAN_DATA_PATH.exists():
        generate_all_stores_data(
            saving_path=CLEAN_DIR,
            global_seed=global_seed,
            anomaly_config=clean_config,
        )
    else:
        print(f"Dataset clean già presente: {CLEAN_DATA_PATH}")

    # =========================================================
    # 2. LEVEL SHIFT SENSITIVITY
    # =========================================================

    generate_level_shift_sensitivity_experiment_from_clean(
        clean_path=CLEAN_DATA_PATH,
        saving_path=LEVEL_SHIFT_SENSITIVITY_DIR,
        force_recompute=force_recompute,
    )

    # =========================================================
    # 3. POS DELAY SENSITIVITY
    # =========================================================

    generate_pos_delay_sensitivity_experiment_from_clean(
        clean_path=CLEAN_DATA_PATH,
        saving_path=POS_DELAY_SENSITIVITY_DIR,
        force_recompute=force_recompute,
    )

    # =========================================================
    # 4. LEVEL SHIFT CONTAMINATION
    # =========================================================

    generate_level_shift_contamination_experiment_from_clean(
        clean_path=CLEAN_DATA_PATH,
        saving_path=LEVEL_SHIFT_CONTAMINATION_DIR,
        contamination_levels=(0, 5, 10, 20, 30, 50),
        global_seed=global_seed,
        force_recompute=force_recompute,
    )

    # =========================================================
    # 5. POS DELAY CONTAMINATION
    # =========================================================

    generate_pos_delay_contamination_experiment_from_clean(
        clean_path=CLEAN_DATA_PATH,
        saving_path=POS_DELAY_CONTAMINATION_DIR,
        contamination_levels=(0, 1, 2, 3, 5, 10),
        global_seed=global_seed,
        force_recompute=force_recompute,
    )


if __name__ == "__main__":
    main()
# ==============================================================================
# supermarkets.py
# Parametri di configurazione per ogni tipologia di punto vendita.
#
# STRUTTURA DEI PARAMETRI AGGIORNATA:
# ─────────────────────────────────────────────────────────────────────────────
# ENTRATE
#   base_food_sales         → Vendite giornaliere medie reparto Alimentari (€)
#   base_nonfood_sales      → Vendite giornaliere medie reparto No-Food (€)
#   annual_trend            → Tasso di crescita annuo delle vendite
#   other_inflow_in         → Altri ricavi trimestrali generici (€/trimestre)
#   promotional_inflow_prob → Probabilità giornaliera di accordo promozionale fornitore
#   supplier_revenue_monthly→ Ricavi mensili da fornitori (listing/slotting fee, contributi)
#   service_revenue_rate    → Commissioni giornaliere su servizi al cliente (% sulle vendite)
#
# USCITE - VARIABILI
#   cogs_ratio              → % delle vendite come costo della merce (COGS)
#   pos_commission_rate     → Commissioni POS giornaliere (% sulle vendite)
#   waste_rate              → Tasso di scarto/shrinkage giornaliero (% sulle vendite)
#
# USCITE - FISSE MENSILI
#   base_salary             → Costo totale mensile del personale (stipendi + contributi + TFR)
#   base_services           → Utenze mensili (energia, gas, acqua, pulizie, manutenzione)
#   base_rent               → Canone mensile di locazione (o quota ammortamento se di proprietà)
#   base_logistics          → Costi mensili di logistica e trasporto (CDO, delivery, carburante)
#   base_marketing          → Spese mensili di marketing (volantini, digital, fidelity)
#   base_it                 → Costi mensili IT (licenze software, hardware, connettività)
#   base_admin              → Spese mensili amministrative (consulenze, commissioni bancarie fisse)
#   base_other              → Altre spese mensili residuali non classificabili altrove
#
# USCITE - PERIODICHE
#   insurance_annual        → Premio assicurativo annuale (RC, furto, incendio, fabbricato)
#
# USCITE - FISCALI
#   base_fixed_tax          → Quota fissa trimestrale di tributi locali (TARI, TOSAP, bolli)
#   tax_rate                → Aliquota sugli utili netti del trimestre (IRES + IRAP approssimati)
# ==============================================================================


params_large = {
    # --- PROFILO: Ipermercato di grandi dimensioni (~100 dipendenti, >5.000 mq) ---

    # Entrate (Mix stimato: 70% Food, 30% Non-Food)
    "base_food_sales": 56000,
    "base_nonfood_sales": 24000,
    "fresh_food_share": 0.38,
    "annual_trend": 0.02,
    "other_inflow_in": 10000,
    "promotional_inflow_prob": 0.04,
    "supplier_revenue_monthly": 25000,      # Grandi volumi = maggiori contributi da fornitori
    "service_revenue_rate": 0.003,          # Alto traffico → più commissioni su servizi

    # Uscite variabili
    "cogs_ratio": 0.738,                    # cogs attuale
    "pos_commission_rate": 0.008,           # 0.8% sulle vendite (mix carte/contanti)
    "waste_rate": 0.012,                    # 1.2% → scala ma volume assoluto alto

    # Uscite fisse mensili
    "base_salary": 250000,
    "base_monthly_kwh_need": 450_000,                 # Utenze molto alte (refrigerazione, HVAC)
    "base_rent": 60000,                     # Affitto elevato per superfici grandi
    "base_logistics": 35000,                # Più frequenze di rifornimento dal CDO
    "base_marketing": 18000,                # Volantini su ampia area, digital marketing
    "base_it": 12000,                       # Più casse, sistemi WMS complessi
    "base_admin": 10000,
    "base_other": 20000,

    # Uscite periodiche
    "insurance_annual": 80000,              # Assicurazione su struttura grande e contenuto elevato

    # Fiscalità
    "base_fixed_tax": 15000,
    "tax_rate": 0.25
}


params_medium = {
    # --- PROFILO: Supermercato standard (~50 dipendenti, 1.000-3.000 mq) ---

    # Entrate (Mix stimato: 80% Food, 20% Non-Food)
    "base_food_sales": 40000,
    "base_nonfood_sales": 10000,
    "fresh_food_share": 0.45,
    "annual_trend": 0.02,
    "other_inflow_in": 5000,
    "promotional_inflow_prob": 0.03,
    "supplier_revenue_monthly": 12000,
    "service_revenue_rate": 0.002,

    # Uscite variabili
    "cogs_ratio": 0.725,
    "pos_commission_rate": 0.008,
    "waste_rate": 0.015,

    # Uscite fisse mensili
    "base_salary": 150000,
    "base_monthly_kwh_need": 275000,
    "base_rent": 30000,
    "base_logistics": 18000,
    "base_marketing": 8000,
    "base_it": 6000,
    "base_admin": 5000,
    "base_other": 10000,

    # Uscite periodiche
    "insurance_annual": 35000,

    # Fiscalità
    "base_fixed_tax": 10000,
    "tax_rate": 0.25
}


params_small = {
    # --- PROFILO: Piccolo supermercato di prossimità (~20-30 dipendenti, <1.000 mq) ---

    # Entrate (Mix stimato: 90% Food, 10% Non-Food)
    "base_food_sales": 22500,
    "base_nonfood_sales": 2500,
    "fresh_food_share": 0.50,
    "annual_trend": 0.01,
    "other_inflow_in": 2000,
    "promotional_inflow_prob": 0.02,
    "supplier_revenue_monthly": 4000,       # Pochi accordi commerciali con fornitori
    "service_revenue_rate": 0.002,

    # Uscite variabili
    "cogs_ratio": 0.712,                     # Margine lordo leggermente migliore (assortimento curato)
    "pos_commission_rate": 0.009,           # Più alto: clientela usa più il bancomat
    "waste_rate": 0.018,                    # Più alto: meno rotazione, più scarti freschi

    # Uscite fisse mensili
    "base_salary": 80000,
    "base_monthly_kwh_need": 150000,
    "base_rent": 12000,
    "base_logistics": 8000,
    "base_marketing": 3000,
    "base_it": 2500,
    "base_admin": 2500,
    "base_other": 6000,

    # Uscite periodiche
    "insurance_annual": 15000,

    # Fiscalità
    "base_fixed_tax": 6000,
    "tax_rate": 0.25
}


params_discount = {
    # --- PROFILO: Discount hard (es. Lidl/Aldi) - alto volume, bassissimo margine ---

    # Entrate (Mix stimato: 85% Food, 15% Non-Food in cesti promozionali)
    "base_food_sales": 59500,
    "base_nonfood_sales": 10500,
    "fresh_food_share": 0.32,
    "annual_trend": 0.02,
    "other_inflow_in": 2000,
    "promotional_inflow_prob": 0.01,        # I discount pagano pochi fornitori per visibilità
    "supplier_revenue_monthly": 3000,       # Pochissimi accordi promozionali (prodotti a marchio proprio)
    "service_revenue_rate": 0.001,

    # Uscite variabili
    "cogs_ratio": 0.840,                     # Margine molto basso: prezzi competitivi
    "pos_commission_rate": 0.007,           # Meno carte di credito (clientela price-sensitive)
    "waste_rate": 0.010,                    # Meno freschi = meno scarti

    # Uscite fisse mensili
    "base_salary": 120000,                  # Personale minimo
    "base_monthly_kwh_need": 200_000,
    "base_rent": 20000,                     # Spesso in zone periferiche, affitti più bassi
    "base_logistics": 25000,                # Frequenza alta di rifornimento per gestire bassi stock
    "base_marketing": 4000,                 # Marketing quasi zero: il prezzo è la comunicazione
    "base_it": 4000,
    "base_admin": 4000,
    "base_other": 5000,

    # Uscite periodiche
    "insurance_annual": 25000,

    # Fiscalità
    "base_fixed_tax": 8000,
    "tax_rate": 0.25
}


params_premium = {
    # --- PROFILO: Supermercato premium (es. Eataly-style, biologico) - meno clienti, alta spesa ---

    # Entrate (Mix stimato: 85% Food premium, 15% Non-Food legato al food es. utensili)
    "base_food_sales": 38000,
    "base_nonfood_sales": 7000,
    "fresh_food_share": 0.60,
    "annual_trend": 0.025,
    "other_inflow_in": 8000,
    "promotional_inflow_prob": 0.05,        # I brand premium pagano molto per la visibilità
    "supplier_revenue_monthly": 18000,      # Accordi premium con produttori artigianali
    "service_revenue_rate": 0.004,          # Servizi premium: personal shopper, consegne prioritarie

    # Uscite variabili
    "cogs_ratio": 0.628,                     # Margine lordo alto: prodotti ad alto valore aggiunto
    "pos_commission_rate": 0.012,           # Più alto: scontrino medio alto, più carte di credito premium
    "waste_rate": 0.022,                    # Alto: prodotti freschi e biologici con vita breve

    # Uscite fisse mensili
    "base_salary": 180000,                  # Personale più qualificato, specialisti di reparto
    "base_monthly_kwh_need": 350_000,                 # Manutenzione alta (espositori premium, climatizzazione)
    "base_rent": 45000,                     # Posizioni centrali/commerciali pregiate
    "base_logistics": 20000,                # Fornitori locali, spesso consegne più piccole e frequenti
    "base_marketing": 15000,                # Comunicazione curata: eventi, influencer, PR
    "base_it": 8000,
    "base_admin": 7000,
    "base_other": 15000,

    # Uscite periodiche
    "insurance_annual": 45000,

    # Fiscalità
    "base_fixed_tax": 12000,
    "tax_rate": 0.25
}


params_tourist = {
    # --- PROFILO: Store in area turistica (mare, montagna, centro storico) - forte stagionalità ---

    # Entrate (Mix stimato: 80% Food, 20% Non-Food come souvenir/articoli estivi-invernali)
    "base_food_sales": 48000,
    "base_nonfood_sales": 12000,
    "fresh_food_share": 0.46,
    "annual_trend": 0.02,
    "other_inflow_in": 7000,
    "promotional_inflow_prob": 0.04,
    "supplier_revenue_monthly": 10000,
    "service_revenue_rate": 0.003,          # Turisti usano molto servizi (valuta, ricariche)

    # Uscite variabili
    "cogs_ratio": 0.734,
    "pos_commission_rate": 0.010,           # Più transazioni estere con commissioni più alte
    "waste_rate": 0.016,                    # Stagionalità crea difficoltà nella gestione scorte

    # Uscite fisse mensili
    "base_salary": 200000,                  # Personale extra per gestire i picchi estivi/invernali
    "base_monthly_kwh_need": 325_000,
    "base_rent": 40000,                     # Canoni elevati nelle zone turistiche
    "base_logistics": 22000,
    "base_marketing": 12000,                # Promozioni stagionali, guide turistiche, digitale
    "base_it": 7000,
    "base_admin": 6000,
    "base_other": 12000,

    # Uscite periodiche
    "insurance_annual": 40000,

    # Fiscalità
    "base_fixed_tax": 11000,
    "tax_rate": 0.25
}


params_low_income = {
    # --- PROFILO: Store in area a basso reddito - prezzi contenuti, margini ridotti ---

    # Entrate (Mix stimato: 90% Food di prima necessità, 10% Non-Food basico)
    "base_food_sales": 31500,
    "base_nonfood_sales": 3500,
    "fresh_food_share": 0.38,
    "annual_trend": 0.01,
    "other_inflow_in": 3000,
    "promotional_inflow_prob": 0.02,
    "supplier_revenue_monthly": 5000,
    "service_revenue_rate": 0.003,          # Alto: clientela usa spesso servizi (bollette, ricariche)

    # Uscite variabili
    "cogs_ratio": 0.760,                     # Margine ridotto per mantenere prezzi accessibili
    "pos_commission_rate": 0.007,
    "waste_rate": 0.020,

    # Uscite fisse mensili
    "base_salary": 100000,
    "base_monthly_kwh_need": 175_000,
    "base_rent": 15000,                     # Canoni più bassi in zone periferiche
    "base_logistics": 12000,
    "base_marketing": 4000,
    "base_it": 3500,
    "base_admin": 3500,
    "base_other": 7000,

    # Uscite periodiche
    "insurance_annual": 18000,

    # Fiscalità
    "base_fixed_tax": 7000,
    "tax_rate": 0.25
}


params_online = {
    # --- PROFILO: Store ibrido con forte componente e-commerce / dark store ---

    # Entrate (Mix stimato: 75% Food, 25% Non-Food, l'online facilita ordini di scorte domestiche)
    "base_food_sales": 41250,
    "base_nonfood_sales": 13750,
    "fresh_food_share": 0.35,
    "annual_trend": 0.03,                   # Crescita più sostenuta grazie all'online
    "other_inflow_in": 5000,
    "promotional_inflow_prob": 0.03,
    "supplier_revenue_monthly": 9000,
    "service_revenue_rate": 0.002,

    # Uscite variabili
    "cogs_ratio": 0.747,
    "pos_commission_rate": 0.010,           # Alto: quasi tutto digitale (carte, pagamenti online)
    "waste_rate": 0.013,                    # Picking ottimizzato = meno sprechi

    # Uscite fisse mensili
    "base_salary": 170000,                  # Picking, confezionamento, gestione resi
    "base_monthly_kwh_need": 300_000,
    "base_rent": 25000,                     # Magazzino + area ritiro
    "base_logistics": 40000,                # Il più alto: corrieri, ultimo miglio, costi delivery
    "base_marketing": 16000,                # Digital marketing, SEO, app, campagne online
    "base_it": 15000,                       # Il più alto: piattaforma e-commerce, API, cybersecurity
    "base_admin": 7000,
    "base_other": 18000,

    # Uscite periodiche
    "insurance_annual": 30000,

    # Fiscalità
    "base_fixed_tax": 10000,
    "tax_rate": 0.25
}


params_urban = {
    # --- PROFILO: Store urbano di città (alta densità, scontrino basso, alta frequenza) ---

    # Entrate (Mix stimato: 85% Food veloce/pronto, 15% Non-Food emergenza)
    "base_food_sales": 59500,
    "base_nonfood_sales": 10500,
    "fresh_food_share": 0.55,
    "annual_trend": 0.02,
    "other_inflow_in": 8000,
    "promotional_inflow_prob": 0.04,
    "supplier_revenue_monthly": 15000,
    "service_revenue_rate": 0.004,          # Alta frequenza clientela → più commissioni servizi

    # Uscite variabili
    "cogs_ratio": 0.726,
    "pos_commission_rate": 0.009,
    "waste_rate": 0.014,

    # Uscite fisse mensili
    "base_salary": 200000,
    "base_monthly_kwh_need": 400_000,                 # Energia cara in centro città
    "base_rent": 70000,                     # Il costo più alto: locazioni prime location urbane
    "base_logistics": 28000,                # Rifornimenti frequenti (no spazio per grandi scorte)
    "base_marketing": 10000,
    "base_it": 9000,
    "base_admin": 8000,
    "base_other": 15000,

    # Uscite periodiche
    "insurance_annual": 50000,              # Zone urbane: più rischio furto, valori assicurati più alti

    # Fiscalità
    "base_fixed_tax": 12000,
    "tax_rate": 0.25
}


params_new = {
    # --- PROFILO: Nuovo punto vendita  ---
    # Punto vendita giovane, ancora in crescita

    # Entrate (Mix stimato: 80% Food, 20% Non-Food)
    "base_food_sales": 30000,               # Vendite iniziali più alte: store già parzialmente avviato
    "base_nonfood_sales": 7500,
    "fresh_food_share": 0.45,
    "annual_trend": 0.035,                  # Crescita superiore alla media
    "other_inflow_in": 4000,
    "promotional_inflow_prob": 0.025,       # Leggera crescita degli accordi promozionali
    "supplier_revenue_monthly": 6000,       # Reputazione commerciale in costruzione, ma non nulla
    "service_revenue_rate": 0.002,

    # Uscite variabili
    "cogs_ratio": 0.720,                    # Margine realistico per store standard/giovane
    "pos_commission_rate": 0.009,
    "waste_rate": 0.017,                    # Ancora alto, ma meno estremo rispetto alla fase iniziale

    # Uscite fisse mensili
    "base_salary": 125000,
    "base_monthly_kwh_need": 225_000,
    "base_rent": 28000,
    "base_logistics": 16000,
    "base_marketing": 10000,                # Marketing sostenuto ma non dominante
    "base_it": 7000,
    "base_admin": 6000,
    "base_other": 8000,

    # Uscite periodiche
    "insurance_annual": 22000,

    # Fiscalità
    "base_fixed_tax": 8000,
    "tax_rate": 0.25
}


# ==============================================================================
# Registry di tutti i punti vendita
# ==============================================================================
all_params = [
    {"store_id": 0, "params": params_large},
    {"store_id": 1, "params": params_medium},
    {"store_id": 2, "params": params_small},
    {"store_id": 3, "params": params_discount},
    {"store_id": 4, "params": params_premium},
    {"store_id": 5, "params": params_tourist},
    {"store_id": 6, "params": params_low_income},
    {"store_id": 7, "params": params_online},
    {"store_id": 8, "params": params_urban},
    {"store_id": 9, "params": params_new}
]
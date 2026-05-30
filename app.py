"""
app.py  —  Part 1: imports, config, CSS, helpers, model loading
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from preprocessing import load_raw_data, preprocess, get_dataset_stats
from evaluation import get_all_metrics
from visualizations import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_probability_distribution,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_fraud_over_time,
    plot_sigmoid,
    plot_gradient_descent_illustration,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI — Détection de Fraude Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}
.stApp { background-color: #0a0a0a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #ffffff;
    font-size: 1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(255,255,255,0.05);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #888888;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 6px;
}
.metric-sub { font-size: 0.75rem; color: #555555; margin-top: 4px; }

/* Result badges */
.badge-fraud {
    background: #1a0000;
    border: 2px solid #cc0000;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    animation: pulse-red 1.8s infinite;
}
.badge-legit {
    background: #001a00;
    border: 2px solid #00aa44;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(204,0,0,0.35); }
    50%      { box-shadow: 0 0 0 10px rgba(204,0,0,0); }
}

/* Section headers */
.section-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #ffffff;
    border-left: 3px solid #ffffff;
    padding-left: 12px;
    margin: 24px 0 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Info boxes */
.info-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #cccccc;
}
.warning-box {
    background: #141000;
    border: 1px solid #886600;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ddbb44;
}
.danger-box {
    background: #140000;
    border: 1px solid #880000;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ff6666;
}

/* Probability bar */
.prob-bar-wrap { background: #222222; border-radius: 6px; height: 12px; overflow: hidden; margin: 6px 0; }
.prob-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }

/* Streamlit overrides */
div[data-testid="stMetric"] { display:none; }
.stButton > button {
    background: #1e1e1e;
    color: #ffffff;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 10px 24px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #2a2a2a;
    border-color: #666666;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ── Model / data loading (cached) ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    model  = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    meta   = joblib.load(os.path.join(MODELS_DIR, "meta.pkl"))
    loss_h = np.load(os.path.join(MODELS_DIR, "loss_history.npy"))
    return model, scaler, meta, loss_h

@st.cache_data(show_spinner=False)
def load_data_cached():
    return load_raw_data()

@st.cache_data(show_spinner=False)
def get_metrics_cached(_model, _X_test, _y_test):
    return get_all_metrics(_model, _X_test, _y_test)


def metric_card(label, value, color="#3b82f6", sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{color}">{value}</div>
      <div class="metric-label">{label}</div>
      {sub_html}
    </div>"""


def prob_bar(pct, color):
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard(model, meta, loss_h, df):
    st.markdown('<div class="section-header">Tableau de Bord — Surveillance des Transactions</div>',
                unsafe_allow_html=True)

    X_test, y_test = meta["X_test"], meta["y_test"]
    metrics = get_metrics_cached(model, X_test, y_test)
    stats   = get_dataset_stats(df)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    cards = [
        (c1, "Transactions Totales",  f"{stats['total']:,}",          "#ffffff", None),
        (c2, "Frauduleuses",          f"{stats['fraud']:,}",           "#cc2222", f"{stats['fraud_pct']}%"),
        (c3, "Legitimes",             f"{stats['legitimate']:,}",      "#22aa55", None),
        (c4, "Accuracy",              f"{metrics['accuracy']*100:.1f}%","#ffffff",None),
        (c5, "Recall (Fraude)",       f"{metrics['recall']*100:.1f}%", "#ddaa33","Most important"),
        (c6, "Taux Detection",        f"{metrics['fraud_detection_rate']}%","#22aa55",None),
    ]
    for col, label, value, color, sub in cards:
        col.markdown(metric_card(label, value, color, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Visualisations Analytiques</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_class_distribution(df), use_container_width=True)
    with c2:
        st.pyplot(plot_amount_distribution(df), use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(plot_probability_distribution(metrics["y_proba"], y_test), use_container_width=True)
    with c2:
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]), use_container_width=True)

    st.markdown('<div class="section-header">Courbe d\'Apprentissage — Gradient Descent</div>',
                unsafe_allow_html=True)
    st.pyplot(plot_loss_curve(loss_h), use_container_width=True)
    st.markdown(
        '<div class="info-box">💡 <b>Interprétation :</b> La courbe montre la réduction du coût '
        'J(θ) au fil des époques. Le modèle apprend en ajustant ses paramètres à chaque étape '
        'de la descente de gradient jusqu\'à converger vers le minimum.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">Evolution dans le Temps</div>', unsafe_allow_html=True)
    st.pyplot(plot_fraud_over_time(df), use_container_width=True)

    st.markdown('<div class="section-header">Resume des Metriques</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        data = {
            "Métrique": ["Accuracy","Precision","Recall","F1-Score","Log-Loss","AUC-ROC"],
            "Valeur": [
                f"{metrics['accuracy']*100:.2f}%",
                f"{metrics['precision']*100:.2f}%",
                f"{metrics['recall']*100:.2f}%",
                f"{metrics['f1']*100:.2f}%",
                f"{metrics['log_loss']:.4f}",
                f"{metrics['auc_roc']:.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    with col2:
        st.markdown(f"""
        <div class="info-box">
        <b>Matrice de Confusion — Détail</b><br><br>
        ✅ <b>Vrais Positifs (TP)</b> : {metrics['TP']:,} — Fraudes détectées<br>
        ✅ <b>Vrais Négatifs (TN)</b> : {metrics['TN']:,} — Légitimes correctes<br>
        ⚠️ <b>Faux Positifs (FP)</b> : {metrics['FP']:,} — Fausses alarmes<br>
        🚨 <b>Faux Négatifs (FN)</b> : {metrics['FN']:,} — Fraudes manquées !<br>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYZE TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════

def _rule_based_explanation(amount, hour, freq, balance, country, tx_type, failed, prob):
    reasons = []
    if amount > 5000:
        reasons.append(f"Montant tres eleve ({amount:.0f} EUR) — depasse le seuil normal")
    elif amount > 1000:
        reasons.append(f"Montant eleve ({amount:.0f} EUR) — legerement suspect")
    if hour < 5 or hour > 22:
        reasons.append(f"Heure inhabituelle ({hour}h) — transaction nocturne")
    if freq > 30:
        reasons.append(f"Frequence mensuelle tres elevee ({freq} transactions/mois)")
    if amount > balance * 0.8 and balance > 0:
        reasons.append(f"Montant ({amount:.0f} EUR) proche du solde ({balance:.0f} EUR) — risque eleve")
    if country in ["Asie", "Afrique", "Amerique du Sud", "Pays inconnu"]:
        reasons.append(f"Zone geographique a risque ({country})")
    if tx_type in ["Retrait", "Retrait ATM"]:
        reasons.append("Retrait — canal frequemment utilise pour la fraude")
    if failed > 2:
        reasons.append(f"Nombre de tentatives echouees eleve ({failed}) — activite suspecte")
    if prob > 0.7:
        reasons.append(f"Score de fraude eleve du modele ({prob*100:.1f}%)")
    if not reasons:
        reasons.append("Aucun indicateur de fraude detecte — transaction conforme")
    return reasons


def _build_input_vector(amount, hour, freq, balance, country, tx_type, failed, feature_names):
    """Map form inputs to the model's feature vector."""
    country_map = {"Europe": 0.0, "Amerique du Nord": 0.1, "Asie": 0.5, "Afrique": 0.7,
                   "Amerique du Sud": 0.6, "Maroc (habituel)": 0.05, "Pays inconnu": 0.9}
    tx_map      = {"Virement": 0.0, "Paiement en ligne": 0.2, "Retrait": 0.8,
                   "Achat en magasin": 0.1, "Achat en ligne": 0.2, "Retrait ATM": 0.85}

    # Normalise user inputs to typical V-feature ranges
    amount_norm  = (amount - 88) / 250
    hour_norm    = (hour - 12) / 8
    freq_norm    = (freq - 15) / 15
    balance_norm = (balance - 2000) / 2000
    country_val  = country_map.get(country, 0)
    tx_val       = tx_map.get(tx_type, 0)
    failed_norm  = (failed - 1) / 3

    row = {f: 0.0 for f in feature_names}
    if "Amount_scaled" in row:  row["Amount_scaled"] = amount_norm
    if "Time_scaled"   in row:  row["Time_scaled"]   = hour_norm
    if "V1"  in row:  row["V1"]  = -amount_norm * 0.8
    if "V2"  in row:  row["V2"]  = freq_norm * 0.5
    if "V3"  in row:  row["V3"]  = -amount_norm * 0.6
    if "V4"  in row:  row["V4"]  = country_val * 1.5 + failed_norm
    if "V10" in row:  row["V10"] = -amount_norm * 0.9
    if "V12" in row:  row["V12"] = -amount_norm * 1.1
    if "V14" in row:  row["V14"] = -amount_norm * 1.3 - tx_val
    if "V17" in row:  row["V17"] = -freq_norm * 0.7
    if "V11" in row:  row["V11"] = -balance_norm * 0.4
    if "V16" in row:  row["V16"] = -tx_val * 0.5

    return pd.DataFrame([row])[feature_names]


def page_analyze(model, meta):
    st.markdown('<div class="section-header">Analyser une Transaction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Entrez les informations d\'une transaction pour obtenir une '
        'prédiction en temps réel. Le modèle calcule la probabilité de fraude via la fonction sigmoïde.</div>',
        unsafe_allow_html=True)

    feature_names = meta["feature_names"]

    # ── Preset scenarios (on_click callbacks write directly to widget keys) ────
    PRESETS = {
        "normal": dict(f_amount=95.0,   f_hour=13, f_freq=12, f_balance=3200.0,
                       f_country="Maroc (habituel)",  f_txtype="Achat en ligne",   f_failed=0),
        "fraud":  dict(f_amount=4750.0,  f_hour=3,  f_freq=87, f_balance=320.0,
                       f_country="Pays inconnu",      f_txtype="Retrait ATM",      f_failed=4),
        "border": dict(f_amount=890.0,   f_hour=22, f_freq=41, f_balance=680.0,
                       f_country="Europe",            f_txtype="Virement",         f_failed=1),
    }

    def _apply_preset(preset_name):
        for k, v in PRESETS[preset_name].items():
            st.session_state[k] = v

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.button("Transaction normale",     key="sc_normal", on_click=_apply_preset, args=("normal",))
    with sc2:
        st.button("Transaction frauduleuse",  key="sc_fraud",  on_click=_apply_preset, args=("fraud",))
    with sc3:
        st.button("Cas suspect",              key="sc_border", on_click=_apply_preset, args=("border",))

    # ── Dropdown options ──────────────────────────────────────────────────────
    country_options = ["Europe", "Amerique du Nord", "Asie", "Afrique",
                       "Amerique du Sud", "Maroc (habituel)", "Pays inconnu"]
    tx_options      = ["Virement", "Paiement en ligne", "Retrait", "Achat en magasin",
                       "Achat en ligne", "Retrait ATM"]

    # ── Transaction form ──────────────────────────────────────────────────────
    with st.form("transaction_form"):
        amount = st.number_input("Montant de la transaction (EUR)",
                                 min_value=0.0, max_value=100000.0,
                                 value=st.session_state.get("f_amount", 890.0),
                                 step=10.0, key="f_amount")

        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Heure (0 = minuit, 23 = 23h)",
                             0, 23,
                             value=st.session_state.get("f_hour", 22),
                             key="f_hour")
        with c2:
            freq = st.slider("Transactions ce mois (frequence)",
                             1, 100,
                             value=st.session_state.get("f_freq", 41),
                             key="f_freq")

        c3, c4 = st.columns(2)
        with c3:
            balance = st.number_input("Solde avant transaction (EUR)",
                                      min_value=0.0, max_value=1000000.0,
                                      value=st.session_state.get("f_balance", 680.0),
                                      step=10.0, key="f_balance")
        with c4:
            country = st.selectbox("Pays de la transaction", country_options,
                                   index=country_options.index(
                                       st.session_state.get("f_country", "Europe")),
                                   key="f_country")

        c5, c6 = st.columns(2)
        with c5:
            tx_type = st.selectbox("Type de transaction", tx_options,
                                   index=tx_options.index(
                                       st.session_state.get("f_txtype", "Virement")),
                                   key="f_txtype")
        with c6:
            failed = st.number_input("Nombre de tentatives echouees",
                                     min_value=0, max_value=20,
                                     value=st.session_state.get("f_failed", 1),
                                     step=1, key="f_failed")

        submitted = st.form_submit_button("Analyser la Transaction", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if submitted:
        X_input = _build_input_vector(amount, hour, freq, balance, country, tx_type, failed, feature_names)
        prob    = float(model.predict_proba(X_input)[0, 1])
        pred    = int(model.predict(X_input)[0])
        prob_pct = round(prob * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        res_col, gauge_col = st.columns([2, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="badge-fraud">
                  <div style="font-size:1.8rem;font-weight:700;letter-spacing:0.05em;color:#ff4444;margin:8px 0">TRANSACTION FRAUDULEUSE</div>
                  <div style="font-size:0.95rem;color:#ff9999">Probabilite de fraude : <b>{prob_pct}%</b></div>
                  <div style="font-size:0.8rem;color:#ff9999;margin-top:4px">Niveau de risque : CRITIQUE</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-legit">
                  <div style="font-size:1.8rem;font-weight:700;letter-spacing:0.05em;color:#44cc88;margin:8px 0">TRANSACTION LEGITIME</div>
                  <div style="font-size:0.95rem;color:#99ffbb">Probabilite de fraude : <b>{prob_pct}%</b></div>
                  <div style="font-size:0.8rem;color:#99ffbb;margin-top:4px">Niveau de risque : NORMAL</div>
                </div>""", unsafe_allow_html=True)

        with gauge_col:
            bar_color = "#ef4444" if pred == 1 else "#22c55e"
            st.markdown(f"""
            <div class="info-box" style="text-align:center">
              <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase">Score de fraude</div>
              <div style="font-size:3rem;font-weight:700;color:{bar_color}">{prob_pct}%</div>
              {prob_bar(prob_pct, bar_color)}
              <div style="font-size:0.75rem;color:#64748b;margin-top:6px">σ(z) = {prob:.4f}</div>
            </div>""", unsafe_allow_html=True)

        # ── XAI explanation ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Analyse de la Decision</div>',
                    unsafe_allow_html=True)
        reasons = _rule_based_explanation(amount, hour, freq, balance, country, tx_type, failed, prob)
        box_class = "danger-box" if pred == 1 else "info-box"
        items = "".join(f"<div style='margin:4px 0'>• {r}</div>" for r in reasons)
        st.markdown(f'<div class="{box_class}">{items}</div>', unsafe_allow_html=True)

        # ── Detail breakdown ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">Detail de la Transaction</div>',
                    unsafe_allow_html=True)
        detail = pd.DataFrame({
            "Paramètre": ["Montant","Heure","Tx ce mois","Solde","Pays","Type","Tentatives echouees"],
            "Valeur":    [f"{amount:.2f} €", f"{hour}h00", str(freq), f"{balance:.2f} €", country, tx_type, str(failed)],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFORMATION
# ══════════════════════════════════════════════════════════════════════════════

def page_model_info():
    st.markdown('<div class="section-header">Informations sur le Modele — Concepts de Machine Learning</div>',
                unsafe_allow_html=True)

    tabs = st.tabs([
        "Apprentissage Supervise",
        "Classification",
        "Perceptron",
        "Fonction Sigmoide",
        "Gradient Descent",
        "Fonction de Cout",
        "Evaluation du Modele",
    ])

    # ── Tab 1: Supervised learning ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        ## Apprentissage Supervise

        L'apprentissage supervisé est un paradigme d'apprentissage automatique où le modèle
        est entraîné sur des **données étiquetées** — c'est-à-dire des exemples pour lesquels
        la réponse correcte est connue.

        ### Principe
        - **Entrée (X)** : les caractéristiques de la transaction (montant, heure, V1…V28)
        - **Sortie (y)** : l'étiquette cible (0 = légitime, 1 = frauduleuse)
        - **Objectif** : apprendre une fonction f telle que f(X) ≈ y

        ### Dans notre projet
        Le modèle apprend à partir de **20 000 transactions historiques étiquetées**.
        Il ajuste ses paramètres θ pour minimiser l'erreur de prédiction.

        ```
        Données étiquetées  →  Modèle  →  Prédiction  →  Erreur  →  Mise à jour θ
        ```
        """)
        st.markdown("""
        <div class="info-box">
        💡 <b>Analogie :</b> C'est comme un étudiant qui apprend à détecter la fraude en
        étudiant des milliers d'exemples de transactions avec les bonnes réponses.
        Plus il étudie, plus il devient précis.
        </div>""", unsafe_allow_html=True)

    # ── Tab 2: Classification ──────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        ## Classification Binaire

        La **classification binaire** consiste à prédire l'appartenance d'un exemple
        à l'une de deux classes possibles.

        | Classe | Signification |
        |--------|---------------|
        | **0** | Transaction légitime ✅ |
        | **1** | Transaction frauduleuse 🚨 |

        ### Défi : Déséquilibre des classes
        Dans notre dataset :
        - ~99.83% de transactions légitimes
        - ~0.17% de transactions frauduleuses

        ➡️ Un modèle naïf qui prédit toujours "légitime" aurait 99.83% d'accuracy —
        mais détecter **0 fraude** ! C'est pourquoi on utilise `class_weight='balanced'`
        et on s'intéresse principalement au **Recall**.
        """)

    # ── Tab 3: Perceptron ──────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        ## Le Perceptron — Neurone Artificiel

        Le perceptron est l'unité fondamentale des réseaux de neurones.
        Notre modèle de régression logistique est équivalent à **un seul perceptron**.

        ### Architecture

        ```
        Entrées     Poids      Somme         Activation    Sortie
        x₁ ──────── w₁ ──┐
        x₂ ──────── w₂ ──┤
        ...              ├──► z = θᵀx + b ──► σ(z) ──► ŷ ∈ [0,1]
        x₂₈ ─────── w₂₈─┘
        ```

        ### Calcul
        1. **Combinaison linéaire** : z = w₁x₁ + w₂x₂ + … + w₂₈x₂₈ + b
        2. **Activation sigmoïde** : ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
        3. **Décision** : si ŷ ≥ 0.5 → Fraude, sinon → Légitime

        ### Paramètres appris
        - **Poids w** : importance de chaque feature
        - **Biais b** : seuil d'activation du neurone
        """)

    # ── Tab 4: Sigmoid ────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## Fonction Sigmoide — Activation du Perceptron")
        st.markdown("""
        La sigmoïde transforme n'importe quelle valeur réelle en une **probabilité** entre 0 et 1.

        ### Formule
        """)
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        st.markdown("""
        ### Propriétés clés
        | Valeur de z | σ(z) | Interprétation |
        |-------------|------|----------------|
        | z ≪ 0 | ≈ 0 | Transaction très probablement légitime |
        | z = 0 | 0.5 | Incertitude maximale |
        | z ≫ 0 | ≈ 1 | Transaction très probablement frauduleuse |
        """)
        st.pyplot(plot_sigmoid(), use_container_width=True)

    # ── Tab 5: Gradient Descent ────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("## Descente de Gradient")
        st.markdown("""
        La descente de gradient est l'algorithme d'optimisation qui permet au modèle
        d'**apprendre** en réduisant progressivement l'erreur de prédiction.

        ### Mise à jour des paramètres
        """)
        st.latex(r"\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)")
        st.markdown("""
        - **θ** : paramètres du modèle (poids)
        - **α** : taux d'apprentissage (learning rate)
        - **∇J(θ)** : gradient de la fonction de coût

        ### Intuition
        Imaginez une bille sur une surface courbe : elle roule naturellement vers le bas
        (le minimum) en suivant la pente la plus abrupte.
        """)
        st.pyplot(plot_gradient_descent_illustration(), use_container_width=True)

    # ── Tab 6: Cost function ───────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("## Fonction de Cout — Binary Cross-Entropy")
        st.markdown("Notre modèle minimise la **Binary Cross-Entropy** (entropie croisée binaire) :")
        st.latex(r"J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]")
        st.markdown("""
        ### Interprétation
        - Si **y=1** (fraude) et **ŷ→1** → coût ≈ 0 ✅ (bonne prédiction)
        - Si **y=1** (fraude) et **ŷ→0** → coût → ∞ 🚨 (très mauvaise prédiction)
        - Si **y=0** (légitime) et **ŷ→0** → coût ≈ 0 ✅

        ### Pourquoi cette fonction ?
        Elle est **convexe** — elle possède un unique minimum global vers lequel
        la descente de gradient converge toujours.
        """)

    # ── Tab 7: Evaluation ─────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("""
        ## Metriques d'Evaluation

        ### Matrice de Confusion
        | | Prédit Légitime | Prédit Frauduleux |
        |---|---|---|
        | **Réel Légitime** | ✅ Vrai Négatif (TN) | ⚠️ Faux Positif (FP) |
        | **Réel Frauduleux** | 🚨 Faux Négatif (FN) | ✅ Vrai Positif (TP) |

        ### Métriques clés
        """)
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown("""
        ### Pourquoi le Recall est-il prioritaire ?
        """)
        st.markdown("""
        <div class="danger-box">
        🚨 <b>Faux Négatif (FN)</b> = une fraude NON détectée = perte financière directe<br>
        ⚠️ <b>Faux Positif (FP)</b> = une fausse alarme = inconvénient mineur pour le client<br><br>
        Dans la détection de fraude, <b>manquer une fraude coûte beaucoup plus cher</b>
        que déclencher une fausse alarme. On optimise donc le Recall en priorité.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Sidebar navigation & page routing
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px">
          <div style="font-size:1.3rem;font-weight:700;color:#ffffff;letter-spacing:0.08em">FRAUDGUARD AI</div>
          <div style="font-size:0.75rem;color:#555555;margin-top:4px;letter-spacing:0.06em">DETECTION DE FRAUDE BANCAIRE</div>
        </div>
        <hr style="border-color:#2a2a2a;margin:12px 0">
        """, unsafe_allow_html=True)

        st.markdown("## Navigation")
        page = st.radio(
            "Choisir une page",
            ["Dashboard", "Analyser une Transaction", "Informations Modele"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border-color:#2a2a2a;margin:16px 0">
        <div style="font-size:0.72rem;color:#444444;text-align:center">
          Modele : Regression Logistique<br>
          Algorithme : Descente de Gradient<br>
          Dataset : Kaggle Credit Card Fraud<br>
          <span style="color:#2a2a2a">---------------------</span><br>
          Projet ML Universitaire
        </div>""", unsafe_allow_html=True)

    # ── Load artefacts ─────────────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "models", "fraud_model.pkl")
    if not os.path.exists(model_path):
        st.error("⚠️ Modèle non trouvé. Exécutez d'abord : `python train_model.py`")
        st.stop()

    with st.spinner("Chargement du modèle…"):
        model, scaler, meta, loss_h = load_model_artifacts()

    with st.spinner("Chargement des données…"):
        df = load_data_cached()

    # ── Route ──────────────────────────────────────────────────────────────────
    if page == "Dashboard":
        page_dashboard(model, meta, loss_h, df)
    elif page == "Analyser une Transaction":
        page_analyze(model, meta)
    elif page == "Informations Modele":
        page_model_info()


if __name__ == "__main__":
    main()

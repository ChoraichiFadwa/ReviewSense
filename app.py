"""Démo interactive du projet « Analyse de la réputation des entreprises ».

    streamlit run app.py

L'app ne réentraîne rien : elle lit le modèle (models/) et les sorties du
pipeline (outputs/, reports/) produits par `python -m src.pipeline`.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT, REPORTS, MODEL = ROOT / "outputs", ROOT / "reports", ROOT / "models" / "sentiment_model.joblib"

# Palette : bleu = positif, rouge = négatif (paire lisible par les daltoniens), or = étoiles
INK, MUTED, LINE, PAPER = "#172033", "#5f6b80", "#e3e7ef", "#ffffff"
POS, NEG, GOLD = "#2459c8", "#c93a3a", "#e9a23b"

st.set_page_config(page_title="Réputation des entreprises", page_icon="⭐", layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, .stMarkdown, .stDataFrame, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
h1, h2, h3, h4, .display {{ font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif !important; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* en-tête */
.hero {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 2rem; align-items: end;
        padding-bottom: 1.4rem; border-bottom: 1px solid {LINE}; margin-bottom: .6rem; }}
.hero .htitle {{ font-family: 'Bricolage Grotesque', sans-serif; color: {INK}; font-size: 2.6rem; line-height: 1.05; font-weight: 700; margin: 0 0 .7rem !important; padding: 0 !important;
            letter-spacing: -.02em; }}
.hero p {{ color: {MUTED}; font-size: 1rem; line-height: 1.55; margin: 0; max-width: 34rem; }}
.stars {{ color: {GOLD}; letter-spacing: .12em; font-size: 1.05rem; margin-bottom: .5rem; }}
.score {{ display: grid; grid-template-columns: repeat(2, 1fr); border-top: 1px solid {LINE}; border-left: 1px solid {LINE}; }}
.score div {{ border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: .75rem .9rem; background: {PAPER}; }}
.score b {{ display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.65rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.score span {{ font-size: .8rem; color: {MUTED}; }}

/* onglets */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 0; font-weight: 500; }}

/* chaîne de traitement : vraie séquence, donc numérotée */
.flow {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 0; margin: 1.2rem 0 1.8rem; }}
.flow .st {{ position: relative; padding: 2.1rem .7rem 0 0; }}
.flow .st::before {{ content: ""; position: absolute; top: .72rem; left: 1.6rem; right: 0; height: 2px; background: {LINE}; }}
.flow .st:last-child::before {{ display: none; }}
.flow .n {{ position: absolute; top: 0; left: 0; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: {INK}; color: #fff;
           font-size: .78rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.flow .t {{ font-size: .82rem; color: {MUTED}; }}
.flow .v {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; color: {INK}; margin: .15rem 0 .3rem; line-height: 1.2; }}
.flow .d {{ font-size: .8rem; color: {MUTED}; line-height: 1.4; }}

/* constats */
.note {{ border-left: 3px solid {INK}; padding: .2rem 0 .2rem 1rem; margin: 0 0 1.1rem; }}
.note.find {{ border-color: {GOLD}; }}
.note h4 {{ font-size: 1.02rem; margin: 0 0 .25rem; font-weight: 700; }}
.note p {{ margin: 0; color: #2c3446; font-size: .93rem; line-height: 1.55; }}

/* testeur : l'élément mis en avant */
.review {{ background: {PAPER}; border: 1px solid {LINE}; border-radius: 14px; padding: 1.4rem 1.6rem 1.2rem;
           box-shadow: 0 10px 30px -18px rgba(23,32,51,.35); }}
.review .txt {{ font-size: 1.2rem; line-height: 2.05; color: {INK}; }}
.review .w {{ border-radius: 5px; padding: 1px 4px; }}
.verdict {{ display: flex; align-items: baseline; gap: .8rem; margin-bottom: .9rem; }}
.verdict .lab {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.9rem; font-weight: 700; }}
.verdict .p {{ color: {MUTED}; font-size: .95rem; }}
.gauge {{ position: relative; height: 10px; border-radius: 6px; margin: 1.1rem 0 .35rem;
          background: linear-gradient(90deg, {NEG} 0%, #e7c9c9 47%, #c9d5ee 53%, {POS} 100%); }}
.gauge .mk {{ position: absolute; top: -5px; width: 4px; height: 20px; background: {INK}; border-radius: 2px; transform: translateX(-2px); }}
.gauge-l {{ display: flex; justify-content: space-between; font-size: .75rem; color: {MUTED}; }}
.legend {{ font-size: .82rem; color: {MUTED}; margin-top: .9rem; }}
.legend i {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 3px; vertical-align: -1px; margin: 0 .3rem 0 .8rem; }}
.step {{ display: grid; grid-template-columns: 2rem 1fr; gap: .2rem .6rem; margin: 1.3rem 0 .5rem; }}
.step .n {{ width: 1.6rem; height: 1.6rem; border-radius: 50%; border: 2px solid {INK}; font-weight: 600; font-size: .85rem;
            display: flex; align-items: center; justify-content: center; }}
.step h4 {{ margin: 0; font-size: 1.05rem; font-weight: 700; padding-top: .1rem; }}
.step p {{ grid-column: 2; margin: 0; color: #2c3446; font-size: .93rem; line-height: 1.55; max-width: 46rem; }}
.calc {{ font-family: 'IBM Plex Sans', sans-serif; background: {PAPER}; border: 1px solid {LINE}; border-radius: 8px;
         padding: .8rem 1rem; margin-left: 2.6rem; font-size: .95rem; }}
.calc b {{ font-weight: 600; }}

/* encadré d'avertissement sobre */
.warn {{ background: #fdf6e9; border: 1px solid #f1dcb3; border-radius: 8px; padding: .8rem 1rem; color: #5b4317;
         font-size: .92rem; line-height: 1.5; margin: .8rem 0 1.2rem; }}
.small {{ font-size: .85rem; color: {MUTED}; }}

@media (max-width: 900px) {{
  .hero {{ grid-template-columns: 1fr; }}
  .flow {{ grid-template-columns: 1fr; }}
  .flow .st::before {{ display: none; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(markup: str) -> None:
    """Affiche du HTML (lignes collées : sinon le Markdown prend l'indentation pour du code)."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------------------- chargement


@st.cache_resource
def load_model():
    model = joblib.load(MODEL)
    inner = model.named_steps["model"]
    vec, clf = inner.named_steps["tfidf"], inner.named_steps["clf"]
    sign = 1 if clf.classes_[1] == 1 else -1
    return {
        "cleaner": model.named_steps["clean"],
        "vec": vec,
        "coef": clf.coef_.ravel() * sign,
        "intercept": float(clf.intercept_[0]) * sign,
        "names": np.asarray(vec.get_feature_names_out()),
        "token_re": re.compile(vec.token_pattern),
    }


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def fmt(n) -> str:
    return f"{n:,.0f}".replace(",", " ")


def pct(x, d=1) -> str:
    return f"{x * 100:.{d}f} %".replace(".", ",")


def dec(x, d=3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def sg(x) -> str:
    return f"{x:+.2f}".replace(".", ",")


if not MODEL.exists() or not (OUT / "metrics.json").exists():
    st.error("Le modèle ou les résultats sont introuvables. Lancez d'abord `python -m src.pipeline`.")
    st.stop()

M = load_model()
metrics = load_json(OUT / "metrics.json")
quality = load_json(REPORTS / "data_quality_report.json")
rep = load_json(OUT / "reputation_summary.json")
T = metrics["test"]
AV, AP = quality["avant_nettoyage"], quality["après_nettoyage"]
N_QUARANTINE = sum(quality["rejets_par_motif"].values())

# --------------------------------------------------------------------------- explication d'un avis


def explain(text: str) -> dict:
    """Décompose la prédiction : texte nettoyé -> termes reconnus -> contributions -> probabilité.
    Régression logistique : score = constante + somme(poids du terme x valeur TF-IDF)."""
    cleaned = M["cleaner"].transform([text])[0]
    X = M["vec"].transform([cleaned])
    idx, vals = X.indices, X.data
    contrib = vals * M["coef"][idx]
    terms = pd.DataFrame({"terme": M["names"][idx], "contribution": contrib})
    terms["type"] = np.where(terms["terme"].str.contains(" "), "bigramme", "mot")
    logit = M["intercept"] + contrib.sum()
    tokens = M["token_re"].findall(cleaned.lower())
    known = set(terms["terme"])
    return {"cleaned": cleaned, "terms": terms.sort_values("contribution"), "logit": logit,
            "proba": 1 / (1 + np.exp(-logit)), "unknown": sorted({t for t in tokens if t not in known})}


def highlight(cleaned: str, terms: pd.DataFrame) -> str:
    """Teinte chaque mot selon son poids propre (mot seul). Les paires de mots (bigrammes),
    qui peuvent inverser le sens (« not good »), sont détaillées dans le graphique."""
    words = terms[terms["type"] == "mot"]
    per_word = dict(zip(words["terme"], words["contribution"]))
    scale = max([abs(v) for v in per_word.values()] + [1e-9])
    out = []
    for piece in re.split(r"(\w+)", cleaned):
        v = per_word.get(piece.lower())
        if not piece or v is None or abs(v) < 0.04 * scale:
            out.append(html.escape(piece))
            continue
        a = 0.12 + 0.55 * abs(v) / scale
        rgb = "201,58,58" if v < 0 else "36,89,200"
        out.append(f'<span class="w" title="contribution {sg(v)}" style="background:rgba({rgb},{a:.2f})">'
                   f"{html.escape(piece)}</span>")
    return "".join(out)


def bar_chart(df: pd.DataFrame, x: str, y: str, color_field: str, domain: list, colors: list,
              x_title: str, height: int, fmt_tip: str = "+.2f", labels: str | None = None):
    """Barres horizontales ; `labels` = format d3 des valeurs affichées au bout des barres."""
    layers = [{"mark": {"type": "bar", "cornerRadiusEnd": 3, "height": {"band": 0.7}},
               "encoding": {"color": {"field": color_field, "type": "nominal",
                                      "scale": {"domain": domain, "range": colors}, "legend": None},
                            "tooltip": [{"field": y}, {"field": x, "format": fmt_tip}]}}]
    if labels:
        layers.append({"mark": {"type": "text", "fontSize": 11, "color": MUTED,
                                "align": {"expr": f"datum['{x}'] < 0 ? 'right' : 'left'"},
                                "dx": {"expr": f"datum['{x}'] < 0 ? -4 : 4"}},
                       "encoding": {"text": {"field": x, "format": labels}}})
    st.vega_lite_chart(df, {
        "layer": layers,
        "encoding": {
            "y": {"field": y, "type": "nominal", "sort": None, "title": None,
                  "axis": {"labelOverlap": False, "labelLimit": 240, "labelFontSize": 12, "labelColor": INK,
                           "ticks": False, "domain": False}},
            "x": {"field": x, "type": "quantitative", "title": x_title,
                  "scale": {"padding": 34 if labels else 0},
                  "axis": {"gridColor": LINE, "labelColor": MUTED, "titleColor": MUTED, "domain": False,
                           "titleFontWeight": "normal"}},
        },
        "height": height,
        "config": {"view": {"stroke": None}, "font": "IBM Plex Sans, system-ui, sans-serif",
                   "locale": {"number": {"decimal": ",", "thousands": "\u202f", "grouping": [3],
                                         "currency": ["", " €"]}}},
    }, width="stretch")


# --------------------------------------------------------------------------- en-tête

H(f"""
<div class="hero">
  <div>
    <div class="stars" aria-hidden="true">★★★★★</div>
    <div class="htitle" role="heading" aria-level="1">Réputation des entreprises</div>
    <p>Un modèle lit un avis client et dit s'il est positif (1) ou négatif (-1). On agrège ensuite ces
    prédictions pour mesurer la réputation de chaque entreprise. Données : {fmt(AV['lignes'])} avis
    Trustpilot en anglais, sur {fmt(AV['entreprises distinctes'])} entreprises.</p>
  </div>
  <div class="score">
    <div><b>{dec(T['accuracy'])}</b><span>accuracy sur le test</span></div>
    <div><b>{dec(T['f1_macro'])}</b><span>F1 macro (référence naïve : 0,366)</span></div>
    <div><b>{dec(rep['correlation_predit_reel'])}</b><span>corrélation réputation prédite / réelle</span></div>
    <div><b>{fmt(N_QUARANTINE)}</b><span>avis écartés, chacun avec son motif</span></div>
  </div>
</div>
""")

tabs = st.tabs(["Le projet", "Tester un avis", "Réputation", "Qualité des données", "Modèles et erreurs"])

# --------------------------------------------------------------------------- 1. le projet

with tabs[0]:
    sp = quality["split"]["splits"]
    steps = [
        ("Ingestion", f"{fmt(AV['lignes'])} avis", "Fichier brut conservé tel quel, empreinte SHA-256, contrat de colonnes"),
        ("Contrôle qualité", f"{len(quality['journal_nettoyage'])} règles", "Mesures avant / après, rapport qualité"),
        ("Nettoyage", f"{fmt(AP['lignes'])} gardés", f"{fmt(N_QUARANTINE)} en quarantaine avec un motif"),
        ("Découpage", f"{sp['test']['entités']} entreprises test", "Groupé par entreprise, aucune fuite"),
        ("Modèle", f"F1 {dec(T['f1_macro'])}", "8 modèles comparés, règle de choix fixée avant le test"),
        ("Réputation", f"r = {dec(rep['correlation_predit_reel'])}", "Score de Wilson par entreprise et par secteur"),
        ("Datamart", "Schéma en étoile", "SQLite, vues prêtes pour la BI, contrôles de recette"),
    ]
    H('<div class="flow">' + "".join(
        f'<div class="st"><div class="n">{i}</div><div class="t">{t}</div><div class="v">{v}</div>'
        f'<div class="d">{d}</div></div>' for i, (t, v, d) in enumerate(steps, 1)) + "</div>")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### La donnée d'abord")
        H(f"""
<div class="note"><h4>Les avis à 3 étoiles sont écartés</h4>
<p>Ni positifs ni négatifs : les ranger d'un côté ajouterait du bruit dans les labels.
{fmt(quality['rejets_par_motif'].get('excluded_label', 0))} avis concernés, tracés en quarantaine.</p></div>
<div class="note"><h4>Le titre n'est ajouté que s'il apporte une information</h4>
<p>58 768 titres ne sont que le début de l'avis coupé par « … » : les ajouter compterait ces mots deux fois.</p></div>
<div class="note"><h4>Les mots-outils sont conservés</h4>
<p>« not » est le terme négatif le plus important du modèle. Le supprimer transformerait « not good » en « good ».</p></div>
<div class="note"><h4>Aucune fuite entre entraînement et test</h4>
<p>Le test porte sur des entreprises jamais vues, comme une nouvelle entreprise à analyser.
Un contrôle bloque le pipeline si une entreprise ou un texte apparaît des deux côtés.</p></div>
""")
    with right:
        st.markdown("### Ce que l'audit des données a révélé")
        H("""
<div class="note find"><h4>La source est échantillonnée</h4>
<p>Aucune entreprise n'a plus de 20 avis pour une même note. Le modèle n'en souffre pas, mais les proportions
par entreprise ne sont pas la vraie réputation : la méthode est validée, pas les valeurs absolues.</p></div>
<div class="note find"><h4>Beaucoup d'erreurs « sûres » sont des erreurs de label</h4>
<p>Des textes clairement positifs notés 1 étoile. Le modèle a raison, la donnée a tort :
ces incohérences peuvent devenir un contrôle qualité automatique.</p></div>
""")
        st.markdown("### Le modèle")
        H("""
<div class="note"><h4>TF-IDF et régression logistique</h4>
<p>Rapide, interprétable, et donne une probabilité utilisée pour le score de réputation.</p></div>
<div class="note"><h4>Le nettoyage est intégré au modèle</h4>
<p>Le même traitement s'applique à l'entraînement et en production : pas d'écart entre les deux.</p></div>
""")
    st.markdown('<p class="small">Pour voir le modèle décider, ouvrez l\'onglet « Tester un avis ».</p>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- 2. tester un avis

EXAMPLES = {
    "Positif": "Delivery was quick and the staff were really helpful,will definitely order again!!!",
    "Négatif": "Parcel arrived damaged and nobody answered my emails. Never again.",
    "Négation": "The product is not good and the support was not helpful at all.",
    "Nuancé": "Good product but the price went up without any warning, not happy about that.",
}

with tabs[1]:
    if "review" not in st.session_state:
        st.session_state["review"] = EXAMPLES["Négation"]
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("### Écrivez un avis")
        st.markdown('<p class="small">En anglais : le modèle a été entraîné sur des avis en anglais. '
                    "Ou partez d'un exemple.</p>", unsafe_allow_html=True)
        b = st.columns(2)
        for i, (label, ex) in enumerate(EXAMPLES.items()):
            if b[i % 2].button(label, width="stretch"):
                st.session_state["review"] = ex
        text = st.text_area("Avis client", key="review", height=140, label_visibility="collapsed")

    r = explain(text) if text.strip() else None
    with right:
        if r is None:
            st.info("Écrivez un avis pour voir la prédiction.")
        else:
            pred = 1 if r["proba"] >= 0.5 else -1
            color = POS if pred == 1 else NEG
            H(f"""
<div class="review">
  <div class="verdict"><span class="lab" style="color:{color}">{"Positif (1)" if pred == 1 else "Négatif (-1)"}</span>
  <span class="p">probabilité d'être positif : <b>{pct(r['proba'])}</b></span></div>
  <div class="txt">{highlight(r['cleaned'], r['terms'])}</div>
  <div class="gauge"><div class="mk" style="left:{r['proba'] * 100:.1f}%"></div></div>
  <div class="gauge-l"><span>négatif</span><span>seuil 50 %</span><span>positif</span></div>
  <div class="legend">Teinte de chaque mot pris seul : <i style="background:rgba(36,89,200,.5)"></i>vers positif
  <i style="background:rgba(201,58,58,.5)"></i>vers négatif. Les paires de mots sont détaillées plus bas.</div>
</div>
""")

    if r is not None:
        terms = r["terms"]
        st.markdown("### Comment le modèle arrive à cette décision")

        same = r["cleaned"] == text.strip()
        H(f"""<div class="step"><div class="n">1</div><h4>Nettoyage du texte</h4>
<p>Le même traitement qu'à l'entraînement : ponctuation collée, lettres répétées, HTML, liens.
{"Ici, le texte n'avait pas besoin d'être modifié." if same else "Le texte a été modifié :"}</p></div>""")
        if not same:
            a, c2 = st.columns(2)
            a.caption("Texte reçu")
            a.code(text, language=None, wrap_lines=True)
            c2.caption("Texte nettoyé")
            c2.code(r["cleaned"], language=None, wrap_lines=True)

        n_words, n_bi = int((terms["type"] == "mot").sum()), int((terms["type"] == "bigramme").sum())
        unknown = (f" Mots absents du vocabulaire, ignorés : {html.escape(', '.join(r['unknown'][:12]))}."
                   if r["unknown"] else " Tous les mots sont connus du vocabulaire.")
        H(f"""<div class="step"><div class="n">2</div><h4>Découpage en termes</h4>
<p>Le modèle reconnaît <b>{n_words} mots</b> et <b>{n_bi} bigrammes</b>, des paires de mots qui captent
« not good » ou « never again ».{unknown}</p></div>""")

        H("""<div class="step"><div class="n">3</div><h4>Poids de chaque terme</h4>
<p>Chaque terme pousse vers le positif ou le négatif. Un même mot peut changer de sens en bigramme :
« good » seul est positif, « not good » est fortement négatif. C'est pour cela qu'on garde les bigrammes
et les mots-outils.</p></div>""")
        top = pd.concat([terms.head(7), terms.tail(7)]).drop_duplicates("terme")
        top = top[top["contribution"].abs() > 1e-6].sort_values("contribution")
        if not top.empty:
            chart = top.assign(sens=np.where(top["contribution"] > 0, "positif", "négatif"))
            _, mid = st.columns([1, 20])
            with mid:
                bar_chart(chart, "contribution", "terme", "sens", ["négatif", "positif"], [NEG, POS],
                          "contribution au score", max(150, 28 * len(chart)), labels="+.2f")

        H(f"""<div class="step"><div class="n">4</div><h4>Décision</h4>
<p>On additionne la constante du modèle et les contributions, puis la fonction logistique transforme ce score
en probabilité.</p></div>
<div class="calc">constante <b>{sg(M['intercept'])}</b> + contributions <b>{sg(terms['contribution'].sum())}</b>
= score <b>{sg(r['logit'])}</b> → probabilité <b>{pct(r['proba'])}</b> →
<b style="color:{POS if r['proba'] >= .5 else NEG}">{"positif" if r['proba'] >= .5 else "négatif"}</b></div>""")

# --------------------------------------------------------------------------- 3. réputation

with tabs[2]:
    rc = load_csv(OUT / "reputation_by_company.csv")
    st.markdown("### De l'avis à la réputation de l'entreprise")
    H(f"""<div class="warn">La source contient au plus 20 avis par note et par entreprise. Les pourcentages
ci-dessous sont ceux de cet échantillon, pas la vraie réputation de ces entreprises : ils illustrent la méthode,
validée sur {rep['entreprises_publiables']} entreprises jamais vues à l'entraînement
(corrélation {dec(rep['correlation_predit_reel'])}, écart moyen {dec(rep['ecart_absolu_moyen_points'], 1)} points).</div>""")

    left, right = st.columns([7, 5], gap="large")
    with left:
        sectors = ["Tous les secteurs"] + sorted(rc["entity_category"].dropna().unique())
        sel = st.selectbox("Secteur", sectors)
        view = rc[rc["publiable"]] if sel == "Tous les secteurs" else rc[rc["publiable"] & (rc["entity_category"] == sel)]
        view = view.assign(part_positifs_predite=view["part_positifs_predite"] * 100,
                           part_positifs_reelle=view["part_positifs_reelle"] * 100)
        show = view.rename(columns={
            "entity_id": "Entreprise", "entity_category": "Secteur", "nb_avis": "Avis",
            "part_positifs_predite": "Positifs prédits", "part_positifs_reelle": "Positifs réels",
            "score_reputation_wilson": "Score"})
        st.dataframe(
            show[["Entreprise", "Avis", "Score", "Positifs prédits", "Positifs réels", "Secteur"]],
            hide_index=True, width="stretch", height=420,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0, max_value=1,
                    help="Borne basse de Wilson (95 %) de la part d'avis positifs prédits"),
                "Positifs prédits": st.column_config.NumberColumn(format="%.0f %%"),
                "Positifs réels": st.column_config.NumberColumn(format="%.0f %%"),
            })
        st.markdown('<p class="small">Le score est la borne basse de Wilson : une entreprise à 3 avis positifs '
                    "sur 3 n'est pas « meilleure » qu'une entreprise à 90 % sur 200 avis.</p>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("**Prédit ou réel : le modèle suit-il la réalité ?**")
        pts = rc[rc["publiable"]].rename(columns={"entity_id": "entreprise"})
        st.vega_lite_chart(pts, {
            "layer": [
                {"data": {"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
                 "mark": {"type": "line", "color": "#c3c8d4", "strokeDash": [4, 4]},
                 "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}},
                {"mark": {"type": "circle", "size": 70, "opacity": 0.75, "color": POS,
                          "stroke": "#fff", "strokeWidth": 1},
                 "encoding": {
                     "x": {"field": "part_positifs_reelle", "type": "quantitative", "title": "part de positifs réelle",
                           "scale": {"domain": [0, 1]}, "axis": {"format": "%"}},
                     "y": {"field": "part_positifs_predite", "type": "quantitative", "title": "part de positifs prédite",
                           "scale": {"domain": [0, 1]}, "axis": {"format": "%"}},
                     "tooltip": [{"field": "entreprise"}, {"field": "entity_category", "title": "secteur"},
                                 {"field": "nb_avis", "title": "avis"},
                                 {"field": "part_positifs_reelle", "title": "réel", "format": ".0%"},
                                 {"field": "part_positifs_predite", "title": "prédit", "format": ".0%"}]}},
            ],
            "height": 380,
            "config": {"view": {"stroke": None}, "font": "IBM Plex Sans, system-ui, sans-serif",
                       "axis": {"gridColor": LINE, "labelColor": MUTED, "titleColor": MUTED, "domain": False,
                                "titleFontWeight": "normal"}},
        }, width="stretch")
        st.markdown('<p class="small">Un point par entreprise. Plus il est proche de la diagonale, plus la '
                    "réputation prédite est juste. Le regroupement autour de 50 % vient de l'échantillonnage.</p>",
                    unsafe_allow_html=True)

    st.markdown("**Par secteur**")
    cat = load_csv(OUT / "reputation_by_category.csv").rename(columns={
        "entity_category": "Secteur", "nb_avis": "Avis", "part_positifs_predite": "Positifs prédits",
        "part_positifs_reelle": "Positifs réels"})
    cat["Positifs prédits"] *= 100
    cat["Positifs réels"] *= 100
    st.dataframe(cat[["Secteur", "Avis", "Positifs prédits", "Positifs réels"]], hide_index=True, width="stretch",
                 column_config={"Positifs prédits": st.column_config.NumberColumn(format="%.1f %%"),
                                "Positifs réels": st.column_config.NumberColumn(format="%.1f %%")})

# --------------------------------------------------------------------------- 4. qualité des données

RULE_LABELS = {"excluded_label": "Note neutre (3 étoiles)", "wrong_language": "Autre langue",
               "duplicate": "Doublon", "duplicate_conflicting_labels": "Doublon à labels contradictoires",
               "no_letters": "Aucune lettre", "empty_text": "Texte vide", "invalid_label": "Note invalide"}

with tabs[3]:
    st.markdown("### Chaque règle est nommée, comptée et traçable")
    st.markdown('<p class="small">Une ligne écartée part en quarantaine avec son motif : rien n\'est supprimé '
                "en silence, et on peut toujours expliquer combien de lignes ont été perdues et pourquoi.</p>",
                unsafe_allow_html=True)

    log = pd.DataFrame(quality["journal_nettoyage"])
    log["effet"] = np.where(log["rows_rejected"] > 0, log["rows_rejected"], log["rows_modified"])
    log["action"] = np.where(log["rows_rejected"] > 0, "lignes écartées", "lignes modifiées")
    log = log[log["effet"] > 0]
    bar_chart(log.rename(columns={"step": "règle"}), "effet", "règle", "action",
              ["lignes modifiées", "lignes écartées"], ["#8a96ad", NEG], "nombre de lignes",
              34 * len(log) + 40, fmt_tip=",.0f", labels=",.0f")
    st.markdown('<p class="small">Les règles sans effet sur ces données (note invalide, texte vide, doublons '
                "contradictoires) restent actives pour une autre source.</p>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Indicateurs avant et après nettoyage**")
        nb = lambda v: "" if v is None else (fmt(v) if float(v).is_integer() else f"{v:.1f}")
        keys = [k for k in AV if not isinstance(AV[k], dict)]
        st.dataframe(pd.DataFrame({"Indicateur": keys, "Avant": [nb(AV[k]) for k in keys],
                                   "Après": [nb(AP.get(k)) for k in keys]}),
                     hide_index=True, width="stretch", height=35 * (len(keys) + 1) + 3)
    with right:
        st.markdown("**Constats sur la source**")
        for k, v in quality["autres_constats"].items():
            H(f'<div class="note find" style="margin-bottom:.7rem"><p>{html.escape(k[0].upper() + k[1:])} : '
              f"<b>{fmt(v)}</b></p></div>")

    st.markdown("**Exemples de normalisation**")
    st.dataframe(load_csv(REPORTS / "cleaning_examples.csv").rename(columns={"text_raw": "Avant", "text": "Après"}),
                 hide_index=True, width="stretch")
    st.markdown("**Exemples de lignes en quarantaine**")
    qx = load_csv(REPORTS / "quarantine_examples.csv").groupby("reject_reason").head(3)
    qx["Motif"] = qx["reject_reason"].map(RULE_LABELS).fillna(qx["reject_reason"])
    st.dataframe(qx[["Motif", "label_raw", "language", "text_raw"]].rename(
        columns={"label_raw": "Note", "language": "Langue détectée", "text_raw": "Texte"}),
        hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 5. modèles et erreurs

with tabs[4]:
    comp = load_csv(OUT / "model_comparison.csv").sort_values("f1_macro", ascending=False)
    chosen = metrics["modele_retenu"]
    st.markdown("### Huit modèles comparés, un seul passage sur le test")
    st.markdown(f'<p class="small">Modèle retenu : <b>{chosen}</b>. Règle fixée avant de regarder le test : '
                "meilleur F1 macro en validation ; à moins de 0,005 près, on préfère la régression logistique, "
                "qui donne une probabilité et reste interprétable.</p>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        c = comp.assign(retenu=np.where(comp["modele"] == chosen, "retenu", "autre"))
        bar_chart(c, "f1_macro", "modele", "retenu", ["retenu", "autre"], [POS, "#b8c0cf"],
                  "F1 macro en validation", 34 * len(c) + 40, fmt_tip=".4f", labels=".3f")
    with right:
        pred = load_parquet(OUT / "test_predictions.parquet")
        cm = (pred.assign(Réel=pred["label"].map({-1: "Négatif", 1: "Positif"}),
                          Prédit=pred["pred"].map({-1: "Négatif", 1: "Positif"}))
              .groupby(["Réel", "Prédit"]).size().rename("avis").reset_index())
        cm["part"] = cm["avis"] / cm.groupby("Réel")["avis"].transform("sum")
        cm["label"] = cm.apply(lambda r: f"{fmt(r['avis'])} ({r['part'] * 100:.0f} %)", axis=1)
        st.markdown("**Matrice de confusion (test)**")
        st.vega_lite_chart(cm, {
            "layer": [
                {"mark": {"type": "rect", "cornerRadius": 4},
                 "encoding": {"color": {"field": "part", "type": "quantitative",
                                        "scale": {"range": ["#eef1f6", POS]}, "legend": None}}},
                {"mark": {"type": "text", "fontSize": 14, "fontWeight": 600},
                 "encoding": {"text": {"field": "label"},
                              "color": {"condition": {"test": "datum.part > 0.5", "value": "white"}, "value": INK}}},
            ],
            "encoding": {"x": {"field": "Prédit", "type": "nominal", "axis": {"labelAngle": 0, "orient": "top"}},
                         "y": {"field": "Réel", "type": "nominal"}},
            "height": 230,
            "config": {"view": {"stroke": None}, "font": "IBM Plex Sans, system-ui, sans-serif",
                       "axis": {"labelColor": INK, "titleColor": MUTED, "domain": False, "ticks": False,
                                "labelFontSize": 12, "titleFontWeight": "normal"}},
        }, width="stretch")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Les erreurs se concentrent sur les notes intermédiaires**")
        by_star = (pred.assign(ok=pred["label"] == pred["pred"]).groupby("label_raw")["ok"].mean().reset_index())
        by_star["Note"] = by_star["label_raw"].astype(int).map(lambda n: "★" * n)
        by_star["ok"] *= 100
        st.dataframe(by_star[["Note", "ok"]].rename(columns={"ok": "Bonnes prédictions"}), hide_index=True,
                     width="stretch", column_config={"Bonnes prédictions": st.column_config.ProgressColumn(
                         format="%.0f %%", min_value=0, max_value=100)})
        st.markdown('<p class="small">Les avis à 2 et 4 étoiles sont souvent nuancés.</p>', unsafe_allow_html=True)
    with right:
        st.markdown("**Les termes qui pèsent le plus**")
        tf = load_csv(OUT / "top_features.csv")
        tf = pd.concat([tf[tf["polarite"] == "positif"].head(8), tf[tf["polarite"] == "négatif"].head(8)])
        tf = tf.sort_values("poids")
        bar_chart(tf, "poids", "terme", "polarite", ["négatif", "positif"], [NEG, POS],
                  "poids dans la régression logistique", 22 * len(tf) + 40)

    st.markdown("**Les erreurs où le modèle était le plus sûr de lui**")
    st.markdown('<p class="small">Une grande partie sont des erreurs de label : des textes clairement positifs '
                "notés 1 ou 2 étoiles. Le modèle a raison, la donnée a tort.</p>", unsafe_allow_html=True)
    err = load_csv(OUT / "error_analysis.csv").head(12)
    err["Note"] = err["label"].map({-1: "1-2 étoiles (négatif)", 1: "4-5 étoiles (positif)"})
    err["confiance_erreur"] *= 100
    st.dataframe(err[["Note", "confiance_erreur", "text"]].rename(columns={
        "confiance_erreur": "Confiance du modèle", "text": "Texte"}),
        hide_index=True, width="stretch",
        column_config={"Confiance du modèle": st.column_config.NumberColumn(format="%.1f %%")})
"""Démo du projet « Analyse de la réputation des entreprises ».

    streamlit run app.py

L'app lit le modèle (models/) et les sorties du
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
NEG, POS, GREY = "#e34948", "#2a78d6", "#898781"

st.set_page_config(page_title="Réputation des entreprises – démo", page_icon="⭐", layout="wide")

# --------------------------------------------------------------------------- chargement


@st.cache_resource
def load_model():
    model = joblib.load(MODEL)
    inner = model.named_steps["model"]
    vec, clf = inner.named_steps["tfidf"], inner.named_steps["clf"]
    sign = 1 if clf.classes_[1] == 1 else -1
    return {
        "model": model,
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


if not MODEL.exists() or not (OUT / "metrics.json").exists():
    st.error("Modèle ou sorties introuvables. Lancer d'abord : `python -m src.pipeline`")
    st.stop()

M = load_model()
metrics = load_json(OUT / "metrics.json")
quality = load_json(REPORTS / "data_quality_report.json")
rep_summary = load_json(OUT / "reputation_summary.json")



def explain(text: str) -> dict:
    """Décompose la prédiction : texte nettoyé -> termes reconnus -> contributions -> probabilité.
    Pour une régression logistique, score = constante + somme(poids du terme x valeur TF-IDF)."""
    cleaned = M["cleaner"].transform([text])[0]
    X = M["vec"].transform([cleaned])
    idx, vals = X.indices, X.data
    contrib = vals * M["coef"][idx]
    terms = pd.DataFrame({"terme": M["names"][idx], "tfidf": vals, "poids": M["coef"][idx],
                          "contribution": contrib})
    terms["type"] = np.where(terms["terme"].str.contains(" "), "bigramme", "mot")
    logit = M["intercept"] + contrib.sum()
    proba = 1 / (1 + np.exp(-logit))
    tokens = M["token_re"].findall(cleaned.lower())
    known = set(terms["terme"])
    unknown = sorted({t for t in tokens if t not in known})
    return {"cleaned": cleaned, "terms": terms.sort_values("contribution"), "logit": logit,
            "proba": proba, "tokens": tokens, "unknown": unknown}


def highlight(cleaned: str, terms: pd.DataFrame) -> str:
    """Colore chaque mot selon sa contribution (le poids d'un bigramme est partagé entre ses deux mots)."""
    per_word: dict[str, float] = {}
    for t, c in zip(terms["terme"], terms["contribution"]):
        parts = t.split(" ")
        for p in parts:
            per_word[p] = per_word.get(p, 0.0) + c / len(parts)
    scale = max([abs(v) for v in per_word.values()] + [1e-9])
    out = []
    for piece in re.split(r"(\w+)", cleaned):
        v = per_word.get(piece.lower())
        if not piece or v is None or abs(v) < 0.02 * scale:
            out.append(html.escape(piece))
            continue
        a = 0.15 + 0.6 * abs(v) / scale
        rgb = "227,73,72" if v < 0 else "42,120,214"
        out.append(f'<span title="{v:+.2f}" style="background:rgba({rgb},{a:.2f});'
                   f'border-radius:4px;padding:1px 3px">{html.escape(piece)}</span>')
    return "".join(out)


# --------------------------------------------------------------------------- en-tête

st.title("Analyse de la réputation des entreprises")
st.caption("Prédire si un avis client est positif (1) ou négatif (-1), puis mesurer la réputation de chaque "
           "entreprise. Données : 123 181 avis Trustpilot (anglais), 1 680 entreprises.")

t = metrics["test"]
c = st.columns(4)
c[0].metric("Accuracy (test)", dec(t["accuracy"]))
c[1].metric("F1 macro (test)", dec(t["f1_macro"]), help="Référence naïve (classe majoritaire) : 0,366")
c[2].metric("Corrélation réputation prédite / réelle", dec(rep_summary["correlation_predit_reel"]),
            help=f"Sur {rep_summary['entreprises_publiables']} entreprises jamais vues à l'entraînement")
c[3].metric("Avis en quarantaine (tracés)", fmt(sum(quality["rejets_par_motif"].values())),
            help="Écartés avec un motif, jamais supprimés en silence")

tabs = st.tabs(["🧭 Le projet en 2 minutes", "✍️ Tester un avis", "🏢 Réputation des entreprises",
                "🧹 Qualité des données", "📊 Modèles et erreurs"])


with tabs[0]:
    av, ap = quality["avant_nettoyage"], quality["après_nettoyage"]
    sp = quality["split"]["splits"]
    steps = [
        ("1. Ingestion", f"{fmt(av['lignes'])} avis bruts",
         "Copie immuable + empreinte SHA-256, contrat de colonnes, schéma standard"),
        ("2. Qualité", f"{len(quality['journal_nettoyage'])} règles tracées",
         "Profilage avant / après, rapport qualité"),
        ("3. Nettoyage", f"{fmt(ap['lignes'])} avis gardés",
         f"{fmt(sum(quality['rejets_par_motif'].values()))} en quarantaine avec un motif"),
        ("4. Découpage", f"{sp['test']['entités']} entreprises en test",
         "Groupé par entreprise : 0 entreprise et 0 texte partagés"),
        ("5. Modèle", f"F1 macro {dec(t['f1_macro'])}",
         "8 modèles comparés, règle de choix fixée avant le test"),
        ("6. Réputation", f"r = {dec(rep_summary['correlation_predit_reel'])}",
         "Score de Wilson par entreprise et par secteur"),
        ("7. Datamart", "Schéma en étoile",
         "SQLite + vues prêtes pour la BI, contrôles de recette"),
    ]
    boxes = "".join(
        f'<div style="flex:1;min-width:150px;border:1px solid rgba(128,128,128,.35);border-radius:10px;'
        f'padding:10px 12px"><div style="font-size:.8rem;opacity:.7">{h}</div>'
        f'<div style="font-weight:600;margin:2px 0 4px">{v}</div>'
        f'<div style="font-size:.8rem;opacity:.8">{d}</div></div>'
        for h, v, d in steps)
    st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap">{boxes}</div>', unsafe_allow_html=True)

    st.subheader("Ce qu'il faut retenir")
    left, right = st.columns(2)
    with left:
        st.markdown(f"""
**La donnée d'abord**
- **3 étoiles écartées** ({fmt(quality['rejets_par_motif'].get('excluded_label', 0))} avis) : ni positives
  ni négatives, les forcer d'un côté ajouterait du bruit.
- **Titre ajouté à l'avis seulement s'il apporte une info** : 58 768 titres ne sont que le début de
  l'avis tronqué par « … ».
- **Ponctuation collée corrigée** sur {fmt(av['ponctuation collée au mot suivant'])} avis
  (« fast,great » → « fast, great »).
- **Mots-outils conservés** : « not » est le mot négatif le plus important du modèle.
- **Aucune fuite** : les entreprises du test n'ont jamais été vues à l'entraînement.
""")
    with right:
        st.markdown("""
**Deux découvertes en auditant les données**
- **La source est échantillonnée** : au plus 20 avis par note et par entreprise. Les proportions par
  entreprise ne sont donc pas la vraie réputation ; la méthode est validée, pas les valeurs absolues.
- **Beaucoup d'erreurs « sûres » du modèle sont des erreurs de label** : des textes clairement positifs
  notés 1 étoile. Le modèle a raison, la donnée a tort — une piste de contrôle qualité automatique.

**Le modèle**
- TF-IDF (mots + bigrammes) + régression logistique : rapide, interprétable, donne une probabilité.
- Le nettoyage du texte est **embarqué dans le modèle** : même traitement à l'entraînement et en production.
""")
    st.info("➡️ Onglet **Tester un avis** : écrivez un avis (en anglais) et voyez, étape par étape, comment "
            "le modèle arrive à sa décision.")

# --------------------------------------------------------------------------- 2. tester un avis

EXAMPLES = {
    "Positif": "Delivery was quick and the staff were really helpful,will definitely order again!!!",
    "Négatif": "Parcel arrived damaged and nobody answered my emails. Never again.",
    "Négation": "The product is not good and the support was not helpful at all.",
    "Nuancé": "Good product but the price went up without any warning, not happy about that.",
}

with tabs[1]:
    st.caption("Le modèle a été entraîné sur des avis en anglais.")
    if "review" not in st.session_state:
        st.session_state["review"] = EXAMPLES["Négation"]
    bcols = st.columns(len(EXAMPLES))
    for col, (label, ex) in zip(bcols, EXAMPLES.items()):
        if col.button(f"Exemple : {label}", width="stretch"):
            st.session_state["review"] = ex
    text = st.text_area("Avis client", key="review", height=110)

    if text.strip():
        r = explain(text)
        pred = 1 if r["proba"] >= 0.5 else -1
        color = POS if pred == 1 else NEG
        st.markdown(
            f'<div style="border-left:6px solid {color};padding:8px 14px;margin:6px 0 14px">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{color}">'
            f'{"Positif (1)" if pred == 1 else "Négatif (-1)"}</div>'
            f'<div>Probabilité que l\'avis soit positif : <b>{pct(r["proba"])}</b></div></div>',
            unsafe_allow_html=True)

        st.markdown("#### Ce qui se passe, étape par étape")
        st.markdown("**① Nettoyage du texte** — exactement le même qu'à l'entraînement "
                    "(ponctuation collée, lettres répétées, HTML, URL…).")
        a, b = st.columns(2)
        a.markdown("*Texte reçu*")
        a.code(text, language=None, wrap_lines=True)
        b.markdown("*Texte nettoyé*" + (" — identique" if r["cleaned"] == text.strip() else ""))
        b.code(r["cleaned"], language=None, wrap_lines=True)

        terms = r["terms"]
        n_words, n_bi = int((terms["type"] == "mot").sum()), int((terms["type"] == "bigramme").sum())
        st.markdown(f"**② Découpage en termes** — le modèle reconnaît **{n_words} mots** et **{n_bi} bigrammes** "
                    "(paires de mots, pour capter « not good », « never again »). "
                    + (f"Mots inconnus du vocabulaire, ignorés : *{', '.join(r['unknown'][:12])}*."
                       if r["unknown"] else "Tous les mots sont connus du vocabulaire."))

        st.markdown("**③ Poids de chaque terme** — chaque terme reconnu pousse vers le positif (bleu) "
                    "ou le négatif (rouge). Survoler un mot affiche sa contribution.")
        st.markdown(f'<div style="line-height:2;font-size:1.05rem">{highlight(r["cleaned"], terms)}</div>',
                    unsafe_allow_html=True)
        st.caption("Un mot peut pousser dans un sens seul et dans l'autre en bigramme : « good » seul est positif, "
                   "mais « not good » est fortement négatif. C'est pour cela que le modèle utilise des bigrammes "
                   "et qu'on ne supprime pas les mots-outils.")
        top = pd.concat([terms.head(6), terms.tail(6)]).drop_duplicates("terme")
        top = top[top["contribution"].abs() > 1e-6].sort_values("contribution")
        if not top.empty:
            chart = top.assign(sens=np.where(top["contribution"] > 0, "vers positif", "vers négatif"))
            st.vega_lite_chart(chart, {
                "mark": {"type": "bar", "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {"field": "terme", "type": "nominal", "sort": None, "title": None,
                          "axis": {"labelOverlap": False, "labelLimit": 220}},
                    "x": {"field": "contribution", "type": "quantitative", "title": "contribution au score"},
                    "color": {"field": "sens", "type": "nominal",
                              "scale": {"domain": ["vers négatif", "vers positif"], "range": [NEG, POS]},
                              "legend": {"orient": "bottom", "title": None}},
                    "tooltip": [{"field": "terme"}, {"field": "type"},
                                {"field": "contribution", "format": "+.2f"}],
                },
                "height": max(140, 30 * len(chart)),
            }, width="stretch")

        sg = lambda x: f"{x:+.2f}".replace(".", ",")
        st.markdown(f"**④ Décision** — score = constante ({sg(M['intercept'])}) + somme des contributions "
                    f"({sg(terms['contribution'].sum())}) = **{sg(r['logit'])}**. "
                    f"La fonction logistique le transforme en probabilité : **{pct(r['proba'])}** → "
                    f"{'positif' if pred == 1 else 'négatif'} (seuil 50 %).")

# --------------------------------------------------------------------------- 3. réputation

with tabs[2]:
    st.warning("La source contient **au plus 20 avis par note et par entreprise**. Les pourcentages ci-dessous "
               "sont ceux de cet échantillon, pas la vraie réputation de ces entreprises : ils illustrent la méthode.")
    rc = load_csv(OUT / "reputation_by_company.csv")
    a, b, c3 = st.columns(3)
    a.metric("Entreprises évaluées (jamais vues à l'entraînement)", rep_summary["entreprises_publiables"])
    b.metric("Corrélation prédit / réel", dec(rep_summary["correlation_predit_reel"]))
    c3.metric("Écart moyen", f"{dec(rep_summary['ecart_absolu_moyen_points'], 1)} points")
    st.markdown("**Score de réputation = borne basse de Wilson** de la part d'avis positifs prédits : une entreprise "
                "à 3 avis positifs sur 3 n'est pas « meilleure » qu'une entreprise à 90 % sur 200 avis.")

    sectors = ["Tous les secteurs"] + sorted(rc["entity_category"].dropna().unique())
    sel = st.selectbox("Secteur", sectors)
    view = rc[rc["publiable"]] if sel == "Tous les secteurs" else rc[rc["publiable"] & (rc["entity_category"] == sel)]
    show = view.rename(columns={
        "entity_id": "entreprise", "entity_category": "secteur", "nb_avis": "avis",
        "part_positifs_predite": "positifs prédits", "part_positifs_reelle": "positifs réels",
        "score_reputation_wilson": "score de réputation"})
    st.dataframe(
        show[["entreprise", "secteur", "avis", "score de réputation", "positifs prédits", "positifs réels"]],
        hide_index=True, width="stretch",
        column_config={
            "score de réputation": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "positifs prédits": st.column_config.NumberColumn(format="%.2f"),
            "positifs réels": st.column_config.NumberColumn(format="%.2f"),
        })
    a, b = st.columns(2)
    a.image(str(OUT / "figures" / "reputation_pred_vs_true.png"),
            caption="Chaque point est une entreprise du test. Le regroupement autour de 50 % vient de l'échantillonnage.")
    cat = load_csv(OUT / "reputation_by_category.csv").rename(columns={
        "entity_category": "secteur", "nb_avis": "avis", "part_positifs_predite": "positifs prédits",
        "part_positifs_reelle": "positifs réels"})
    b.markdown("**Par secteur**")
    b.dataframe(cat[["secteur", "avis", "positifs prédits", "positifs réels"]], hide_index=True,
                width="stretch", column_config={
                    "positifs prédits": st.column_config.NumberColumn(format="%.2f"),
                    "positifs réels": st.column_config.NumberColumn(format="%.2f")})

# --------------------------------------------------------------------------- 4. qualité des données

RULE_LABELS = {"excluded_label": "note neutre (3 étoiles)", "wrong_language": "autre langue",
               "duplicate": "doublon", "duplicate_conflicting_labels": "doublon à labels contradictoires",
               "no_letters": "aucune lettre", "empty_text": "texte vide", "invalid_label": "note invalide"}

with tabs[3]:
    st.markdown("Chaque règle est **nommée et comptée** ; chaque ligne écartée part en **quarantaine** avec "
                "un motif. Rien n'est supprimé en silence.")
    av, ap = quality["avant_nettoyage"], quality["après_nettoyage"]
    keys = [k for k in av if not isinstance(av[k], dict)]
    a, b = st.columns(2)
    nb = lambda v: "" if v is None else (f"{v:,.0f}".replace(",", " ") if float(v).is_integer() else f"{v:.1f}")
    a.markdown("**Indicateurs avant / après nettoyage**")
    a.dataframe(pd.DataFrame({"indicateur": keys, "avant": [nb(av[k]) for k in keys],
                              "après": [nb(ap.get(k)) for k in keys]}),
                hide_index=True, width="stretch", height=35 * (len(keys) + 1) + 3)
    b.markdown("**Journal des règles**")
    b.dataframe(pd.DataFrame(quality["journal_nettoyage"]).rename(columns={
        "step": "règle", "rows_in": "entrée", "rows_modified": "modifiées", "rows_rejected": "rejetées"}),
        hide_index=True, width="stretch")
    b.markdown("**Autres constats**")
    for k, v in quality["autres_constats"].items():
        b.markdown(f"- {k} : **{fmt(v)}**")

    st.markdown("**Exemples de normalisation**")
    st.dataframe(load_csv(REPORTS / "cleaning_examples.csv").rename(
        columns={"text_raw": "avant", "text": "après"}), hide_index=True, width="stretch")
    st.markdown("**Exemples de lignes en quarantaine**")
    qx = load_csv(REPORTS / "quarantine_examples.csv")
    qx["motif"] = qx["reject_reason"].map(RULE_LABELS).fillna(qx["reject_reason"])
    st.dataframe(qx[["motif", "label_raw", "language", "text_raw"]].rename(
        columns={"label_raw": "note", "language": "langue détectée", "text_raw": "texte"}),
        hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 5. modèles et erreurs

with tabs[4]:
    comp = load_csv(OUT / "model_comparison.csv").sort_values("f1_macro", ascending=False)
    st.markdown(f"**Modèle retenu : `{metrics['modele_retenu']}`** — règle fixée avant de regarder le test : "
                "meilleur F1 macro en validation ; à moins de 0,005 près, on préfère la régression logistique "
                "(probabilité + interprétable). Le test n'est utilisé qu'une fois.")
    a, b = st.columns([3, 2])
    a.dataframe(comp[["modele", "f1_macro", "accuracy", "roc_auc", "temps_entrainement_s"]].rename(columns={
        "modele": "modèle", "f1_macro": "F1 macro (val.)", "temps_entrainement_s": "entraînement (s)"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.4f")
                       for c in ["F1 macro (val.)", "accuracy", "roc_auc"]})
    b.image(str(OUT / "figures" / "confusion_matrix.png"))

    pred = load_parquet(OUT / "test_predictions.parquet")
    by_star = (pred.assign(ok=pred["label"] == pred["pred"]).groupby("label_raw")["ok"].mean()
               .rename("bonnes prédictions").reset_index().rename(columns={"label_raw": "note (étoiles)"}))
    a, b = st.columns([2, 3])
    a.markdown("**Les erreurs se concentrent sur les notes intermédiaires**")
    a.dataframe(by_star, hide_index=True, width="stretch",
                column_config={"bonnes prédictions": st.column_config.ProgressColumn(
                    format="%.3f", min_value=0, max_value=1)})
    b.image(str(OUT / "figures" / "top_features.png"),
            caption="Les négations (« not », « not good ») sont parmi les termes les plus négatifs.")

    st.markdown("**Erreurs où le modèle est le plus sûr de lui** — une grande partie sont des erreurs de "
                "label (texte positif noté 1-2 étoiles), pas du modèle.")
    err = load_csv(OUT / "error_analysis.csv").head(12)
    st.dataframe(err[["type_erreur", "label", "confiance_erreur", "text"]].rename(columns={
        "type_erreur": "erreur", "label": "label réel", "confiance_erreur": "confiance", "text": "texte"}),
        hide_index=True, width="stretch",
        column_config={"confiance": st.column_config.NumberColumn(format="%.3f")})

st.divider()
st.caption("Code, rapport qualité et notebook : voir le README du projet. "
           "Données : Trustpilot Reviews 123k (licence MIT).")

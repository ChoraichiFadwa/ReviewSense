# Analyse de la réputation des entreprises à partir des avis clients

Prédire la polarité d'un avis client (**1 = positif, -1 = négatif**) à partir de son texte,
puis s'en servir pour mesurer la réputation de chaque entreprise.

 Démo en ligne : [LIEN_DE_LA_DEMO](https://reviewsense-fadwa.streamlit.app/)** : tester un avis et voir, étape par étape,
comment le modèle décide ; réputation par entreprise ; qualité des données.
En local : `streamlit run app.py`.

![Aperçu de la démo](docs/demo_apercu.png)

## Résumé

| | |
|---|---|
| **Données** | 123 181 avis Trustpilot, 1 680 entreprises, 22 secteurs → 101 861 après nettoyage |
| **Test – accuracy / F1 macro** | **0,935 / 0,934** (référence naïve : 0,577 / 0,366) |
| **Test – ROC AUC** | 0,978 |
| **Réputation par entreprise** | corrélation prédit / réel **0,981**, écart moyen **2,5 points** |
| **Conditions du test** | 168 entreprises **jamais vues** à l'entraînement (découpage groupé, contrôle anti-fuite) |

**Ce qui fait la différence**

1. **La donnée d'abord** : 9 règles de nettoyage nommées et comptées, 21 320 lignes en **quarantaine avec un motif**
   (rien n'est supprimé en silence), rapport qualité avant / après (`reports/data_quality_report.md`).
2. **Des règles de gestion justifiées** : 3 étoiles écartées (ni positives ni négatives) ; titre ajouté à l'avis
   seulement s'il apporte une info (58 768 titres n'étaient que le début de l'avis tronqué) ; mots-outils conservés
   (« not » est le mot négatif le plus important du modèle).
3. **Deux découvertes en auditant les données** : la source est **échantillonnée** (au plus 20 avis par note et par
   entreprise, donc les proportions ne sont pas la vraie réputation) ; une grande partie des erreurs « sûres » du
   modèle sont des **erreurs de label** (texte positif noté 1 étoile).
4. **Un modèle prêt pour la production** : comparaison de 8 modèles avec une règle de choix fixée avant le test ;
   nettoyage **embarqué dans le modèle** (même traitement à l'entraînement et en production) ; prédiction en ligne
   de commande.
5. **Jusqu'à la BI** : score de réputation (borne de Wilson) par entreprise et par secteur, **datamart en étoile**
   (SQLite) avec contrôles de recette.

**Où regarder** : la démo → ce README → `notebooks/analyse_reputation.ipynb` (déjà exécuté) → `src/`.

La chaîne complète :

```
ingestion → contrôles qualité → nettoyage tracé → découpage sans fuite
          → comparaison de modèles → évaluation → réputation par entreprise → datamart BI
```

## Lancer le projet

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

**Démonstration rapide (moins d'une minute, échantillon inclus de 160 entreprises)** :

```bash
python -m src.pipeline --config config.sample.yaml     # sorties dans data/sample_run/
```

**Jeu complet (~4 min)** : télécharger le CSV depuis
[Trustpilot Reviews 123k](https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k)
(onglet *Files and versions*, licence MIT), puis :

```bash
python scripts/prepare_data.py --file chemin/vers/le_fichier.csv
python -m src.pipeline
```

**Prédire de nouveaux avis** (le modèle entraîné est fourni dans `models/`) :

```bash
python -m src.predict "Parcel arrived damaged and nobody answered my emails."
python -m src.predict --input nouveaux_avis.csv --text-col review
```

**Démo interactive** : `streamlit run app.py` (lit le modèle et les sorties fournis, rien à réentraîner).

**Tests** : `python -m pytest` (17 tests : règles de nettoyage, contrat d'ingestion, absence de fuite).

**Notebook de présentation** : `notebooks/analyse_reputation.ipynb` (déjà exécuté ; il relit les sorties du pipeline).

## Structure

```
app.py                   démo interactive (Streamlit)
config.yaml              paramètres ; seule la section `source` change pour une autre source
config.sample.yaml       même chose sur l'échantillon de démonstration
scripts/prepare_data.py  dépôt du fichier brut + empreinte SHA-256 + échantillon
src/
  ingestion.py           lecture, contrat de colonnes, mise au schéma standard, lignage
  cleaning.py            normalisation du texte + règles de rejet (quarantaine avec motif)
  quality.py             profilage avant / après, rapport qualité
  split.py               découpage groupé par entreprise + contrôle anti-fuite bloquant
  data_pipeline.py       orchestration de l'étape données
  models.py              modèles candidats + modèle d'inférence (nettoyage embarqué)
  train.py               comparaison, sélection, réentraînement, évaluation unique sur le test
  evaluate.py            métriques, graphiques, analyse d'erreurs
  reputation.py          agrégation par entreprise et par secteur (score de Wilson)
  datamart.py            chargement SQLite en étoile + contrôles de recette
  predict.py             prédiction en ligne de commande
  pipeline.py            lance toute la chaîne
reports/                 rapport qualité des données (.md et .json)
outputs/                 métriques, comparaison des modèles, erreurs, réputation, datamart, figures
models/                  modèle entraîné
tests/                   tests unitaires
docs/                    captures d'écran de la démo
data/sample/             échantillon de démonstration
```

## 1. Ingestion

- Le fichier source est **copié tel quel** dans `data/raw/` avec son empreinte SHA-256 (`data/raw/SOURCE.md`) :
  on peut prouver sur quelle version des données le modèle a été entraîné.
- **Contrat d'interface** : si une colonne attendue manque, le pipeline s'arrête avec un message clair
  au lieu de produire un résultat faux.
- **Schéma standard** : chaque avis reçoit un identifiant stable (hash du contenu), garde sa note d'origine
  (`label_raw`) et son lignage (fichier, ligne, date d'ingestion).
- **Brancher une autre source** (par exemple les données d'un client) : il suffit de modifier la section
  `source` de `config.yaml` — format (csv, jsonl, xlsx, parquet), noms des colonnes, correspondance des notes.
  Le reste de la chaîne ne change pas.

## 2. Contrôle qualité et nettoyage

Chaque règle est nommée et comptée ; chaque ligne écartée part en **quarantaine** (`data/rejected/`)
avec son motif. Rien n'est supprimé en silence. Rapport complet : `reports/data_quality_report.md`.

### Règles de gestion

| Règle | Pourquoi | Lignes concernées |
|---|---|---:|
| 1-2 étoiles → -1, 4-5 étoiles → 1 | correspondance demandée par le sujet | 101 917 |
| 3 étoiles écartées | ni positives ni négatives : les forcer d'un côté ajouterait du bruit | 21 264 |
| Titre ajouté devant l'avis seulement s'il apporte une info | 58 768 titres ne sont que le début de l'avis tronqué par « … » : les ajouter compterait ces mots deux fois | 64 413 titres ajoutés |
| Avis clairement dans une autre langue écartés | modèle entraîné sur l'anglais ; heuristique explicable (mots-outils), testée | 42 |
| Doublons (texte normalisé) : 1ère occurrence gardée | éviter qu'un même texte pèse double, ou soit à la fois en train et en test | 13 |
| Doublons à labels contradictoires : tous écartés | impossible de savoir lequel est juste | 0 ici (règle prête) |
| Textes sans aucune lettre écartés | aucun signal pour le modèle | 1 |

### Normalisation du texte (62 343 avis modifiés)

Correction d'encodage, entités et balises HTML, URL remplacées par un token, **espace ajouté après une
ponctuation collée** (26 291 avis : « service,very fast.Will order » → « service, very fast. Will order »),
lettres répétées réduites (« sooooo » → « sooo »), espaces normalisés.

**Ce qu'on ne fait volontairement pas** : supprimer les mots-outils (« not good » ≠ « good » — et le modèle
confirme que « not » est son mot négatif le plus important) ni supprimer les avis courts (« Never again »
est un avis très clair).

### Constat majeur : la source est échantillonnée par note

Aucune entreprise n'a plus de **20 avis pour une même note**, et les 558 entreprises à 100 avis en ont
exactement 20 par niveau d'étoiles. Le jeu a donc été constitué en plafonnant le nombre d'avis par note.

- Pour **entraîner** le modèle : plutôt favorable (pas de sur-représentation des 5 étoiles).
- Pour **mesurer la réputation** : les proportions de l'échantillon ne sont **pas** la réalité. Une entreprise
  avec 20 avis de chaque note ressort à 50 % de positifs quelle que soit sa vraie réputation. La méthode est
  valable ; les valeurs absolues ne doivent pas être interprétées comme la réputation réelle.

Autres constats : la description de l'entreprise est répétée sur chaque avis (elle est rangée une seule
fois dans la dimension du datamart) ; aucune entreprise n'a plusieurs secteurs ou plusieurs descriptions.

## 3. Découpage sans fuite de données

- Dédoublonnage **avant** le découpage.
- Découpage **groupé par entreprise** (80 / 10 / 10) : 1 344 entreprises en entraînement, 168 en validation,
  168 en test, aucune en commun. C'est la situation réelle (analyser une nouvelle entreprise) et cela
  empêche le modèle de « tricher » en apprenant le nom ou les produits d'une entreprise.
- Un **contrôle bloquant** arrête le pipeline si une entreprise ou un texte se retrouve dans deux splits.

## 4. Modélisation

Même représentation pour tous (TF-IDF mots + bigrammes, mots-outils conservés), seul le classifieur change.

| Modèle | F1 macro (validation) |
|---|---:|
| Référence : classe majoritaire | 0,366 |
| Naive Bayes | 0,907 |
| Régression logistique C=1 / C=4 / **C=10** | 0,924 / 0,928 / **0,929** |
| Régression logistique C=4, classes rééquilibrées | 0,927 |
| SVM linéaire C=0,1 / C=0,5 | 0,926 / 0,930 |

**Règle de sélection fixée avant de regarder le test** : meilleur F1 macro en validation ; à moins de 0,005
près, on préfère la régression logistique, qui donne une **probabilité** (utilisée pour la réputation) et
dont les poids sont **interprétables**. Le modèle retenu est réentraîné sur train + validation, puis évalué
**une seule fois** sur le test.

Pourquoi le **F1 macro** plutôt que l'accuracy : les classes sont à ~43 % / 57 % ; la référence naïve atteint
58 % d'accuracy sans rien apprendre, mais seulement 0,37 de F1 macro.

Essai écarté : ajouter des n-grammes de caractères (robustes aux fautes) fait passer le F1 de 0,9232 à 0,9234
sur un sous-échantillon de 40 000 avis, pour un temps d'entraînement 6 fois plus long.

## 5. Évaluation et analyse d'erreurs

| Test (9 904 avis, 168 entreprises jamais vues) | Négatif | Positif |
|---|---:|---:|
| Précision | 0,917 | 0,949 |
| Rappel | 0,930 | 0,940 |
| F1 | 0,923 | 0,944 |

- Les erreurs se concentrent sur les **notes intermédiaires** : 97 % de bonnes prédictions sur les 1 étoile
  et 98 % sur les 5 étoiles, mais environ 88-89 % sur les 2 et 4 étoiles, souvent des avis nuancés.
- Parmi les erreurs où le modèle est le plus sûr de lui, **une grande partie sont des erreurs de label** :
  des textes clairement positifs (« Absolutely brilliant service… Highly recommended ») notés 1 ou 2 étoiles.
  Le modèle a raison, la donnée a tort. Ces incohérences note / texte pourraient servir de **contrôle qualité**.
- Les autres erreurs : avis nuancés (« Good experience… but I was not informed of later charges »),
  avis mis à jour où le client change d'avis, sujets d'opinion plutôt que de service.

Détail : `outputs/error_analysis.csv`, `outputs/figures/`.

## 6. De l'avis à la réputation de l'entreprise

- Agrégation des prédictions par entreprise et par secteur.
- **Score de réputation = borne basse de Wilson** (95 %) de la part d'avis positifs : une entreprise à 3 avis
  positifs sur 3 n'est pas « meilleure » qu'une entreprise à 90 % sur 200 avis.
- Score publié seulement à partir de 20 avis (`reputation.min_reviews`).
- **Validation pour ce besoin** : sur 163 entreprises jamais vues, la part de positifs prédite suit la réelle
  (corrélation **0,981**, écart moyen **2,5 points**). Le modèle est donc fiable pour une réputation agrégée,
  même s'il se trompe sur quelques avis individuels.

Sorties : `outputs/reputation_by_company.csv`, `outputs/reputation_by_category.csv`.

## 7. Datamart pour la BI

`outputs/reputation_datamart.sqlite`, modèle en étoile :

- `dim_entreprise` : une ligne par entreprise (secteur, description) ;
- `fait_avis` : une ligne par avis (note source, label réel, prédiction, probabilité) ;
- vues `v_reputation_entreprise` et `v_reputation_secteur`, prêtes pour Power BI ou une requête SQL.

Contrôles exécutés après chargement : nombre de lignes, clés orphelines, cohérence des totaux des vues.
Seuls les avis du test sont chargés : des prédictions sur les données d'entraînement seraient trop optimistes.

## Limites et pistes

**Limites**
- Échantillonnage de la source (20 avis max par note) : les proportions par entreprise ne sont pas réelles.
- Bruit dans les labels (note incohérente avec le texte), qui pénalise l'entraînement et l'évaluation.
- Avis en anglais uniquement : pour des avis en français ou en darija, il faudrait des données labellisées
  dans ces langues (la chaîne, elle, reste la même).
- Les 3 étoiles sont écartées : le modèle force tout avis en positif ou négatif.

**Pistes**
- Élargir la grille de C (le meilleur est en bord de grille) et calibrer les probabilités.
- Comparer à un modèle pré-entraîné (BERT ; CamemBERT pour le français) : gain attendu sur les avis nuancés,
  coût de calcul bien plus élevé.
- Exploiter les incohérences note / texte comme contrôle qualité automatique.
- Suivre la performance dans le temps (dérive du vocabulaire) sur des avis récents notés.
- Passer à 3 classes (positif / neutre / négatif) si le besoin métier l'exige.

## Source des données

Trustpilot Reviews 123k — https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k — licence MIT.

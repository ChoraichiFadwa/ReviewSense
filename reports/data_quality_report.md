# Rapport qualité des données

Source : `trustpilot` (trustpilot_reviews.csv)

## Indicateurs avant / après nettoyage

| Indicateur | Avant | Après |
|---|---:|---:|
| lignes | 123 181 | 101 861 |
| entreprises distinctes | 1 680 | 1 680 |
| secteurs distincts | 22 | 22 |
| avis par entreprise (médiane) | 89 | 72 |
| avis par entreprise (max) | 100 | 80 |
| répartition des notes source | 1: 24 065, 2: 19 687, 3: 21 264, 4: 26 013, 5: 32 152 | 1: 24 058, 2: 19 686, 4: 25 997, 5: 32 120 |
| répartition des labels (1 / -1) | -1: 43 752, 1: 58 165, exclu / invalide: 21 264 | -1: 43 744, 1: 58 117 |
| textes vides | 0 | 0 |
| textes sans aucune lettre | 2 | 0 |
| longueur (car.) médiane | 306 | 302 |
| longueur (car.) p95 | 889 | 884 |
| avis < 20 caractères | 96 | 81 |
| doublons (lignes en trop) | 14 | 0 |
| groupes de doublons | 13 | 0 |
| groupes de doublons à labels contradictoires | 0 | 0 |
| groupes de doublons sur plusieurs entreprises | 7 | 0 |
| entités HTML (&amp; …) | 1 | 0 |
| balises HTML | 0 | 0 |
| URL | 5 | 0 |
| ponctuation collée au mot suivant | 26 291 | 0 |
| lettres répétées (sooooo) | 254 | 0 |
| avis avec émojis | 3 508 | 3 131 |
| avis dans une autre langue que 'en' | 44 | 0 |

## Journal des règles de nettoyage

| Étape | Lignes en entrée | Lignes modifiées | Lignes rejetées |
|---|---:|---:|---:|
| titre ajouté devant l'avis (non redondant) | 123 181 | 64 413 | 0 |
| normalisation du texte | 123 181 | 62 343 | 0 |
| note neutre écartée (règle de gestion) | 123 181 | 0 | 21 264 |
| note hors du référentiel | 101 917 | 0 | 0 |
| texte vide ou inexploitable | 101 917 | 0 | 0 |
| aucune lettre (ponctuation, chiffres ou émojis seuls) | 101 917 | 0 | 1 |
| avis dans une autre langue que 'en' | 101 916 | 0 | 42 |
| doublons avec labels contradictoires (tous écartés) | 101 874 | 0 | 0 |
| doublons (1ère occurrence conservée) | 101 874 | 0 | 13 |

## Lignes en quarantaine (data/rejected/)

| Motif | Lignes |
|---|---:|
| `excluded_label` | 21 264 |
| `wrong_language` | 42 |
| `duplicate` | 13 |
| `no_letters` | 1 |

## Découpage train / validation / test

| Split | Avis | Entités | % positifs |
|---|---:|---:|---:|
| train | 81 895 | 1 344 | 56.8 % |
| val | 10 062 | 168 | 57.7 % |
| test | 9 904 | 168 | 58.2 % |

- Entités présentes dans plusieurs splits : **0**
- Textes présents dans plusieurs splits : **0**

## Autres constats

- titres redondants (début de l'avis tronqué) non ajoutés au texte : **58 768**
- entreprises au plafond de la source (nb d'avis = max observé) : **558**
- maximum d'avis pour une même entreprise et une même note : **20**
- entreprises avec exactement 20 avis pour chacune des 5 notes : **558**
- entreprises avec plusieurs valeurs de « category » : **0**
- entreprises avec plusieurs valeurs de « description » : **0**

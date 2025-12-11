# OPCO-ATLAS-Module-4-Brief-1

_M4 – Brief 1 : Benchmark de modèles de régression_

## 🎯 Objectif

Construire et comparer plusieurs modèles de régression pour prédire le **prix médian des logements** (en milliers de dollars) à Boston, à partir de données socio-économiques et urbaines, en respectant :

- une **analyse éthique** des variables,
- un **pipeline de préparation** propre et reproductible,
- une **validation croisée** rigoureuse pour le benchmark.

---
## 📝 Notebook

Pour ce projet j'ai utilisé un Notebook pour consigner mon travail. Pour cela j'ai utilisé JupyterLab.

````
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
python3 -m ipykernel install --user
jupyter lab
````

Le Notebook est disponible dans ce dépôt : [Notebook](notebook.ipynb)
---

## 🧩 Méthodologie

### 1. Chargement & exploration

- Chargement du dataset `bostonhousing.csv`.
- Inspection des dimensions, types de variables et statistiques descriptives.
- Visualisation des distributions pour chaque feature.
- Vérification des valeurs manquantes (via `missingno`) : **aucune imputation nécessaire**.

---

### 2. Nettoyage & analyse éthique

#### Variable supprimée

Dans la version finale du notebook, **une seule variable est retirée du jeu de données avant la modélisation** :

| Variable | Description (classique Boston)                                           | Décision       | Raison principale |
|----------|---------------------------------------------------------------------------|----------------|-------------------|
| `b`      | Indice lié à la proportion de population afro-américaine dans le quartier | ❌ Supprimée    | Variable éthiquement sensible, risque de discrimination dans un contexte d’aide à la décision publique ou privée (urbanisme, crédit, politique du logement). |

Cette suppression est un **choix éthique conscient** : le modèle ne doit pas exploiter une information explicitement ou implicitement raciale pour prédire des prix immobiliers.

---

### 3. Gestion des outliers

Les distributions étant souvent **asymétriques**, une simple règle IQR n’est pas adaptée partout.

J'ai donc mis en place une **fonction hybride** de détection/traitement :

- configuration par variable (`summary`) indiquant la méthode à appliquer :
  - `none` : aucune action,
  - `iqr_standard` : borne IQR classique,
  - `iqr_asymmetric` : borne IQR adaptée aux distributions asymétriques,
  - `percentile` : capping par percentiles (ex. 1ᵉʳ–99ᵉ).
- application d’un **capping (winsorisation)** sur les valeurs extrêmes, au lieu de les supprimer.

Résultat : on **atténue l’influence des valeurs extrêmes** sans perdre d’observations.

---

### 4. Préparation des données

Sur le dataset final (`df_final`) :

- Séparation **features** / **cible** (`medv`).
- Identification des variables :
  - **numériques** (continues),
  - **catégorielles** (si présentes).
- Construction d’un pipeline Scikit-learn combinant :
  - **StandardScaler** sur les variables continues,
  - **OneHotEncoder** sur les variables catégorielles,
  - le modèle de régression choisi.

Cette approche garantit :

- une **préparation cohérente** pour tous les modèles,
- aucune fuite de données (fit uniquement sur le train).

---

## 🤖 Modèles et validation

### Modèles testés

Le notebook compare les modèles suivants :

- `Linear Regression`
- `Ridge (alpha=1.0)`
- `Lasso (alpha=0.1)`
- `Decision Tree` (max_depth=5)
- `Random Forest` (100 arbres)
- `Gradient Boosting` (100 itérations)
- `K-Neighbors (k=5)`
- `SVR (rbf)`
- `XGBoost`
- `LightGBM`

### Schéma d’évaluation

- **Validation croisée 5-fold (KFold)** pour chaque modèle.
- Métriques calculées :
  - **R²**
  - **RMSE**
  - **MAE**

Cela permet une **comparaison**, moins dépendante d’un simple split train/test.

---

## 📊 Résultats principaux

D’après la cellule de synthèse du notebook :

- 🥇 **Meilleur modèle : Gradient Boosting**
  - **R² moyen ≈ 0.8498**
  - **RMSE ≈ 3.53** (en milliers de dollars)
- Les modèles ensemblistes (**Random Forest, Gradient Boosting, XGBoost, LightGBM**) **surclassent nettement** les modèles linéaires.
- **Gradient Boosting** et **XGBoost** présentent les **écarts-types les plus faibles** en validation croisée → bonne stabilité.
- Les modèles **linéaires** (Linear, Ridge, Lasso) restent corrects mais clairement en dessous des ensemblistes.
- **SVR (rbf)** est le **moins performant** dans ce benchmark.

---

## 🛠️ Choix techniques clés

- Suppression de la variable **`b`** pour des raisons **éthiques** (éviter un modèle discriminatoire).
- Gestion des outliers par **stratégie hybride** configurable plutôt que règle unique.
- **Standardisation** systématique des variables continues (important pour les modèles à régularisation et SVR).
- **Encodage One-Hot** des variables catégorielles.
- **Validation croisée 5-fold** pour une évaluation robuste.
- Conservation d’une **baseline linéaire** pour mesurer les gains des modèles plus complexes.

---

## ✅ Conclusion

- Le **Gradient Boosting** est retenu comme **modèle optimal** sur ce dataset, avec un **R² ≈ 0.85** et un **RMSE ≈ 3.53**, ce qui correspond à une erreur moyenne d’environ **3–4 milliers de dollars** sur la prédiction du prix médian.
- Les **choix éthiques** (notamment la suppression de `b`) et la **gestion fine des outliers** renforcent la crédibilité du modèle dans un contexte d’usage réel.



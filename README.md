# 📊 Prédiction de la Variable `p0q0` – Analyse, Nettoyage et Modélisation

## 🧾 Description du Projet

Ce projet vise à prédire la variable continue **`p0q0`** à partir de données issues de capteurs ou de systèmes liés à des **gares** et des **trains**. L’objectif est de construire un pipeline complet incluant l’exploration, le nettoyage, l’encodage, la normalisation et l'entraînement de plusieurs modèles de régression, avec un accent particulier sur l’optimisation des hyperparamètres du modèle **XGBoost** à l’aide de **Optuna**.

---

## 📁 Fichiers

- `x_train_final.csv` : Jeu de données contenant les variables explicatives.
- `y_train_final_j5KGWWK.csv` : Fichier contenant la variable cible `p0q0`.
- `main.py` : Code principal du projet.
- `README.md` : Ce fichier.

---

## ⚙️ Étapes de Traitement

### 1. 🔍 Exploration et Prétraitement

- Suppression des colonnes `Unnamed`.
- Analyse de la structure des données (`info`, `describe`, histogrammes).
- Visualisation des outliers via **boxplots**.

### 2. 🧹 Nettoyage

- **Suppression des outliers** : méthode de l’**IQR**.
- Vérification de la distribution de la cible avant/après nettoyage.

### 3. 📆 Traitement des Dates

- Extraction du jour de la semaine.
- **Encodage cyclique** avec sinus et cosinus.

### 4. 🔢 Encodage des Variables Catégorielles

- **Encodage OneHot** de la variable `gare`.
- **Label Encoding** de la variable `train`.

### 5. ⚖️ Normalisation

- Utilisation du **RobustScaler** (résistant aux outliers).

### 6. 🧠 Entraînement de Modèles

- **Régression Linéaire** (baseline).
- **XGBoost Regressor** avec **tuning d’hyperparamètres via Optuna**.

---

## 🧪 Modèles Utilisés

| Modèle                | MAE (exemple) 
|-----------------------|---------------
| Régression Linéaire   | 0,66        |
| XGBoost (tuned) Optuna     |  0,58       |



---

## 🧠 Optimisation avec Optuna

- **Optuna** est utilisé pour optimiser automatiquement les hyperparamètres de **XGBoost**.
- Paramètres optimisés : `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
- Métrique de validation : **MAE** (erreur absolue moyenne) via **validation croisée**.

---

## 📈 Visualisations Clés

- Histogrammes et boxplots de la cible `p0q0`.
- Matrice de corrélation entre variables numériques.
- Distributions après nettoyage.

---

## 🛠️ Librairies Utilisées

```python
pandas, numpy, matplotlib, seaborn, sklearn, xgboost, optuna

---




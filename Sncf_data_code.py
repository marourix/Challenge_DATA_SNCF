#Importation des Bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
#Chargement de la data
dfx=pd.read_csv('/content/x_train_final.csv')
dfy=pd.read_csv('/content/y_train_final_j5KGWWK.csv')
print(dfx.shape)
print(dfy.shape)
print(dfx.head())
print(dfy.head())
dfx=dfx.drop('Unnamed: 0.1',axis=1)
dfx=dfx.drop('Unnamed: 0',axis=1)
dfy=dfy.drop('Unnamed: 0',axis=1)
#Exploration des Données (EDA)
print(dfx.info())
print(dfx.describe())
# 2. Distribution de la cible
plt.figure(figsize=(8, 4))
sns.histplot(dfy['p0q0'], kde=True, bins=30)
plt.title("Distribution de la cible p0q0 (AVANT traitement)")
plt.xlabel("p0q0")
plt.ylabel("Fréquence")
plt.show()
#concaténation des nos variable avec le label pour faire le nettoyage
df_concat = pd.concat([dfx, dfy], axis=1)
# 3. Boxplot sur quelques colonnes numériques
numeric_cols = ['p0q0', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']

plt.figure(figsize=(12, 5))
sns.boxplot(data=df_concat[numeric_cols])
plt.xticks(rotation=45)
plt.title("Boxplot des variables numériques AVANT traitement des outliers")
plt.show()
train=dfx["train"].value_counts()
gare=dfx["gare"].value_counts()
date=dfx["date"].value_counts()
print(train)
print(gare)
print(date)
#Traitement des valeur aberrantes
df_concat.head()
df_concat.describe()
print(df_concat.describe(include='object'))
df_concat.isna().sum()
def remove_outliers_iqr(df):
    """
    Supprime les outliers des colonnes numériques du DataFrame en utilisant la méthode IQR.
    """
    df_clean = df.copy()  # Copier le DataFrame pour éviter les modifications en place
    for col in df.select_dtypes(include=['number']).columns:  # Appliquer aux colonnes numériques
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtrer les valeurs dans l'intervalle acceptable
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

# Application de la fonction
df_cleaned = remove_outliers_iqr(df_concat)
df_cleaned.describe()
df_concat.describe()
#EDA APRÈS le traitement des outliers
print("📊 Structure des données APRÈS traitement des outliers")
print(df_cleaned.info())
print(df_cleaned.describe())

# Distribution de la cible après traitement
plt.figure(figsize=(8, 4))
sns.histplot(df_cleaned['p0q0'], kde=True, bins=30)
plt.title("Distribution de la cible p0q0 (APRÈS traitement)")
plt.xlabel("p0q0")
plt.ylabel("Fréquence")
plt.show()

# Boxplot des colonnes numériques après nettoyage
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_cleaned[numeric_cols])
plt.xticks(rotation=45)
plt.title("Boxplot des variables numériques APRÈS traitement des outliers")
plt.show()
# Sélection des colonnes numériques
numeric_features = df_cleaned.select_dtypes(include='number')

# Matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_features.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title(" Matrice de corrélation APRÈS suppression des outliers")
plt.show()
#Visualisation des données
# Histogramme pour voir la distribution après nettoyage
df_cleaned.hist(figsize=(12, 8), bins=30)
plt.suptitle("Distribution des variables après suppression des outliers")
plt.show()
#Préparation des Données
# Conversion de la colonne en type datetime
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

# Extraction du jour de la semaine (lundi=0, dimanche=6)
df_cleaned['jour_num'] = df_cleaned['date'].dt.dayofweek

# Encodage cyclique : transformation en coordonnées sin et cos
df_cleaned['jour_sin'] = np.sin(2 * np.pi * df_cleaned['jour_num'] / 7)
df_cleaned['jour_cos'] = np.cos(2 * np.pi * df_cleaned['jour_num'] / 7)

# Suppression de la colonne date si elle n'est plus utile
df_cleaned.drop(columns=['date', 'jour_num'], inplace=True)
df_cleaned.head()
# 'handle_unknown="ignore"' permet d’éviter les erreurs si des valeurs inconnues apparaissent en test.
# 'sparse_output=False' signifie que la sortie sera un tableau dense (de type numpy array), plus facile à manipuler.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


encoder.fit(df_cleaned[['gare']])

encoded_train = encoder.transform(df_cleaned[['gare']])

# Création d’un DataFrame à partir des données encodées, avec :
# - des noms de colonnes explicites (ex : "gare_Paris", "gare_Lyon", etc.)
# - les mêmes index que df_cleaned pour conserver l’alignement des lignes.
encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=[f"gare_{cat}" for cat in encoder.categories_[0]],
    index=df_cleaned.index
)

# Fusion du DataFrame original (sans la colonne 'gare') avec le DataFrame contenant les colonnes encodées.
df_cleaned = pd.concat([df_cleaned.drop(columns=['gare']), encoded_train_df], axis=1)
df_cleaned.head()
# Initialiser LabelEncoder
le = LabelEncoder()

# Appliquer Label Encoding aux colonnes 'train'
df_cleaned['train_encoded'] = le.fit_transform(df_cleaned['train'])
df_cleaned=df_cleaned.drop('train',axis=1)
df_cleaned.head()
df_cleaned.head()
"""Séparation de données

On sépare le jeu de données en deux parties :

X contient toutes les variables explicatives
y contient la variable cible (p0q0)
80% des données seront utilisées pour l'entraînement (X_train, y_train)
20% seront réservées pour le test (X_test, y_test)"""
# On suppose que df_cleaned est déjà chargé et prêt à l'emploi
dff = df_cleaned.copy()

# X = toutes les colonnes sauf 'p0q0'
X = dff.drop('p0q0', axis=1)
feature_names = X.columns.tolist() ## Enregistrement de la structure des colonnes utilisées pour l'entraînement
# y = colonne cible
y = dff['p0q0']
# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  # pour la reproductibilité
)
scaler = RobustScaler()

# On suppose que toutes les colonnes de X sont numériques après encodage (ce qui est le cas ici)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Régression Linéaire

# Création et entraînement du modèle de régression linéaire
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Prédictions sur le jeu de test
y_pred = model_lr.predict(X_test)

# Calcul des métriques : MAE et R²
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE (Mean Absolute Error) :", mae)
print("R² Score :", r2)
#XGBOOST regressor
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
#Définition de l'objectif Optuna avec validation croisée
def objective(trial):
param = {
 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
 'max_depth': trial.suggest_int('max_depth', 3, 10),
 'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
 'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1.0),
 'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 1.0),
   }
model = XGBRegressor(**param, random_state=42)
#Cross-validation sur 5 folds
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
return -scores.mean()  # Minimiser le MAE
#Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Optimisation avec Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
#Meilleurs hyperparamètres trouvés
best_params = study.best_params
print(f"Meilleurs hyperparamètres Optuna : {best_params}")
#Entraîner le modèle final avec les meilleurs hyperparamètres
best_model = XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)
#Prédictions finales
y_pred_best = best_model.predict(X_test)
#Évaluation
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print("📌 XGBoost avec Optuna et Cross Validation")
print("🔹 MAE:", mae_best)
print("🔹 R² Score:", r2_best)
print("-" * 40)

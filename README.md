# SMS Spam Detection — Pipeline Data Mining

Détection de spam SMS via **NLP + Machine Learning**, plutôt que des règles manuelles fixes.

## Dataset
- **Source** : UCI SMS Spam Collection
- **Total** : 5572 messages — 4825 ham (86.6%) / 747 spam (13.4%)
- Dataset déséquilibré → métriques principales : **F1-Score** et **ROC-AUC**

## Pipeline

**1. Data Selection**
- Chargement et aperçu du dataset (TSV)

**2. Data Cleaning**
- Nettoyage texte (stopwords, ponctuation, casse)
- Réduction moyenne : ~44% du texte supprimé (bruit)

**3. EDA (Exploratory Data Analysis)**
- Heatmap de corrélation des features avec le label
- Variables clés : `has_number`, `has_free`, `msg_length`, `has_url`

**4. Feature Selection / Vectorisation**
- TF-IDF (Term Frequency – Inverse Document Frequency)
- Transforme le texte en vecteurs numériques pondérés

**5. Comparaison des modèles**
- Naive Bayes, Logistic Regression, Linear SVC
- Validation croisée 5-fold
- **Meilleur modèle : Linear SVC (F1 = 0.9230)**

**6. Évaluation finale**
- Matrice de confusion, rapport de classification
- Courbe ROC (AUC > 0.98)

## Stack technique
- Python, Jupyter Lab
- pandas, scikit-learn, matplotlib, seaborn
- Flask (web app de démo)
- joblib (sérialisation du modèle)

## Structure
```
├── spam_pipeline_final.ipynb   # notebook complet (standalone)
├── data/                       # dataset UCI
└── output/                     # graphiques générés
```

## Lancer le projet
```bash
jupyter lab spam_pipeline_final.ipynb
# ou
python app.py
```

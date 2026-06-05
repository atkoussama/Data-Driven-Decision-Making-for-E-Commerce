# Data-Driven Decision Making for E-Commerce

## Overview

L'objectif est de construire un système complet d'aide à la décision permettant d'analyser l'activité d'une entreprise e-commerce, d'identifier les opportunités de croissance et de transformer les résultats analytiques en recommandations métier exploitables.

Le projet couvre l'ensemble du pipeline DDDM :

```text
Business Question
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Analysis
        ↓
Machine Learning
        ↓
Interpretability
        ↓
Dashboarding
        ↓
Business Recommendations
        ↓
A/B Testing Plan
```

---

# Business Problem

Le projet répond aux questions métier suivantes :

- Quels produits génèrent le plus de revenus ?
- Quels clients sont les plus rentables ?
- Quels segments doivent être ciblés par les campagnes marketing ?
- Quels facteurs influencent les retours produits ?
- Quelles actions peuvent améliorer les ventes et la rentabilité ?

---

# Datasets

Le projet combine deux sources de données :

## 1. E-commerce Sales Transactions

Environ 34 500 transactions contenant :

- commandes
- produits
- clients
- catégories
- paiements
- marges
- retours
- délais de livraison

## 2. Online Retail Dataset

Environ 541 909 transactions contenant :

- factures
- produits
- quantités
- prix unitaires
- pays

Le volume total dépasse :

```text
576 000 lignes
```

---

# Data Processing

Les données ont été :

- nettoyées
- validées
- transformées
- enrichies

Variables créées :

- net_price
- shipping_ratio
- revenue_per_item
- is_high_discount
- is_slow_delivery
- is_negative_margin
- age_group

---

# Machine Learning

## Customer Segmentation

Algorithme :

```text
K-Means Clustering
```

Variables utilisées :

- nombre de commandes
- dépenses totales
- taux de retour
- délai moyen
- marge moyenne

Segments identifiés :

- Clients fidèles premium
- Clients sensibles aux remises
- Acheteurs ponctuels
- Acheteurs à haut risque

---

## Return Risk Prediction

Objectif :

Prédire le risque de retour d'une commande.

Modèles évalués :

- Logistic Regression
- Random Forest
- Gradient Boosting

Métriques :

- AUC-ROC
- F1 Score
- Precision
- Recall
- Accuracy

---
# Installation

Pré-requis : Python 3.8+ et `git`.

Installer les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

# Exécution rapide

Pour lancer le pipeline (génère les fichiers dans `outputs/`) :

```bash
python run_pipeline.py
```

Pour ouvrir l'analyse interactive :

```bash
jupyter notebook analysis.ipynb
```

# Architecture

Le projet suit un pipeline de traitement de données et de modélisation :

- Ingestion (fichiers sources) → Prétraitement (`ddm/`)
- Analyse exploratoire (`analysis.ipynb`)
- Modélisation (scripts et modules dans `ddm/`)
- Export des résultats → `outputs/`

Diagramme simplifié (ASCII) :

```text
raw_data/  -->  ddm.preprocessing  --> features/  --> models/  --> outputs/
                                                        |                                  |
                                                notebooks                         reports / pbix
```
# Dashboard Power BI

Le dashboard contient 5 vues distinctes.

## 1. Executive Overview

Vue globale de l'activité :

- CA total
- nombre de commandes
- nombre de clients
- panier moyen
- évolution du CA

## 2. Direction Dashboard

Vue stratégique :

- profit
- marge
- top clients
- rentabilité

## 3. Operations Dashboard

Vue opérationnelle :

- volume des ventes
- délais
- retours
- régions

## 4. Marketing Dashboard

Vue marketing :

- segmentation
- catégories
- profils clients
- analyse produit

## 5. AI Recommendations Dashboard

Vue décisionnelle :

- résultats IA
- importance des variables
- recommandations
- plan d'action

---

# Project Structure

```text
repo/
├── analysis.ipynb        
├── run_pipeline.py       
├── ddm/                  
├── requirements.txt 
├── outputs/              
├── README.md     
└── powerbi/
    └── DDDM_dashboards.pbix
```
## Livrables

- [A/B Test Plan](docs/AB_Test_Plan.pdf)
---

## Fichiers générés

Après exécution, les fichiers suivants sont créés dans le dossier **`outputs/`** :
- `orders_enriched.xlsx` — Commandes enrichies avec montants totaux et dates
- `customer_segments.xlsx` — Segmentation client (clustering K-Means)
- `feature_importances.xlsx` — Importance des variables pour modélisation
- `model_comparison.xlsx` — Comparaison des modèles (LogReg, Random Forest, Decision Tree)
- `monthly_statistics.xlsx` — Statistiques mensuelles (CA, nombre de commandes)

# Key Business Recommendations

## Recommendation 1

Fidéliser les clients premium.

Impact attendu :

- augmentation du CA
- amélioration de la rétention

## Recommendation 2

Réduire les retours produits.

Impact attendu :

- réduction des coûts logistiques
- amélioration de la satisfaction client

## Recommendation 3

Cibler les clients sensibles aux remises.

Impact attendu :

- augmentation du taux de conversion

---

# A/B Testing Plan

Hypothèse :

Une remise de 10 % appliquée aux clients sensibles aux remises augmente le chiffre d'affaires sans dégrader significativement la marge.

Groupe A :

- sans promotion

Groupe B :

- promotion de 10 %

KPI suivis :

- chiffre d'affaires
- panier moyen
- taux de retour
- marge

Durée :

```text
4 semaines
```

---

# Authors

Oussama AIT ELKABIR

Akram AL OUMAMI

Ayoub JNIEH

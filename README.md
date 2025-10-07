# 📡 Tableau de bord Streamlit – ETA & Couverture RT

Cette version ajoute l'analyse du **fichier de couverture de l'information voyageur en temps réel** (suivi expliqué / manquements) **dans la même app** que l'accuracy ETA, avec un onglet de **corrélation** entre les deux.

## Démarrer
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement (streamlit.app)
1. Poussez ces fichiers sur GitHub.
2. Sur https://streamlit.app → *New app* → choisissez ce dépôt → `app.py`.
3. Déployez, puis téléversez vos deux CSV dans l'interface web.

## Format attendu – Couverture
Colonnes: `startDate`, `endDate`, `route`, `timePeriod`, `scheduledTripStops`, `countTrackedExplained`, `countOnFullyMissingTrips`, `countMissingOther`, `fractionTrackedExplained`, `fractionOnFullyMissingTrips`, `fractionMissingOther`.

## Fonctionnalités ajoutées
- KPI couverture (suivi expliqué, trajets entièrement manquants, autres manquements)
- Barres empilées par **période**
- Barres (avec **IC de Wilson 95%**) par **route**
- **Heatmap** route × période
- **Pareto** des manquements
- Onglet **Corrélation** ETA ↔ Couverture (scatter + r de Pearson)

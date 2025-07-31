# nba

DATA:
- setup cron job on railway (with retry que + email)
- use a-shell on iphone as cron job to pull data
	- have a-shell pull the github before running the job. email self for failure to manual run

GENERAL: 
- create a data loader for cleaner scripting (calcs spreads, sets player buckets etc.)

MODEL:
- model minutes for a player (to be used in point prediction)
- model points given current feature sets
	- need to try both raw feature values, and feature values noramlized (besides points to provide scale)
	- Regression, XGBoost, SVM, RandomForest, NN, Attention Transformer? 
- create backtesting framework (shadow bet baseline / RMSE)

ODDS:
- scrape draftkings, betmgm, fanduel



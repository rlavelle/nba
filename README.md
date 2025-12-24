# nba

TODO:
- Rewrite model building and prediction framework
	- needs ability for offline historical testing
	- dont recompute features daily for the entire dataset
	- collect both sides of ML and spread (need away team ids)

- Framework to grade previous games predictions, and rolling seasomal predictions
- Criterion for unit betting based on all provided predictions

- Make email output look better (embeded HTML or markdown?)

Backlog:
- spread model needs a lot more EDA
- look into pre season data for all types of modeling
- look into win streak as a predictor
- write process to merge prod DB with local?
	- or just overwrite local with prod?
- process to repair data scrape failures (create queue to rerun next day?)



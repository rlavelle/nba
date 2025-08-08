# nba

INTRA SEASON PROD:
- At some point need to put a pin in modeling, take best current attempt, and build prod framework around that
- finish off data scraping scripts (trigger model building / prediction on complete)
- iterative model building (if LM) or shadow training (if deep learning)
- automated results storing (if scraping odds successful)

DATA:
- setup cron job on railway (with retry que + email)
- use a-shell on iphone as cron job to pull data
	- have a-shell pull the github before running the job. email self for failure to manual run

ODDS:
- scrape draftkings, betmgm, fanduel
- free tier odds api? pull data right before games (should be under limit, but no opening line)


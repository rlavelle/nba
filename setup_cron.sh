#!/bin/bash

# Get absolute path to project root
PROJECT_DIR="$(pwd)"

# Cron entries
CRON_JOB_1="0 5 * * * cd $PROJECT_DIR && python3 -m src.scrapers.jobs.get_prev_nba_data"
CRON_JOB_2="5 5 * * * cd $PROJECT_DIR && python3 -m src.scrapers.jobs.get_next_nba_data1"

CRON_JOB_3="0 7 * * * cd $PROJECT_DIR && python3 -m src.modeling_framework.jobs.build_models"
CRON_JOB_4="0 10 * * * cd $PROJECT_DIR && python3 -m src.modeling_framework.jobs.predict"

# Install cron jobs (prevent duplicates)
( crontab -l | grep -v "get_prev_nba_data"; echo "$CRON_JOB_1" ) | crontab -
( crontab -l | grep -v "get_next_nba_data"; echo "$CRON_JOB_2" ) | crontab -
( crontab -l | grep -v "build_models"; echo "$CRON_JOB_3" ) | crontab -
( crontab -l | grep -v "predict"; echo "$CRON_JOB_4" ) | crontab -

echo "Cron jobs installed"
crontab -l

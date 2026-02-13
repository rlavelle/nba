#!/bin/bash
YESTERDAY=$(date -v -1d +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)
VENV_PATH="venv/bin/python3"

$VENV_PATH -m src.scrapers.jobs.get_prev_nba_data --date=$YESTERDAY && \
$VENV_PATH -m src.scrapers.jobs.get_next_nba_data && \
$VENV_PATH -m src.modeling_framework.jobs.build_models --cache --date=$TODAY && \
$VENV_PATH -m src.modeling_framework.jobs.predict --cache --clean-up --date=$TODAY && \
$VENV_PATH -m src.modeling_framework.jobs.grade_predictions --admin --date=$TODAY && \
$VENV_PATH -m src.scrapers.jobs.get_failed_nba_data

#!/bin/bash
venv/bin/python3 -m src.scrapers.jobs.get_prev_nba_data --date=$(date -v -1d +%Y-%m-%d) && \
venv/bin/python3 -m src.scrapers.jobs.get_next_nba_data && \
venv/bin/python3 -m src.modeling_framework.jobs.build_models && \
venv/bin/python3 -m src.modeling_framework.jobs.predict

import datetime

import numpy as np
import pandas as pd

from src.db.db_manager import DBManager
from src.logging.email_sender import EmailSender
from src.modeling_framework.jobs.utils.grading import hit_by_delta, get_pct_results
from src.utils.date import fmt_iso_dint


def fmt_diff_data(data):
    meta_cols = [
        "season", "season_type", "season_type_code", "dint",
        "date", "team_id", "is_home", "stat_type"
    ]

    home = data[data["is_home"] == 1].set_index("game_id")
    away = data[data["is_home"] == 0].set_index("game_id")
    meta = home[meta_cols].reset_index()

    home_stats = home.drop(columns=meta_cols)
    away_stats = away.drop(columns=meta_cols)

    diff_stats = home_stats - away_stats
    diff_stats = diff_stats.reset_index()

    df_diff = pd.concat([meta, diff_stats.drop(columns=["game_id"])], axis=1)

    return df_diff


def fmt_player_data(data):
    data = data[(data.points > 0) | (data.points.isna())].copy()

    data = data.sort_values(by=['player_id', 'season', 'date'])

    data['ppm_s1'] = (
        data.groupby(['player_id', 'season'])['ppm']
        .expanding()
        .mean()
        .shift(1)
        .fillna(0)
        .reset_index(drop=True)
        .values
    )
    data['ppm_diff'] = data.ppm - data.ppm_s1
    return data


# TODO: this is a bad way to reconcile games.... lol
def format_testing_data(odds, test_data):
    tmp = test_data.copy()

    odds['is_home'] = (odds.index % 2 == 0).astype(int)
    odds['game_id'] = odds.index // 2

    tmp = tmp.drop(columns=['is_home', 'game_id'])

    odds['team_id'] = odds.team_id.astype(int)
    tmp['team_id'] = tmp.team_id.astype(int)

    tmp_home = pd.merge(odds[['is_home', 'game_id', 'team_id']], tmp,  on=['team_id'],  how='left')

    odds['is_home'] = np.where(odds.is_home==1, 0, 1)
    tmp_away = pd.merge(odds[['is_home', 'game_id', 'team_id']], tmp,  on=['team_id'],  how='left')

    return fmt_diff_data(tmp_home), fmt_diff_data(tmp_away)


def pretty_print_results(prop_r, ml_r):
    prop_r = prop_r.copy()
    ml_r = ml_r.copy()

    dbm = DBManager()
    players = dbm.get_players()
    teams = dbm.get_teams()

    if prop_r is not None:
        prop_r = prop_r[~prop_r.price.isna()]
        prop_r['delta'] = prop_r.preds - prop_r.point
        prop_r['delta_pct'] = (prop_r.preds / prop_r.point) - 1

        prop_r['preds'] = prop_r['preds'].round(1)
        prop_r['delta'] = prop_r['delta'].round(1)
        prop_r['delta_pct'] = prop_r['delta_pct'].apply(lambda x: f"{x:.1%}")

        prop_r = prop_r[['player_id', 'bookmaker',
                         'description', 'price', 'point', 'preds', 'delta']].copy()
        prop_r = pd.merge(prop_r, players, on='player_id', how='left')
        prop_r = prop_r.drop(columns=['player_slug'])



    # if spread_r is not None:
    #     spread_r = spread_r[['team_id', 'bookmaker', 'price', 'point', 'preds']].copy()
    #     spread_r = pd.merge(spread_r, teams, on='team_id', how='left')

    if ml_r is not None:
        ml_r['vegas_preds'] = 1 / ml_r.price
        ml_r['price_pred'] = 1 / ml_r.preds

        ml_r['price_pred'] = ml_r['price_pred'].round(1)
        ml_r['vegas_preds'] = ml_r['vegas_preds'].apply(lambda x: f"{x:.1%}")
        ml_r['preds'] = ml_r['preds'].apply(lambda x: f"{x:.1%}")

        ml_r = ml_r[['team_id', 'bookmaker', 'price', 'price_pred', 'vegas_preds', 'preds', 'is_home']].copy()
        ml_r = pd.merge(ml_r, teams, on='team_id', how='left')

    md = f"""
    # NBA Results {datetime.date.today()}
    Column descriptions:
    - price: decimal odds (1.83 means 83 cents back per dollar bet)
    - point: vegas line for prop bet
    - preds: model prediction (price_pred is model predicted odds for moneyline)
    
    prediction for prop bets is the predicted points scored for a player, prediction for moneyline is percent chance of win, or predicted odds.
    
    
    ### PLAYER PROPS
    {prop_r.to_markdown(index=False)}
    
    ### MONEY LINE
    {ml_r.to_markdown(index=False)}
    
    -------------------------------------
    *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    html = f"""
    # NBA Results {datetime.date.today()}
    Column descriptions:
    - price: decimal odds (1.83 means 83 cents back per dollar bet)
    - point: vegas line for prop bet
    - preds: model prediction (price_pred is model predicted odds for moneyline)
    
    prediction for prop bets is the predicted points scored for a player, prediction for moneyline is percent chance of win, or predicted odds.

    ### PLAYER PROPS
    {prop_r.to_html(index=False)}

    ### MONEY LINE
    {ml_r.to_html(index=False)}

    -------------------------------------
    *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    if ml_r.empty and prop_r.empty:
        return "No odds available today, go home", "No odds available today, go home"

    return md, html


def pretty_print_grading(game_wins, player_wins, game_wins_prev, player_wins_prev):
    # TODO: this if very sloppy thrown together, plz fix later, thx
    rtotal = hit_by_delta(player_wins)
    rprev = hit_by_delta(player_wins_prev)

    vegas_ml_res, model_ml_res, diff, ngames, model_prop_res, nplayers = get_pct_results(game_wins, player_wins)
    vegas_ml_res1, model_ml_res1, diff1, ngames1, model_prop_res1, nplayers1 = get_pct_results(game_wins_prev, player_wins_prev)

    md = f"""
    # NBA Bet Grading {datetime.date.today() - datetime.timedelta(days=-1)}
       
    ### Yesterdays Results
    Vegas favorites: {round(vegas_ml_res1, 2)*100}%
    Model favorites: {round(model_ml_res1, 2)*100}%
    Vegas fade Model favorite: {round(diff1, 2)*100}%
    N Moneyline bets: {ngames1}
   
    Model Prop O/U: {round(model_prop_res1, 2)*100}%
    N Prop bets: {nplayers1}
   
    {rprev.to_markdown(index=False)}
   
    ### Total Season Results
    Vegas favorites: {round(vegas_ml_res, 2)*100}%
    Model favorites: {round(model_ml_res, 2)*100}%
    Vegas fade Model favorite: {round(diff, 2)*100}%
    N Moneyline bets: {ngames}
   
    Model Prop O/U: {round(model_prop_res, 2)*100}%
    N Prop bets: {nplayers}
   
    {rtotal.to_markdown(index=False)}
   

    -------------------------------------
    *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    html = f"""
    # NBA Bet Grading {datetime.date.today() - datetime.timedelta(days=-1)}
       
    ### Yesterdays Results
    Vegas favorites: {round(vegas_ml_res1, 2)*100}%
    Model favorites: {round(model_ml_res1, 2)*100}%
    Vegas fade Model favorite: {round(diff1, 2)*100}%
    N Moneyline bets: {ngames1}
   
    Model Prop O/U: {round(model_prop_res1, 2)*100}%
    N Prop bets: {nplayers1}
   
    {rprev.to_html(index=False)}
   
    ### Total Season Results
    Vegas favorites: {round(vegas_ml_res, 2)*100}%
    Model favorites: {round(model_ml_res, 2)*100}%
    Vegas fade Model favorite: {round(diff, 2)*100}%
    N Moneyline bets: {ngames}
   
    Model Prop O/U: {round(model_prop_res, 2)*100}%
    N Prop bets: {nplayers}
   
    {rtotal.to_html(index=False)}
   

    -------------------------------------
    *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    if game_wins_prev.empty and player_wins_prev.empty:
        return "No odds available today, go home", "No odds available today, go home"

    return md, html


def prep_odds(odds: pd.DataFrame, bookmakers: list[str], curr_date:int):
    odds = odds.copy()

    # TODO: patch for now... needs a cleaner solution based on commence_time and better
    #       timezone handling...
    #       this does not work at all, patched the patch
    odds['dint_tmp'] = odds.last_update.apply(fmt_iso_dint)
    odds = odds[(odds.bookmaker.isin(bookmakers)) & (odds.dint_tmp == curr_date)]
    odds = odds.drop(columns=['last_update', 'dint', 'dint_tmp'])
    return odds.drop_duplicates(keep='first')


def send_results(subject, msg, admin):
    email_sender = EmailSender()
    email_sender.read_recipients_from_file()
    email_sender.set_subject(subject)
    email_sender.set_body(msg)
    email_sender.send_email(admin=admin)

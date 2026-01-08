import numpy as np
import pandas as pd


def format_game_data(games, ml_results):
    games = games.dropna()
    games = games.sort_values(by=['team_id', 'season', 'date'])

    games['team_id'] = games.team_id.astype(str)
    ml_results['team_id'] = ml_results.team_id.astype(str)
    game_data = pd.merge(ml_results, games,
                         on=['team_id', 'dint'],
                         how='left')

    game_data = game_data[~game_data.game_id.isna()].copy()
    game_data['vegas_preds'] = 1.0 / game_data.price

    winners_idx = game_data.groupby(game_data.game_id).points.idxmax()
    pred_winners_idx = game_data.groupby(game_data.game_id).preds.idxmax()
    vegas_winners_idx = game_data.groupby(game_data.game_id).vegas_preds.idxmax()

    game_data['win'] = 0
    game_data.loc[winners_idx, 'win'] = 1

    game_data['win_pred'] = 0
    game_data.loc[pred_winners_idx, 'win_pred'] = 1

    game_data['win_vegas'] = 0
    game_data.loc[vegas_winners_idx, 'win_vegas'] = 1

    return game_data


def format_player_data(player_data, prop_results):
    player_data = player_data[~player_data.spread.isna()].copy()
    player_data = player_data.drop(columns=['position'])
    player_data = player_data.dropna()
    player_data = player_data.sort_values(by=['player_id', 'season', 'date'])

    player_data = pd.merge(prop_results, player_data,
                           on=['player_id', 'dint'],
                           how='left')

    tmp = player_data[~player_data.points.isna()].copy()
    tmp['bet'] = np.where(((tmp.description == 'Over') & (tmp.preds > tmp.point)) | (
            (tmp.description == 'Under') & (tmp.preds < tmp.point)), 1, 0)
    tmp['win'] = np.where(((tmp.description == 'Over') & (tmp.points > tmp.point)) | (
            (tmp.description == 'Under') & (tmp.points < tmp.point)), 1, 0)
    tmp['delta'] = np.abs(tmp.preds - tmp.point)

    return tmp


def hit_by_delta(wins):
    results = []
    max_floor = int(wins.delta.max())

    for threshold in range(0, max_floor + 1):
        subset = wins[wins.delta > threshold].copy()
        subset2 = wins[(wins.delta >= threshold) & (wins.delta < threshold + 1)]
        group_name = f"delta > {threshold} | bucket = [{threshold}, {threshold + 1})"

        mean_bet = subset.bet.mean()
        mean_bet2 = subset2.bet.mean()
        count = len(subset)
        count2 = len(subset2)
        results.append({
            'group': group_name,
            'cum_bet': mean_bet,
            'bucket_bet': mean_bet2,
            'n_total': count,
            'n_bucket': count2
        })

    return pd.DataFrame(results)


def get_pct_results(win1, win2):
    vegas_ml_res = win1.win_vegas.mean()
    model_ml_res = win1.win_pred.mean()
    diff = win1[win1.win_vegas != win1.win_pred].win_pred.mean()
    ngames = win1.shape[0]

    model_prop_res = win2.bet.mean()
    nplayers = win2.shape[0]

    return vegas_ml_res, model_ml_res, diff, ngames, model_prop_res, nplayers

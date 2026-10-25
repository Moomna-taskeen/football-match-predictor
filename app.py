import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from scipy.stats import poisson

BUNDLE_PATH = "football_predictor.pkl"
if not os.path.exists(BUNDLE_PATH):
    raise FileNotFoundError(
        "football_predictor.pkl is missing. Upload your exported Kaggle model bundle "
        "to the root of this Hugging Face Space."
    )

bundle = joblib.load(BUNDLE_PATH)
outcome_model = bundle["outcome_model"]
home_goal_model = bundle["home_goal_model"]
away_goal_model = bundle["away_goal_model"]
history = bundle["history"].copy()
ratings = bundle["elo_ratings"]
feature_columns = bundle["feature_columns"]

history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
history = history.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
history = history.sort_values("Date").reset_index(drop=True)

teams = sorted(set(history["HomeTeam"].astype(str)) | set(history["AwayTeam"].astype(str)))
OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}

def team_match_history(team):
    """Return matches in the team's perspective, oldest to newest."""
    home = history[history["HomeTeam"] == team].copy()
    home_rows = pd.DataFrame({
        "Date": home["Date"], "Team": team, "Opponent": home["AwayTeam"],
        "GoalsFor": home["FTHG"].astype(float), "GoalsAgainst": home["FTAG"].astype(float),
        "Points": np.where(home["FTR"] == "H", 3, np.where(home["FTR"] == "D", 1, 0)),
        "Venue": "home"
    })
    away = history[history["AwayTeam"] == team].copy()
    away_rows = pd.DataFrame({
        "Date": away["Date"], "Team": team, "Opponent": away["HomeTeam"],
        "GoalsFor": away["FTAG"].astype(float), "GoalsAgainst": away["FTHG"].astype(float),
        "Points": np.where(away["FTR"] == "A", 3, np.where(away["FTR"] == "D", 1, 0)),
        "Venue": "away"
    })
    return pd.concat([home_rows, away_rows], ignore_index=True).sort_values("Date")

team_histories = {team: team_match_history(team) for team in teams}

def recent_features(team, venue=None):
    rows = team_histories[team]
    if venue is not None:
        rows = rows[rows["Venue"] == venue]
    rows = rows.tail(5)
    if rows.empty:
        return 0.0, 0.0, 0.0
    return (
        float(rows["Points"].sum()),
        float(rows["GoalsFor"].mean()),
        float(rows["GoalsAgainst"].mean())
    )

def build_feature_row(home_team, away_team):
    h_points, h_for, h_against = recent_features(home_team)
    a_points, a_for, a_against = recent_features(away_team)
    hv_points, hv_for, hv_against = recent_features(home_team, "home")
    av_points, av_for, av_against = recent_features(away_team, "away")

    home_elo = float(ratings.get(home_team, 1500.0))
    away_elo = float(ratings.get(away_team, 1500.0))
    elo_diff = home_elo - away_elo

    values = {
        "HomeFormPoints5": h_points,
        "HomeAvgGoalsFor5": h_for,
        "HomeAvgGoalsAgainst5": h_against,
        "AwayFormPoints5": a_points,
        "AwayAvgGoalsFor5": a_for,
        "AwayAvgGoalsAgainst5": a_against,
        "HomeVenuePoints5": hv_points,
        "HomeVenueGoalsFor5": hv_for,
        "HomeVenueGoalsAgainst5": hv_against,
        "AwayVenuePoints5": av_points,
        "AwayVenueGoalsFor5": av_for,
        "AwayVenueGoalsAgainst5": av_against,
        "EloDifference": elo_diff,
        "AbsEloDifference": abs(elo_diff),
        "EloDifferenceSquared": elo_diff ** 2,
    }
    missing = [c for c in feature_columns if c not in values]
    if missing:
        raise ValueError("Missing expected model features: " + ", ".join(missing))
    return pd.DataFrame([{c: values[c] for c in feature_columns}])

def predict_match(home_team, away_team):
    if not home_team or not away_team:
        raise gr.Error("Please select both teams.")
    if home_team == away_team:
        raise gr.Error("Choose two different teams.")

    X = build_feature_row(home_team, away_team)

    # The model was trained with classes H, D, A; explicitly align probability order.
    raw_probs = outcome_model.predict_proba(X)[0]
    model_classes = list(outcome_model.classes_)
    probs = {label: float(raw_probs[model_classes.index(label)]) for label in ["H", "D", "A"]}

    home_goals = max(0.01, float(home_goal_model.predict(X)[0]))
    away_goals = max(0.01, float(away_goal_model.predict(X)[0]))

    goal_values = np.arange(7)
    h_goal_probs = poisson.pmf(goal_values, home_goals)
    a_goal_probs = poisson.pmf(goal_values, away_goals)
    score_matrix = np.outer(h_goal_probs, a_goal_probs)
    best_index = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    likely_score = f"{best_index[0]} – {best_index[1]}"

    winner = max(probs, key=probs.get)
    probability_table = pd.DataFrame({
        "Outcome": ["Home win", "Draw", "Away win"],
        "Estimated probability": [probs["H"], probs["D"], probs["A"]]
    })
    probability_table["Estimated probability"] = (
        probability_table["Estimated probability"].map(lambda p: f"{p:.1%}")
    )

    summary = (
        f"### {home_team} vs {away_team}\n"
        f"**Most likely outcome (classification model):** {OUTCOME_NAMES[winner]}\n\n"
        f"**Expected goals:** {home_team} {home_goals:.2f} — {away_goals:.2f} {away_team}\n\n"
        f"**Most likely scoreline from the goal model:** {likely_score}\n\n"
        "**Important:** These are experimental model estimates, not guarantees. "
        "The probability estimates have not yet demonstrated strong log-loss performance."
    )
    return summary, probability_table

with gr.Blocks(title="Football Match Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Football Match Predictor\n"
        "### Explore estimated outcomes and expected goals\n"
        "Select two teams from the historical dataset."
    )
    with gr.Row():
        home_dropdown = gr.Dropdown(choices=teams, value=teams[0] if teams else None, label="Home team")
        away_default = teams[1] if len(teams) > 1 else (teams[0] if teams else None)
        away_dropdown = gr.Dropdown(choices=teams, value=away_default, label="Away team")
    predict_button = gr.Button("Predict match", variant="primary")
    result = gr.Markdown()
    probabilities = gr.Dataframe(
        headers=["Outcome", "Estimated probability"],
        label="Outcome probability estimates",
        interactive=False
    )
    predict_button.click(
        fn=predict_match,
        inputs=[home_dropdown, away_dropdown],
        outputs=[result, probabilities]
    )
    gr.Markdown(
        "---\n"
        "**About this prototype:** Trained on historical English football results. "
        "It does not use live injuries, lineups, odds, or confirmed future fixtures."
    )

if __name__ == "__main__":
    demo.launch()

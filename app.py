import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
import os
import datetime
from google import genai
from google.genai import types

# --- API Key Setup ---
# NEVER commit your real API key to GitHub. Use environment variables.
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Premier League ML Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Premier League ML & Hybrid AI Predictor")
st.markdown("Neural Network predicts Expected Goals (xG) ➡️ Gemini AI adjusts based on live news & odds.")


# --- 1. Fetch ONLY UNPLAYED Fixtures (With Lookahead Logic) ---
def fetch_unplayed_fixtures():
    bootstrap_res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = {t['id']: t['name'] for t in bootstrap_res['teams']}
    fixtures_res = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()

    # Filter matches that have been played or lack a date/gameweek (e.g., postponed matches)
    unplayed = [f for f in fixtures_res if
                not f['finished'] and f['kickoff_time'] is not None and f['event'] is not None]
    unplayed = sorted(unplayed, key=lambda x: x['kickoff_time'])

    if not unplayed:
        return [], "None", False

    # Group matches by Gameweek
    gws = {}
    for f in unplayed:
        gw = f['event']
        if gw not in gws:
            gws[gw] = []
        gws[gw].append(f)

    sorted_gws = sorted(list(gws.keys()))
    current_gw = sorted_gws[0]
    matches_to_predict = gws[current_gw]

    target_gw_str = str(current_gw)
    show_time_warning = False

    # Logic: If fewer than 4 matches remain in the current Gameweek, include the next Gameweek's matches
    if len(matches_to_predict) < 4 and len(sorted_gws) > 1:
        next_gw = sorted_gws[1]
        next_gw_matches = gws[next_gw]
        matches_to_predict.extend(next_gw_matches)
        target_gw_str = f"{current_gw} & {next_gw}"

        # Check if the first match of the next Gameweek is more than 4 days away
        first_match_time = pd.to_datetime(next_gw_matches[0]['kickoff_time']).tz_localize(None)
        now = datetime.datetime.utcnow()
        if (first_match_time - now).days > 4:
            show_time_warning = True

    name_mapping = {"Man Utd": "Man United", "Spurs": "Tottenham"}
    upcoming_matches = []

    for f in matches_to_predict:
        home = name_mapping.get(teams[f['team_h']], teams[f['team_h']])
        away = name_mapping.get(teams[f['team_a']], teams[f['team_a']])
        upcoming_matches.append((home, away, f['event']))

    return upcoming_matches, target_gw_str, show_time_warning


# --- 2. Data Engineering & Advanced Features ---
def fetch_premier_league_data():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    ]
    dfs = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            dfs.append(df)
    matches_df = pd.concat(dfs, ignore_index=True)

    # Convert Date to datetime to calculate chronological form correctly
    matches_df['Date'] = pd.to_datetime(matches_df['Date'], dayfirst=True, errors='coerce')
    matches_df = matches_df.sort_values('Date').dropna(subset=['Date'])

    columns_needed = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HST', 'AST']
    return matches_df[columns_needed].dropna()


def engineer_features_and_split(df, train_ratio=0.7):
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    team_stats = {}
    teams = pd.unique(train_df[['HomeTeam', 'AwayTeam']].values.ravel())

    for team in teams:
        home_games = train_df[train_df['HomeTeam'] == team]
        away_games = train_df[train_df['AwayTeam'] == team]
        total_games = len(home_games) + len(away_games)
        if total_games == 0:
            continue

        home_wins = len(home_games[home_games['FTR'] == 'H'])
        away_wins = len(away_games[away_games['FTR'] == 'A'])

        # Calculate recent FORM (Approximation using last 10 combined games of the team)
        all_team_games = train_df[(train_df['HomeTeam'] == team) | (train_df['AwayTeam'] == team)].tail(10)
        form_points = 0
        for _, row in all_team_games.iterrows():
            if row['HomeTeam'] == team and row['FTR'] == 'H':
                form_points += 3
            elif row['AwayTeam'] == team and row['FTR'] == 'A':
                form_points += 3
            elif row['FTR'] == 'D':
                form_points += 1

        team_stats[team] = {
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0.3,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0.2,
            'attack_power': (home_games['FTHG'].sum() + away_games['FTAG'].sum()) / total_games,
            'defense_power': (home_games['FTAG'].sum() + away_games['FTHG'].sum()) / total_games,
            'recent_form': form_points / 30.0  # Normalized 0-1
        }

    def create_dataset(data_split):
        X, y_goals = [], []
        for _, row in data_split.iterrows():
            h_team, a_team = row['HomeTeam'], row['AwayTeam']

            # Default stats if team is newly promoted or lacks historical data
            default_stats = {'home_win_rate': 0.3, 'away_win_rate': 0.2, 'attack_power': 1.0, 'defense_power': 1.5,
                             'recent_form': 0.5}
            h_stats = team_stats.get(h_team, default_stats)
            a_stats = team_stats.get(a_team, default_stats)

            features = [
                h_stats['home_win_rate'], a_stats['away_win_rate'],
                h_stats['attack_power'], a_stats['attack_power'],
                h_stats['defense_power'], a_stats['defense_power'],
                h_stats.get('recent_form', 0.5), a_stats.get('recent_form', 0.5)
            ]
            X.append(features)
            y_goals.append([int(row['FTHG']), int(row['FTAG'])])
        return np.array(X), np.array(y_goals)

    X_train, y_train = create_dataset(train_df)
    X_test, y_test = create_dataset(test_df)
    return X_train, y_train, X_test, y_test, team_stats


# --- 3. Model Building & UI Evaluation ---
def build_score_neural_network():
    model = Sequential([
        Input(shape=(8,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def evaluate_model_ui(ai_model, X_test, y_test):
    predictions = ai_model.predict(X_test, verbose=0)
    total_matches = len(y_test)
    exact_score_hits = 0
    correct_direction_hits = 0

    for i in range(total_matches):
        true_h, true_a = int(round(y_test[i][0])), int(round(y_test[i][1]))
        pred_h, pred_a = int(round(predictions[i][0])), int(round(predictions[i][1]))

        # 1. Exact Score Match
        if true_h == pred_h and true_a == pred_a:
            exact_score_hits += 1
        # 2. Correct Match Direction (Win/Draw/Loss)
        if np.sign(true_h - true_a) == np.sign(pred_h - pred_a):
            correct_direction_hits += 1

    return {
        "total": total_matches,
        "exact_acc": (exact_score_hits / total_matches) * 100 if total_matches > 0 else 0,
        "direction_acc": (correct_direction_hits / total_matches) * 100 if total_matches > 0 else 0
    }


def predict_match_xg(ai_model, team_stats, home_team, away_team):
    default_stats = {'home_win_rate': 0.4, 'away_win_rate': 0.3, 'attack_power': 1.2, 'defense_power': 1.5,
                     'recent_form': 0.5}
    h_stats = team_stats.get(home_team, default_stats)
    a_stats = team_stats.get(away_team, default_stats)

    match_features = np.array([[
        h_stats['home_win_rate'], a_stats['away_win_rate'],
        h_stats['attack_power'], a_stats['attack_power'],
        h_stats['defense_power'], a_stats['defense_power'],
        h_stats.get('recent_form', 0.5), a_stats.get('recent_form', 0.5)
    ]])

    prediction = ai_model.predict(match_features, verbose=0)[0]
    return {
        "raw_home": round(float(prediction[0]), 2),
        "raw_away": round(float(prediction[1]), 2),
        "rounded_home": int(round(float(prediction[0]))),
        "rounded_away": int(round(float(prediction[1])))
    }


# --- 4. Agentic Post-Processing ---
# --- 4. Agentic Post-Processing ---
def generate_irish_guy_report(xg_predictions_list, gw_number_str):
    matches_text = ""
    for item in xg_predictions_list:
        matches_text += f"- {item['home']} vs {item['away']} | Base NN Prediction: {item['xg']['rounded_home']}-{item['xg']['rounded_away']}\n"

    prompt = f"""
    You are an elite, bold Football Betting Analyst.
    Here are the base mathematical predictions for unplayed matches in GW {gw_number_str}:
    {matches_text}

    INSTRUCTIONS:
    1. Search the live web for CURRENT injuries, suspensions, manager changes, and odds for these matches.
    2. The base NN predictions are mathematically safe (mostly 1-1 or 2-1). YOUR JOB is to add variance! 
    3. If a top team (e.g., Man City, Arsenal) is playing a weak team, or if the web search shows a key defender is injured, DO NOT be afraid to predict high-scoring games like 3-0, 3-1, or 4-1. 
    4. Conversely, if two defensive teams play, 0-0 is acceptable.

    Provide a bulleted list exactly like this:
    * **[Home] [Score] - [Score] [Away]** | Reason: [1-sentence analysis explaining the bold score based on live news].
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.5)
        )
        return response.text
    except Exception as e:
        return f"Could not reach experts. Error: {e}"

# --- Cache Model Training to avoid re-training on every button click ---
@st.cache_resource
def load_and_train_model():
    raw_data = fetch_premier_league_data()
    X_train, y_train, X_test, y_test, global_team_stats = engineer_features_and_split(raw_data, train_ratio=0.7)

    ai_brain = build_score_neural_network()
    ai_brain.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    # Run evaluation on the 30% Test Set
    eval_metrics = evaluate_model_ui(ai_brain, X_test, y_test)
    return ai_brain, global_team_stats, eval_metrics


# --- STREAMLIT UI EXECUTION ---
if st.button("🚀 Initialize System & Train Neural Network"):
    with st.spinner("Downloading data, engineering form features, & training Neural Network..."):
        ai_brain, global_team_stats, metrics = load_and_train_model()

    # Display the Test Set Accuracy Metrics
    st.success("✅ Neural Network Trained Successfully!")
    st.markdown("### 📊 Model Evaluation Results (Tested on unseen 30% data)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Matches Tested", f"{metrics['total']}")
    col2.metric("Direction Accuracy (Win/Draw/Loss)", f"{metrics['direction_acc']:.1f}%",
                "Industry standard is ~50-55%")
    col3.metric("Exact Score Accuracy", f"{metrics['exact_acc']:.1f}%", "Extremely hard to predict")
    st.markdown("---")

    with st.spinner("Fetching upcoming unplayed fixtures..."):
        upcoming_matches, current_gw_str, show_time_warning = fetch_unplayed_fixtures()

    st.subheader(f"📅 Gameweek {current_gw_str} - Base ML Predictions (xG)")

    if not upcoming_matches:
        st.warning("No unplayed fixtures found right now.")
    else:
        all_xg_predictions = []
        for home, away, match_gw in upcoming_matches:
            xg_result = predict_match_xg(ai_brain, global_team_stats, home, away)
            all_xg_predictions.append({"home": home, "away": away, "xg": xg_result})

            st.write(
                f"**GW {match_gw} | {home}** vs **{away}** | ML Predicts: {xg_result['rounded_home']} - {xg_result['rounded_away']} *(Raw: {xg_result['raw_home']} - {xg_result['raw_away']})*")

        st.markdown("---")
        st.subheader("🧠 Gemini Expert Analysis (Live Web Context)")

        with st.spinner("Gemini is searching the web for live injuries, news, and odds to adjust predictions..."):
            final_report = generate_irish_guy_report(all_xg_predictions, current_gw_str)
            st.info(final_report)

        # Print the smart warning if next Gameweek matches are far away
        if show_time_warning:
            st.warning(
                "⚠️ **Note:** Some of the predicted matches are more than 4 days away. The AI's live context (injuries, odds) might change significantly closer to kickoff. Check back later for updated predictions!")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from google import genai
from google.genai import types

# --- הגדרות האתר (Streamlit) ---
st.set_page_config(page_title="FPL AI Predictor", page_icon="⚽")
st.title("⚽ חיזוי משחקי פרמייר ליג עם AI")
st.write(
    "ברוכים הבאים! האפליקציה מאמנת רשת עצבית על נתוני עבר, ומשתמשת ב-Gemini כדי להוסיף הקשר מהעולם האמיתי (פציעות, חדשות).")

# --- חיבור ל-GEMINI דרך Streamlit Secrets ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error("❌ שגיאה: לא מצאתי את המפתח הסודי! ודא שהוספת 'GEMINI_API_KEY' להגדרות ה-Secrets ב-Streamlit.")
    st.stop()


# ==========================================
# הפונקציות שלך (בלי שינוי לוגי, רק הותאמו לתצוגה)
# ==========================================
def fetch_upcoming_fixtures():
    bootstrap_res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = {t['id']: t['name'] for t in bootstrap_res['teams']}
    next_gw = next(gw['id'] for gw in bootstrap_res['events'] if not gw['finished'])

    fixtures_res = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    gw_fixtures = [f for f in fixtures_res if f['event'] == next_gw]

    name_mapping = {"Man Utd": "Man United", "Spurs": "Tottenham"}
    upcoming_matches = []
    for f in gw_fixtures:
        home = name_mapping.get(teams[f['team_h']], teams[f['team_h']])
        away = name_mapping.get(teams[f['team_a']], teams[f['team_a']])
        upcoming_matches.append((home, away))
    return upcoming_matches, next_gw


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
        if total_games == 0: continue

        home_wins = len(home_games[home_games['FTR'] == 'H'])
        away_wins = len(away_games[away_games['FTR'] == 'A'])

        home_win_rate = home_wins / len(home_games) if len(home_games) > 0 else 0.3
        away_win_rate = away_wins / len(away_games) if len(away_games) > 0 else 0.2

        avg_goals_scored = (home_games['FTHG'].sum() + away_games['FTAG'].sum()) / total_games
        avg_goals_conceded = (home_games['FTAG'].sum() + away_games['FTHG'].sum()) / total_games
        avg_xg_created = (home_games['HST'].sum() + away_games['AST'].sum()) / total_games
        avg_xg_conceded = (home_games['AST'].sum() + away_games['HST'].sum()) / total_games

        team_stats[team] = {
            'home_win_rate': home_win_rate, 'away_win_rate': away_win_rate,
            'attack_power': avg_goals_scored, 'defense_power': avg_goals_conceded,
            'xg_created': avg_xg_created, 'xg_conceded': avg_xg_conceded
        }

    def create_dataset(data_split):
        X, y_goals = [], []
        for _, row in data_split.iterrows():
            h_team, a_team = row['HomeTeam'], row['AwayTeam']
            h_goals, a_goals = int(row['FTHG']), int(row['FTAG'])
            h_stats = team_stats.get(h_team, {'home_win_rate': 0.3, 'attack_power': 1.0, 'defense_power': 1.5,
                                              'xg_created': 4.0, 'xg_conceded': 4.0})
            a_stats = team_stats.get(a_team, {'away_win_rate': 0.2, 'attack_power': 1.0, 'defense_power': 1.5,
                                              'xg_created': 4.0, 'xg_conceded': 4.0})

            features = [
                h_stats['home_win_rate'], a_stats['away_win_rate'],
                h_stats['attack_power'], a_stats['attack_power'],
                h_stats['defense_power'], a_stats['defense_power'],
                h_stats['xg_created'], a_stats['xg_created']
            ]
            X.append(features)
            y_goals.append([h_goals, a_goals])
        return np.array(X), np.array(y_goals)

    X_train, y_train = create_dataset(train_df)
    X_test, y_test = create_dataset(test_df)
    return X_train, y_train, X_test, y_test, team_stats


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


def predict_match_xg(ai_model, team_stats, home_team, away_team):
    h_stats = team_stats.get(home_team,
                             {'home_win_rate': 0.4, 'attack_power': 1.2, 'defense_power': 1.5, 'xg_created': 4.0})
    a_stats = team_stats.get(away_team,
                             {'away_win_rate': 0.3, 'attack_power': 1.2, 'defense_power': 1.5, 'xg_created': 4.0})
    match_features = np.array([[
        h_stats['home_win_rate'], a_stats['away_win_rate'],
        h_stats['attack_power'], a_stats['attack_power'],
        h_stats['defense_power'], a_stats['defense_power'],
        h_stats['xg_created'], a_stats['xg_created']
    ]])
    prediction = ai_model.predict(match_features, verbose=0)[0]
    return {
        "raw_home": round(float(prediction[0]), 2), "raw_away": round(float(prediction[1]), 2),
        "rounded_home": int(round(float(prediction[0]))), "rounded_away": int(round(float(prediction[1])))
    }


def generate_irish_guy_report(xg_predictions_list, gw_number):
    matches_text = ""
    for item in xg_predictions_list:
        matches_text += f"- {item['home']} vs {item['away']} | Base Prediction: {item['xg']['rounded_home']}-{item['xg']['rounded_away']}\n"

    prompt = f"""
    You are an elite Football Betting Analyst. Today is March 2026.
    My Neural Network has generated base mathematical predictions for Premier League Gameweek {gw_number}.
    Here are the base predictions:
    {matches_text}

    YOUR TASK (Post-Processing):
    1. Search the live web for the latest March 2026 news for these specific matches.
    2. Adjust the Neural Network's base predictions based on real-world context (injuries/odds).
    3. Determine the FINAL EXACT INTEGER SCORE for each match.
    4. Write a 1-sentence reason for your final score.

    OUTPUT FORMAT:
    Provide a bulleted list where each match follows this structure EXACTLY:
    * **[Home Team] [Home Score] - [Away Score] [Away Team]** | Reason: [Your 1-sentence analysis].
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.5)
        )
        return response.text
    except Exception as e:
        return f"שגיאה בתקשורת עם Gemini: {e}"


# --- כפתור ההפעלה של האפליקציה ---
if st.button("🚀 הפעל מודל וקבל תחזית עכשיו!"):
    with st.spinner('מושך נתונים, מאמן רשת עצבית ושואל את ג'ימיני...(זה ייקח בערך דקה)'):

    # 1
    upcoming_matches, current_gw = fetch_upcoming_fixtures()
    st.info(f"📅 נמצאו {len(upcoming_matches)} משחקים למחזור {current_gw}")

    # 2
    raw_data = fetch_premier_league_data()
    X_train, y_train, X_test, y_test, global_team_stats = engineer_features_and_split(raw_data)

    # 3
    ai_brain = build_score_neural_network()
    ai_brain.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    # 4
    all_xg_predictions =[]
    for home, away in upcoming_matches:
        xg_result = predict_match_xg(ai_brain, global_team_stats, home, away)
    all_xg_predictions.append({"home": home, "away": away, "xg": xg_result})

    # 5
    final_report = generate_irish_guy_report(all_xg_predictions, current_gw)

    st.success(f"🏆 התחזית הסופית למחזור {current_gw} 🏆")
    st.markdown(final_report)
    st.balloons()  # קצת חגיגה במסך!
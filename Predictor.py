"""
R6 Pro League Prediction Model
Run after scraper.py: python model.py
pip install pandas scikit-learn xgboost flask flask-cors
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df):
    """
    Builds team-level features from raw match data.
    Returns a feature dict per (team, map) pair.
    """
    features = {}
    
    all_teams = set(df["team_a"].tolist() + df["team_b"].tolist())
    all_maps = df["map"].unique()
    
    for team in all_teams:
        features[team] = {
            "overall_win_rate": 0,
            "total_maps": 0,
            "map_stats": {},
            "recent_form": 0,  # last 5 maps win rate
            "h2h": {},         # head-to-head vs other teams
        }
        
        # All maps this team played (as either team_a or team_b)
        team_maps = df[(df["team_a"] == team) | (df["team_b"] == team)].copy()
        team_maps["won"] = team_maps["winner"] == team
        
        if len(team_maps) == 0:
            continue
        
        # Overall win rate
        features[team]["overall_win_rate"] = team_maps["won"].mean()
        features[team]["total_maps"] = len(team_maps)
        
        # Recent form — last 5 maps played
        recent = team_maps.tail(5)
        features[team]["recent_form"] = recent["won"].mean()
        
        # Per-map win rate
        for map_name in all_maps:
            map_games = team_maps[team_maps["map"] == map_name]
            if len(map_games) > 0:
                features[team]["map_stats"][map_name] = {
                    "win_rate": map_games["won"].mean(),
                    "games_played": len(map_games),
                    "avg_rounds_scored": map_games.apply(
                        lambda r: r["score_a"] if r["team_a"] == team else r["score_b"], axis=1
                    ).mean() if "score_a" in map_games.columns else 7.0,
                    "avg_rounds_conceded": map_games.apply(
                        lambda r: r["score_b"] if r["team_a"] == team else r["score_a"], axis=1
                    ).mean() if "score_b" in map_games.columns else 5.0,
                }
        
        # Head-to-head
        for opponent in all_teams:
            if opponent == team:
                continue
            h2h_maps = team_maps[(team_maps["team_a"] == opponent) | (team_maps["team_b"] == opponent)]
            if len(h2h_maps) > 0:
                features[team]["h2h"][opponent] = h2h_maps["won"].mean()
    
    return features


def get_matchup_features(team_a, team_b, map_name, features):
    """
    Generates a feature vector for a team_a vs team_b matchup on a specific map.
    """
    fa = features.get(team_a, {})
    fb = features.get(team_b, {})
    
    # Map-specific win rates
    fa_map = fa.get("map_stats", {}).get(map_name, {})
    fb_map = fb.get("map_stats", {}).get(map_name, {})
    
    fa_map_wr = fa_map.get("win_rate", fa.get("overall_win_rate", 0.5))
    fb_map_wr = fb_map.get("win_rate", fb.get("overall_win_rate", 0.5))
    
    # Games played on this map (confidence factor)
    fa_map_games = fa_map.get("games_played", 0)
    fb_map_games = fb_map.get("games_played", 0)
    
    # Round differential on this map
    fa_round_diff = fa_map.get("avg_rounds_scored", 6) - fa_map.get("avg_rounds_conceded", 6)
    fb_round_diff = fb_map.get("avg_rounds_scored", 6) - fb_map.get("avg_rounds_conceded", 6)
    
    # Head-to-head
    fa_h2h = fa.get("h2h", {}).get(team_b, 0.5)
    fb_h2h = fb.get("h2h", {}).get(team_a, 0.5)
    
    return [
        fa_map_wr,                                    # Team A map win rate
        fb_map_wr,                                    # Team B map win rate
        fa_map_wr - fb_map_wr,                        # Win rate delta
        fa.get("overall_win_rate", 0.5),              # Team A overall
        fb.get("overall_win_rate", 0.5),              # Team B overall
        fa.get("recent_form", 0.5),                   # Team A recent form
        fb.get("recent_form", 0.5),                   # Team B recent form
        fa.get("recent_form", 0.5) - fb.get("recent_form", 0.5),  # Form delta
        fa_h2h,                                       # H2H win rate
        fa_round_diff,                                # Team A round diff on map
        fb_round_diff,                                # Team B round diff on map
        fa_round_diff - fb_round_diff,                # Round diff delta
        min(fa_map_games, 10) / 10,                   # Map experience (normalized)
        min(fb_map_games, 10) / 10,
        fa.get("total_maps", 0) / 50,                 # Overall experience
        fb.get("total_maps", 0) / 50,
    ]

FEATURE_NAMES = [
    "team_a_map_wr", "team_b_map_wr", "map_wr_delta",
    "team_a_overall_wr", "team_b_overall_wr",
    "team_a_recent_form", "team_b_recent_form", "form_delta",
    "h2h_win_rate",
    "team_a_round_diff", "team_b_round_diff", "round_diff_delta",
    "team_a_map_experience", "team_b_map_experience",
    "team_a_total_experience", "team_b_total_experience",
]


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df):
    """
    Trains XGBoost + Logistic Regression ensemble.
    Returns trained models + team features.
    """
    print("Building features...")
    team_features = build_features(df)
    
    # Build training set
    X, y = [], []
    for _, row in df.iterrows():
        try:
            feat = get_matchup_features(row["team_a"], row["team_b"], row["map"], team_features)
            label = 1 if row["winner"] == row["team_a"] else 0
            X.append(feat)
            y.append(label)
        except:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training on {len(X)} samples...")
    
    # Logistic Regression (fast, interpretable)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X, y)
    
    lr_cv = cross_val_score(lr, X, y, cv=min(5, len(X)//10), scoring="accuracy")
    print(f"Logistic Regression CV accuracy: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")
    
    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X, y)
        xgb_cv = cross_val_score(xgb, X, y, cv=min(5, len(X)//10), scoring="accuracy")
        print(f"XGBoost CV accuracy: {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f}")
        primary_model = xgb
        model_name = "XGBoost"
    except ImportError:
        print("XGBoost not available, using Logistic Regression")
        primary_model = lr
        model_name = "LogisticRegression"
    
    # Save everything
    model_data = {
        "model": primary_model,
        "lr_model": lr,
        "features": team_features,
        "model_name": model_name,
        "training_samples": len(X),
        "cv_accuracy": float(lr_cv.mean()),
        "teams": sorted(list(team_features.keys())),
        "maps": sorted(df["map"].unique().tolist()),
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Also save as JSON for the frontend
    export_for_frontend(model_data, df)
    
    print(f"\n✅ Model saved → model.pkl")
    return model_data


def export_for_frontend(model_data, df):
    """
    Exports team stats + model metadata as JSON for the React frontend.
    """
    team_features = model_data["features"]
    
    # Build team summary stats
    teams_json = {}
    for team, feat in team_features.items():
        teams_json[team] = {
            "overall_win_rate": round(feat["overall_win_rate"], 3),
            "recent_form": round(feat["recent_form"], 3),
            "total_maps": feat["total_maps"],
            "map_win_rates": {
                m: round(s["win_rate"], 3) 
                for m, s in feat.get("map_stats", {}).items()
            },
            "best_map": max(feat.get("map_stats", {}).items(), 
                          key=lambda x: x[1]["win_rate"], default=(None, {}))[0],
            "worst_map": min(feat.get("map_stats", {}).items(), 
                           key=lambda x: x[1]["win_rate"], default=(None, {}))[0],
        }
    
    frontend_data = {
        "teams": teams_json,
        "maps": model_data["maps"],
        "model_accuracy": model_data["cv_accuracy"],
        "training_samples": model_data["training_samples"],
    }
    
    with open("model_data.json", "w") as f:
        json.dump(frontend_data, f, indent=2)
    
    print("✅ Frontend data exported → model_data.json")


# ─────────────────────────────────────────────
# FLASK API SERVER
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this

# Load model on startup
model_data = None

def load_model():
    global model_data
    try:
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
        print("✅ Model loaded")
    except FileNotFoundError:
        print("⚠️  model.pkl not found — run training first")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "team_a": "G2 Esports", "team_b": "Team BDS", "map": "Clubhouse" }
    Returns: win probability + key stats
    """
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    team_a = data.get("team_a")
    team_b = data.get("team_b")
    map_name = data.get("map")
    
    if not all([team_a, team_b, map_name]):
        return jsonify({"error": "Missing team_a, team_b, or map"}), 400
    
    features = model_data["features"]
    
    # Get features
    feat_vector = get_matchup_features(team_a, team_b, map_name, features)
    X = np.array([feat_vector])
    
    # Predict
    model = model_data["model"]
    prob_a = float(model.predict_proba(X)[0][1])
    prob_b = 1 - prob_a
    
    # Confidence interval (±1 std based on feature uncertainty)
    confidence_margin = 0.05 + (0.1 * (1 - min(feat_vector[12], feat_vector[13])))
    
    # Team stats for display
    fa = features.get(team_a, {})
    fb = features.get(team_b, {})
    fa_map = fa.get("map_stats", {}).get(map_name, {})
    fb_map = fb.get("map_stats", {}).get(map_name, {})
    
    return jsonify({
        "team_a": team_a,
        "team_b": team_b,
        "map": map_name,
        "team_a_win_prob": round(prob_a, 3),
        "team_b_win_prob": round(prob_b, 3),
        "confidence_low": round(max(0, prob_a - confidence_margin), 3),
        "confidence_high": round(min(1, prob_a + confidence_margin), 3),
        "predicted_winner": team_a if prob_a > 0.5 else team_b,
        "model_accuracy": round(model_data["cv_accuracy"], 3),
        "key_stats": {
            "team_a_map_wr": round(fa_map.get("win_rate", fa.get("overall_win_rate", 0.5)), 3),
            "team_b_map_wr": round(fb_map.get("win_rate", fb.get("overall_win_rate", 0.5)), 3),
            "team_a_recent_form": round(fa.get("recent_form", 0.5), 3),
            "team_b_recent_form": round(fb.get("recent_form", 0.5), 3),
            "team_a_map_games": fa_map.get("games_played", 0),
            "team_b_map_games": fb_map.get("games_played", 0),
            "h2h": round(fa.get("h2h", {}).get(team_b, 0.5), 3),
        }
    })


@app.route("/teams", methods=["GET"])
def get_teams():
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"teams": model_data["teams"]})


@app.route("/maps", methods=["GET"])
def get_maps():
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"maps": model_data["maps"]})


@app.route("/rankings", methods=["GET"])
def get_rankings():
    """Returns team rankings sorted by overall win rate."""
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    
    features = model_data["features"]
    rankings = []
    
    for team, feat in features.items():
        if feat["total_maps"] < 2:
            continue
        rankings.append({
            "team": team,
            "overall_win_rate": round(feat["overall_win_rate"], 3),
            "recent_form": round(feat["recent_form"], 3),
            "total_maps": feat["total_maps"],
            "best_map": max(feat.get("map_stats", {}).items(),
                          key=lambda x: x[1]["win_rate"], default=("N/A", {}))[0],
        })
    
    rankings.sort(key=lambda x: (x["overall_win_rate"] * 0.5 + x["recent_form"] * 0.5), reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1
    
    return jsonify({"rankings": rankings})


@app.route("/series-predict", methods=["POST"])
def predict_series():
    """
    Predicts full series outcome given a map pool.
    Body: { "team_a": "...", "team_b": "...", "maps": ["Clubhouse", "Oregon", "Bank"] }
    """
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    team_a = data.get("team_a")
    team_b = data.get("team_b")
    maps = data.get("maps", [])
    
    features = model_data["features"]
    model = model_data["model"]
    
    map_results = []
    team_a_wins = 0
    team_b_wins = 0
    
    for map_name in maps:
        feat = get_matchup_features(team_a, team_b, map_name, features)
        prob_a = float(model.predict_proba(np.array([feat]))[0][1])
        map_results.append({
            "map": map_name,
            "team_a_prob": round(prob_a, 3),
            "team_b_prob": round(1 - prob_a, 3),
            "predicted_winner": team_a if prob_a > 0.5 else team_b,
        })
        if prob_a > 0.5:
            team_a_wins += 1
        else:
            team_b_wins += 1
    
    # Monte Carlo series win probability
    n_sim = 10000
    team_a_series_wins = 0
    probs = [r["team_a_prob"] for r in map_results]
    
    for _ in range(n_sim):
        a_wins = sum(1 for p in probs if np.random.random() < p)
        if a_wins > len(probs) / 2:
            team_a_series_wins += 1
    
    series_prob_a = team_a_series_wins / n_sim
    
    return jsonify({
        "team_a": team_a,
        "team_b": team_b,
        "map_results": map_results,
        "team_a_series_prob": round(series_prob_a, 3),
        "team_b_series_prob": round(1 - series_prob_a, 3),
        "predicted_series_winner": team_a if series_prob_a > 0.5 else team_b,
        "map_score_prediction": f"{team_a_wins}-{team_b_wins}",
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if "--train" in sys.argv or "--serve" not in sys.argv:
        # Train mode
        print("=" * 50)
        print("R6 Pro League Prediction Model — Training")
        print("=" * 50)
        
        try:
            df = pd.read_csv("pro_matches.csv")
            print(f"Loaded {len(df)} match records")
        except FileNotFoundError:
            print("pro_matches.csv not found — run scraper.py first")
            print("Loading seed data directly...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("scraper", "scraper.py")
            scraper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scraper)
            df = pd.DataFrame(scraper.get_seed_data())
            df.to_csv("pro_matches.csv", index=False)
        
        model_data = train_model(df)
        print("\nTo start the API server: python model.py --serve")
    
    if "--serve" in sys.argv:
        load_model()
        print("\n🚀 API server running on http://localhost:5000")
        print("Endpoints: /predict, /teams, /maps, /rankings, /series-predict")
        app.run(debug=False, port=5000)
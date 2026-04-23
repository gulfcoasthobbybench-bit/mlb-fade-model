import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from pybaseball import statcast

# =========================================================
# CONFIG
# =========================================================
RECENT_LOOKBACK_DAYS = 7
BASELINE_LOOKBACK_DAYS = 30
PITCHER_LOOKBACK_DAYS = 30

MIN_RECENT_BBE = 3
MIN_BASELINE_BBE = 8
MIN_PITCHER_BBE_AGAINST = 20
MIN_BULLPEN_BBE_AGAINST = 40

TOP_BETS_COUNT = 15
BEST_OPPORTUNITY_COUNT = 12
MOST_STABLE_COUNT = 12
HIGH_RISK_COUNT = 12
PARLAY_COUNT = 6

MIN_PARLAY_CONFIDENCE = 65.0
MIN_PARLAY_FIT = 60.0

REQUEST_TIMEOUT = 30

OUTPUT_DIR = "output"
ENABLE_CSV_EXPORT = True

TEAM_NAME_TO_ABBR = {
    "Houston Astros": "HOU",
    "New York Yankees": "NYY",
    "Boston Red Sox": "BOS",
    "Los Angeles Dodgers": "LAD",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "San Francisco Giants": "SF",
    "San Diego Padres": "SD",
    "Seattle Mariners": "SEA",
    "Texas Rangers": "TEX",
    "Arizona Diamondbacks": "ARI",
    "Colorado Rockies": "COL",
    "Miami Marlins": "MIA",
    "Atlanta Braves": "ATL",
    "Washington Nationals": "WSH",
    "New York Mets": "NYM",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "St. Louis Cardinals": "STL",
    "Milwaukee Brewers": "MIL",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Detroit Tigers": "DET",
    "Kansas City Royals": "KC",
    "Minnesota Twins": "MIN",
    "Baltimore Orioles": "BAL",
    "Tampa Bay Rays": "TB",
    "Toronto Blue Jays": "TOR",
    "Los Angeles Angels": "LAA",
    "Athletics": "ATH",
}

VALID_BATTER_POSITIONS = {
    "C", "1B", "2B", "3B", "SS",
    "LF", "CF", "RF", "OF", "DH",
    "IF", "UT", "PH"
}

PARK_FACTORS = {
    "ARI": 102, "ATH": 96, "ATL": 101, "BAL": 100, "BOS": 104,
    "CHC": 102, "CIN": 105, "CLE": 98, "COL": 118, "CWS": 101,
    "DET": 97, "HOU": 100, "KC": 99, "LAA": 101, "LAD": 101,
    "MIA": 95, "MIL": 100, "MIN": 99, "NYM": 98, "NYY": 104,
    "PHI": 103, "PIT": 97, "SD": 96, "SEA": 95, "SF": 94,
    "STL": 100, "TB": 97, "TEX": 104, "TOR": 101, "WSH": 100,
}

EXPECTED_LINEUP_COLS = [
    "batter",
    "player",
    "team",
    "opponent_team",
    "position",
    "batting_order",
    "lineup_slot",
    "bat_side",
    "opposing_pitcher",
    "source",
    "source_mode",
    "game_id",
]

PROP_FAMILY_VOLATILITY = {
    "HITTER_FADE": "LOW",
    "TEAM_TOTAL_UNDER": "LOW",
    "FIRST5_TEAM_TOTAL_UNDER": "LOW",
    "PITCHER_OUTS": "LOW",
    "PITCHER_STRIKEOUTS": "MED",
    "PITCHER_HITS_ALLOWED": "MED",
    "PITCHER_WALKS_ALLOWED": "MED",
}

PROP_FAMILY_STABILITY_BONUS = {
    "HITTER_FADE": 8,
    "TEAM_TOTAL_UNDER": 12,
    "FIRST5_TEAM_TOTAL_UNDER": 14,
    "PITCHER_OUTS": 10,
    "PITCHER_STRIKEOUTS": 6,
    "PITCHER_HITS_ALLOWED": 6,
    "PITCHER_WALKS_ALLOWED": 4,
}


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Prop:
    prop: str
    entity_name: str
    team: str
    opponent_team: str
    matchup: str
    game_id: str
    prop_family: str
    model_prob: float
    stability: float
    confidence: float
    parlay_fit: float
    opportunity_score: float
    volatility: str
    script_tag: str
    market_score: float
    reason: str
    source_mode: str
    opposing_pitcher: Optional[str] = None
    lineup_slot: Optional[float] = None
    raw: Dict = field(default_factory=dict)


@dataclass
class Parlay:
    legs: List[Prop]
    avg_conf: float
    min_conf: float
    structural: float
    avg_opportunity: float
    estimated_hit_rate: float
    final_score: float


# =========================================================
# HELPERS
# =========================================================
def safe_get_json(url: str) -> dict:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def ensure_output_dir() -> None:
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


def batting_order_to_slot(batting_order_value):
    if batting_order_value is None:
        return None
    try:
        return int(str(batting_order_value)[0])
    except Exception:
        return None


def get_batting_team(row):
    return row["away_team"] if row["inning_topbot"] == "Top" else row["home_team"]


def get_defending_team(row):
    return row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"]


def handedness_matchup(bat_side, pitch_hand):
    if not bat_side or not pitch_hand:
        return "Unknown"
    if bat_side == "S":
        return "Switch"
    if bat_side == pitch_hand:
        return "Same-Handed"
    return "Platoon-Edge"


def clamp(value, low, high):
    return max(low, min(high, value))


def normalized_market_probability(market_score: float) -> float:
    prob = 0.45 + (market_score / 140.0)
    return clamp(prob, 0.48, 0.84)


def compute_stability(
    prop_family: str,
    source_mode: str,
    lineup_slot: Optional[float],
    recent_bbe: Optional[float],
    park_factor: Optional[float],
) -> float:
    stability = 68.0
    stability += PROP_FAMILY_STABILITY_BONUS.get(prop_family, 0)

    if source_mode == "confirmed_lineup":
        stability += 8
    elif source_mode == "preliminary_team_pool":
        stability -= 5

    if lineup_slot in [7, 8, 9]:
        stability += 6
    elif lineup_slot in [5, 6]:
        stability += 3

    if recent_bbe is not None and not pd.isna(recent_bbe):
        if recent_bbe >= 8:
            stability += 5
        elif recent_bbe <= 4:
            stability -= 6

    if park_factor is not None and not pd.isna(park_factor):
        if park_factor <= 96:
            stability += 3
        elif park_factor >= 106:
            stability -= 3

    return round(clamp(stability, 40, 98), 1)


def compute_confidence(model_prob: float, stability: float, volatility: str, market_score: float) -> float:
    volatility_penalty = {"LOW": 4, "MED": 9, "HIGH": 16}.get(volatility, 10)
    score = (
        0.40 * (model_prob * 100)
        + 0.30 * stability
        + 0.20 * market_score
        - 0.10 * volatility_penalty
    )
    return round(clamp(score, 0, 100), 1)


def compute_opportunity_score(market_score: float, confidence: float, stability: float, volatility: str) -> float:
    volatility_penalty = {"LOW": 2, "MED": 6, "HIGH": 12}.get(volatility, 6)
    score = (
        0.45 * market_score +
        0.30 * confidence +
        0.25 * stability -
        volatility_penalty
    )
    return round(clamp(score, 0, 100), 1)


def compute_parlay_fit(confidence: float, stability: float, volatility: str, prop_family: str) -> float:
    fit = confidence * 0.55 + stability * 0.45

    if volatility == "HIGH":
        fit -= 12
    elif volatility == "MED":
        fit -= 5

    if prop_family in {"FIRST5_TEAM_TOTAL_UNDER", "TEAM_TOTAL_UNDER", "HITTER_FADE", "PITCHER_OUTS"}:
        fit += 3

    return round(clamp(fit, 0, 100), 1)


def derive_script_tag(prop_family: str, market_score: float, lineup_slot=None) -> str:
    if prop_family == "HITTER_FADE":
        if lineup_slot in [7, 8, 9]:
            return "Bottom-Order Suppression"
        return "Hitter Fade"
    if prop_family == "TEAM_TOTAL_UNDER":
        return "Suppression Environment"
    if prop_family == "FIRST5_TEAM_TOTAL_UNDER":
        return "Early Suppression"
    if prop_family == "PITCHER_OUTS":
        return "Outs Efficiency"
    if prop_family == "PITCHER_STRIKEOUTS":
        return "Strikeout Funnel"
    if prop_family == "PITCHER_HITS_ALLOWED":
        return "Contact Suppression"
    if prop_family == "PITCHER_WALKS_ALLOWED":
        return "Command Edge"
    if market_score >= 90:
        return "Top Opportunity"
    return "Neutral"


def build_matchup(team: str, opponent_team: str) -> str:
    return f"{team} vs {opponent_team}"


def get_person_bat_side(person_id):
    if pd.isna(person_id):
        return None
    try:
        pdata = safe_get_json(f"https://statsapi.mlb.com/api/v1/people/{int(person_id)}")
        people = pdata.get("people", [])
        if people:
            return people[0].get("batSide", {}).get("code")
    except Exception:
        return None
    return None


def get_person_pitch_hand(person_id):
    if pd.isna(person_id):
        return None
    try:
        pdata = safe_get_json(f"https://statsapi.mlb.com/api/v1/people/{int(person_id)}")
        people = pdata.get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code")
    except Exception:
        return None
    return None


def safe_statcast_pull(start_dt: str, end_dt: str, needed_cols: List[str]) -> pd.DataFrame:
    try:
        sc = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception:
        return pd.DataFrame(columns=needed_cols)

    if sc is None or sc.empty:
        return pd.DataFrame(columns=needed_cols)

    for col in needed_cols:
        if col not in sc.columns:
            sc[col] = pd.NA

    return sc[needed_cols].copy()


# =========================================================
# SCHEDULE / LINEUPS
# =========================================================
def get_actual_or_probable_starters(game, game_pk):
    home_pitcher = "TBD"
    away_pitcher = "TBD"
    home_pitcher_id = None
    away_pitcher_id = None

    feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        feed_data = safe_get_json(feed_url)
        box_teams = feed_data.get("liveData", {}).get("boxscore", {}).get("teams", {})

        def extract_starter(players_dict):
            candidates = []
            for p in players_dict.values():
                if p.get("position", {}).get("code") == "1":
                    person = p.get("person", {})
                    pitching_stats = p.get("stats", {}).get("pitching", {})
                    games_started = pitching_stats.get("gamesStarted", 0)
                    if games_started >= 1:
                        return person.get("fullName", "TBD"), person.get("id")
                    candidates.append((person.get("fullName", "TBD"), person.get("id")))
            if candidates:
                return candidates[0]
            return "TBD", None

        home_pitcher, home_pitcher_id = extract_starter(box_teams.get("home", {}).get("players", {}))
        away_pitcher, away_pitcher_id = extract_starter(box_teams.get("away", {}).get("players", {}))
    except Exception:
        pass

    if home_pitcher == "TBD":
        home_pitcher = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher_id = game["teams"]["home"].get("probablePitcher", {}).get("id")

    if away_pitcher == "TBD":
        away_pitcher = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
        away_pitcher_id = game["teams"]["away"].get("probablePitcher", {}).get("id")

    return home_pitcher, away_pitcher, home_pitcher_id, away_pitcher_id


def load_schedule(target_date: Optional[str] = None) -> pd.DataFrame:
    if not target_date:
        target_date = datetime.today().strftime("%Y-%m-%d")

    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}"
    schedule_data = safe_get_json(schedule_url)

    games = []
    for date_block in schedule_data.get("dates", []):
        for game in date_block.get("games", []):
            game_pk = game["gamePk"]
            home_team = game["teams"]["home"]["team"]["name"]
            away_team = game["teams"]["away"]["team"]["name"]

            home_abbr = TEAM_NAME_TO_ABBR.get(home_team)
            away_abbr = TEAM_NAME_TO_ABBR.get(away_team)

            home_pitcher, away_pitcher, home_pitcher_id, away_pitcher_id = get_actual_or_probable_starters(game, game_pk)
            status = game.get("status", {}).get("detailedState", "Unknown")

            games.append({
                "game_pk": str(game_pk),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": game["teams"]["home"]["team"].get("id"),
                "away_team_id": game["teams"]["away"]["team"].get("id"),
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
                "status": status,
                "park_factor": PARK_FACTORS.get(home_abbr, 100),
            })

    return pd.DataFrame(games)


def get_preliminary_batter_pool(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    batter_hand_cache = {}

    for _, game_row in games_df.iterrows():
        for side in ["home", "away"]:
            team_id = game_row["home_team_id"] if side == "home" else game_row["away_team_id"]
            this_team_abbr = game_row["home_abbr"] if side == "home" else game_row["away_abbr"]
            opp_team_abbr = game_row["away_abbr"] if side == "home" else game_row["home_abbr"]
            opp_pitcher = game_row["away_pitcher"] if side == "home" else game_row["home_pitcher"]

            if pd.isna(team_id):
                continue

            roster_url = f"https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster/active"
            try:
                roster_data = safe_get_json(roster_url)
            except Exception:
                continue

            for player in roster_data.get("roster", []):
                person = player.get("person", {})
                position = player.get("position", {})
                position_abbr = position.get("abbreviation", "")
                player_id = person.get("id")
                player_name = person.get("fullName", "")

                if player_id is None or position_abbr not in VALID_BATTER_POSITIONS:
                    continue

                pid = int(player_id)
                if pid not in batter_hand_cache:
                    batter_hand_cache[pid] = get_person_bat_side(pid)

                rows.append({
                    "batter": pid,
                    "player": player_name,
                    "team": this_team_abbr,
                    "opponent_team": opp_team_abbr,
                    "position": position_abbr,
                    "batting_order": None,
                    "lineup_slot": None,
                    "bat_side": batter_hand_cache.get(pid),
                    "opposing_pitcher": opp_pitcher,
                    "source": "preliminary",
                    "source_mode": "preliminary_team_pool",
                    "game_id": str(game_row["game_pk"]),
                })

    preliminary_df = pd.DataFrame(rows)
    if preliminary_df.empty:
        return pd.DataFrame(columns=EXPECTED_LINEUP_COLS)

    for col in EXPECTED_LINEUP_COLS:
        if col not in preliminary_df.columns:
            preliminary_df[col] = None

    return preliminary_df[EXPECTED_LINEUP_COLS].drop_duplicates(subset=["batter", "game_id"])


def load_batter_pool(games_df: pd.DataFrame) -> pd.DataFrame:
    lineup_rows = []
    batter_hand_cache = {}

    for _, game_row in games_df.iterrows():
        game_pk = game_row["game_pk"]
        home_abbr = game_row["home_abbr"]
        away_abbr = game_row["away_abbr"]
        home_pitcher = game_row["home_pitcher"]
        away_pitcher = game_row["away_pitcher"]

        feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            feed_data = safe_get_json(feed_url)
        except Exception:
            continue

        teams_block = feed_data.get("liveData", {}).get("boxscore", {}).get("teams", {})
        for side in ["home", "away"]:
            team_block = teams_block.get(side, {})
            players = team_block.get("players", {})

            this_team_abbr = home_abbr if side == "home" else away_abbr
            opp_team_abbr = away_abbr if side == "home" else home_abbr
            opp_pitcher = away_pitcher if side == "home" else home_pitcher

            for _, player in players.items():
                person = player.get("person", {})
                position = player.get("position", {})
                batting_order = player.get("battingOrder")
                player_id = person.get("id")
                player_name = person.get("fullName", "")
                position_abbr = position.get("abbreviation", "")
                bat_side = player.get("batSide", {}).get("code")

                if not bat_side and player_id is not None:
                    pid = int(player_id)
                    if pid not in batter_hand_cache:
                        batter_hand_cache[pid] = get_person_bat_side(pid)
                    bat_side = batter_hand_cache.get(pid)

                if player_id is None:
                    continue

                if batting_order and position_abbr in VALID_BATTER_POSITIONS:
                    lineup_rows.append({
                        "batter": int(player_id),
                        "player": player_name,
                        "team": this_team_abbr,
                        "opponent_team": opp_team_abbr,
                        "position": position_abbr,
                        "batting_order": str(batting_order),
                        "lineup_slot": batting_order_to_slot(batting_order),
                        "bat_side": bat_side,
                        "opposing_pitcher": opp_pitcher,
                        "source": "lineup",
                        "source_mode": "confirmed_lineup",
                        "game_id": str(game_pk),
                    })

    lineups_df = pd.DataFrame(lineup_rows)
    if lineups_df.empty:
        return get_preliminary_batter_pool(games_df)

    for col in EXPECTED_LINEUP_COLS:
        if col not in lineups_df.columns:
            lineups_df[col] = None

    return lineups_df[EXPECTED_LINEUP_COLS].drop_duplicates(subset=["batter", "game_id"])


# =========================================================
# STATCAST LOADERS
# =========================================================
def pull_hitter_window(days_back: int, teams_today_abbr: set, target_batter_ids: set) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    needed_cols = [
        "batter", "pitcher", "home_team", "away_team", "inning_topbot",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "launch_speed", "pitch_type", "description", "events"
    ]
    df = safe_statcast_pull(
        start_dt=start_date.strftime("%Y-%m-%d"),
        end_dt=end_date.strftime("%Y-%m-%d"),
        needed_cols=needed_cols,
    )

    if df.empty:
        return df

    df["batter"] = pd.to_numeric(df["batter"], errors="coerce")
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    df = df.dropna(subset=["batter", "home_team", "away_team", "inning_topbot"]).copy()
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").fillna(0).astype(int)
    df["team"] = df.apply(get_batting_team, axis=1)
    df["def_team"] = df.apply(get_defending_team, axis=1)
    df = df.rename(columns={
        "estimated_ba_using_speedangle": "xBA",
        "estimated_woba_using_speedangle": "xwOBA",
        "launch_speed": "exit_velocity",
    })
    df = df[df["team"].isin(teams_today_abbr)].copy()
    df = df[df["batter"].isin(target_batter_ids)].copy()
    return df


def pull_pitcher_window(days_back: int) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    needed_cols = [
        "game_pk", "inning_topbot", "pitcher", "home_team", "away_team",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "launch_speed", "pitch_type", "description", "events"
    ]
    df = safe_statcast_pull(
        start_dt=start_date.strftime("%Y-%m-%d"),
        end_dt=end_date.strftime("%Y-%m-%d"),
        needed_cols=needed_cols,
    )

    if df.empty:
        return df

    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    df = df.dropna(subset=["pitcher", "game_pk", "inning_topbot", "home_team", "away_team"]).copy()
    df["pitcher"] = df["pitcher"].astype(int)
    return df


# =========================================================
# SCORING HELPERS
# =========================================================
def score_pitcher_quality(row):
    score = 0
    if pd.notna(row["pitcher_xBA_allowed"]) and row["pitcher_xBA_allowed"] < 0.230:
        score += 8
    elif pd.notna(row["pitcher_xBA_allowed"]) and row["pitcher_xBA_allowed"] < 0.245:
        score += 4

    if pd.notna(row["pitcher_xwOBA_allowed"]) and row["pitcher_xwOBA_allowed"] < 0.300:
        score += 8
    elif pd.notna(row["pitcher_xwOBA_allowed"]) and row["pitcher_xwOBA_allowed"] < 0.315:
        score += 4

    if pd.notna(row["pitcher_ev_allowed"]) and row["pitcher_ev_allowed"] < 89:
        score += 5
    elif pd.notna(row["pitcher_ev_allowed"]) and row["pitcher_ev_allowed"] < 90:
        score += 2

    if pd.notna(row["pitcher_bbe_against"]) and row["pitcher_bbe_against"] < MIN_PITCHER_BBE_AGAINST:
        score -= 3

    return max(score, 0)


def score_bullpen_quality(row):
    score = 0
    if pd.notna(row["bullpen_xBA_allowed"]) and row["bullpen_xBA_allowed"] < 0.230:
        score += 4
    elif pd.notna(row["bullpen_xBA_allowed"]) and row["bullpen_xBA_allowed"] < 0.245:
        score += 2

    if pd.notna(row["bullpen_xwOBA_allowed"]) and row["bullpen_xwOBA_allowed"] < 0.305:
        score += 4
    elif pd.notna(row["bullpen_xwOBA_allowed"]) and row["bullpen_xwOBA_allowed"] < 0.320:
        score += 2

    if pd.notna(row["bullpen_ev_allowed"]) and row["bullpen_ev_allowed"] < 89.5:
        score += 3
    elif pd.notna(row["bullpen_ev_allowed"]) and row["bullpen_ev_allowed"] < 90.5:
        score += 1

    if pd.notna(row["bullpen_bbe_against"]) and row["bullpen_bbe_against"] < MIN_BULLPEN_BBE_AGAINST:
        score -= 2

    return max(score, 0)


def score_team_offense_environment(row):
    score = 0
    if pd.notna(row["team_recent_xwOBA"]) and row["team_recent_xwOBA"] < 0.300:
        score += 7
    elif pd.notna(row["team_recent_xwOBA"]) and row["team_recent_xwOBA"] < 0.315:
        score += 3

    if pd.notna(row["team_recent_xBA"]) and row["team_recent_xBA"] < 0.230:
        score += 5
    elif pd.notna(row["team_recent_xBA"]) and row["team_recent_xBA"] < 0.245:
        score += 2

    if pd.notna(row["lineup_pocket_xwOBA"]) and row["lineup_pocket_xwOBA"] < 0.305:
        score += 6
    elif pd.notna(row["lineup_pocket_xwOBA"]) and row["lineup_pocket_xwOBA"] < 0.320:
        score += 3

    if pd.notna(row["lineup_pocket_xBA"]) and row["lineup_pocket_xBA"] < 0.235:
        score += 4
    elif pd.notna(row["lineup_pocket_xBA"]) and row["lineup_pocket_xBA"] < 0.245:
        score += 2

    return max(score, 0)


def score_game_environment(row):
    score = 0
    park_factor = row.get("park_factor")
    if pd.notna(park_factor):
        if park_factor <= 96:
            score += 6
        elif park_factor <= 99:
            score += 3
        elif park_factor >= 106:
            score -= 4
        elif park_factor >= 103:
            score -= 2

    combined_team_env = row.get("combined_offense_env", 0)
    full_game_suppression = row.get("full_game_suppression_score", 0)

    if combined_team_env <= 0:
        score += 2
    elif combined_team_env >= 12:
        score -= 2

    if full_game_suppression >= 26:
        score += 4
    elif full_game_suppression >= 20:
        score += 2

    return score


def calc_no_hit_score(row):
    score = 0

    if row["recent_xBA"] < 0.220:
        score += 26
    elif row["recent_xBA"] < 0.240:
        score += 14

    if row["recent_xwOBA"] < 0.300:
        score += 12
    elif row["recent_xwOBA"] < 0.320:
        score += 6

    if row["pitcher_k_pressure_score"] >= 8:
        score += 9
    elif row["pitcher_k_pressure_score"] >= 4:
        score += 4

    if row["pitcher_zone_contact_suppression_score"] >= 6:
        score += 8
    elif row["pitcher_zone_contact_suppression_score"] >= 3:
        score += 4

    if row["pitch_mix_penalty"] >= 6:
        score += 8
    elif row["pitch_mix_penalty"] >= 3:
        score += 4

    slot = row["lineup_slot"]
    if slot in [7, 8, 9]:
        score += 10
    elif slot in [5, 6]:
        score += 4
    elif pd.isna(slot) and row.get("source_mode") == "preliminary_team_pool":
        score += 2

    if row["handedness_matchup"] == "Same-Handed":
        score += 5

    if row["recent_bbe"] <= 4:
        score -= 8
    elif row["recent_bbe"] <= 6:
        score -= 4

    score += row.get("pitcher_quality_score", 0)
    score += min(row.get("game_environment_score", 0), 5)

    if row.get("source_mode") == "preliminary_team_pool":
        score -= 2

    return max(score, 0)


def calc_tb_score(row):
    score = 0

    if row["recent_xwOBA"] < 0.300:
        score += 22
    elif row["recent_xwOBA"] < 0.320:
        score += 11

    if row["recent_exit_velocity"] < 90:
        score += 16
    elif row["recent_exit_velocity"] < 91.5:
        score += 8

    if row["recent_xBA"] < 0.220:
        score += 10
    elif row["recent_xBA"] < 0.240:
        score += 5

    if row["slump_delta_xwOBA"] < -0.025:
        score += 9
    elif row["slump_delta_xwOBA"] < -0.015:
        score += 4

    if row["pitch_mix_penalty"] >= 6:
        score += 8
    elif row["pitch_mix_penalty"] >= 3:
        score += 4

    score += row.get("pitcher_quality_score", 0)
    score += row.get("bullpen_quality_score", 0)
    score += row.get("game_environment_score", 0)

    if row["recent_bbe"] <= 4:
        score -= 7
    elif row["recent_bbe"] <= 6:
        score -= 3

    if row.get("source_mode") == "preliminary_team_pool":
        score -= 2

    return max(score, 0)


def calc_runs_score(row):
    score = 0

    slot = row["lineup_slot"]
    if slot in [7, 8, 9]:
        score += 18
    elif slot in [5, 6]:
        score += 10
    elif pd.isna(slot) and row.get("source_mode") == "preliminary_team_pool":
        score += 4

    if row["team_offense_env_score"] >= 12:
        score += 14
    elif row["team_offense_env_score"] >= 8:
        score += 8

    if row["full_game_suppression_score"] >= 24:
        score += 10
    elif row["full_game_suppression_score"] >= 18:
        score += 5

    if row["recent_xwOBA"] < 0.300:
        score += 14
    elif row["recent_xwOBA"] < 0.320:
        score += 7

    if row["recent_xBA"] < 0.220:
        score += 9
    elif row["recent_xBA"] < 0.240:
        score += 4

    if row["slump_delta_xBA"] < -0.020:
        score += 7
    elif row["slump_delta_xBA"] < -0.010:
        score += 3

    score += row.get("bullpen_quality_score", 0)
    score += row.get("game_environment_score", 0)

    if row.get("source_mode") == "preliminary_team_pool":
        score -= 3

    return max(score, 0)


def classify_fade_type(row):
    nh = row["no_hit_score"]
    tb = row["tb_suppression_score"]
    rs = row["runs_suppression_score"]

    if nh >= tb and nh >= rs:
        return "Fade for No Hit Profile"
    if tb >= nh and tb >= rs:
        return "Fade for Total Bases Suppression"
    return "Fade for Run Scored Suppression"


def assign_best_bet_type(row):
    if row["best_fade_type"] == "Fade for No Hit Profile":
        return "Under 0.5 Hits"
    if row["best_fade_type"] == "Fade for Total Bases Suppression":
        return "Under 1.5 Total Bases"
    return "Under 0.5 Runs"


def build_reason(row):
    tags = []

    if row["best_fade_type"] == "Fade for No Hit Profile":
        tags.append("no-hit profile")
    elif row["best_fade_type"] == "Fade for Total Bases Suppression":
        tags.append("TB suppression profile")
    else:
        tags.append("runs suppression profile")

    if row["recent_xBA"] < 0.220:
        tags.append("weak recent xBA")
    if row["recent_xwOBA"] < 0.300:
        tags.append("weak recent xwOBA")
    if row["recent_exit_velocity"] < 90:
        tags.append("low EV")
    if row["lineup_slot"] in [7, 8, 9]:
        tags.append("bottom-order")
    elif row["lineup_slot"] in [5, 6]:
        tags.append("limited lineup spot")
    elif pd.isna(row["lineup_slot"]) and row.get("source_mode") == "preliminary_team_pool":
        tags.append("preliminary lineup")
    if row["handedness_matchup"] == "Same-Handed":
        tags.append("same-handed matchup")
    if row["slump_delta_xBA"] < -0.015:
        tags.append("slumping xBA")
    if row["slump_delta_xwOBA"] < -0.020:
        tags.append("slumping xwOBA")
    if row["pitcher_quality_score"] >= 18:
        tags.append("tough starter")
    if row["bullpen_quality_score"] >= 8:
        tags.append("strong bullpen")
    if row["pitch_mix_penalty"] >= 6:
        tags.append("pitch-mix mismatch")
    if row["team_offense_env_score"] >= 8:
        tags.append("weak lineup context")
    if row["game_environment_score"] >= 5:
        tags.append("low-scoring park/env")

    return ", ".join(tags[:7]) if tags else "mixed profile"


# =========================================================
# FEATURE BUILDERS
# =========================================================
def get_pitcher_recent_summary(df, pitcher_id):
    subset = df[df["pitcher"] == pitcher_id].copy()
    if subset.empty:
        return {}

    subset["is_strikeout"] = subset["events"].fillna("").isin(["strikeout", "strikeout_double_play"]).astype(int)
    subset["is_walk"] = subset["events"].fillna("").isin(["walk", "intent_walk"]).astype(int)
    subset["is_hit"] = subset["events"].fillna("").isin(["single", "double", "triple", "home_run"]).astype(int)
    subset["is_out_event"] = subset["events"].fillna("").isin([
        "field_out", "force_out", "grounded_into_double_play", "double_play",
        "strikeout", "strikeout_double_play", "flyout", "lineout", "pop_out",
        "fielders_choice_out", "sac_fly", "sac_bunt", "triple_play"
    ]).astype(int)

    total_pitches = len(subset)
    return {
        "pitcher_total_pitches_recent": total_pitches,
        "pitcher_recent_k_rate": subset["is_strikeout"].mean() if total_pitches else 0,
        "pitcher_recent_out_event_rate": subset["is_out_event"].mean() if total_pitches else 0,
        "pitcher_recent_hit_event_rate": subset["is_hit"].mean() if total_pitches else 0,
        "pitcher_recent_walk_event_rate": subset["is_walk"].mean() if total_pitches else 0,
    }


def get_opp_pitch_hand(row, games_df, pitcher_hand_cache):
    game_id = str(row.get("game_id"))
    match_rows = games_df[games_df["game_pk"] == game_id]
    if match_rows.empty:
        return None
    game = match_rows.iloc[0]
    pitcher_id = game["away_pitcher_id"] if game["home_abbr"] == row["team"] else game["home_pitcher_id"]
    if pd.isna(pitcher_id):
        return None
    pitcher_id = int(pitcher_id)
    if pitcher_id not in pitcher_hand_cache:
        pitcher_hand_cache[pitcher_id] = get_person_pitch_hand(pitcher_id)
    return pitcher_hand_cache.get(pitcher_id)


def get_opp_pitcher_id(row, games_df):
    game_id = str(row.get("game_id"))
    match_rows = games_df[games_df["game_pk"] == game_id]
    if match_rows.empty:
        return None
    game = match_rows.iloc[0]
    return game["away_pitcher_id"] if game["home_abbr"] == row["team"] else game["home_pitcher_id"]


def get_park_factor(row, games_df):
    game_id = str(row.get("game_id"))
    match_rows = games_df[games_df["game_pk"] == game_id]
    if match_rows.empty:
        return 100
    return match_rows.iloc[0]["park_factor"]


# =========================================================
# BOARD BUILDERS
# =========================================================
def build_hitter_board(
    games_df: pd.DataFrame,
    batter_meta: pd.DataFrame,
    recent_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    pitch_df: pd.DataFrame,
) -> pd.DataFrame:
    pitcher_hand_cache = {}
    for _, row in games_df.iterrows():
        for pitcher_id in [row["home_pitcher_id"], row["away_pitcher_id"]]:
            if pd.notna(pitcher_id):
                pid = int(pitcher_id)
                if pid not in pitcher_hand_cache:
                    pitcher_hand_cache[pid] = get_person_pitch_hand(pid)

    target_pitcher_ids = set()
    for _, row in games_df.iterrows():
        if pd.notna(row["home_pitcher_id"]):
            target_pitcher_ids.add(int(row["home_pitcher_id"]))
        if pd.notna(row["away_pitcher_id"]):
            target_pitcher_ids.add(int(row["away_pitcher_id"]))

    starter_df = pitch_df[pitch_df["pitcher"].isin(target_pitcher_ids)].copy()

    pitcher_agg = starter_df.groupby("pitcher").agg(
        pitcher_xBA_allowed=("estimated_ba_using_speedangle", "mean"),
        pitcher_xwOBA_allowed=("estimated_woba_using_speedangle", "mean"),
        pitcher_ev_allowed=("launch_speed", "mean"),
        pitcher_bbe_against=("estimated_ba_using_speedangle", lambda s: s.notna().sum())
    ).reset_index()

    starter_df["is_strikeout"] = starter_df["events"].fillna("").isin(["strikeout", "strikeout_double_play"]).astype(int)
    starter_df["is_ball_in_play_or_contact"] = starter_df["description"].fillna("").isin([
        "hit_into_play", "foul", "foul_tip", "hit_into_play_no_out", "hit_into_play_score"
    ]).astype(int)

    pitcher_contact_agg = starter_df.groupby("pitcher").agg(
        pitcher_k_events=("is_strikeout", "sum"),
        pitcher_contact_events=("is_ball_in_play_or_contact", "sum"),
        pitcher_total_pitches=("pitcher", "count")
    ).reset_index()

    pitcher_contact_agg["pitcher_k_pressure_raw"] = (
        pitcher_contact_agg["pitcher_k_events"] /
        pitcher_contact_agg["pitcher_total_pitches"].replace(0, pd.NA)
    ).fillna(0)

    pitcher_contact_agg["pitcher_contact_suppression_raw"] = (
        1 - (
            pitcher_contact_agg["pitcher_contact_events"] /
            pitcher_contact_agg["pitcher_total_pitches"].replace(0, pd.NA)
        )
    ).fillna(0)

    pitch_mix = starter_df.dropna(subset=["pitch_type"]).copy()
    pitch_mix_counts = pitch_mix.groupby(["pitcher", "pitch_type"]).size().reset_index(name="pitch_count")
    pitch_mix_totals = pitch_mix_counts.groupby("pitcher")["pitch_count"].sum().reset_index(name="total_pitch_count")
    pitch_mix_counts = pitch_mix_counts.merge(pitch_mix_totals, on="pitcher", how="left")
    pitch_mix_counts["usage_rate"] = pitch_mix_counts["pitch_count"] / pitch_mix_counts["total_pitch_count"]
    pitch_mix_counts = pitch_mix_counts.sort_values(["pitcher", "usage_rate"], ascending=[True, False])
    top_pitch_mix = pitch_mix_counts.groupby("pitcher").head(3).copy()

    pitch_df = pitch_df.copy()
    pitch_df["def_team"] = pitch_df.apply(
        lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"],
        axis=1
    )

    starter_team_map = []
    for _, row in games_df.iterrows():
        if pd.notna(row["home_pitcher_id"]):
            starter_team_map.append({"pitcher": int(row["home_pitcher_id"]), "def_team": row["home_abbr"]})
        if pd.notna(row["away_pitcher_id"]):
            starter_team_map.append({"pitcher": int(row["away_pitcher_id"]), "def_team": row["away_abbr"]})

    starter_team_df = (
        pd.DataFrame(starter_team_map).drop_duplicates()
        if starter_team_map else pd.DataFrame(columns=["pitcher", "def_team"])
    )

    bullpen_df = pitch_df.merge(starter_team_df, on=["pitcher", "def_team"], how="left", indicator=True)
    bullpen_df = bullpen_df[bullpen_df["_merge"] == "left_only"].copy()

    bullpen_agg = bullpen_df.groupby("def_team").agg(
        bullpen_xBA_allowed=("estimated_ba_using_speedangle", "mean"),
        bullpen_xwOBA_allowed=("estimated_woba_using_speedangle", "mean"),
        bullpen_ev_allowed=("launch_speed", "mean"),
        bullpen_bbe_against=("estimated_ba_using_speedangle", lambda s: s.notna().sum())
    ).reset_index().rename(columns={"def_team": "opponent_team"})

    recent_stat_agg = recent_df.groupby("batter").agg(
        recent_xBA=("xBA", "mean"),
        recent_xwOBA=("xwOBA", "mean"),
        recent_exit_velocity=("exit_velocity", "mean"),
        recent_bbe=("xBA", lambda s: s.notna().sum())
    ).reset_index()

    baseline_stat_agg = baseline_df.groupby("batter").agg(
        baseline_xBA=("xBA", "mean"),
        baseline_xwOBA=("xwOBA", "mean"),
        baseline_exit_velocity=("exit_velocity", "mean"),
        baseline_bbe=("xBA", lambda s: s.notna().sum())
    ).reset_index()

    hitter_pitch_df = recent_df.dropna(subset=["pitch_type"]).copy()
    hitter_pitch_agg = hitter_pitch_df.groupby(["batter", "pitch_type"]).agg(
        pitchtype_xwOBA=("xwOBA", "mean"),
        pitchtype_xBA=("xBA", "mean"),
        pitchtype_samples=("pitch_type", "count")
    ).reset_index()

    team_recent_agg = recent_df.groupby("team").agg(
        team_recent_xBA=("xBA", "mean"),
        team_recent_xwOBA=("xwOBA", "mean"),
        team_recent_ev=("exit_velocity", "mean"),
        team_recent_bbe=("xBA", lambda s: s.notna().sum())
    ).reset_index()

    lineup_skill = batter_meta.merge(recent_stat_agg, on="batter", how="left")
    lineup_skill["lineup_slot"] = pd.to_numeric(lineup_skill["lineup_slot"], errors="coerce")

    pocket_rows = []
    for _, hitter_row in lineup_skill.iterrows():
        team = hitter_row["team"]
        slot = hitter_row["lineup_slot"]
        batter = hitter_row["batter"]

        if pd.isna(slot):
            pocket_rows.append({
                "batter": batter,
                "lineup_pocket_xBA": None,
                "lineup_pocket_xwOBA": None,
                "lineup_pocket_ev": None
            })
            continue

        team_lineup = lineup_skill[lineup_skill["team"] == team].copy()
        team_lineup = team_lineup[team_lineup["lineup_slot"].notna()].copy()
        team_lineup["slot_distance"] = (team_lineup["lineup_slot"] - slot).abs()
        pocket = team_lineup[team_lineup["slot_distance"] <= 1].copy()
        pocket = pocket[pocket["batter"] != batter].copy()

        pocket_rows.append({
            "batter": batter,
            "lineup_pocket_xBA": pocket["recent_xBA"].mean() if not pocket.empty else None,
            "lineup_pocket_xwOBA": pocket["recent_xwOBA"].mean() if not pocket.empty else None,
            "lineup_pocket_ev": pocket["recent_exit_velocity"].mean() if not pocket.empty else None,
        })

    lineup_pocket_df = pd.DataFrame(pocket_rows)

    hitters = batter_meta.merge(recent_stat_agg, on="batter", how="left")
    hitters = hitters.merge(baseline_stat_agg, on="batter", how="left")
    hitters = hitters.merge(team_recent_agg, on="team", how="left")
    hitters = hitters.merge(lineup_pocket_df, on="batter", how="left")

    hitters = hitters[
        (hitters["recent_bbe"].fillna(0) >= MIN_RECENT_BBE) &
        (hitters["baseline_bbe"].fillna(0) >= MIN_BASELINE_BBE)
    ].copy()

    if hitters.empty:
        return pd.DataFrame()

    hitters["opp_pitch_hand"] = hitters.apply(lambda r: get_opp_pitch_hand(r, games_df, pitcher_hand_cache), axis=1)
    hitters["handedness_matchup"] = hitters.apply(
        lambda row: handedness_matchup(row["bat_side"], row["opp_pitch_hand"]),
        axis=1
    )

    hitters["slump_delta_xBA"] = hitters["recent_xBA"] - hitters["baseline_xBA"]
    hitters["slump_delta_xwOBA"] = hitters["recent_xwOBA"] - hitters["baseline_xwOBA"]
    hitters["slump_delta_EV"] = hitters["recent_exit_velocity"] - hitters["baseline_exit_velocity"]

    hitters["opp_pitcher_id"] = hitters.apply(lambda r: get_opp_pitcher_id(r, games_df), axis=1)
    hitters["opp_pitcher_id"] = hitters["opp_pitcher_id"].apply(lambda x: int(x) if pd.notna(x) else None)
    hitters["park_factor"] = hitters.apply(lambda r: get_park_factor(r, games_df), axis=1)

    hitters = hitters.merge(
        pitcher_agg,
        left_on="opp_pitcher_id",
        right_on="pitcher",
        how="left"
    )
    hitters = hitters.merge(
        pitcher_contact_agg[["pitcher", "pitcher_k_pressure_raw", "pitcher_contact_suppression_raw"]],
        left_on="opp_pitcher_id",
        right_on="pitcher",
        how="left",
        suffixes=("", "_contact")
    )
    hitters["pitcher_quality_score"] = hitters.apply(score_pitcher_quality, axis=1)

    hitters = hitters.merge(bullpen_agg, on="opponent_team", how="left")
    hitters["bullpen_quality_score"] = hitters.apply(score_bullpen_quality, axis=1)
    hitters["full_game_suppression_score"] = hitters["pitcher_quality_score"].fillna(0) + hitters["bullpen_quality_score"].fillna(0)

    hitters["team_offense_env_score"] = hitters.apply(score_team_offense_environment, axis=1)
    hitters["combined_offense_env"] = hitters["team_offense_env_score"]
    hitters["game_environment_score"] = hitters.apply(score_game_environment, axis=1)

    pitch_mix_penalties = []
    for _, hitter_row in hitters.iterrows():
        batter_id = hitter_row["batter"]
        opp_pitcher_id = hitter_row["opp_pitcher_id"]

        if pd.isna(opp_pitcher_id):
            pitch_mix_penalties.append(0)
            continue

        pitcher_top = top_pitch_mix[top_pitch_mix["pitcher"] == opp_pitcher_id]
        hitter_pitch_stats = hitter_pitch_agg[hitter_pitch_agg["batter"] == batter_id]

        penalty = 0.0
        for _, p_row in pitcher_top.iterrows():
            ptype = p_row["pitch_type"]
            usage = p_row["usage_rate"]
            hmatch = hitter_pitch_stats[hitter_pitch_stats["pitch_type"] == ptype]
            if hmatch.empty:
                continue

            h_xwoba = hmatch.iloc[0]["pitchtype_xwOBA"]
            h_xba = hmatch.iloc[0]["pitchtype_xBA"]
            h_samples = hmatch.iloc[0]["pitchtype_samples"]

            if pd.isna(h_xwoba) or h_samples < 3:
                continue

            if h_xwoba < 0.280:
                penalty += 8 * usage
            elif h_xwoba < 0.310:
                penalty += 5 * usage
            elif h_xwoba < 0.330:
                penalty += 2 * usage

            if pd.notna(h_xba) and h_xba < 0.220:
                penalty += 2 * usage

        pitch_mix_penalties.append(round(penalty, 2))

    hitters["pitch_mix_penalty"] = pitch_mix_penalties

    hitters["pitcher_k_pressure_score"] = hitters["pitcher_k_pressure_raw"].fillna(0).apply(
        lambda x: 10 if x >= 0.08 else 7 if x >= 0.06 else 4 if x >= 0.045 else 0
    )
    hitters["pitcher_zone_contact_suppression_score"] = hitters["pitcher_contact_suppression_raw"].fillna(0).apply(
        lambda x: 8 if x >= 0.70 else 6 if x >= 0.64 else 3 if x >= 0.58 else 0
    )

    hitters["no_hit_score"] = hitters.apply(calc_no_hit_score, axis=1)
    hitters["tb_suppression_score"] = hitters.apply(calc_tb_score, axis=1)
    hitters["runs_suppression_score"] = hitters.apply(calc_runs_score, axis=1)
    hitters["failure_score"] = hitters[["no_hit_score", "tb_suppression_score", "runs_suppression_score"]].max(axis=1)
    hitters["failure_score"] = hitters["failure_score"].apply(lambda x: min(x, 110))

    hitters["best_fade_type"] = hitters.apply(classify_fade_type, axis=1)
    hitters["best_bet_type"] = hitters.apply(assign_best_bet_type, axis=1)
    hitters["reason"] = hitters.apply(build_reason, axis=1)

    return hitters.sort_values(
        by=["failure_score", "lineup_slot", "recent_bbe"],
        ascending=[False, True, False]
    ).copy()


def build_team_total_unders(
    games_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    pitch_df: pd.DataFrame,
    first5: bool = False
) -> pd.DataFrame:
    if games_df.empty or recent_df.empty or pitch_df.empty:
        return pd.DataFrame()

    team_recent_agg = recent_df.groupby("team").agg(
        team_recent_xBA=("xBA", "mean"),
        team_recent_xwOBA=("xwOBA", "mean"),
        team_recent_ev=("exit_velocity", "mean"),
        team_recent_bbe=("xBA", lambda s: s.notna().sum())
    ).reset_index()

    target_pitcher_ids = set()
    for _, row in games_df.iterrows():
        if pd.notna(row["home_pitcher_id"]):
            target_pitcher_ids.add(int(row["home_pitcher_id"]))
        if pd.notna(row["away_pitcher_id"]):
            target_pitcher_ids.add(int(row["away_pitcher_id"]))

    starter_df = pitch_df[pitch_df["pitcher"].isin(target_pitcher_ids)].copy()

    pitcher_agg = starter_df.groupby("pitcher").agg(
        pitcher_xBA_allowed=("estimated_ba_using_speedangle", "mean"),
        pitcher_xwOBA_allowed=("estimated_woba_using_speedangle", "mean"),
        pitcher_ev_allowed=("launch_speed", "mean"),
        pitcher_bbe_against=("estimated_ba_using_speedangle", lambda s: s.notna().sum())
    ).reset_index()

    pitch_df = pitch_df.copy()
    pitch_df["def_team"] = pitch_df.apply(
        lambda row: row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"],
        axis=1
    )

    starter_team_map = []
    for _, row in games_df.iterrows():
        if pd.notna(row["home_pitcher_id"]):
            starter_team_map.append({"pitcher": int(row["home_pitcher_id"]), "def_team": row["home_abbr"]})
        if pd.notna(row["away_pitcher_id"]):
            starter_team_map.append({"pitcher": int(row["away_pitcher_id"]), "def_team": row["away_abbr"]})

    starter_team_df = (
        pd.DataFrame(starter_team_map).drop_duplicates()
        if starter_team_map else pd.DataFrame(columns=["pitcher", "def_team"])
    )

    bullpen_df = pitch_df.merge(starter_team_df, on=["pitcher", "def_team"], how="left", indicator=True)
    bullpen_df = bullpen_df[bullpen_df["_merge"] == "left_only"].copy()

    bullpen_agg = bullpen_df.groupby("def_team").agg(
        bullpen_xBA_allowed=("estimated_ba_using_speedangle", "mean"),
        bullpen_xwOBA_allowed=("estimated_woba_using_speedangle", "mean"),
        bullpen_ev_allowed=("launch_speed", "mean"),
        bullpen_bbe_against=("estimated_ba_using_speedangle", lambda s: s.notna().sum())
    ).reset_index().rename(columns={"def_team": "opponent_team"})

    rows = []

    for _, game in games_df.iterrows():
        for side in ["home", "away"]:
            batting_team = game["home_abbr"] if side == "home" else game["away_abbr"]
            opp_team = game["away_abbr"] if side == "home" else game["home_abbr"]
            opp_pitcher_id = game["away_pitcher_id"] if side == "home" else game["home_pitcher_id"]
            opp_pitcher_name = game["away_pitcher"] if side == "home" else game["home_pitcher"]

            team_row = team_recent_agg[team_recent_agg["team"] == batting_team]
            sp_row = pitcher_agg[pitcher_agg["pitcher"] == opp_pitcher_id] if pd.notna(opp_pitcher_id) else pd.DataFrame()
            bp_row = bullpen_agg[bullpen_agg["opponent_team"] == opp_team]

            if team_row.empty:
                continue

            score = 0
            reasons = []

            team_xba = team_row.iloc[0]["team_recent_xBA"]
            team_xwoba = team_row.iloc[0]["team_recent_xwOBA"]
            team_ev = team_row.iloc[0]["team_recent_ev"]

            if pd.notna(team_xwoba) and team_xwoba < 0.300:
                score += 12
                reasons.append("weak team xwOBA")
            elif pd.notna(team_xwoba) and team_xwoba < 0.315:
                score += 6
                reasons.append("below-average team xwOBA")

            if pd.notna(team_xba) and team_xba < 0.230:
                score += 9
                reasons.append("weak team xBA")
            elif pd.notna(team_xba) and team_xba < 0.245:
                score += 4
                reasons.append("below-average team xBA")

            if pd.notna(team_ev) and team_ev < 89.5:
                score += 5
                reasons.append("low team EV")

            if not sp_row.empty:
                sp_xwoba = sp_row.iloc[0]["pitcher_xwOBA_allowed"]
                sp_xba = sp_row.iloc[0]["pitcher_xBA_allowed"]
                if pd.notna(sp_xwoba) and sp_xwoba < 0.300:
                    score += 11
                    reasons.append("strong starter")
                elif pd.notna(sp_xwoba) and sp_xwoba < 0.315:
                    score += 5
                if pd.notna(sp_xba) and sp_xba < 0.230:
                    score += 5

            if not first5 and not bp_row.empty:
                bp_xwoba = bp_row.iloc[0]["bullpen_xwOBA_allowed"]
                if pd.notna(bp_xwoba) and bp_xwoba < 0.305:
                    score += 7
                    reasons.append("strong bullpen")
                elif pd.notna(bp_xwoba) and bp_xwoba < 0.320:
                    score += 3

            park_factor = game["park_factor"]
            if park_factor <= 96:
                score += 6
                reasons.append("pitcher-friendly park")
            elif park_factor <= 99:
                score += 3

            rows.append({
                "prop_family": "FIRST5_TEAM_TOTAL_UNDER" if first5 else "TEAM_TOTAL_UNDER",
                "entity_name": batting_team,
                "team": batting_team,
                "opponent_team": opp_team,
                "game_id": str(game["game_pk"]),
                "prop": "F5 Team Total Under" if first5 else "Team Total Under",
                "market_score": score,
                "reason": ", ".join(reasons[:5]),
                "source_mode": "team_model",
                "lineup_slot": None,
                "opposing_pitcher": opp_pitcher_name,
                "park_factor": park_factor,
                "recent_bbe": None,
                "matchup": build_matchup(batting_team, opp_team),
            })

    return pd.DataFrame(rows).sort_values(by=["market_score"], ascending=False).head(12) if rows else pd.DataFrame()


def build_pitcher_prop_boards(
    games_df: pd.DataFrame,
    pitch_df: pd.DataFrame,
    recent_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if games_df.empty or pitch_df.empty or recent_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    target_pitcher_ids = set()
    for _, row in games_df.iterrows():
        if pd.notna(row["home_pitcher_id"]):
            target_pitcher_ids.add(int(row["home_pitcher_id"]))
        if pd.notna(row["away_pitcher_id"]):
            target_pitcher_ids.add(int(row["away_pitcher_id"]))

    starter_df = pitch_df[pitch_df["pitcher"].isin(target_pitcher_ids)].copy()

    pitcher_agg = starter_df.groupby("pitcher").agg(
        pitcher_xBA_allowed=("estimated_ba_using_speedangle", "mean"),
        pitcher_xwOBA_allowed=("estimated_woba_using_speedangle", "mean"),
        pitcher_ev_allowed=("launch_speed", "mean"),
        pitcher_bbe_against=("estimated_ba_using_speedangle", lambda s: s.notna().sum())
    ).reset_index()

    team_recent_agg = recent_df.groupby("team").agg(
        team_recent_xBA=("xBA", "mean"),
        team_recent_xwOBA=("xwOBA", "mean"),
        team_recent_ev=("exit_velocity", "mean"),
        team_recent_bbe=("xBA", lambda s: s.notna().sum())
    ).reset_index()

    outs_rows = []
    k_rows = []
    hits_rows = []
    walks_rows = []

    for _, game in games_df.iterrows():
        for side in ["home", "away"]:
            pitcher_id = game["home_pitcher_id"] if side == "home" else game["away_pitcher_id"]
            pitcher_name = game["home_pitcher"] if side == "home" else game["away_pitcher"]
            pitcher_team = game["home_abbr"] if side == "home" else game["away_abbr"]
            opp_team = game["away_abbr"] if side == "home" else game["home_abbr"]
            park_factor = game["park_factor"]

            if pd.isna(pitcher_id):
                continue

            pitcher_id = int(pitcher_id)
            recent_pitch_summary = get_pitcher_recent_summary(pitch_df, pitcher_id)
            sp_row = pitcher_agg[pitcher_agg["pitcher"] == pitcher_id]
            opp_team_row = team_recent_agg[team_recent_agg["team"] == opp_team]

            if sp_row.empty:
                continue

            sp_xwoba = sp_row.iloc[0]["pitcher_xwOBA_allowed"]
            sp_xba = sp_row.iloc[0]["pitcher_xBA_allowed"]
            sp_ev = sp_row.iloc[0]["pitcher_ev_allowed"]

            outs_score = 0
            k_score = 0
            hits_score = 0
            walks_score = 0

            outs_reasons = []
            k_reasons = []
            hits_reasons = []
            walks_reasons = []

            if pd.notna(sp_xwoba) and sp_xwoba < 0.300:
                outs_score += 12
                k_score += 8
                hits_score += 10
                walks_score += 3
                outs_reasons.append("suppresses quality contact")
                k_reasons.append("quality run prevention")
                hits_reasons.append("suppresses quality contact")
            elif pd.notna(sp_xwoba) and sp_xwoba < 0.315:
                outs_score += 6
                k_score += 4
                hits_score += 5

            if pd.notna(sp_xba) and sp_xba < 0.230:
                outs_score += 8
                k_score += 4
                hits_score += 10
                hits_reasons.append("low xBA allowed")
            elif pd.notna(sp_xba) and sp_xba < 0.245:
                hits_score += 5

            if pd.notna(sp_ev) and sp_ev < 89:
                outs_score += 4
                hits_score += 4
                outs_reasons.append("low EV allowed")

            recent_k_rate = recent_pitch_summary.get("pitcher_recent_k_rate", 0)
            recent_out_rate = recent_pitch_summary.get("pitcher_recent_out_event_rate", 0)
            recent_hit_rate = recent_pitch_summary.get("pitcher_recent_hit_event_rate", 0)
            recent_walk_rate = recent_pitch_summary.get("pitcher_recent_walk_event_rate", 0)

            if recent_out_rate >= 0.72:
                outs_score += 10
                outs_reasons.append("high recent out conversion")
            elif recent_out_rate >= 0.66:
                outs_score += 5

            if recent_k_rate >= 0.08:
                k_score += 12
                k_reasons.append("strong recent K rate")
            elif recent_k_rate >= 0.06:
                k_score += 6
                k_reasons.append("solid recent K rate")

            if recent_hit_rate <= 0.11:
                hits_score += 10
                hits_reasons.append("low recent hit rate")
            elif recent_hit_rate <= 0.14:
                hits_score += 5

            if recent_walk_rate <= 0.05:
                walks_score += 10
                walks_reasons.append("low recent walk rate")
            elif recent_walk_rate <= 0.065:
                walks_score += 5

            if not opp_team_row.empty:
                opp_xwoba = opp_team_row.iloc[0]["team_recent_xwOBA"]
                opp_xba = opp_team_row.iloc[0]["team_recent_xBA"]

                if pd.notna(opp_xwoba) and opp_xwoba < 0.300:
                    outs_score += 8
                    k_score += 8
                    hits_score += 8
                    outs_reasons.append("weak opposing offense")
                    k_reasons.append("weak opposing offense")
                    hits_reasons.append("weak opposing offense")
                elif pd.notna(opp_xwoba) and opp_xwoba < 0.315:
                    outs_score += 4
                    k_score += 4
                    hits_score += 4

                if pd.notna(opp_xba) and opp_xba < 0.230:
                    outs_score += 5
                    k_score += 4
                    hits_score += 6

            if park_factor <= 96:
                outs_score += 5
                hits_score += 5
                k_score += 2
                outs_reasons.append("pitcher-friendly park")
                hits_reasons.append("pitcher-friendly park")
            elif park_factor >= 106:
                outs_score -= 4
                hits_score -= 4

            game_id = str(game["game_pk"])
            matchup = build_matchup(pitcher_team, opp_team)

            outs_rows.append({
                "prop_family": "PITCHER_OUTS",
                "entity_name": pitcher_name,
                "team": pitcher_team,
                "opponent_team": opp_team,
                "game_id": game_id,
                "prop": "Pitcher Outs Over",
                "market_score": outs_score,
                "reason": ", ".join(outs_reasons[:5]),
                "source_mode": "pitcher_model",
                "lineup_slot": None,
                "opposing_pitcher": None,
                "park_factor": park_factor,
                "recent_bbe": None,
                "matchup": matchup,
            })

            k_rows.append({
                "prop_family": "PITCHER_STRIKEOUTS",
                "entity_name": pitcher_name,
                "team": pitcher_team,
                "opponent_team": opp_team,
                "game_id": game_id,
                "prop": "Pitcher Strikeouts Over",
                "market_score": k_score,
                "reason": ", ".join(k_reasons[:5]),
                "source_mode": "pitcher_model",
                "lineup_slot": None,
                "opposing_pitcher": None,
                "park_factor": park_factor,
                "recent_bbe": None,
                "matchup": matchup,
            })

            hits_rows.append({
                "prop_family": "PITCHER_HITS_ALLOWED",
                "entity_name": pitcher_name,
                "team": pitcher_team,
                "opponent_team": opp_team,
                "game_id": game_id,
                "prop": "Pitcher Hits Allowed Under",
                "market_score": hits_score,
                "reason": ", ".join(hits_reasons[:5]),
                "source_mode": "pitcher_model",
                "lineup_slot": None,
                "opposing_pitcher": None,
                "park_factor": park_factor,
                "recent_bbe": None,
                "matchup": matchup,
            })

            walks_rows.append({
                "prop_family": "PITCHER_WALKS_ALLOWED",
                "entity_name": pitcher_name,
                "team": pitcher_team,
                "opponent_team": opp_team,
                "game_id": game_id,
                "prop": "Pitcher Walks Allowed Under",
                "market_score": walks_score,
                "reason": ", ".join(walks_reasons[:5]),
                "source_mode": "pitcher_model",
                "lineup_slot": None,
                "opposing_pitcher": None,
                "park_factor": park_factor,
                "recent_bbe": None,
                "matchup": matchup,
            })

    def top_df(rows):
        return pd.DataFrame(rows).sort_values(by=["market_score"], ascending=False).head(12) if rows else pd.DataFrame()

    return top_df(outs_rows), top_df(k_rows), top_df(hits_rows), top_df(walks_rows)


# =========================================================
# PROP CONVERSION
# =========================================================
def hitter_board_to_props(hitter_board: pd.DataFrame) -> List[Prop]:
    if hitter_board.empty:
        return []

    props = []
    for _, row in hitter_board.iterrows():
        prop_family = "HITTER_FADE"
        model_prob = normalized_market_probability(row["failure_score"])
        volatility = PROP_FAMILY_VOLATILITY[prop_family]
        stability = compute_stability(
            prop_family=prop_family,
            source_mode=row.get("source_mode"),
            lineup_slot=row.get("lineup_slot"),
            recent_bbe=row.get("recent_bbe"),
            park_factor=row.get("park_factor"),
        )
        confidence = compute_confidence(model_prob, stability, volatility, row["failure_score"])
        opportunity_score = compute_opportunity_score(row["failure_score"], confidence, stability, volatility)
        parlay_fit = compute_parlay_fit(confidence, stability, volatility, prop_family)
        script_tag = derive_script_tag(prop_family, row["failure_score"], row.get("lineup_slot"))

        props.append(Prop(
            prop=row["best_bet_type"],
            entity_name=row["player"],
            team=row["team"],
            opponent_team=row["opponent_team"],
            matchup=build_matchup(row["team"], row["opponent_team"]),
            game_id=str(row.get("game_id", f"{row['team']}_{row['opponent_team']}")),
            prop_family=prop_family,
            model_prob=model_prob,
            stability=stability,
            confidence=confidence,
            parlay_fit=parlay_fit,
            opportunity_score=opportunity_score,
            volatility=volatility,
            script_tag=script_tag,
            market_score=row["failure_score"],
            reason=row["reason"],
            source_mode=row["source_mode"],
            opposing_pitcher=row.get("opposing_pitcher"),
            lineup_slot=row.get("lineup_slot"),
            raw=row.to_dict(),
        ))

    return props


def generic_board_to_props(board: pd.DataFrame) -> List[Prop]:
    if board.empty:
        return []

    props = []
    for _, row in board.iterrows():
        prop_family = row["prop_family"]
        model_prob = normalized_market_probability(row["market_score"])
        volatility = PROP_FAMILY_VOLATILITY[prop_family]
        stability = compute_stability(
            prop_family=prop_family,
            source_mode=row.get("source_mode"),
            lineup_slot=row.get("lineup_slot"),
            recent_bbe=row.get("recent_bbe"),
            park_factor=row.get("park_factor"),
        )
        confidence = compute_confidence(model_prob, stability, volatility, row["market_score"])
        opportunity_score = compute_opportunity_score(row["market_score"], confidence, stability, volatility)
        parlay_fit = compute_parlay_fit(confidence, stability, volatility, prop_family)
        script_tag = derive_script_tag(prop_family, row["market_score"], row.get("lineup_slot"))

        props.append(Prop(
            prop=row["prop"],
            entity_name=row["entity_name"],
            team=row["team"],
            opponent_team=row["opponent_team"],
            matchup=row.get("matchup", build_matchup(row["team"], row["opponent_team"])),
            game_id=str(row["game_id"]),
            prop_family=prop_family,
            model_prob=model_prob,
            stability=stability,
            confidence=confidence,
            parlay_fit=parlay_fit,
            opportunity_score=opportunity_score,
            volatility=volatility,
            script_tag=script_tag,
            market_score=row["market_score"],
            reason=row["reason"],
            source_mode=row["source_mode"],
            opposing_pitcher=row.get("opposing_pitcher"),
            lineup_slot=row.get("lineup_slot"),
            raw=row.to_dict(),
        ))

    return props


# =========================================================
# PARLAYS
# =========================================================
def estimate_parlay_hit_rate(legs: List[Prop]) -> float:
    if not legs:
        return 0.0

    joint_prob = 1.0
    for leg in legs:
        joint_prob *= leg.model_prob

    game_ids = [leg.game_id for leg in legs]
    unique_games = len(set(game_ids))
    repeated_games = len(game_ids) - unique_games

    correlation_penalty = 1.0 - (0.08 * repeated_games)
    correlation_penalty = max(0.78, correlation_penalty)

    est = joint_prob * correlation_penalty
    return round(clamp(est * 100, 0, 100), 1)


def build_blended_parlays(props: List[Prop], count: int = PARLAY_COUNT) -> List[Parlay]:
    if not props:
        return []

    candidates = [
        p for p in props
        if p.confidence >= MIN_PARLAY_CONFIDENCE and p.parlay_fit >= MIN_PARLAY_FIT
    ]

    if len(candidates) < 2:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: (x.parlay_fit, x.confidence, x.opportunity_score),
        reverse=True
    )[:30]

    parlays: List[Parlay] = []

    # 2-leg parlays
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            legs = [candidates[i], candidates[j]]

            avg_conf = sum(l.confidence for l in legs) / len(legs)
            min_conf = min(l.confidence for l in legs)
            avg_opportunity = sum(l.opportunity_score for l in legs) / len(legs)
            estimated_hit_rate = estimate_parlay_hit_rate(legs)

            structural = 100.0

            if len({l.game_id for l in legs}) < len(legs):
                structural -= 6

            if len({l.prop_family for l in legs}) == 1:
                structural -= 4

            for leg in legs:
                if leg.volatility == "HIGH":
                    structural -= 8
                elif leg.volatility == "MED":
                    structural -= 3

            structural = clamp(structural, 55, 100)

            final_score = round(
                0.30 * avg_conf +
                0.20 * min_conf +
                0.20 * avg_opportunity +
                0.15 * structural +
                0.15 * estimated_hit_rate,
                1
            )

            parlays.append(Parlay(
                legs=legs,
                avg_conf=round(avg_conf, 1),
                min_conf=round(min_conf, 1),
                structural=round(structural, 1),
                avg_opportunity=round(avg_opportunity, 1),
                estimated_hit_rate=estimated_hit_rate,
                final_score=final_score,
            ))

    # 3-leg parlays
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                legs = [candidates[i], candidates[j], candidates[k]]

                avg_conf = sum(l.confidence for l in legs) / len(legs)
                min_conf = min(l.confidence for l in legs)
                avg_opportunity = sum(l.opportunity_score for l in legs) / len(legs)
                estimated_hit_rate = estimate_parlay_hit_rate(legs)

                structural = 100.0
                unique_games = len({l.game_id for l in legs})

                if unique_games == 2:
                    structural -= 7
                elif unique_games == 1:
                    structural -= 12

                families = [l.prop_family for l in legs]
                if len(set(families)) == 1:
                    structural -= 6
                elif len(set(families)) == 2:
                    structural -= 2

                for leg in legs:
                    if leg.volatility == "HIGH":
                        structural -= 8
                    elif leg.volatility == "MED":
                        structural -= 3

                structural = clamp(structural, 50, 100)

                final_score = round(
                    0.28 * avg_conf +
                    0.20 * min_conf +
                    0.20 * avg_opportunity +
                    0.14 * structural +
                    0.18 * estimated_hit_rate,
                    1
                )

                parlays.append(Parlay(
                    legs=legs,
                    avg_conf=round(avg_conf, 1),
                    min_conf=round(min_conf, 1),
                    structural=round(structural, 1),
                    avg_opportunity=round(avg_opportunity, 1),
                    estimated_hit_rate=estimated_hit_rate,
                    final_score=final_score,
                ))

    parlays = sorted(
        parlays,
        key=lambda x: (x.final_score, x.estimated_hit_rate, x.avg_conf),
        reverse=True
    )

    unique = []
    seen = set()
    for parlay in parlays:
        sig = tuple(sorted((leg.prop, leg.entity_name, leg.team, leg.game_id) for leg in parlay.legs))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(parlay)
        if len(unique) >= count:
            break

    return unique


# =========================================================
# SORTING / OUTPUT
# =========================================================
def sort_top_bets(props: List[Prop]) -> List[Prop]:
    return sorted(props, key=lambda x: (x.confidence, x.opportunity_score, x.stability), reverse=True)[:TOP_BETS_COUNT]


def sort_best_opportunity(props: List[Prop]) -> List[Prop]:
    return sorted(props, key=lambda x: (x.opportunity_score, x.confidence), reverse=True)[:BEST_OPPORTUNITY_COUNT]


def sort_most_stable(props: List[Prop]) -> List[Prop]:
    return sorted(props, key=lambda x: (x.stability, x.confidence), reverse=True)[:MOST_STABLE_COUNT]


def sort_high_risk(props: List[Prop]) -> List[Prop]:
    high_risk = [p for p in props if p.volatility != "LOW" or p.confidence < 62]
    return sorted(
        high_risk,
        key=lambda x: (x.opportunity_score, -x.confidence),
        reverse=True
    )[:HIGH_RISK_COUNT]


def build_metrics(props: List[Prop]) -> Dict:
    if not props:
        return {
            "top": 0,
            "avg_opportunity": 0.0,
            "best_opportunity": 0.0,
            "avg_conf": 0.0,
            "total": 0,
        }

    return {
        "top": len([p for p in props if p.confidence >= 70]),
        "avg_opportunity": round(sum(p.opportunity_score for p in props) / len(props), 2),
        "best_opportunity": round(max(p.opportunity_score for p in props), 2),
        "avg_conf": round(sum(p.confidence for p in props) / len(props), 1),
        "total": len(props),
    }


def build_summary_tables(props: List[Prop]) -> Dict[str, pd.DataFrame]:
    if not props:
        empty = pd.DataFrame()
        return {
            "by_game": empty,
            "by_family": empty,
            "tiered": empty,
        }

    rows = []
    for p in props:
        rows.append({
            "Game ID": p.game_id,
            "Matchup": p.matchup,
            "Prop": p.prop,
            "Entity": p.entity_name,
            "Family": p.prop_family,
            "Confidence": p.confidence,
            "Opportunity": p.opportunity_score,
            "Parlay Fit": p.parlay_fit,
            "Volatility": p.volatility,
        })
    df = pd.DataFrame(rows)

    by_game = (
        df.groupby(["Game ID", "Matchup"], as_index=False)
        .agg(
            Playable_Props=("Prop", "count"),
            Avg_Confidence=("Confidence", "mean"),
            Avg_Opportunity=("Opportunity", "mean"),
            Best_Prop=("Opportunity", "max"),
        )
        .sort_values(["Best_Prop", "Avg_Confidence"], ascending=False)
    )

    by_family = (
        df.groupby("Family", as_index=False)
        .agg(
            Props=("Prop", "count"),
            Avg_Confidence=("Confidence", "mean"),
            Avg_Opportunity=("Opportunity", "mean"),
        )
        .sort_values(["Avg_Opportunity", "Avg_Confidence"], ascending=False)
    )

    tier_df = df.copy()

    def assign_tier(row):
        if row["Confidence"] >= 75 and row["Opportunity"] >= 78:
            return "Top Tier"
        if row["Confidence"] >= 68 and row["Opportunity"] >= 70:
            return "Strong"
        if row["Confidence"] >= 60 and row["Opportunity"] >= 60:
            return "Fringe"
        return "Avoid"

    tier_df["Tier"] = tier_df.apply(assign_tier, axis=1)
    tiered = (
        tier_df.groupby("Tier", as_index=False)
        .agg(
            Props=("Prop", "count"),
            Avg_Confidence=("Confidence", "mean"),
            Avg_Opportunity=("Opportunity", "mean"),
        )
    )

    tier_order = {"Top Tier": 0, "Strong": 1, "Fringe": 2, "Avoid": 3}
    tiered["sort_order"] = tiered["Tier"].map(tier_order)
    tiered = tiered.sort_values("sort_order").drop(columns=["sort_order"])

    return {
        "by_game": by_game,
        "by_family": by_family,
        "tiered": tiered,
    }


def export_debug_csvs(results: Dict):
    if not ENABLE_CSV_EXPORT:
        return

    ensure_output_dir()
    today = datetime.today().strftime("%Y-%m-%d")

    def props_to_df(props: List[Prop]) -> pd.DataFrame:
        rows = []
        for p in props:
            rows.append({
                "prop": p.prop,
                "entity_name": p.entity_name,
                "team": p.team,
                "opponent_team": p.opponent_team,
                "matchup": p.matchup,
                "game_id": p.game_id,
                "prop_family": p.prop_family,
                "model_prob": p.model_prob,
                "stability": p.stability,
                "confidence": p.confidence,
                "parlay_fit": p.parlay_fit,
                "opportunity_score": p.opportunity_score,
                "volatility": p.volatility,
                "script_tag": p.script_tag,
                "market_score": p.market_score,
                "reason": p.reason,
                "source_mode": p.source_mode,
                "opposing_pitcher": p.opposing_pitcher,
                "lineup_slot": p.lineup_slot,
            })
        return pd.DataFrame(rows)

    props_to_df(results["all_props"]).to_csv(os.path.join(OUTPUT_DIR, f"all_props_{today}.csv"), index=False)
    props_to_df(results["top_bets"]).to_csv(os.path.join(OUTPUT_DIR, f"top_bets_{today}.csv"), index=False)
    props_to_df(results["best_opportunity"]).to_csv(os.path.join(OUTPUT_DIR, f"best_opportunity_{today}.csv"), index=False)
    props_to_df(results["most_stable"]).to_csv(os.path.join(OUTPUT_DIR, f"most_stable_{today}.csv"), index=False)
    props_to_df(results["high_risk"]).to_csv(os.path.join(OUTPUT_DIR, f"high_risk_{today}.csv"), index=False)

    parlay_rows = []
    for p in results["parlays"]:
        parlay_rows.append({
            "legs": " | ".join([f"{leg.prop} - {leg.entity_name}" for leg in p.legs]),
            "avg_conf": p.avg_conf,
            "min_conf": p.min_conf,
            "structural": p.structural,
            "avg_opportunity": p.avg_opportunity,
            "estimated_hit_rate": p.estimated_hit_rate,
            "final_score": p.final_score,
        })
    pd.DataFrame(parlay_rows).to_csv(os.path.join(OUTPUT_DIR, f"parlays_{today}.csv"), index=False)


# =========================================================
# MAIN RUNNER
# =========================================================
def empty_results(games_found: int = 0, source_mode: str = "none") -> Dict:
    return {
        "top_bets": [],
        "best_opportunity": [],
        "most_stable": [],
        "high_risk": [],
        "parlays": [],
        "all_props": [],
        "metrics": {
            "top": 0,
            "avg_opportunity": 0.0,
            "best_opportunity": 0.0,
            "avg_conf": 0.0,
            "total": 0,
        },
        "summary_tables": {
            "by_game": pd.DataFrame(),
            "by_family": pd.DataFrame(),
            "tiered": pd.DataFrame(),
        },
        "meta": {
            "run_time": datetime.now().isoformat(),
            "games_found": games_found,
            "source_mode": source_mode,
        }
    }


def run_model(target_date: Optional[str] = None) -> Dict:
    games_df = load_schedule(target_date)

    if games_df.empty:
        return empty_results(0, "none")

    batter_meta = load_batter_pool(games_df)
    if batter_meta.empty:
        return empty_results(len(games_df), "none")

    teams_today_abbr = set(games_df["home_abbr"].dropna().tolist() + games_df["away_abbr"].dropna().tolist())
    target_batter_ids = set(batter_meta["batter"].dropna().astype(int).tolist())

    source_mode = "confirmed_lineup" if (
        not batter_meta.empty and (batter_meta["source_mode"] == "confirmed_lineup").any()
    ) else "preliminary_team_pool"

    recent_df = pull_hitter_window(RECENT_LOOKBACK_DAYS, teams_today_abbr, target_batter_ids)
    baseline_df = pull_hitter_window(BASELINE_LOOKBACK_DAYS, teams_today_abbr, target_batter_ids)
    pitch_df = pull_pitcher_window(PITCHER_LOOKBACK_DAYS)

    if recent_df.empty or baseline_df.empty or pitch_df.empty:
        return empty_results(len(games_df), source_mode)

    hitter_board = build_hitter_board(games_df, batter_meta, recent_df, baseline_df, pitch_df)
    team_total_under_board = build_team_total_unders(games_df, recent_df, pitch_df, first5=False)
    first5_team_total_under_board = build_team_total_unders(games_df, recent_df, pitch_df, first5=True)
    pitcher_outs_board, pitcher_k_board, pitcher_hits_board, pitcher_walks_board = build_pitcher_prop_boards(
        games_df, pitch_df, recent_df
    )

    all_props: List[Prop] = []
    all_props.extend(hitter_board_to_props(hitter_board.head(25) if not hitter_board.empty else hitter_board))
    all_props.extend(generic_board_to_props(team_total_under_board))
    all_props.extend(generic_board_to_props(first5_team_total_under_board))
    all_props.extend(generic_board_to_props(pitcher_outs_board))
    all_props.extend(generic_board_to_props(pitcher_k_board))
    all_props.extend(generic_board_to_props(pitcher_hits_board))
    all_props.extend(generic_board_to_props(pitcher_walks_board))

    all_props = sorted(all_props, key=lambda x: (x.confidence, x.opportunity_score, x.stability), reverse=True)

    top_bets = sort_top_bets(all_props)
    best_opportunity = sort_best_opportunity(all_props)
    most_stable = sort_most_stable(all_props)
    high_risk = sort_high_risk(all_props)
    parlays = build_blended_parlays(all_props, count=PARLAY_COUNT)

    results = {
        "top_bets": top_bets,
        "best_opportunity": best_opportunity,
        "most_stable": most_stable,
        "high_risk": high_risk,
        "parlays": parlays,
        "all_props": all_props,
        "metrics": build_metrics(all_props),
        "summary_tables": build_summary_tables(all_props),
        "meta": {
            "run_time": datetime.now().isoformat(),
            "games_found": len(games_df),
            "source_mode": source_mode,
        }
    }

    export_debug_csvs(results)
    return results


if __name__ == "__main__":
    results = run_model()
    print("\n=== MLB FADE MODEL ===")
    print(f"Games found: {results['meta']['games_found']}")
    print(f"Source mode: {results['meta']['source_mode']}")
    print("\n=== TOP BETS ===")
    for p in results["top_bets"][:10]:
        print(
            f"{p.prop} | {p.entity_name} | {p.matchup} | "
            f"Conf {p.confidence} | Opp {p.opportunity_score}"
        )
    print("\n=== PARLAYS ===")
    for parlay in results["parlays"]:
        legs = " + ".join([f"{leg.prop} ({leg.entity_name})" for leg in parlay.legs])
        print(f"{legs} | Score {parlay.final_score} | Est Hit {parlay.estimated_hit_rate}%")

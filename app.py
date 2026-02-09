import json
import math
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CouponProfile:
    key: str
    title: str
    target_min: float
    target_max: float
    min_legs: int
    max_legs: int
    odd_min: float
    odd_max: float
    min_edge: float
    min_prob: float
    min_books: int


BASE_ODDS = "https://api.the-odds-api.com/v4"
BASE_FOOTBALL = "https://api.football-data.org/v4"
ODDS_MARKETS = ["h2h", "totals", "btts"]

LEAGUES = {
    "soccer_epl": {"label": "Premier League", "fd_comp": "PL"},
    "soccer_spain_la_liga": {"label": "La Liga", "fd_comp": "PD"},
    "soccer_italy_serie_a": {"label": "Serie A", "fd_comp": "SA"},
    "soccer_germany_bundesliga": {"label": "Bundesliga", "fd_comp": "BL1"},
    "soccer_france_ligue_one": {"label": "Ligue 1", "fd_comp": "FL1"},
    "soccer_portugal_primeira_liga": {"label": "Primeira Liga", "fd_comp": "PPL"},
    "soccer_netherlands_eredivisie": {"label": "Eredivisie", "fd_comp": "DED"},
}

PROFILES = [
    CouponProfile(
        key="high",
        title="Yuksek Oran",
        target_min=250.0,
        target_max=1500.0,
        min_legs=6,
        max_legs=12,
        odd_min=1.55,
        odd_max=7.5,
        min_edge=0.025,
        min_prob=0.17,
        min_books=2,
    ),
    CouponProfile(
        key="monster",
        title="Canavar Kupon",
        target_min=1500.0,
        target_max=15000.0,
        min_legs=8,
        max_legs=15,
        odd_min=1.85,
        odd_max=12.0,
        min_edge=0.04,
        min_prob=0.11,
        min_books=2,
    ),
    CouponProfile(
        key="golden",
        title="Altin Vurus",
        target_min=50000.0,
        target_max=100000.0,
        min_legs=9,
        max_legs=20,
        odd_min=1.85,
        odd_max=17.0,
        min_edge=0.035,
        min_prob=0.06,
        min_books=1,
    ),
]

MARKET_TR = {
    "h2h": "MS",
    "totals": "Alt/Ust",
    "btts": "KG",
}

STOPWORDS = {
    "fc",
    "cf",
    "ac",
    "sc",
    "as",
    "club",
    "de",
    "the",
    "ud",
    "afc",
    "fk",
    "sk",
    "calcio",
    "sporting",
}


def require_login_if_enabled() -> None:
    app_password = os.getenv("APP_PASSWORD", "").strip()
    if not app_password:
        return

    if st.session_state.get("authenticated", False):
        return

    st.title("Bet AI Kupon Motoru Pro")
    st.caption("Guvenli erisim aktif. Devam etmek icin sifre gir.")
    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Sifre", type="password")
        submit = st.form_submit_button("Giris")
    if submit and pwd == app_password:
        st.session_state["authenticated"] = True
        st.rerun()
    elif submit:
        st.error("Sifre hatali.")
    st.stop()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #060912;
            --card: #101727;
            --card-2: #182238;
            --line: #2a3654;
            --txt: #f2f6ff;
            --muted: #9fb0d3;
            --ok: #1fcf82;
            --warn: #f4b63c;
            --danger: #ff5d73;
            --gold: #f0c35b;
            --blue: #3f8cff;
        }
        .stApp { background: radial-gradient(1200px 400px at 40% -100px, #1c2b4a 0%, var(--bg) 55%); }
        .kpi-card {
            background: linear-gradient(160deg, var(--card), var(--card-2));
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px 14px;
        }
        .kpi-title { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
        .kpi-value { color: var(--txt); font-size: 24px; font-weight: 800; }
        .kpi-sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
        .panel-card {
            background: linear-gradient(165deg, #0f172b, #0e1322);
            border: 1px solid #253351;
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .chip {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid #2c416a;
            color: #b7cdfa;
            font-size: 11px;
            padding: 2px 8px;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .gold-chip { border-color: #8a6c2a; color: #ffd676; }
        .warn-chip { border-color: #6c5420; color: #f2c96d; }
        .ok-chip { border-color: #1f724f; color: #7ef4be; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def implied_prob(odd: float) -> float:
    if odd <= 1.0001:
        return 0.0
    return 1.0 / odd


def normalize_probs(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return values
    return [v / total for v in values]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 0.001), 0.999)
    return math.log(p / (1.0 - p))


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def normalize_team_name(name: str) -> str:
    text = name.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return " ".join(tokens)


def parse_form(form_text: Optional[str]) -> float:
    if not form_text:
        return 0.5
    points = 0.0
    count = 0
    for token in re.split(r"[-,\s]+", form_text.upper()):
        if token == "W":
            points += 3.0
            count += 1
        elif token == "D":
            points += 1.0
            count += 1
        elif token == "L":
            count += 1
    if count == 0:
        return 0.5
    return clamp(points / (3.0 * count), 0.0, 1.0)


def format_kickoff(iso_text: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
        tr = dt.astimezone(timezone(timedelta(hours=3)))
        return tr.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return iso_text


def market_to_bet_text(market_key: str, pick_text: str, home: str, away: str) -> Tuple[str, str]:
    low = pick_text.lower()
    if market_key == "h2h":
        if low in {"draw", "x"}:
            return "Mac Sonucu", "MSX"
        if low in {"home", "1"} or normalize_team_name(pick_text) == normalize_team_name(home):
            return "Mac Sonucu", "MS1"
        if low in {"away", "2"} or normalize_team_name(pick_text) == normalize_team_name(away):
            return "Mac Sonucu", "MS2"
        return "Mac Sonucu", f"MS ({pick_text})"

    if market_key == "totals":
        line_match = re.search(r"(\d+(?:\.\d+)?)", pick_text)
        line = line_match.group(1) if line_match else "2.5"
        if "over" in low:
            return "Alt/Ust", f"{line} Ust"
        if "under" in low:
            return "Alt/Ust", f"{line} Alt"
        return "Alt/Ust", pick_text

    if market_key == "btts":
        if "yes" in low:
            return "Karsilikli Gol", "KG Var"
        if "no" in low:
            return "Karsilikli Gol", "KG Yok"
        return "Karsilikli Gol", pick_text

    return MARKET_TR.get(market_key, market_key.upper()), pick_text


def poisson_pmf(lmbd: float, k: int) -> float:
    return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)


def match_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "1"
    if home_goals < away_goals:
        return "2"
    return "X"


def expected_goals(home_data: dict, away_data: dict) -> Tuple[float, float]:
    ppg_delta = home_data["ppg"] - away_data["ppg"]
    form_delta = home_data["form_score"] - away_data["form_score"]
    attack_delta = home_data["attack"] - away_data["defense"]
    away_attack_delta = away_data["attack"] - home_data["defense"]
    motivation_delta = home_data.get("motivation", 0.5) - away_data.get("motivation", 0.5)
    recent_form_delta = (home_data.get("recent_form_ppg", 1.2) - away_data.get("recent_form_ppg", 1.2)) / 2.8
    rest_delta = (home_data.get("days_rest", 5.0) - away_data.get("days_rest", 5.0)) / 8.0

    home_xg = (
        1.30
        + (0.24 * ppg_delta)
        + (0.18 * form_delta)
        + (0.27 * attack_delta)
        + (0.10 * motivation_delta)
        + (0.08 * recent_form_delta)
        + (0.05 * rest_delta)
        - (0.12 * home_data.get("injury_risk_proxy", 0.2))
        + (0.08 * away_data.get("injury_risk_proxy", 0.2))
    )
    away_xg = (
        1.02
        - (0.16 * ppg_delta)
        - (0.10 * form_delta)
        + (0.24 * away_attack_delta)
        - (0.07 * motivation_delta)
        - (0.06 * recent_form_delta)
        - (0.04 * rest_delta)
        - (0.10 * away_data.get("injury_risk_proxy", 0.2))
        + (0.08 * home_data.get("injury_risk_proxy", 0.2))
    )

    tempo = ((home_data["attack"] + away_data["attack"]) / 2.4) - ((home_data["defense"] + away_data["defense"]) / 3.0)
    tempo += 0.18 * (home_data.get("congestion_index", 0.0) + away_data.get("congestion_index", 0.0))
    home_xg += 0.12 * tempo
    away_xg += 0.10 * tempo

    return clamp(home_xg, 0.25, 3.8), clamp(away_xg, 0.20, 3.5)


def parse_h2h_pick_to_result(pick_text: str, home: str, away: str) -> Optional[str]:
    low = pick_text.lower()
    if low in {"draw", "x"}:
        return "X"
    if low in {"home", "1"} or normalize_team_name(pick_text) == normalize_team_name(home):
        return "1"
    if low in {"away", "2"} or normalize_team_name(pick_text) == normalize_team_name(away):
        return "2"
    return None


def pick_matches_score(market_key: str, pick_text: str, home: str, away: str, hg: int, ag: int) -> bool:
    low = pick_text.lower()
    total_goals = hg + ag
    if market_key == "totals":
        m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
        line = float(m.group(1)) if m else 2.5
        if "over" in low:
            return total_goals > line
        if "under" in low:
            return total_goals < line
        return True
    if market_key == "btts":
        if "yes" in low:
            return hg > 0 and ag > 0
        if "no" in low:
            return hg == 0 or ag == 0
        return True
    if market_key == "h2h":
        target = parse_h2h_pick_to_result(pick_text, home, away)
        if target is None:
            return True
        return match_outcome(hg, ag) == target
    return True


def build_match_scenarios(home_data: dict, away_data: dict) -> List[dict]:
    home_xg, away_xg = expected_goals(home_data, away_data)
    max_goals = 6

    h1_xg, a1_xg = home_xg * 0.46, away_xg * 0.46
    h2_xg, a2_xg = home_xg * 0.54, away_xg * 0.54
    h1_pmf = [poisson_pmf(h1_xg, k) for k in range(max_goals + 1)]
    a1_pmf = [poisson_pmf(a1_xg, k) for k in range(max_goals + 1)]
    h2_pmf = [poisson_pmf(h2_xg, k) for k in range(max_goals + 1)]
    a2_pmf = [poisson_pmf(a2_xg, k) for k in range(max_goals + 1)]

    scenarios: List[dict] = []
    for h1 in range(max_goals + 1):
        for a1 in range(max_goals + 1):
            p_first = h1_pmf[h1] * a1_pmf[a1]
            if p_first <= 0:
                continue
            iy = match_outcome(h1, a1)
            for h2 in range(max_goals + 1):
                for a2 in range(max_goals + 1):
                    p_second = h2_pmf[h2] * a2_pmf[a2]
                    if p_second <= 0:
                        continue
                    hg, ag = h1 + h2, a1 + a2
                    ms = match_outcome(h1 + h2, a1 + a2)
                    scenarios.append(
                        {
                            "hg": hg,
                            "ag": ag,
                            "score": f"{hg}-{ag}",
                            "iyms": f"{iy}/{ms}",
                            "p": p_first * p_second,
                        }
                    )
    return scenarios


def summarize_conditioned_predictions(
    scenarios: List[dict],
    market_key: str,
    pick_text: str,
    home: str,
    away: str,
) -> Tuple[str, str, str, str, float, float, float]:
    if not scenarios:
        return "1-1", "X/X", "1-1 %0.0", "X/X %0.0", 0.0, 0.0, 0.0

    total_prob = sum(s["p"] for s in scenarios)
    conditioned = [s for s in scenarios if pick_matches_score(market_key, pick_text, home, away, s["hg"], s["ag"])]

    consistency_prob = (sum(s["p"] for s in conditioned) / total_prob) if total_prob > 0 else 0.0
    active = conditioned if conditioned else scenarios
    active_total = sum(s["p"] for s in active)
    if active_total <= 0:
        active_total = 1.0

    score_probs: Dict[str, float] = {}
    iyms_probs: Dict[str, float] = {}
    for s in active:
        score_probs[s["score"]] = score_probs.get(s["score"], 0.0) + (s["p"] / active_total)
        iyms_probs[s["iyms"]] = iyms_probs.get(s["iyms"], 0.0) + (s["p"] / active_total)

    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_iyms = sorted(iyms_probs.items(), key=lambda x: x[1], reverse=True)

    score_pred = sorted_scores[0][0] if sorted_scores else "1-1"
    iyms_pred = sorted_iyms[0][0] if sorted_iyms else "X/X"
    score_pred_prob = sorted_scores[0][1] if sorted_scores else 0.0
    iyms_pred_prob = sorted_iyms[0][1] if sorted_iyms else 0.0
    score_top3 = " | ".join([f"{s} %{p*100:.1f}" for s, p in sorted_scores[:3]])
    iyms_top3 = " | ".join([f"{k} %{v*100:.1f}" for k, v in sorted_iyms[:3]])

    return score_pred, iyms_pred, score_top3, iyms_top3, consistency_prob, score_pred_prob, iyms_pred_prob


def safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 25) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    if response.status_code != 200:
        body = response.text[:250]
        raise requests.HTTPError(f"HTTP {response.status_code} - {body}", response=response)
    return response.json()


def parse_iso_dt(iso_text: str) -> Optional[datetime]:
    if not iso_text:
        return None
    try:
        return datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_odds_events(odds_api_key: str, region: str, sport_keys: List[str], days: int) -> List[dict]:
    events: List[dict] = []
    cutoff = datetime.now(timezone.utc) + timedelta(days=days)
    for sport_key in sport_keys:
        active_markets = list(ODDS_MARKETS)
        data = []
        while active_markets:
            try:
                data = safe_get(
                    f"{BASE_ODDS}/sports/{sport_key}/odds",
                    params={
                        "apiKey": odds_api_key,
                        "regions": region,
                        "markets": ",".join(active_markets),
                        "oddsFormat": "decimal",
                        "dateFormat": "iso",
                    },
                )
                break
            except requests.HTTPError as exc:
                error_text = str(exc).lower()
                if "invalid_market" not in error_text:
                    raise
                removed = None
                for market in list(active_markets):
                    if market in error_text:
                        removed = market
                        break
                if removed is None:
                    removed = active_markets[-1]
                active_markets.remove(removed)
                if not active_markets:
                    raise
                # Retry with reduced market set when endpoint doesn't support one market.
                time.sleep(0.15)
        for event in data:
            kick = event.get("commence_time")
            if not kick:
                continue
            try:
                dt = datetime.fromisoformat(kick.replace("Z", "+00:00"))
                if dt <= cutoff:
                    events.append(event)
            except Exception:
                events.append(event)
        time.sleep(0.2)
    return events


@st.cache_data(ttl=1800)
def fetch_standings(football_data_key: str, competition_codes: List[str]) -> Dict[str, dict]:
    headers = {"X-Auth-Token": football_data_key}
    result: Dict[str, dict] = {}
    for code in competition_codes:
        data = safe_get(f"{BASE_FOOTBALL}/competitions/{code}/standings", headers=headers)
        table = []
        standings = data.get("standings", [])
        if standings:
            table = standings[0].get("table", [])
        league_data = {}
        points_by_pos: Dict[int, float] = {}
        for row in table:
            points_by_pos[int(row.get("position", 99))] = float(row.get("points", 0))
        league_size = max(len(table), 20)
        top_points = points_by_pos.get(1, 0.0)
        relegation_pos = max(league_size - 2, 1)
        relegation_line_points = points_by_pos.get(relegation_pos, 0.0)
        for row in table:
            team = row.get("team", {})
            raw_name = team.get("name", "")
            team_id = int(team.get("id", 0))
            key = normalize_team_name(raw_name)
            won = float(row.get("won", 0))
            draw = float(row.get("draw", 0))
            lost = float(row.get("lost", 0))
            played = max(won + draw + lost, 1.0)
            gf = float(row.get("goalsFor", 0))
            ga = float(row.get("goalsAgainst", 0))
            gd = gf - ga
            ppg = float(row.get("points", 0)) / played
            attack = gf / played
            defense = ga / played
            form_score = parse_form(row.get("form"))
            points = float(row.get("points", 0))
            position = float(row.get("position", 20))
            points_gap_top = top_points - points
            points_gap_relegation = points - relegation_line_points
            title_race = 1.0 if (position <= 3 and points_gap_top <= 7) else 0.0
            relegation_battle = 1.0 if (position >= relegation_pos and points_gap_relegation <= 6) else 0.0
            europe_race = 1.0 if (position <= 8 and points_gap_top <= 12) else 0.0
            motivation = clamp(0.45 + (0.35 * title_race) + (0.25 * relegation_battle) + (0.15 * europe_race), 0.25, 1.0)
            league_data[key] = {
                "raw_name": raw_name,
                "team_id": team_id,
                "position": position,
                "played": played,
                "points": points,
                "ppg": ppg,
                "attack": attack,
                "defense": defense,
                "goal_diff_per_game": gd / played,
                "form_score": form_score,
                "motivation": motivation,
                "title_race": title_race,
                "relegation_battle": relegation_battle,
                "europe_race": europe_race,
                "league_size": float(league_size),
                "top_points": top_points,
                "relegation_line_points": relegation_line_points,
            }
        result[code] = league_data
        time.sleep(0.2)
    return result


def build_team_id_map_for_events(events: List[dict], standings_by_comp: Dict[str, dict]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for event in events:
        sport_key = event.get("sport_key")
        comp = LEAGUES.get(sport_key, {}).get("fd_comp")
        league_data = standings_by_comp.get(comp, {}) if comp else {}
        for team_name in [event.get("home_team"), event.get("away_team")]:
            if not team_name:
                continue
            key = normalize_team_name(team_name)
            if key in league_data and int(league_data[key].get("team_id", 0)) > 0:
                mapping[key] = int(league_data[key]["team_id"])
    return mapping


@st.cache_data(ttl=1800)
def fetch_team_contexts(football_data_key: str, team_ids: Tuple[int, ...]) -> Dict[int, dict]:
    headers = {"X-Auth-Token": football_data_key}
    contexts: Dict[int, dict] = {}
    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=30)).date().isoformat()
    date_to = (now + timedelta(days=8)).date().isoformat()

    for team_id in team_ids:
        try:
            data = safe_get(
                f"{BASE_FOOTBALL}/teams/{team_id}/matches",
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to},
                timeout=25,
            )
            matches = data.get("matches", []) or []
        except Exception:
            contexts[team_id] = {
                "recent_form_ppg": 1.2,
                "recent_goal_diff_pg": 0.0,
                "days_rest": 5.0,
                "congestion_index": 0.0,
                "next_match_days": 7.0,
                "injury_risk_proxy": 0.22,
            }
            continue

        finished = [m for m in matches if m.get("status") == "FINISHED"]
        upcoming = [m for m in matches if m.get("status") in {"SCHEDULED", "TIMED"}]
        finished.sort(key=lambda m: m.get("utcDate", ""))
        upcoming.sort(key=lambda m: m.get("utcDate", ""))

        recent = finished[-5:]
        pts = 0.0
        gd = 0.0
        recent_count = 0
        for m in recent:
            home_id = int((m.get("homeTeam") or {}).get("id", 0))
            away_id = int((m.get("awayTeam") or {}).get("id", 0))
            score = (m.get("score") or {}).get("fullTime") or {}
            hg = int(score.get("home", 0) or 0)
            ag = int(score.get("away", 0) or 0)
            if team_id == home_id:
                team_goals, opp_goals = hg, ag
            else:
                team_goals, opp_goals = ag, hg
            recent_count += 1
            gd += (team_goals - opp_goals)
            if team_goals > opp_goals:
                pts += 3.0
            elif team_goals == opp_goals:
                pts += 1.0

        recent_form_ppg = (pts / max(recent_count, 1))
        recent_goal_diff_pg = gd / max(recent_count, 1)

        last_finished_dt = parse_iso_dt(finished[-1].get("utcDate")) if finished else None
        next_match_dt = parse_iso_dt(upcoming[0].get("utcDate")) if upcoming else None
        days_rest = (now - last_finished_dt).total_seconds() / 86400.0 if last_finished_dt else 5.0
        next_match_days = (next_match_dt - now).total_seconds() / 86400.0 if next_match_dt else 7.0

        recent_10_days = 0
        ten_days_ago = now - timedelta(days=10)
        for m in finished:
            dt = parse_iso_dt(m.get("utcDate"))
            if dt and dt >= ten_days_ago:
                recent_10_days += 1
        congestion_index = clamp((recent_10_days - 2) / 4.0, 0.0, 1.0)
        injury_risk_proxy = clamp((0.26 * max(0.0, 3.0 - days_rest)) + (0.44 * congestion_index), 0.05, 0.95)

        contexts[team_id] = {
            "recent_form_ppg": recent_form_ppg,
            "recent_goal_diff_pg": recent_goal_diff_pg,
            "days_rest": clamp(days_rest, 0.0, 12.0),
            "congestion_index": congestion_index,
            "next_match_days": clamp(next_match_days, 0.0, 12.0),
            "injury_risk_proxy": injury_risk_proxy,
        }
        time.sleep(0.12)

    return contexts


def map_team_strength(
    sport_key: str,
    home: str,
    away: str,
    standings_by_comp: Dict[str, dict],
    team_contexts: Optional[Dict[int, dict]] = None,
) -> Tuple[dict, dict]:
    comp = LEAGUES.get(sport_key, {}).get("fd_comp")
    league_data = standings_by_comp.get(comp, {}) if comp else {}
    home_key = normalize_team_name(home)
    away_key = normalize_team_name(away)
    default_team = {
        "raw_name": "",
        "team_id": 0,
        "position": 12.0,
        "played": 10.0,
        "points": 14.0,
        "ppg": 1.2,
        "attack": 1.2,
        "defense": 1.2,
        "goal_diff_per_game": 0.0,
        "form_score": 0.5,
        "motivation": 0.55,
        "title_race": 0.0,
        "relegation_battle": 0.0,
        "europe_race": 0.0,
        "league_size": 20.0,
        "top_points": 60.0,
        "relegation_line_points": 20.0,
        "recent_form_ppg": 1.2,
        "recent_goal_diff_pg": 0.0,
        "days_rest": 5.0,
        "congestion_index": 0.0,
        "next_match_days": 7.0,
        "injury_risk_proxy": 0.2,
    }
    home_strength = dict(league_data.get(home_key, default_team))
    away_strength = dict(league_data.get(away_key, default_team))

    if team_contexts:
        home_strength.update(team_contexts.get(int(home_strength.get("team_id", 0)), {}))
        away_strength.update(team_contexts.get(int(away_strength.get("team_id", 0)), {}))
    return home_strength, away_strength


def adjust_probabilities(
    market_key: str,
    outcome_name: str,
    base_prob: float,
    home_name: str,
    away_name: str,
    home_team_data: dict,
    away_team_data: dict,
) -> float:
    base_prob = clamp(base_prob, 0.02, 0.98)
    ppg_delta = home_team_data["ppg"] - away_team_data["ppg"]
    form_delta = home_team_data["form_score"] - away_team_data["form_score"]
    goal_delta = home_team_data["goal_diff_per_game"] - away_team_data["goal_diff_per_game"]
    motivation_delta = home_team_data.get("motivation", 0.5) - away_team_data.get("motivation", 0.5)
    recent_form_delta = (home_team_data.get("recent_form_ppg", 1.2) - away_team_data.get("recent_form_ppg", 1.2)) / 3.0
    rest_delta = (home_team_data.get("days_rest", 5.0) - away_team_data.get("days_rest", 5.0)) / 7.0
    injury_delta = away_team_data.get("injury_risk_proxy", 0.2) - home_team_data.get("injury_risk_proxy", 0.2)
    strength_delta = (
        (0.33 * ppg_delta)
        + (0.24 * form_delta)
        + (0.17 * goal_delta)
        + (0.12 * motivation_delta)
        + (0.08 * recent_form_delta)
        + (0.06 * rest_delta)
        + (0.05 * injury_delta)
    )

    if market_key == "h2h":
        out = outcome_name.lower()
        if out == "draw":
            draw_shift = -0.22 * abs(strength_delta)
            return clamp(sigmoid(logit(base_prob) + draw_shift), 0.02, 0.7)
        if out == "home" or out == "1":
            shift = 0.70 * strength_delta
            return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)
        if out == "away" or out == "2":
            shift = -0.70 * strength_delta
            return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)

        # Names can match actual team names in many feeds.
        if normalize_team_name(outcome_name) == normalize_team_name(home_name):
            shift = 0.70 * strength_delta
            return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)
        if normalize_team_name(outcome_name) == normalize_team_name(away_name):
            shift = -0.70 * strength_delta
            return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)

    if market_key == "totals":
        attack_total = home_team_data["attack"] + away_team_data["attack"]
        defense_total = home_team_data["defense"] + away_team_data["defense"]
        pace = (attack_total - defense_total) * 0.45
        is_over = "over" in outcome_name.lower()
        shift = pace if is_over else -pace
        return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)

    if market_key == "btts":
        scoring_tendency = (home_team_data["attack"] * away_team_data["attack"]) / 2.2
        conceding_tendency = (home_team_data["defense"] + away_team_data["defense"]) / 2.4
        btts_bias = (scoring_tendency - conceding_tendency) * 0.35
        yes_pick = "yes" in outcome_name.lower()
        shift = btts_bias if yes_pick else -btts_bias
        return clamp(sigmoid(logit(base_prob) + shift), 0.02, 0.98)

    return base_prob


def extract_candidates(
    events: List[dict],
    standings_by_comp: Dict[str, dict],
    team_contexts: Optional[Dict[int, dict]] = None,
    injury_adjustments: Optional[Dict[str, float]] = None,
) -> List[dict]:
    candidates: List[dict] = []
    for event in events:
        home = event.get("home_team")
        away = event.get("away_team")
        commence = event.get("commence_time")
        sport_key = event.get("sport_key")
        bookmakers = event.get("bookmakers", []) or []
        if not home or not away or not commence:
            continue
        if not bookmakers:
            continue

        home_data, away_data = map_team_strength(sport_key, home, away, standings_by_comp, team_contexts)
        if injury_adjustments:
            h_key = normalize_team_name(home)
            a_key = normalize_team_name(away)
            h_adj = injury_adjustments.get(h_key, 0.0)
            a_adj = injury_adjustments.get(a_key, 0.0)
            home_data["injury_risk_proxy"] = clamp(home_data.get("injury_risk_proxy", 0.2) + h_adj, 0.02, 0.99)
            away_data["injury_risk_proxy"] = clamp(away_data.get("injury_risk_proxy", 0.2) + a_adj, 0.02, 0.99)
            home_data["motivation"] = clamp(home_data.get("motivation", 0.55) - (0.08 * h_adj), 0.2, 1.0)
            away_data["motivation"] = clamp(away_data.get("motivation", 0.55) - (0.08 * a_adj), 0.2, 1.0)
        match_scenarios = build_match_scenarios(home_data, away_data)

        market_map: Dict[Tuple[str, Optional[float]], dict] = {}
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []) or []:
                market_key = market.get("key")
                if market_key not in ODDS_MARKETS:
                    continue
                point = market.get("point")
                key = (market_key, point)
                bucket = market_map.setdefault(
                    key,
                    {"prices": {}, "best_prices": {}, "book_count": {}},
                )
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name")
                    price = outcome.get("price")
                    if not name or not price:
                        continue
                    price = float(price)
                    bucket["prices"].setdefault(name, []).append(price)
                    bucket["best_prices"][name] = max(bucket["best_prices"].get(name, 0.0), price)
                    bucket["book_count"][name] = bucket["book_count"].get(name, 0) + 1

        for (market_key, point), bucket in market_map.items():
            outcomes = list(bucket["prices"].keys())
            if len(outcomes) < 2:
                continue
            avg_probs = []
            for outcome_name in outcomes:
                avg_price = float(np.mean(bucket["prices"][outcome_name]))
                avg_probs.append(implied_prob(avg_price))
            base_probs = normalize_probs(avg_probs)

            adjusted_raw = []
            for idx, outcome_name in enumerate(outcomes):
                adjusted_raw.append(
                    adjust_probabilities(
                        market_key=market_key,
                        outcome_name=outcome_name,
                        base_prob=base_probs[idx],
                        home_name=home,
                        away_name=away,
                        home_team_data=home_data,
                        away_team_data=away_data,
                    )
                )
            adjusted_probs = normalize_probs(adjusted_raw)

            for idx, outcome_name in enumerate(outcomes):
                odd = bucket["best_prices"][outcome_name]
                if odd <= 1.01:
                    continue
                outcome_label = outcome_name
                if market_key == "totals" and point is not None:
                    outcome_label = f"{outcome_name} {point}"
                bet_type, bet_text = market_to_bet_text(market_key, outcome_label, home, away)
                score_pred, iyms_pred, score_top3, iyms_top3, consistency_prob, score_pred_prob, iyms_pred_prob = summarize_conditioned_predictions(
                    scenarios=match_scenarios,
                    market_key=market_key,
                    pick_text=outcome_label,
                    home=home,
                    away=away,
                )
                market_prob = implied_prob(odd)
                model_prob = float(adjusted_probs[idx])
                edge = model_prob - market_prob
                books = int(bucket["book_count"][outcome_name])
                roi = (model_prob * odd) - 1.0
                confidence = clamp(
                    (0.55 * model_prob)
                    + (2.8 * max(edge, 0.0))
                    + (0.4 * max(roi, 0.0))
                    + (0.03 * math.log(max(books, 1) + 1.0)),
                    0.0,
                    1.0,
                )
                risk = clamp(1.0 - confidence + (0.45 / max(odd, 1.2)), 0.0, 1.0)
                value_score = (
                    (edge * math.log(max(odd, 1.1)) * math.sqrt(max(books, 1)))
                    + (0.35 * roi)
                    + (0.30 * consistency_prob)
                    - (0.2 * risk)
                )
                candidates.append(
                    {
                        "match_id": f"{home}__{away}__{commence}",
                        "sport_key": sport_key,
                        "league": LEAGUES.get(sport_key, {}).get("label", sport_key),
                        "home": home,
                        "away": away,
                        "commence": commence,
                        "market_key": market_key,
                        "market": market_key.upper(),
                        "pick": outcome_label,
                        "bet_type": bet_type,
                        "bet_text": bet_text,
                        "odd": float(odd),
                        "model_prob": model_prob,
                        "market_prob": market_prob,
                        "edge": edge,
                        "roi": roi,
                        "confidence": confidence,
                        "risk": risk,
                        "consistency_prob": consistency_prob,
                        "motivation_home": home_data.get("motivation", 0.5),
                        "motivation_away": away_data.get("motivation", 0.5),
                        "injury_risk_home": home_data.get("injury_risk_proxy", 0.2),
                        "injury_risk_away": away_data.get("injury_risk_proxy", 0.2),
                        "days_rest_home": home_data.get("days_rest", 5.0),
                        "days_rest_away": away_data.get("days_rest", 5.0),
                        "bookmakers_count": books,
                        "value_score": value_score,
                        "score_pred": score_pred,
                        "score_pred_prob": score_pred_prob,
                        "iyms_pred": iyms_pred,
                        "iyms_pred_prob": iyms_pred_prob,
                        "score_top3": score_top3,
                        "iyms_top3": iyms_top3,
                    }
                )
    candidates.sort(key=lambda c: (c["value_score"], c["edge"], c["odd"]), reverse=True)
    return candidates


def candidate_pool(
    candidates: List[dict],
    profile: CouponProfile,
    open_mode: bool = False,
    quality_level: str = "Yuksek",
) -> List[dict]:
    if open_mode:
        pool = [c for c in candidates if c["odd"] > 1.01 and c["bookmakers_count"] >= 1]
        pool = sorted(pool, key=lambda c: (c["value_score"], c["roi"], c["edge"]), reverse=True)
        return pool[:650]

    consistency_min = 0.40
    if quality_level == "Maksimum":
        consistency_min = 0.48
    elif quality_level == "Dengeli":
        consistency_min = 0.32

    filtered = [
        c
        for c in candidates
        if profile.odd_min <= c["odd"] <= profile.odd_max
        and c["edge"] >= profile.min_edge
        and c["model_prob"] >= profile.min_prob
        and c["consistency_prob"] >= consistency_min
        and c["bookmakers_count"] >= profile.min_books
    ]
    return filtered[:260]


def relaxed_candidate_pool(
    candidates: List[dict],
    profile: CouponProfile,
    open_mode: bool = False,
) -> List[dict]:
    if open_mode:
        relaxed = [c for c in candidates if c["odd"] > 1.01]
        relaxed = sorted(relaxed, key=lambda c: (c["roi"], c["value_score"], c["consistency_prob"]), reverse=True)
        return relaxed[:700]
    relaxed = [
        c
        for c in candidates
        if (profile.odd_min * 0.85) <= c["odd"] <= (profile.odd_max * 1.10)
        and c["edge"] >= max(profile.min_edge * 0.55, 0.008)
        and c["model_prob"] >= max(profile.min_prob * 0.70, 0.04)
        and c["consistency_prob"] >= 0.22
        and c["bookmakers_count"] >= 1
    ]
    return relaxed[:320]


def coupon_metrics(picks: List[dict], total_odd: float, simulations: int) -> Dict[str, float]:
    if not picks:
        return {"hit_prob": 0.0, "sim_hit": 0.0, "ev_mult": 0.0, "risk_index": 1.0, "roi_pct": -100.0}
    hit_prob = float(np.prod([p["model_prob"] for p in picks]))
    ev_mult = hit_prob * total_odd
    roi_pct = (ev_mult - 1.0) * 100.0
    risk_index = clamp(float(np.mean([p["risk"] for p in picks])), 0.0, 1.0)
    rng = random.Random(44)
    hits = 0
    corr_penalty = 0.97 if len(picks) <= 8 else 0.94
    for _ in range(simulations):
        alive = True
        for p in picks:
            adjusted_prob = clamp(p["model_prob"] * corr_penalty, 0.001, 0.999)
            if rng.random() > adjusted_prob:
                alive = False
                break
        if alive:
            hits += 1
    sim_hit = hits / simulations
    return {"hit_prob": hit_prob, "sim_hit": sim_hit, "ev_mult": ev_mult, "risk_index": risk_index, "roi_pct": roi_pct}


def quality_score(total_odd: float, picks: List[dict], profile: CouponProfile) -> float:
    if not picks:
        return -1e9
    target_mid = math.sqrt(profile.target_min * profile.target_max)
    dist = abs(math.log(max(total_odd, 1.001)) - math.log(target_mid))
    avg_value = float(np.mean([p["value_score"] for p in picks]))
    avg_conf = float(np.mean([p["confidence"] for p in picks]))
    legs_bonus = 0.06 * len(picks)
    range_bonus = 2.2 if profile.target_min <= total_odd <= profile.target_max else 0.0
    return (2.0 * avg_value) + (1.2 * avg_conf) + legs_bonus + range_bonus - (1.05 * dist)


def tune_coupon_to_target(
    picks: List[dict],
    total_odd: float,
    pool: List[dict],
    profile: CouponProfile,
) -> Tuple[List[dict], float]:
    if not picks:
        return picks, total_odd

    tuned = list(picks)
    used = {p["match_id"] for p in tuned}

    if total_odd < profile.target_min:
        boosters = [
            c
            for c in sorted(pool, key=lambda x: (x["odd"] * (1.0 + x["value_score"])), reverse=True)
            if c["match_id"] not in used
        ]
        for b in boosters:
            if len(tuned) >= profile.max_legs:
                break
            tuned.append(b)
            used.add(b["match_id"])
            total_odd *= b["odd"]
            if total_odd >= profile.target_min:
                break

    if total_odd > profile.target_max:
        tuned.sort(key=lambda x: x["odd"])
        idx = 0
        while total_odd > profile.target_max and len(tuned) > profile.min_legs and idx < len(tuned):
            drop = tuned[idx]
            total_odd /= max(drop["odd"], 1.01)
            used.discard(drop["match_id"])
            tuned.pop(idx)

    if profile.key == "golden" and total_odd < profile.target_min:
        replacements = sorted(pool, key=lambda x: x["odd"], reverse=True)
        for i, low in enumerate(sorted(tuned, key=lambda x: x["odd"])):
            if total_odd >= profile.target_min:
                break
            for rep in replacements:
                if rep["match_id"] in used:
                    continue
                if rep["odd"] <= low["odd"] * 1.30:
                    continue
                # Swap a low odd pick with a much higher odd alternative.
                tuned.remove(low)
                used.discard(low["match_id"])
                total_odd /= max(low["odd"], 1.01)
                tuned.append(rep)
                used.add(rep["match_id"])
                total_odd *= rep["odd"]
                break

    tuned = sorted(tuned, key=lambda x: (x["commence"], -x["value_score"]))
    return tuned, total_odd


def generate_coupon(
    candidates: List[dict],
    profile: CouponProfile,
    seed: int,
    open_mode: bool = False,
    quality_level: str = "Yuksek",
    depth_level: str = "Standart",
) -> Tuple[List[dict], float]:
    pool = candidate_pool(candidates, profile, open_mode=open_mode, quality_level=quality_level)
    if not pool:
        pool = relaxed_candidate_pool(candidates, profile, open_mode=open_mode)
    if not pool:
        return [], 1.0

    rng = random.Random(seed)
    best_picks: List[dict] = []
    best_odd = 1.0
    best_quality = -1e9

    base_iter = 2200 if profile.key == "golden" else 1400
    if depth_level == "Derin":
        iterations = int(base_iter * 1.5)
    elif depth_level == "Maksimum":
        iterations = int(base_iter * 2.1)
    else:
        iterations = base_iter
    target_mid = math.sqrt(profile.target_min * profile.target_max)

    for _ in range(iterations):
        leg_target = rng.randint(profile.min_legs, profile.max_legs)
        used_matches = set()
        picks: List[dict] = []
        total_odd = 1.0

        ranked = sorted(pool, key=lambda x: x["value_score"], reverse=True)
        sample_size = min(len(ranked), 110 + profile.max_legs * 5) if open_mode else min(len(ranked), 70 + profile.max_legs * 3)
        subset = ranked[:sample_size]
        rng.shuffle(subset)
        subset.sort(key=lambda x: x["value_score"] + (0.2 * rng.random()), reverse=True)

        for cand in subset:
            if len(picks) >= leg_target:
                break
            if cand["match_id"] in used_matches:
                continue
            projected = total_odd * cand["odd"]
            remaining = max(leg_target - len(picks) - 1, 0)
            if remaining > 0:
                max_possible = projected * (profile.odd_max ** remaining)
                if max_possible < profile.target_min * 0.55:
                    continue
            picks.append(cand)
            used_matches.add(cand["match_id"])
            total_odd = projected
            if total_odd > profile.target_max * 2.5:
                break

        if len(picks) < profile.min_legs:
            continue

        if total_odd < profile.target_min:
            boosters = [c for c in pool if c["match_id"] not in used_matches and c["odd"] >= 2.7]
            boosters = sorted(boosters, key=lambda x: x["value_score"], reverse=True)
            for b in boosters:
                if len(picks) >= profile.max_legs:
                    break
                picks.append(b)
                used_matches.add(b["match_id"])
                total_odd *= b["odd"]
                if total_odd >= profile.target_min:
                    break

        q = quality_score(total_odd, picks, profile)
        if total_odd > 0:
            proximity = abs(math.log(total_odd) - math.log(target_mid))
            q -= 0.2 * proximity
        if q > best_quality:
            best_quality = q
            best_picks = picks
            best_odd = total_odd

    best_picks, best_odd = tune_coupon_to_target(best_picks, best_odd, pool, profile)
    return best_picks, best_odd


def ai_leg_comment(pick: dict) -> str:
    conf = pick["confidence"] * 100.0
    edge = pick["edge"] * 100.0
    roi = pick["roi"] * 100.0
    risk = pick["risk"] * 100.0
    consistency = pick["consistency_prob"] * 100.0
    if conf >= 62:
        trust = "guven yuksek"
    elif conf >= 45:
        trust = "guven orta"
    else:
        trust = "agresif secim"
    return f"{trust}; edge %{edge:.2f}; roi %{roi:.2f}; tutarlilik %{consistency:.1f}; risk %{risk:.1f}; kaynak {pick['bookmakers_count']} bookmaker"


def ai_coupon_summary(profile: CouponProfile, picks: List[dict], total_odd: float, metrics: Dict[str, float]) -> str:
    if not picks:
        return f"{profile.title}: Bu profil icin uygun kombinasyon olusmadi."
    in_range = profile.target_min <= total_odd <= profile.target_max
    range_text = "hedef aralikta" if in_range else "hedef aralik disinda"
    return (
        f"{profile.title}: {len(picks)} mac, toplam oran {total_odd:,.2f} ({range_text}). "
        f"Model tutma %{metrics['hit_prob']*100:.5f}, Monte Carlo %{metrics['sim_hit']*100:.5f}, EV {metrics['ev_mult']:.4f}, ROI %{metrics['roi_pct']:.2f}."
    )


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coupon_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            profile_title TEXT NOT NULL,
            total_odd REAL NOT NULL,
            hit_prob REAL NOT NULL,
            sim_hit REAL NOT NULL,
            ev_mult REAL NOT NULL,
            risk_index REAL NOT NULL,
            legs_count INTEGER NOT NULL,
            picks_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bet_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            league TEXT NOT NULL,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            kickoff TEXT NOT NULL,
            market_key TEXT NOT NULL,
            bet_text TEXT NOT NULL,
            odd_open REAL NOT NULL,
            model_prob REAL NOT NULL,
            settled INTEGER NOT NULL DEFAULT 0,
            won INTEGER,
            home_goals INTEGER,
            away_goals INTEGER,
            odd_close REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            kickoff TEXT NOT NULL,
            market_key TEXT NOT NULL,
            bet_text TEXT NOT NULL,
            odd REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            report_path TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_coupon(db_path: Path, profile: CouponProfile, picks: List[dict], total_odd: float, metrics: Dict[str, float]) -> None:
    if not picks:
        return
    payload = [
        {
            "league": p["league"],
            "home": p["home"],
            "away": p["away"],
            "market": p["market"],
            "pick": p["pick"],
            "odd": p["odd"],
            "edge": p["edge"],
            "model_prob": p["model_prob"],
            "kickoff": p["commence"],
        }
        for p in picks
    ]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    created_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO coupon_history (
            created_at, profile_key, profile_title, total_odd,
            hit_prob, sim_hit, ev_mult, risk_index, legs_count, picks_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            profile.key,
            profile.title,
            float(total_odd),
            float(metrics["hit_prob"]),
            float(metrics["sim_hit"]),
            float(metrics["ev_mult"]),
            float(metrics["risk_index"]),
            int(len(picks)),
            json.dumps(payload, ensure_ascii=True),
        ),
    )
    for p in picks:
        conn.execute(
            """
            INSERT INTO bet_legs (
                run_id, created_at, profile_key, league, home, away, kickoff,
                market_key, bet_text, odd_open, model_prob, settled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run_id,
                created_at,
                profile.key,
                p["league"],
                p["home"],
                p["away"],
                p["commence"],
                p["market_key"],
                p["pick"],
                float(p["odd"]),
                float(p["model_prob"]),
            ),
        )
        conn.execute(
            """
            INSERT INTO odds_snapshots (
                run_id, captured_at, home, away, kickoff, market_key, bet_text, odd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                p["home"],
                p["away"],
                p["commence"],
                p["market_key"],
                p["pick"],
                float(p["odd"]),
            ),
        )
    conn.commit()
    conn.close()


def load_recent_history(db_path: Path, limit: int = 25) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT id, created_at, profile_title, total_odd, hit_prob, sim_hit, ev_mult, risk_index, legs_count
        FROM coupon_history
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )
    conn.close()
    return df


def parse_pick_line(pick_text: str, default_line: float = 2.5) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
    return float(m.group(1)) if m else default_line


def leg_won(market_key: str, pick_text: str, home: str, away: str, hg: int, ag: int) -> bool:
    low = pick_text.lower()
    if market_key == "totals":
        line = parse_pick_line(pick_text)
        total = hg + ag
        if "over" in low:
            return total > line
        if "under" in low:
            return total < line
        return False
    if market_key == "btts":
        if "yes" in low:
            return hg > 0 and ag > 0
        if "no" in low:
            return hg == 0 or ag == 0
        return False
    if market_key == "h2h":
        target = parse_h2h_pick_to_result(pick_text, home, away)
        if target is None:
            return False
        return match_outcome(hg, ag) == target
    return False


def fetch_finished_matches_by_competitions(football_data_key: str, comp_codes: List[str], days_back: int = 7) -> Dict[str, tuple]:
    headers = {"X-Auth-Token": football_data_key}
    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=days_back)).date().isoformat()
    date_to = now.date().isoformat()
    out: Dict[str, tuple] = {}
    for code in comp_codes:
        try:
            data = safe_get(
                f"{BASE_FOOTBALL}/competitions/{code}/matches",
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
            )
        except Exception:
            continue
        for m in data.get("matches", []) or []:
            ht = ((m.get("homeTeam") or {}).get("name") or "").strip()
            at = ((m.get("awayTeam") or {}).get("name") or "").strip()
            ft = (m.get("score") or {}).get("fullTime") or {}
            hg = int(ft.get("home", 0) or 0)
            ag = int(ft.get("away", 0) or 0)
            key = f"{normalize_team_name(ht)}__{normalize_team_name(at)}"
            out[key] = (hg, ag)
    return out


def lookup_closing_odd(conn: sqlite3.Connection, home: str, away: str, kickoff: str, market_key: str, pick_text: str) -> Optional[float]:
    row = conn.execute(
        """
        SELECT odd FROM odds_snapshots
        WHERE home = ? AND away = ? AND kickoff = ? AND market_key = ? AND bet_text = ?
        ORDER BY id DESC LIMIT 1
        """,
        (home, away, kickoff, market_key, pick_text),
    ).fetchone()
    return float(row[0]) if row else None


def settle_open_legs(
    db_path: Path,
    football_data_key: str,
    leagues: List[str],
) -> int:
    if not football_data_key:
        return 0
    comp_codes = sorted({LEAGUES[k]["fd_comp"] for k in leagues if k in LEAGUES})
    finished_map = fetch_finished_matches_by_competitions(football_data_key, comp_codes, days_back=10)
    if not finished_map:
        return 0
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT id, home, away, kickoff, market_key, bet_text
        FROM bet_legs
        WHERE settled = 0
        """
    ).fetchall()
    settled_count = 0
    for rid, home, away, kickoff, market_key, bet_text in rows:
        key = f"{normalize_team_name(home)}__{normalize_team_name(away)}"
        if key not in finished_map:
            continue
        hg, ag = finished_map[key]
        won = 1 if leg_won(market_key, bet_text, home, away, hg, ag) else 0
        odd_close = lookup_closing_odd(conn, home, away, kickoff, market_key, bet_text)
        conn.execute(
            """
            UPDATE bet_legs
            SET settled = 1, won = ?, home_goals = ?, away_goals = ?, odd_close = ?
            WHERE id = ?
            """,
            (won, int(hg), int(ag), odd_close, int(rid)),
        )
        settled_count += 1
    conn.commit()
    conn.close()
    return settled_count


def load_settled_performance(db_path: Path) -> Dict[str, float]:
    if not db_path.exists():
        return {"n": 0.0, "hit_rate": 0.0, "avg_roi": 0.0, "avg_clv": 0.0}
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(CASE WHEN won = 1 THEN 1.0 ELSE 0.0 END) AS hit_rate,
            AVG(CASE WHEN won = 1 THEN odd_open - 1.0 ELSE -1.0 END) AS avg_roi,
            AVG(CASE WHEN odd_close IS NOT NULL THEN (1.0/odd_open) - (1.0/odd_close) END) AS avg_clv
        FROM bet_legs
        WHERE settled = 1
        """
    ).fetchone()
    conn.close()
    n, hit_rate, avg_roi, avg_clv = row if row else (0, 0, 0, 0)
    return {
        "n": float(n or 0),
        "hit_rate": float(hit_rate or 0.0),
        "avg_roi": float(avg_roi or 0.0),
        "avg_clv": float(avg_clv or 0.0),
    }


def load_calibration_bins(db_path: Path, bins: int = 10) -> List[Tuple[float, float, float]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT model_prob, won
        FROM bet_legs
        WHERE settled = 1 AND model_prob > 0
        """
    ).fetchall()
    conn.close()
    if len(rows) < 60:
        return []
    arr = np.array(rows, dtype=float)
    probs = arr[:, 0]
    outcomes = arr[:, 1]
    bins_edges = np.linspace(0.0, 1.0, bins + 1)
    out: List[Tuple[float, float, float]] = []
    for i in range(bins):
        lo, hi = bins_edges[i], bins_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if mask.sum() < 3:
            continue
        pred_mean = float(probs[mask].mean())
        real_mean = float(outcomes[mask].mean())
        out.append((lo, pred_mean, real_mean))
    return out


def calibrate_prob(prob: float, bins_data: List[Tuple[float, float, float]]) -> float:
    if not bins_data:
        return prob
    best = min(bins_data, key=lambda x: abs(prob - x[1]))
    raw = best[2]
    # blend to avoid overfitting
    return clamp((0.6 * prob) + (0.4 * raw), 0.01, 0.99)


def apply_calibration_to_candidates(candidates: List[dict], bins_data: List[Tuple[float, float, float]]) -> None:
    if not bins_data:
        return
    for c in candidates:
        c["model_prob"] = calibrate_prob(float(c["model_prob"]), bins_data)
        c["edge"] = c["model_prob"] - c["market_prob"]
        c["roi"] = (c["model_prob"] * c["odd"]) - 1.0
        c["confidence"] = clamp(
            (0.55 * c["model_prob"])
            + (2.8 * max(c["edge"], 0.0))
            + (0.4 * max(c["roi"], 0.0))
            + (0.03 * math.log(max(c["bookmakers_count"], 1) + 1.0)),
            0.0,
            1.0,
        )
        c["risk"] = clamp(1.0 - c["confidence"] + (0.45 / max(c["odd"], 1.2)), 0.0, 1.0)
        c["value_score"] = (
            (c["edge"] * math.log(max(c["odd"], 1.1)) * math.sqrt(max(c["bookmakers_count"], 1)))
            + (0.35 * c["roi"])
            + (0.30 * c["consistency_prob"])
            - (0.2 * c["risk"])
        )


def render_coupon_panel(profile: CouponProfile, picks: List[dict], total_odd: float, metrics: Dict[str, float]) -> None:
    st.subheader(f"{profile.title}")
    st.caption(f"Hedef Oran: {profile.target_min:,.0f} - {profile.target_max:,.0f}")
    if not picks:
        st.warning("Bu profil icin kupon uretilemedi.")
        return

    in_range = profile.target_min <= total_odd <= profile.target_max
    st.metric("Toplam Oran", f"{total_odd:,.2f}")
    st.metric("Model Tutma", f"%{metrics['hit_prob']*100:.5f}")
    st.metric("Monte Carlo Tutma", f"%{metrics['sim_hit']*100:.5f}")
    st.metric("EV Carpani", f"{metrics['ev_mult']:.4f}")
    st.metric("Beklenen ROI", f"%{metrics['roi_pct']:.2f}")
    st.metric("Risk Endeksi", f"%{metrics['risk_index']*100:.2f}")
    progress = clamp(total_odd / max(profile.target_min, 1.0), 0.0, 1.0)
    st.progress(progress, text=f"Hedefe yakinlik: %{progress*100:.1f}")
    if in_range:
        st.success("Kupon hedef oran araliginda.")
    else:
        st.warning("Kupon hedef aralik disinda. Daha fazla lig ve daha uzun bulten sec.")

    for p in picks:
        st.markdown(
            "<div class='panel-card'>"
            f"<div><span class='chip'>{p['league']}</span><span class='chip warn-chip'>{format_kickoff(p['commence'])}</span></div>"
            f"<div><b>{p['home']} - {p['away']}</b></div>"
            f"<div style='margin-top:6px'><span class='chip ok-chip'>{p['bet_type']} -> {p['bet_text']}</span>"
            f"<span class='chip gold-chip'>Oran {p['odd']:.2f}</span></div>"
            f"<div style='margin-top:6px;color:#c8d7ff'>Skor Tahmini: <b>{p['score_pred']}</b> | IY/MS: <b>{p['iyms_pred']}</b></div>"
            f"<div style='margin-top:6px;color:#9eb3de'>Skor Senaryolari: {p['score_top3']}</div>"
            f"<div style='margin-top:4px;color:#9eb3de'>IY/MS Olasiliklari: {p['iyms_top3']}</div>"
            f"<div style='margin-top:4px;color:#9eb3de'>Motivasyon H/A: %{p['motivation_home']*100:.0f}/%{p['motivation_away']*100:.0f} | "
            f"Dinlenme H/A: {p['days_rest_home']:.1f}/{p['days_rest_away']:.1f} gun | "
            f"Sakatlik-Rotasyon Riski H/A: %{p['injury_risk_home']*100:.0f}/%{p['injury_risk_away']*100:.0f}</div>"
            f"<div style='margin-top:6px;color:#9eb3de'>Edge %{p['edge']*100:.2f} | Model %{p['model_prob']*100:.2f} | ROI %{p['roi']*100:.2f} | {ai_leg_comment(p)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def build_visuals(candidates: List[dict]) -> None:
    if not candidates:
        return
    df = pd.DataFrame(candidates)
    st.subheader("Piyasa Analizi")
    col1, col2 = st.columns(2)

    with col1:
        top = (
            df.sort_values("value_score", ascending=False)
            .head(20)
            .loc[:, ["pick", "odd", "edge"]]
            .copy()
        )
        top["edge_pct"] = top["edge"] * 100.0
        st.bar_chart(top.set_index("pick")[["odd", "edge_pct"]])
        st.caption("Value skoruna gore ilk 20 secim")

    with col2:
        scatter = df[["odd", "edge", "confidence"]].copy()
        scatter["edge_pct"] = scatter["edge"] * 100.0
        scatter["confidence_pct"] = scatter["confidence"] * 100.0
        st.scatter_chart(scatter, x="odd", y="edge_pct")
        st.caption("Oran ve edge iliskisi")

    st.subheader("Aday Maclar")
    show = (
        df.sort_values(["value_score", "edge"], ascending=False)
        .head(120)
        .loc[
            :,
            [
                "league",
                "home",
                "away",
                "commence",
                "market",
                "pick",
                "bet_type",
                "bet_text",
                "score_pred",
                "score_pred_prob",
                "iyms_pred",
                "iyms_pred_prob",
                "score_top3",
                "iyms_top3",
                "odd",
                "model_prob",
                "market_prob",
                "edge",
                "roi",
                "consistency_prob",
                "motivation_home",
                "motivation_away",
                "days_rest_home",
                "days_rest_away",
                "injury_risk_home",
                "injury_risk_away",
                "confidence",
                "risk",
                "bookmakers_count",
                "value_score",
            ],
        ]
        .copy()
    )
    show["commence"] = show["commence"].apply(format_kickoff)
    show["model_prob"] = (show["model_prob"] * 100.0).round(3)
    show["score_pred_prob"] = (show["score_pred_prob"] * 100.0).round(2)
    show["iyms_pred_prob"] = (show["iyms_pred_prob"] * 100.0).round(2)
    show["market_prob"] = (show["market_prob"] * 100.0).round(3)
    show["edge"] = (show["edge"] * 100.0).round(3)
    show["roi"] = (show["roi"] * 100.0).round(3)
    show["consistency_prob"] = (show["consistency_prob"] * 100.0).round(3)
    show["motivation_home"] = (show["motivation_home"] * 100.0).round(1)
    show["motivation_away"] = (show["motivation_away"] * 100.0).round(1)
    show["injury_risk_home"] = (show["injury_risk_home"] * 100.0).round(1)
    show["injury_risk_away"] = (show["injury_risk_away"] * 100.0).round(1)
    show["confidence"] = (show["confidence"] * 100.0).round(3)
    show["risk"] = (show["risk"] * 100.0).round(3)
    st.dataframe(show, use_container_width=True, height=440)


def confidence_grade(row: pd.Series) -> str:
    score = (
        (0.34 * row["consistency_prob"])
        + (0.22 * row["confidence"])
        + (0.20 * max(row["roi"], 0.0))
        + (0.12 * max(row["edge"], 0.0))
        + (0.08 * ((row["motivation_home"] + row["motivation_away"]) / 2.0))
        - (0.12 * row["risk"])
    )
    if score >= 0.42:
        return "A+"
    if score >= 0.32:
        return "A"
    if score >= 0.23:
        return "B"
    return "C"


def build_premium_shortlist(candidates: List[dict], top_n: int = 20) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    df = pd.DataFrame(candidates).copy()
    df["premium_score"] = (
        (0.34 * df["consistency_prob"])
        + (0.22 * df["confidence"])
        + (0.20 * np.maximum(df["roi"], 0.0))
        + (0.12 * np.maximum(df["edge"], 0.0))
        + (0.08 * ((df["motivation_home"] + df["motivation_away"]) / 2.0))
        - (0.12 * df["risk"])
    )
    df["grade"] = df.apply(confidence_grade, axis=1)

    best_rows = []
    for match_id, grp in df.groupby("match_id", sort=False):
        grp = grp.sort_values("premium_score", ascending=False)
        best = grp.iloc[0]
        alt = grp.iloc[1] if len(grp) > 1 else None
        best_rows.append(
            {
                "lig": best["league"],
                "mac": f"{best['home']} - {best['away']}",
                "tarih": format_kickoff(best["commence"]),
                "onerilen_bahis": f"{best['bet_type']} -> {best['bet_text']}",
                "oran": round(float(best["odd"]), 2),
                "model_olasilik_%": round(float(best["model_prob"] * 100), 2),
                "roi_%": round(float(best["roi"] * 100), 2),
                "tutarlilik_%": round(float(best["consistency_prob"] * 100), 2),
                "skor": f"{best['score_pred']} (%{best['score_pred_prob']*100:.1f})",
                "iy_ms": f"{best['iyms_pred']} (%{best['iyms_pred_prob']*100:.1f})",
                "alternatif_bahis": (f"{alt['bet_type']} -> {alt['bet_text']} ({alt['odd']:.2f})" if alt is not None else "-"),
                "guven_notu": best["grade"],
                "premium_skor": round(float(best["premium_score"]), 4),
            }
        )

    out = pd.DataFrame(best_rows).sort_values(["premium_skor", "roi_%"], ascending=False).head(top_n)
    return out


def render_score_iyms_center(candidates: List[dict]) -> None:
    st.subheader("Skor ve IY/MS Merkezi")
    shortlist = build_premium_shortlist(candidates, top_n=25)
    if shortlist.empty:
        st.info("Yeterli aday bulunamadi.")
        return

    topA = shortlist[shortlist["guven_notu"].isin(["A+", "A"])].copy()
    if not topA.empty:
        st.markdown("**Premium A+ / A Listesi**")
        st.dataframe(topA, use_container_width=True, height=320)
    else:
        st.caption("A+ / A filtrede yeterli mac yok, B seviyeleri listelendi.")

    st.markdown("**Tum Premium Shortlist**")
    st.dataframe(shortlist, use_container_width=True, height=420)


def render_history(db_path: Path) -> None:
    st.subheader("Kupon Gecmisi")
    history = load_recent_history(db_path)
    if history.empty:
        st.info("Kayitli kupon yok.")
        return
    hist = history.copy()
    hist["created_at"] = pd.to_datetime(hist["created_at"], utc=True).dt.tz_convert("Europe/Istanbul").dt.strftime("%d.%m.%Y %H:%M")
    hist["hit_prob"] = (hist["hit_prob"] * 100.0).round(5)
    hist["sim_hit"] = (hist["sim_hit"] * 100.0).round(5)
    hist["risk_index"] = (hist["risk_index"] * 100.0).round(2)
    st.dataframe(hist, use_container_width=True, height=280)


def render_overview_cards(candidates: List[dict], generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]]) -> None:
    total_matches = len({c["match_id"] for c in candidates})
    total_candidates = len(candidates)
    golden_total = next((t for p, _, t, _ in generated if p.key == "golden"), 1.0)
    in_target = "Evet" if 50000 <= golden_total <= 100000 else "Hayir"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Toplam Mac</div><div class='kpi-value'>{total_matches}</div>"
            f"<div class='kpi-sub'>Analiz edilen fikstur</div></div>",
            unsafe_allow_html=True,
        )


def stake_plan_for_coupons(
    generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]],
    bankroll: float,
    risk_mode: str,
) -> pd.DataFrame:
    if bankroll <= 0:
        return pd.DataFrame()
    risk_multiplier = {"Dusuk": 0.35, "Orta": 0.55, "Yuksek": 0.8}.get(risk_mode, 0.55)
    rows = []
    for profile, picks, total_odd, metrics in generated:
        if not picks:
            continue
        ev = metrics["ev_mult"]
        raw_kelly = 0.0
        if total_odd > 1.01:
            b = total_odd - 1.0
            p = metrics["hit_prob"]
            q = 1.0 - p
            raw_kelly = max((b * p - q) / max(b, 1e-6), 0.0)
        fractional = clamp(raw_kelly * risk_multiplier, 0.0, 0.06)
        recommended = bankroll * fractional
        rows.append(
            {
                "Profil": profile.title,
                "Toplam Oran": round(total_odd, 2),
                "Model Tutma %": round(metrics["hit_prob"] * 100, 4),
                "EV": round(ev, 4),
                "Kelly %": round(fractional * 100, 3),
                "Onerilen Tutar": round(recommended, 2),
            }
        )
    return pd.DataFrame(rows)


def best_single_bets(candidates: List[dict], bankroll: float, risk_mode: str, top_n: int = 15) -> pd.DataFrame:
    if not candidates or bankroll <= 0:
        return pd.DataFrame()
    risk_multiplier = {"Dusuk": 0.22, "Orta": 0.35, "Yuksek": 0.5}.get(risk_mode, 0.35)
    df = pd.DataFrame(candidates).copy()
    df["single_score"] = (
        (0.34 * df["consistency_prob"])
        + (0.24 * df["confidence"])
        + (0.24 * np.maximum(df["roi"], 0.0))
        + (0.10 * np.maximum(df["edge"], 0.0))
        - (0.12 * df["risk"])
    )
    df = df.sort_values("single_score", ascending=False).head(120)
    rows = []
    for _, r in df.iterrows():
        b = max(float(r["odd"]) - 1.0, 1e-6)
        p = float(r["model_prob"])
        q = 1.0 - p
        kelly = max((b * p - q) / b, 0.0)
        frac = clamp(kelly * risk_multiplier, 0.0, 0.03)
        stake = bankroll * frac
        rows.append(
            {
                "lig": r["league"],
                "mac": f"{r['home']} - {r['away']}",
                "bahis": f"{r['bet_type']} -> {r['bet_text']}",
                "oran": round(float(r["odd"]), 2),
                "model_%": round(float(r["model_prob"] * 100), 2),
                "roi_%": round(float(r["roi"] * 100), 2),
                "guven_notu": confidence_grade(r),
                "onerilen_stake": round(stake, 2),
            }
        )
    out = pd.DataFrame(rows).sort_values(["roi_%", "model_%"], ascending=False).head(top_n)
    return out


def portfolio_diagnostics(generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]]) -> Dict[str, float]:
    picks = []
    for _, legs, _, _ in generated:
        picks.extend(legs)
    if not picks:
        return {"total_legs": 0.0, "league_concentration": 0.0, "same_day_ratio": 0.0, "avg_consistency": 0.0}
    total = len(picks)
    league_counts: Dict[str, int] = {}
    day_counts: Dict[str, int] = {}
    cons = []
    for p in picks:
        league_counts[p["league"]] = league_counts.get(p["league"], 0) + 1
        d = (p.get("commence") or "")[:10]
        day_counts[d] = day_counts.get(d, 0) + 1
        cons.append(float(p.get("consistency_prob", 0.0)))
    max_league_share = max(league_counts.values()) / total
    max_day_share = max(day_counts.values()) / total if day_counts else 0.0
    return {
        "total_legs": float(total),
        "league_concentration": float(max_league_share),
        "same_day_ratio": float(max_day_share),
        "avg_consistency": float(np.mean(cons) if cons else 0.0),
    }


def bankroll_stress_test(
    generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]],
    bankroll: float,
    risk_mode: str,
    paths: int = 2000,
) -> Dict[str, float]:
    if bankroll <= 0 or not generated:
        return {"p10": bankroll, "p50": bankroll, "p90": bankroll}
    risk_mult = {"Dusuk": 0.5, "Orta": 0.8, "Yuksek": 1.0}.get(risk_mode, 0.8)
    stakes = stake_plan_for_coupons(generated, bankroll, risk_mode)
    stake_map = {r["Profil"]: float(r["Onerilen Tutar"]) for _, r in stakes.iterrows()} if not stakes.empty else {}
    rng = random.Random(123)
    end_vals = []
    for _ in range(paths):
        bal = bankroll
        for profile, _, total_odd, metrics in generated:
            stake = stake_map.get(profile.title, 0.0) * risk_mult
            if stake <= 0:
                continue
            p = clamp(float(metrics["hit_prob"]), 0.001, 0.999)
            if rng.random() <= p:
                bal += stake * (total_odd - 1.0)
            else:
                bal -= stake
        end_vals.append(bal)
    arr = np.array(end_vals)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def generate_machine_report_json(
    db_path: Path,
    candidates: List[dict],
    generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]],
    perf: Dict[str, float],
) -> Path:
    reports_dir = db_path.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"machine_report_{ts}.json"
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "profiles": [
            {
                "profile": p.title,
                "legs": len(legs),
                "total_odd": total,
                "hit_prob": m["hit_prob"],
                "ev_mult": m["ev_mult"],
                "roi_pct": m["roi_pct"],
            }
            for p, legs, total, m in generated
        ],
        "settled_performance": perf,
        "premium_shortlist_top10": build_premium_shortlist(candidates, top_n=10).to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def generate_alternative_sets(
    candidates: List[dict],
    simulations: int,
    base_seed: int = 2026,
) -> Dict[str, List[Tuple[str, float, Dict[str, float], List[dict]]]]:
    alternatives: Dict[str, List[Tuple[str, float, Dict[str, float], List[dict]]]] = {}
    for idx, profile in enumerate(PROFILES):
        packs = []
        for offset in [0, 11, 23]:
            picks, total = generate_coupon(candidates, profile, seed=base_seed + idx + offset)
            metrics = coupon_metrics(picks, total, simulations=max(4000, simulations // 2))
            packs.append((f"Set-{offset//11+1}", total, metrics, picks))
        packs.sort(key=lambda x: x[2]["ev_mult"], reverse=True)
        alternatives[profile.key] = packs
    return alternatives


def render_alert_center(candidates: List[dict]) -> None:
    st.subheader("Alarm Merkezi")
    if not candidates:
        st.info("Alarm olusturmak icin aday yok.")
        return
    df = pd.DataFrame(candidates)
    high_edge = df[df["edge"] >= 0.10].sort_values("edge", ascending=False).head(12)
    fragile = df[(df["risk"] >= 0.70) & (df["odd"] >= 2.5)].sort_values("risk", ascending=False).head(12)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Yuksek Edge Firsatlari**")
        if high_edge.empty:
            st.caption("Su an %10+ edge firsati yok.")
        else:
            for _, row in high_edge.iterrows():
                st.markdown(
                    f"- `{row['league']}` {row['home']} - {row['away']} | **{row['bet_text']}** | "
                    f"oran `{row['odd']:.2f}` | edge `%{row['edge']*100:.2f}`"
                )
    with c2:
        st.markdown("**Yuksek Risk Uyarilari**")
        if fragile.empty:
            st.caption("Kritik riskli secim yok.")
        else:
            for _, row in fragile.iterrows():
                st.markdown(
                    f"- `{row['league']}` {row['home']} - {row['away']} | **{row['bet_text']}** | "
                    f"risk `%{row['risk']*100:.1f}` | oran `{row['odd']:.2f}`"
                )


def parse_injury_csv(uploaded_file) -> Dict[str, float]:
    if uploaded_file is None:
        return {}
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return {}
    if "team" not in df.columns:
        return {}
    missing_col = "missing_count" if "missing_count" in df.columns else None
    if missing_col is None:
        return {}
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        team = str(r.get("team", "")).strip()
        if not team:
            continue
        miss = float(r.get(missing_col, 0.0) or 0.0)
        out[normalize_team_name(team)] = clamp(miss * 0.035, 0.0, 0.35)
    return out


def generate_daily_report(
    db_path: Path,
    candidates: List[dict],
    generated: List[Tuple[CouponProfile, List[dict], float, Dict[str, float]]],
) -> Path:
    reports_dir = db_path.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"daily_report_{ts}.md"

    lines = []
    lines.append(f"# Gunluk Rapor - {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append("")
    lines.append(f"- Analiz edilen aday sayisi: {len(candidates)}")
    perf = load_settled_performance(db_path)
    lines.append(f"- Settled leg: {int(perf['n'])}, hit rate: %{perf['hit_rate']*100:.2f}, avg ROI: %{perf['avg_roi']*100:.2f}, avg CLV: %{perf['avg_clv']*100:.3f}")
    lines.append("")
    lines.append("## Kupon Ozetleri")
    for profile, picks, total, m in generated:
        lines.append(f"- {profile.title}: {len(picks)} mac | oran {total:,.2f} | model %{m['hit_prob']*100:.4f} | ROI %{m['roi_pct']:.2f}")
    lines.append("")
    lines.append("## Premium Shortlist (ilk 10)")
    shortlist = build_premium_shortlist(candidates, top_n=10)
    if shortlist.empty:
        lines.append("- Uygun premium shortlist yok.")
    else:
        for _, r in shortlist.iterrows():
            lines.append(f"- {r['lig']} | {r['mac']} | {r['onerilen_bahis']} | oran {r['oran']} | guven {r['guven_notu']} | ROI %{r['roi_%']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO daily_reports (created_at, report_path) VALUES (?, ?)",
        (datetime.now(timezone.utc).isoformat(), str(report_path)),
    )
    conn.commit()
    conn.close()
    return report_path
def main() -> None:
    st.set_page_config(page_title="Bet AI Kupon Motoru Pro", layout="wide")
    inject_custom_css()
    require_login_if_enabled()
    st.title("Bet AI Kupon Motoru Pro")
    st.caption("Matematiksel edge + takim guc analizi + AI destekli kupon motoru")

    odds_key_env = os.getenv("ODDS_API_KEY", "").strip()
    football_key_env = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    require_user_keys = os.getenv("REQUIRE_USER_KEYS", "true").strip().lower() == "true"
    default_region = os.getenv("REGION", "eu").strip()
    db_path = Path("data/coupon_history.db")
    init_db(db_path)
    odds_key = ""
    football_key = ""

    with st.sidebar:
        st.header("Motor Ayarlari")
        st.subheader("API Anahtarlari")
        st.caption("Kullanici kendi API keylerini girerek sistemi kullanir.")
        with st.expander("Key nereden alinir?"):
            st.markdown("1. The Odds API key al: [the-odds-api.com](https://the-odds-api.com)")
            st.markdown("2. Football-Data key al: [football-data.org/client/register](https://www.football-data.org/client/register)")
            st.markdown("3. Aldigin keyleri asagidaki kutulara yapistir.")

        odds_key_input = st.text_input("The Odds API Key", value="", type="password", placeholder="odds key buraya")
        football_key_input = st.text_input("Football-Data API Key", value="", type="password", placeholder="football-data key buraya")

        if require_user_keys:
            odds_key = odds_key_input.strip()
            football_key = football_key_input.strip()
        else:
            odds_key = odds_key_input.strip() or odds_key_env
            football_key = football_key_input.strip() or football_key_env

        if not odds_key:
            st.error("The Odds API key gerekli. Kutuga yapistir.")
        if not football_key:
            st.warning("Football-Data key yoksa takim guc analizi kisitli calisir.")

        region_choices = ["eu", "uk", "us", "au"]
        region = st.selectbox(
            "Bookmaker bolgesi",
            options=region_choices,
            index=region_choices.index(default_region) if default_region in region_choices else 0,
        )
        leagues = st.multiselect(
            "Ligler",
            options=list(LEAGUES.keys()),
            default=["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a", "soccer_germany_bundesliga"],
            format_func=lambda k: LEAGUES[k]["label"],
        )
        days = st.slider("Bulten suresi (gun)", min_value=1, max_value=7, value=3)
        simulations = st.select_slider("Monte Carlo simulasyon", options=[5000, 10000, 20000], value=10000)
        open_mode = st.toggle("Tum Oranlari Ac (Kisitsiz Mod)", value=True)
        quality_level = st.selectbox("Kalite Filtresi", options=["Dengeli", "Yuksek", "Maksimum"], index=2)
        depth_level = st.selectbox("Analiz Derinligi", options=["Standart", "Derin", "Maksimum"], index=2)
        bankroll = st.number_input("Bakiye (TL)", min_value=100.0, max_value=10000000.0, value=10000.0, step=100.0)
        risk_mode = st.selectbox("Risk Modu", options=["Dusuk", "Orta", "Yuksek"], index=1)
        save_history = st.toggle("Uretilen kuponlari kaydet", value=True)
        run = st.button("Kuponlari Uret", type="primary", use_container_width=True)
        st.divider()
        st.caption("Uzak erisim icin: Render/Railway ile deploy edip URL'yi telefondan acabilirsin.")
        st.caption("Not: Kisitsiz mod acikken daha fazla market kombinasyonu analiz edilir.")
        st.divider()
        st.subheader("Ek Veri")
        injury_file = st.file_uploader("Sakatlik CSV (team,missing_count)", type=["csv"], accept_multiple_files=False)
        auto_report = st.toggle("Gunluk raporu otomatik olustur", value=True)

    if not run:
        st.info("'Kuponlari Uret' butonuna basarak sistemin tam analizini calistir.")
        render_history(db_path)
        return

    if not odds_key or not leagues:
        st.stop()

    try:
        odds_events = fetch_odds_events(odds_api_key=odds_key, region=region, sport_keys=leagues, days=days)
    except Exception as exc:
        st.error(f"Odds API hatasi: {exc}")
        st.stop()

    if not odds_events:
        st.warning("Mac bulunamadi. Bolge, lig veya gun ayarini degistir.")
        st.stop()

    standings = {}
    team_contexts: Dict[int, dict] = {}
    settled_now = 0
    if football_key:
        comp_codes = sorted({LEAGUES[k]["fd_comp"] for k in leagues if k in LEAGUES})
        try:
            standings = fetch_standings(football_data_key=football_key, competition_codes=comp_codes)
            team_map = build_team_id_map_for_events(odds_events, standings)
            if team_map:
                team_contexts = fetch_team_contexts(
                    football_data_key=football_key,
                    team_ids=tuple(sorted(set(team_map.values()))),
                )
            settled_now = settle_open_legs(db_path, football_key, leagues)
        except Exception as exc:
            st.warning(f"Football-data katmani hata verdi: {exc}. Sadece odds modeliyle devam ediliyor.")

    injury_adjustments = parse_injury_csv(injury_file)
    candidates = extract_candidates(odds_events, standings, team_contexts, injury_adjustments)
    calib_bins = load_calibration_bins(db_path, bins=10)
    apply_calibration_to_candidates(candidates, calib_bins)
    candidates.sort(key=lambda c: (c["value_score"], c["edge"], c["odd"]), reverse=True)
    if not candidates:
        st.warning("Model filtrelerinden gecen aday secim bulunamadi.")
        st.stop()

    st.success(f"{len(odds_events)} mac yuklendi, {len(candidates)} aday secim uretildi.")
    if settled_now > 0:
        st.info(f"Otomatik settlement tamamlandi: {settled_now} leg sonuclandirildi.")

    generated = []
    for idx, profile in enumerate(PROFILES):
        picks, total_odd = generate_coupon(
            candidates,
            profile,
            seed=2026 + idx,
            open_mode=open_mode,
            quality_level=quality_level,
            depth_level=depth_level,
        )
        metrics = coupon_metrics(picks, total_odd, simulations=simulations)
        generated.append((profile, picks, total_odd, metrics))
        if save_history:
            save_coupon(db_path, profile, picks, total_odd, metrics)

    render_overview_cards(candidates, generated)
    plan_df = stake_plan_for_coupons(generated, bankroll=bankroll, risk_mode=risk_mode)
    alternatives = generate_alternative_sets(candidates, simulations=simulations)
    single_df = best_single_bets(candidates, bankroll=bankroll, risk_mode=risk_mode, top_n=15)
    diag = portfolio_diagnostics(generated)
    stress = bankroll_stress_test(generated, bankroll=bankroll, risk_mode=risk_mode, paths=2500)
    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Kupon Merkezi", "Piyasa Paneli", "Skor & IYMS", "Operasyon", "Gecmis"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        for col, payload in zip([col1, col2, col3], generated):
            profile, picks, total_odd, metrics = payload
            with col:
                render_coupon_panel(profile, picks, total_odd, metrics)
        st.divider()
        st.subheader("AI Ozet")
        for profile, picks, total_odd, metrics in generated:
            st.markdown(f"- {ai_coupon_summary(profile, picks, total_odd, metrics)}")
        export_rows = []
        for profile, picks, total_odd, metrics in generated:
            for p in picks:
                export_rows.append(
                    {
                        "profil": profile.title,
                        "lig": p["league"],
                        "ev_sahibi": p["home"],
                        "deplasman": p["away"],
                        "tarih": format_kickoff(p["commence"]),
                        "bahis_tipi": p["bet_type"],
                        "bahis": p["bet_text"],
                        "oran": round(p["odd"], 2),
                        "skor_tahmini": p["score_pred"],
                        "skor_tahmin_olasilik": round(p["score_pred_prob"] * 100, 2),
                        "iy_ms": p["iyms_pred"],
                        "iy_ms_olasilik": round(p["iyms_pred_prob"] * 100, 2),
                        "skor_senaryolari": p["score_top3"],
                        "iyms_senaryolari": p["iyms_top3"],
                        "model_olasilik": round(p["model_prob"] * 100, 3),
                        "edge": round(p["edge"] * 100, 3),
                        "roi": round(p["roi"] * 100, 3),
                        "tutarlilik": round(p["consistency_prob"] * 100, 3),
                    }
                )
        if export_rows:
            export_df = pd.DataFrame(export_rows)
            st.download_button(
                "Kuponlari CSV Indir",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="kupon_merkezi_export.csv",
                mime="text/csv",
            )

    with tab2:
        build_visuals(candidates)

    with tab3:
        render_score_iyms_center(candidates)

    with tab4:
        st.subheader("Para Yonetimi")
        if plan_df.empty:
            st.info("Stake plani olusturulamadi.")
        else:
            st.dataframe(plan_df, use_container_width=True, height=220)
            toplam_onerilen = float(plan_df["Onerilen Tutar"].sum())
            st.info(f"Toplam onerilen kupon butcesi: {toplam_onerilen:,.2f} TL")
        st.divider()
        st.subheader("Alternatif Kupon Setleri")
        for profile in PROFILES:
            st.markdown(f"**{profile.title}**")
            packs = alternatives.get(profile.key, [])
            if not packs:
                st.caption("Set bulunamadi.")
                continue
            for tag, total, mtx, picks in packs:
                st.markdown(
                    f"- `{tag}` | oran `{total:,.2f}` | model `%{mtx['hit_prob']*100:.4f}` | EV `{mtx['ev_mult']:.4f}` | bacak `{len(picks)}`"
                )
        st.divider()
        st.subheader("Tek Mac Premium (Top 15)")
        if single_df.empty:
            st.caption("Tek mac premium liste olusturulamadi.")
        else:
            st.dataframe(single_df, use_container_width=True, height=320)
        st.divider()
        st.subheader("Portfoy Diagnostik")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Toplam Bacak", f"{int(diag['total_legs'])}")
        with d2:
            st.metric("Lig Yogunluk", f"%{diag['league_concentration']*100:.1f}")
        with d3:
            st.metric("Ayni Gun Orani", f"%{diag['same_day_ratio']*100:.1f}")
        with d4:
            st.metric("Ort Tutarlilik", f"%{diag['avg_consistency']*100:.1f}")
        st.caption("Lig yogunluk %60+ ise kuponlari liglere daha yaymak riskleri azaltir.")
        st.divider()
        st.subheader("Bankroll Stres Testi")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("P10", f"{stress['p10']:,.0f} TL")
        with s2:
            st.metric("P50", f"{stress['p50']:,.0f} TL")
        with s3:
            st.metric("P90", f"{stress['p90']:,.0f} TL")
        st.caption("Bu dagilim tek tur tahmin sonu bakiye senaryolaridir.")
        st.divider()
        render_alert_center(candidates)
        st.divider()
        st.subheader("Gercek Performans ve Kalibrasyon")
        perf = load_settled_performance(db_path)
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.metric("Settled Leg", f"{int(perf['n'])}")
        with pc2:
            st.metric("Gercek Hit", f"%{perf['hit_rate']*100:.2f}")
        with pc3:
            st.metric("Gercek Ortalama ROI", f"%{perf['avg_roi']*100:.2f}")
        with pc4:
            st.metric("Ortalama CLV", f"%{perf['avg_clv']*100:.3f}")
        if calib_bins:
            cal_df = pd.DataFrame(
                [{"bin_alt": b[0], "pred_ort": b[1], "gercek_ort": b[2]} for b in calib_bins]
            )
            cal_df["pred_ort"] = (cal_df["pred_ort"] * 100).round(2)
            cal_df["gercek_ort"] = (cal_df["gercek_ort"] * 100).round(2)
            st.dataframe(cal_df, use_container_width=True, height=220)
        else:
            st.caption("Kalibrasyon icin yeterli settled veri yok (en az ~60 leg).")

    with tab5:
        render_history(db_path)

    if auto_report:
        rp = generate_daily_report(db_path, candidates, generated)
        st.caption(f"Gunluk rapor olusturuldu: {rp}")
        mr = generate_machine_report_json(db_path, candidates, generated, perf=load_settled_performance(db_path))
        st.caption(f"Makine raporu olusturuldu: {mr}")


if __name__ == "__main__":
    main()

import argparse
import html
import os
import sqlite3
import time
import json
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # Python 3.8 fallback
    ZoneInfo = None

import requests
from dotenv import load_dotenv

from app import (
    LEAGUES,
    ai_coupon_summary,
    best_single_bets,
    fetch_injury_csv_from_url,
    fetch_finished_matches_by_competitions,
    format_kickoff,
    leg_won,
    load_score_mastery_leaderboard,
    normalize_team_name,
    run_coupon_engine,
    settle_open_legs,
)

load_dotenv()


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_leagues(raw: str) -> List[str]:
    if not raw.strip():
        return [
            "soccer_epl",
            "soccer_spain_la_liga",
            "soccer_italy_serie_a",
            "soccer_germany_bundesliga",
        ]
    out = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item in LEAGUES:
            out.append(item)
    return out


def _chunk_text(text: str, max_len: int = 3500) -> List[str]:
    chunks = []
    current = ""
    for line in text.splitlines():
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(line) <= max_len:
                current = line
            else:
                parts = [line[i : i + max_len] for i in range(0, len(line), max_len)]
                chunks.extend(parts[:-1])
                current = parts[-1]
    if current:
        chunks.append(current)
    return chunks or [text]


def _telegram_api(token: str, method: str, data: Optional[dict] = None, files: Optional[dict] = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    if files:
        resp = requests.post(url, data=data or {}, files=files, timeout=45)
    else:
        resp = requests.post(url, data=data or {}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"Telegram API hatasi ({method}): {payload}")
    return payload


def _send_text(token: str, chat_id: int, text: str) -> None:
    for chunk in _chunk_text(text):
        _telegram_api(
            token,
            "sendMessage",
            data={
                "chat_id": str(chat_id),
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": "true",
            },
        )


def _send_document(token: str, chat_id: int, file_path: Path) -> None:
    if not file_path.exists():
        return
    with file_path.open("rb") as fp:
        _telegram_api(token, "sendDocument", data={"chat_id": str(chat_id)}, files={"document": fp})


def _safe(v) -> str:
    return html.escape(str(v if v is not None else "-"))


def _card_mode() -> str:
    mode = os.getenv("TELEGRAM_CARD_MODE", "dual").strip().lower()
    if mode not in {"compact", "full", "dual"}:
        return "dual"
    return mode


def _theme() -> str:
    theme = os.getenv("TELEGRAM_THEME", "gold").strip().lower()
    if theme not in {"gold", "classic"}:
        return "gold"
    return theme


def _confidence_badge(conf: float) -> str:
    if conf >= 0.68:
        return "🟢 GUVEN YUKSEK"
    if conf >= 0.56:
        return "🟡 GUVEN ORTA"
    return "🔴 GUVEN DUSUK"


def _build_coupon_cards(payload: Dict, top_single_count: int = 8) -> List[str]:
    generated = payload["generated"]
    candidates = payload["candidates"]
    mode = _card_mode()
    theme = _theme()

    if theme == "gold":
        line = "╔══════════════════════════════╗"
        sep = "╚══════════════════════════════╝"
        title_emoji = "👑"
        bullet_match = "🏟️"
        bullet_bet = "💰"
        bullet_model = "📊"
        bullet_clock = "🕒"
    else:
        line = "━━━━━━━━━━━━━━━━━━━━"
        sep = "━━━━━━━━━━━━━━━━━━━━"
        title_emoji = "🏆"
        bullet_match = "🏟️"
        bullet_bet = "🎲"
        bullet_model = "📈"
        bullet_clock = "🕒"

    cards: List[str] = []
    header = (
        f"{title_emoji} <b>BET AI ODDS | VIP GOLD EDITION</b>\n"
        f"💎 <b>Tema:</b> <code>{theme.upper()}</code>\n"
        f"🕒 <b>Tarih:</b> {_safe(datetime.now().strftime('%d.%m.%Y %H:%M'))}\n"
        f"📌 <b>Toplam Aday:</b> <code>{len(candidates)}</code>\n"
        f"🧩 <b>Kart Modu:</b> <code>{mode.upper()}</code>\n"
        f"{line}\n"
        "🥇 <b>Yalnizca VIP uyelere ozel premium analiz</b>\n"
        f"{sep}"
    )
    cards.append(header)

    for profile, picks, total_odd, metrics in generated:
        profile_head = (
            f"👑 <b>{_safe(profile.title.upper())} | VIP GOLD</b>\n"
            f"🎯 <b>Toplam Oran:</b> <code>{total_odd:,.2f}</code> | ⚽ <b>Mac:</b> <code>{len(picks)}</code>\n"
            f"{bullet_model} <b>Model:</b> <code>%{metrics['hit_prob']*100:.4f}</code> | "
            f"<b>Sim:</b> <code>%{metrics['sim_hit']*100:.4f}</code> | "
            f"<b>EV:</b> <code>{metrics['ev_mult']:.4f}</code> | "
            f"<b>ROI:</b> <code>%{metrics['roi_pct']:.2f}</code>"
        )
        if not picks:
            cards.append(profile_head + "\n\n❌ Bu profil icin kupon uretilemedi.")
            continue

        if mode in {"compact", "dual"}:
            compact_lines: List[str] = [profile_head, "🧾 <b>KISA KART | HIZLI OYNA</b>"]
            for idx, p in enumerate(picks[:8], start=1):
                compact_lines.append(
                    f"{idx}) <b>{_safe(p['home'])} - {_safe(p['away'])}</b>\n"
                    f"   {bullet_bet} {_safe(p['bet_text'])} @ <code>{p['odd']:.2f}</code> | "
                    f"Model <code>%{p['model_prob']*100:.1f}</code> | "
                    f"Edge <code>%{p['edge']*100:.2f}</code>"
                )
            if len(picks) > 8:
                compact_lines.append(f"… +{len(picks) - 8} secim")
            compact_lines.append(sep)
            cards.append("\n".join(compact_lines))

        if mode in {"full", "dual"}:
            detail_lines: List[str] = [profile_head, "📋 <b>DETAY KART | DERIN ANALIZ</b>"]
            for idx, p in enumerate(picks[:6], start=1):
                score_pred = p.get("score_pred", "-")
                iyms_pred = p.get("iyms_pred", "-")
                score_top3 = p.get("score_top3", "-")
                iyms_top3 = p.get("iyms_top3", "-")
                detail_lines.append(f"\n{bullet_match} <b>MAC {idx}: {_safe(p['home'])} - {_safe(p['away'])}</b>")
                detail_lines.append(
                    f"{bullet_bet} <b>Bahis:</b> {_safe(p['bet_type'])} → <code>{_safe(p['bet_text'])}</code> @ <code>{p['odd']:.2f}</code>"
                )
                detail_lines.append(
                    f"🔢 <b>Skor:</b> <code>{_safe(score_pred)}</code> | <b>IY/MS:</b> <code>{_safe(iyms_pred)}</code>"
                )
                detail_lines.append(f"🧠 <b>Skor Senaryo:</b> <code>{_safe(score_top3)}</code>")
                detail_lines.append(f"🧠 <b>IY/MS Senaryo:</b> <code>{_safe(iyms_top3)}</code>")
                detail_lines.append(
                    f"{bullet_model} <b>Model:</b> <code>%{p['model_prob']*100:.1f}</code> | "
                    f"<b>Edge:</b> <code>%{p['edge']*100:.2f}</code> | "
                    f"<b>ROI:</b> <code>%{p['roi']*100:.2f}</code> | "
                    f"<b>Tutarlilik:</b> <code>%{p['consistency_prob']*100:.1f}</code>"
                )
                detail_lines.append(f"🛡️ <b>Seviye:</b> {_confidence_badge(float(p.get('confidence', 0.0)))}")
                detail_lines.append(f"{bullet_clock} <b>Mac Saati:</b> {_safe(format_kickoff(p['commence']))}")
            if len(picks) > 6:
                detail_lines.append(f"\n… +{len(picks) - 6} secim (detay kartta gosterilmedi)")
            detail_lines.append(sep)
            cards.append("\n".join(detail_lines))

    single_df = best_single_bets(candidates, bankroll=10000.0, risk_mode="Orta", top_n=top_single_count)
    if not single_df.empty:
        lines = ["🔥 <b>TOP TEK MAC FIRSATLARI | GOLD SHORTLIST</b>", line]
        for idx, (_, row) in enumerate(single_df.iterrows(), start=1):
            lines.append(
                f"{idx}) <b>{_safe(row['mac'])}</b>\n"
                f"• {_safe(row['bahis'])} @ <code>{row['oran']:.2f}</code>\n"
                f"• ROI: <code>%{_safe(row['roi_%'])}</code> | Guven: <code>{_safe(row['guven_notu'])}</code>"
            )
        lines.append(sep)
        cards.append("\n".join(lines))
    return cards


def _market_tr(market_key: str) -> str:
    if market_key == "h2h":
        return "MS"
    if market_key == "totals":
        return "ALT/UST"
    if market_key == "btts":
        return "KG"
    return market_key.upper()


def _normalize_pick_text(market_key: str, pick_text: str) -> str:
    low = pick_text.lower().strip()
    if market_key == "h2h":
        if low in {"draw", "x"}:
            return "MSX"
        if low in {"home", "1"}:
            return "MS1"
        if low in {"away", "2"}:
            return "MS2"
    if market_key == "totals":
        if "over" in low:
            return pick_text.replace("Over", "Ust").replace("over", "Ust")
        if "under" in low:
            return pick_text.replace("Under", "Alt").replace("under", "Alt")
    if market_key == "btts":
        if low == "yes":
            return "KG Var"
        if low == "no":
            return "KG Yok"
    return pick_text


def _load_recent_settled(db_path: Path, limit: int = 30) -> Tuple[Dict[str, float], List[dict]]:
    if not db_path.exists():
        return {"n": 0.0, "wins": 0.0, "hit": 0.0, "avg_odd_win": 0.0}, []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, created_at, league, home, away, market_key, bet_text, odd_open, won, home_goals, away_goals
        FROM bet_legs
        WHERE settled = 1
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    if not rows:
        return {"n": 0.0, "wins": 0.0, "hit": 0.0, "avg_odd_win": 0.0}, []
    data = [dict(r) for r in rows]
    n = float(len(data))
    wins = float(sum(1 for r in data if int(r.get("won", 0) or 0) == 1))
    hit = (wins / n) if n > 0 else 0.0
    win_odds = [float(r["odd_open"]) for r in data if int(r.get("won", 0) or 0) == 1 and r.get("odd_open") is not None]
    avg_odd_win = float(sum(win_odds) / len(win_odds)) if win_odds else 0.0
    return {"n": n, "wins": wins, "hit": hit, "avg_odd_win": avg_odd_win}, data


def _build_proof_cards(db_path: Path) -> List[str]:
    theme = _theme()
    if theme == "gold":
        line = "╔══════════════════════════════╗"
        sep = "╚══════════════════════════════╝"
    else:
        line = "━━━━━━━━━━━━━━━━━━━━"
        sep = "━━━━━━━━━━━━━━━━━━━━"

    limit = _env_int("TELEGRAM_PROOF_LIMIT", 30)
    stats, rows = _load_recent_settled(db_path, limit=max(10, limit))
    cards: List[str] = []
    if not rows:
        cards.append(
            "✅ <b>SISTEM KANIT PANELI</b>\n"
            "Henüz yeterli sonuçlanan bahis verisi yok.\n"
            "Veri biriktikçe son isabetler otomatik burada paylaşılır."
        )
        return cards

    head = (
        "✅ <b>SISTEM KANIT PANELI | SISTEM ONERI SONUCLARI</b>\n"
        f"📚 <b>Incelenen Kupon Bacagi:</b> <code>{int(stats['n'])}</code>\n"
        f"🏁 <b>Kazanan:</b> <code>{int(stats['wins'])}</code>\n"
        f"🎯 <b>Tutma Orani:</b> <code>%{stats['hit']*100:.2f}</code>\n"
        f"💰 <b>Kazanan Ortalama Oran:</b> <code>{stats['avg_odd_win']:.2f}</code>\n"
        f"{line}"
    )
    cards.append(head + f"\n{sep}")

    winners = [r for r in rows if int(r.get("won", 0) or 0) == 1][:8]
    if winners:
        lines = ["🏆 <b>SON KAZANANLAR (SISTEM ONERISI)</b>", line]
        for idx, r in enumerate(winners, start=1):
            result = f"{int(r.get('home_goals', 0))}-{int(r.get('away_goals', 0))}"
            mk = _market_tr(str(r.get("market_key", "")))
            bet = _normalize_pick_text(str(r.get("market_key", "")), str(r.get("bet_text", "")))
            lines.append(
                f"{idx}) <b>{_safe(r.get('home'))} - {_safe(r.get('away'))}</b>\n"
                f"• {mk}: <code>{_safe(bet)}</code> @ <code>{float(r.get('odd_open') or 0):.2f}</code>\n"
                f"• Sonuc: <code>{result}</code> | Durum: ✅"
            )
        lines.append(sep)
        cards.append("\n".join(lines))
    else:
        cards.append("🏆 <b>SON KAZANANLAR (SISTEM ONERISI)</b>\nBu periyotta kazanan bacak yok.")

    high_odds_hits = [r for r in rows if int(r.get("won", 0) or 0) == 1 and float(r.get("odd_open") or 0) >= 2.20][:6]
    if high_odds_hits:
        lines = ["🚀 <b>YUKSEK ORAN ISABETLERI (SISTEM ONERISI)</b>", line]
        for idx, r in enumerate(high_odds_hits, start=1):
            mk = _market_tr(str(r.get("market_key", "")))
            bet = _normalize_pick_text(str(r.get("market_key", "")), str(r.get("bet_text", "")))
            lines.append(
                f"{idx}) <b>{_safe(r.get('home'))} - {_safe(r.get('away'))}</b>\n"
                f"• {mk}: <code>{_safe(bet)}</code> @ <code>{float(r.get('odd_open') or 0):.2f}</code> ✅"
            )
        lines.append(sep)
        cards.append("\n".join(lines))

    cards.append(
        "🧠 <b>Not:</b> IY/MS tahminleri gunluk <b>DETAY KART</b> icinde her macta paylasilmaktadir.\n"
        "Bu panel sadece <b>sistemin onerdiği bacaklarin</b> sonuçlarını gösterir. Kisisel kuponun farklıysa sonuç doğal olarak farklı olabilir."
    )
    return cards


def _build_weekly_success_cards(db_path: Path) -> List[str]:
    if not db_path.exists():
        return ["📆 <b>HAFTALIK BASARI RAPORU</b>\nYeterli veri yok."]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, created_at, home, away, market_key, bet_text, odd_open, won, home_goals, away_goals
        FROM bet_legs
        WHERE settled = 1
        ORDER BY id DESC
        LIMIT 800
        """
    ).fetchall()
    conn.close()

    if not rows:
        return ["📆 <b>HAFTALIK BASARI RAPORU</b>\nHenüz sonuçlanan bahis yok."]

    now_utc = datetime.now(timezone.utc)
    recent = []
    for row in rows:
        created_raw = str(row["created_at"] or "")
        try:
            dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if (now_utc - dt).days <= 7:
            recent.append(dict(row))

    if not recent:
        return ["📆 <b>HAFTALIK BASARI RAPORU</b>\nSon 7 günde sonuçlanan bahis verisi yok."]

    n = float(len(recent))
    wins = float(sum(1 for r in recent if int(r.get("won", 0) or 0) == 1))
    hit = (wins / n) if n > 0 else 0.0
    high_hits = [r for r in recent if int(r.get("won", 0) or 0) == 1 and float(r.get("odd_open") or 0.0) >= 2.20]
    avg_odd = (
        sum(float(r.get("odd_open") or 0.0) for r in recent if int(r.get("won", 0) or 0) == 1) / max(int(wins), 1)
        if wins > 0
        else 0.0
    )

    head = (
        "📆 <b>HAFTALIK BASARI RAPORU | VIP GOLD</b>\n"
        f"🧮 <b>Toplam Sonuclanan:</b> <code>{int(n)}</code>\n"
        f"✅ <b>Kazanan:</b> <code>{int(wins)}</code>\n"
        f"🎯 <b>Haftalik Tutma:</b> <code>%{hit*100:.2f}</code>\n"
        f"💰 <b>Kazanan Ort. Oran:</b> <code>{avg_odd:.2f}</code>\n"
        f"🚀 <b>2.20+ Isabet:</b> <code>{len(high_hits)}</code>\n"
        "╔══════════════════════════════╗"
    )

    winners = [r for r in recent if int(r.get("won", 0) or 0) == 1][:10]
    lines = [head, "🏆 <b>HAFTANIN KAZANANLARI</b>"]
    for idx, r in enumerate(winners, start=1):
        mk = _market_tr(str(r.get("market_key", "")))
        bet = _normalize_pick_text(str(r.get("market_key", "")), str(r.get("bet_text", "")))
        hg = int(r.get("home_goals") or 0)
        ag = int(r.get("away_goals") or 0)
        lines.append(
            f"{idx}) <b>{_safe(r.get('home'))} - {_safe(r.get('away'))}</b>\n"
            f"• {mk}: <code>{_safe(bet)}</code> @ <code>{float(r.get('odd_open') or 0.0):.2f}</code>\n"
            f"• Skor: <code>{hg}-{ag}</code> | Durum: ✅"
        )
    lines.append("╚══════════════════════════════╝")
    return ["\n".join(lines)]


def _build_scoreboard_cards(db_path: Path, title_prefix: str = "🏁 <b>SKOR USTALIK LIDERLIGI</b>") -> List[str]:
    leaderboard = load_score_mastery_leaderboard(db_path, played_only=True, min_samples=8, limit=8)
    if not leaderboard:
        return [f"{title_prefix}\nYeterli sonuclanmis veri yok (min 8)."]
    lines = [f"{title_prefix} | <b>VIP GOLD</b>", "╔══════════════════════════════╗"]
    for idx, row in enumerate(leaderboard, start=1):
        lines.append(
            f"{idx}) <b>{_market_tr(str(row.get('market_key', '')))}</b> | "
            f"Mastery <code>%{float(row.get('mastery_score', 0.0))*100:.2f}</code>\n"
            f"• Hit <code>%{float(row.get('hit_rate', 0.0))*100:.2f}</code> | "
            f"IY/MS <code>%{float(row.get('iyms_hit_rate', 0.0))*100:.2f}</code> | "
            f"Skor MAE <code>{float(row.get('score_mae', 0.0)):.2f}</code>\n"
            f"• ROI <code>%{float(row.get('avg_roi', 0.0))*100:.2f}</code> | "
            f"Orneklem <code>{int(row.get('n', 0))}</code>"
        )
    lines.append("╚══════════════════════════════╝")
    lines.append(
        "Not: Mastery puani; hit, IY/MS isabeti, skor hatasi (MAE), ROI ve orneklem agirlikli hesaplanir."
    )
    return ["\n".join(lines)]


def _init_access_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subscribers (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            status TEXT NOT NULL,
            request_note TEXT,
            requested_at TEXT,
            approved_at TEXT,
            approved_by INTEGER,
            paid_until TEXT,
            plan_days INTEGER NOT NULL DEFAULT 30,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_played_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            market_key TEXT NOT NULL,
            bet_text TEXT NOT NULL,
            odd REAL NOT NULL,
            settled INTEGER NOT NULL DEFAULT 0,
            won INTEGER,
            home_goals INTEGER,
            away_goals INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS played_drafts (
            user_id INTEGER PRIMARY KEY,
            step TEXT NOT NULL,
            home TEXT,
            away TEXT,
            market_key TEXT,
            bet_text TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Backward-compatible migrations for existing DBs
    try:
        conn.execute("ALTER TABLE subscribers ADD COLUMN paid_until TEXT")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE subscribers ADD COLUMN plan_days INTEGER NOT NULL DEFAULT 30")
    except Exception:
        pass
    conn.commit()
    conn.close()


def _get_played_draft(db_path: Path, user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT user_id, step, home, away, market_key, bet_text, updated_at FROM played_drafts WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _set_played_draft(
    db_path: Path,
    user_id: int,
    step: str,
    home: str = "",
    away: str = "",
    market_key: str = "",
    bet_text: str = "",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO played_drafts (user_id, step, home, away, market_key, bet_text, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            step=excluded.step,
            home=excluded.home,
            away=excluded.away,
            market_key=excluded.market_key,
            bet_text=excluded.bet_text,
            updated_at=excluded.updated_at
        """,
        (int(user_id), step, home, away, market_key, bet_text, now),
    )
    conn.commit()
    conn.close()


def _clear_played_draft(db_path: Path, user_id: int) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM played_drafts WHERE user_id = ?", (int(user_id),))
    conn.commit()
    conn.close()


def _normalize_market_key(raw: str) -> Optional[str]:
    x = raw.strip().lower()
    if x in {"h2h", "ms", "macsonucu", "mac_sonucu"}:
        return "h2h"
    if x in {"totals", "altust", "alt_ust", "ou", "overunder"}:
        return "totals"
    if x in {"btts", "kg", "karsilikligol", "karsilikli_gol"}:
        return "btts"
    return None


def _normalize_bet_text_for_market(market_key: str, bet_text: str) -> str:
    t = bet_text.strip()
    low = t.lower()
    if market_key == "h2h":
        if low in {"ms1", "1", "home"}:
            return "home"
        if low in {"ms2", "2", "away"}:
            return "away"
        if low in {"msx", "x", "draw"}:
            return "draw"
    if market_key == "totals":
        if low.startswith("alt "):
            return "Under " + t.split(" ", 1)[1]
        if low.startswith("ust "):
            return "Over " + t.split(" ", 1)[1]
    if market_key == "btts":
        if low in {"var", "yes", "kg var"}:
            return "yes"
        if low in {"yok", "no", "kg yok"}:
            return "no"
    return t


def _save_user_played_leg(
    db_path: Path,
    user_id: int,
    home: str,
    away: str,
    market_key: str,
    bet_text: str,
    odd: float,
) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        INSERT INTO user_played_legs (
            user_id, created_at, home, away, market_key, bet_text, odd, settled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            int(user_id),
            datetime.now(timezone.utc).isoformat(),
            home.strip(),
            away.strip(),
            market_key,
            bet_text.strip(),
            float(odd),
        ),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


def _parse_played_command(text: str) -> Optional[dict]:
    # Format: /played Home - Away ; market ; bet_text ; odd
    raw = text.strip()
    if not raw.lower().startswith("/played "):
        return None
    payload = raw[len("/played ") :].strip()
    parts = [p.strip() for p in payload.split(";")]
    if len(parts) != 4:
        return None
    teams, market_raw, bet_raw, odd_raw = parts
    if " - " in teams:
        home, away = [x.strip() for x in teams.split(" - ", 1)]
    elif "-" in teams:
        home, away = [x.strip() for x in teams.split("-", 1)]
    else:
        return None
    if not home or not away:
        return None
    market_key = _normalize_market_key(market_raw)
    if market_key is None:
        return None
    try:
        odd = float(odd_raw.replace(",", "."))
    except ValueError:
        return None
    if odd <= 1.01:
        return None
    bet_text = _normalize_bet_text_for_market(market_key, bet_raw)
    return {
        "home": home,
        "away": away,
        "market_key": market_key,
        "bet_text": bet_text,
        "odd": odd,
    }


def _settle_user_played_legs(db_path: Path, football_data_key: str) -> int:
    if not football_data_key.strip():
        return 0
    comp_codes = sorted({v["fd_comp"] for v in LEAGUES.values()})
    finished_map = fetch_finished_matches_by_competitions(football_data_key, comp_codes, days_back=12)
    if not finished_map:
        return 0
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT id, home, away, market_key, bet_text
        FROM user_played_legs
        WHERE settled = 0
        """
    ).fetchall()
    settled = 0
    for rid, home, away, market_key, bet_text in rows:
        key = f"{normalize_team_name(home)}__{normalize_team_name(away)}"
        if key not in finished_map:
            continue
        hg, ag = finished_map[key]
        won = 1 if leg_won(market_key, bet_text, home, away, int(hg), int(ag)) else 0
        conn.execute(
            """
            UPDATE user_played_legs
            SET settled = 1, won = ?, home_goals = ?, away_goals = ?
            WHERE id = ?
            """,
            (won, int(hg), int(ag), int(rid)),
        )
        settled += 1
    conn.commit()
    conn.close()
    return settled


def _build_user_played_cards(db_path: Path, limit: int = 30) -> List[str]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM user_played_legs
        WHERE settled = 1
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    data = [dict(r) for r in rows]
    n = len(data)
    w = sum(1 for r in data if int(r.get("won", 0) or 0) == 1)
    hit = (w / max(n, 1)) * 100.0
    lines = [
        "🎫 <b>GERCEK OYNANAN KUPON SONUCLARI</b>",
        f"📚 Son {n} oynanan bacak | ✅ Kazanan {w} | 🎯 Tutma %{hit:.2f}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for idx, r in enumerate(data[:8], start=1):
        result = f"{int(r.get('home_goals') or 0)}-{int(r.get('away_goals') or 0)}"
        status = "✅" if int(r.get("won", 0) or 0) == 1 else "❌"
        lines.append(
            f"{idx}) <b>{_safe(r.get('home'))} - {_safe(r.get('away'))}</b>\n"
            f"• {_safe(r.get('market_key'))}: <code>{_safe(r.get('bet_text'))}</code> @ <code>{float(r.get('odd') or 0):.2f}</code>\n"
            f"• Sonuc: <code>{result}</code> | Durum: {status}"
        )
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    return ["\n".join(lines)]


def _set_state(db_path: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO bot_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


def _get_state(db_path: Path, key: str, default: str = "") -> str:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
    conn.close()
    if not row:
        return default
    return str(row[0])


def _upsert_user(db_path: Path, user: dict, status: str, note: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status FROM subscribers WHERE user_id = ?", (int(user["id"]),)).fetchone()
    current_status = row[0] if row else None

    requested_at = now if status == "pending" and current_status is None else None
    if status == "pending" and current_status in {"approved", "blocked"}:
        status = current_status

    if row:
        conn.execute(
            """
            UPDATE subscribers
            SET username = ?, first_name = ?, last_name = ?,
                status = ?, request_note = COALESCE(NULLIF(?, ''), request_note),
                requested_at = COALESCE(?, requested_at), updated_at = ?
            WHERE user_id = ?
            """,
            (
                user.get("username", ""),
                user.get("first_name", ""),
                user.get("last_name", ""),
                status,
                note,
                requested_at,
                now,
                int(user["id"]),
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO subscribers (
                user_id, username, first_name, last_name, status,
                request_note, requested_at, approved_at, approved_by, paid_until, plan_days, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (
                int(user["id"]),
                user.get("username", ""),
                user.get("first_name", ""),
                user.get("last_name", ""),
                status,
                note,
                requested_at or now,
                int(_env_int("TELEGRAM_PLAN_DAYS", 30)),
                now,
            ),
        )
    conn.commit()
    conn.close()


def _parse_iso_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _default_plan_days() -> int:
    return max(1, _env_int("TELEGRAM_PLAN_DAYS", 30))


def _set_subscription_days(db_path: Path, user_id: int, days: int) -> bool:
    days = max(1, int(days))
    now = datetime.now(timezone.utc)
    paid_until = (now + timedelta(days=days)).isoformat()
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT user_id FROM subscribers WHERE user_id = ?", (int(user_id),)).fetchone()
    if not row:
        conn.close()
        return False
    conn.execute(
        """
        UPDATE subscribers
        SET paid_until = ?, plan_days = ?, updated_at = ?
        WHERE user_id = ?
        """,
        (paid_until, days, now.isoformat(), int(user_id)),
    )
    conn.commit()
    conn.close()
    return True


def _subscription_days_left(row: Optional[dict]) -> Optional[int]:
    if not row:
        return None
    paid_until_dt = _parse_iso_dt(row.get("paid_until"))
    if paid_until_dt is None:
        return None
    now = datetime.now(timezone.utc)
    diff = paid_until_dt - now
    days_left = math.ceil(diff.total_seconds() / 86400.0)
    return int(days_left)


def _is_subscription_active(row: Optional[dict]) -> bool:
    if not row:
        return False
    if row.get("status") != "approved":
        return False
    days_left = _subscription_days_left(row)
    if days_left is None:
        return False
    return days_left >= 0


def _list_expiring_users(db_path: Path, days: int = 3, limit: int = 100) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM subscribers
        WHERE status = 'approved' AND paid_until IS NOT NULL
        ORDER BY paid_until ASC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        item = dict(r)
        left = _subscription_days_left(item)
        if left is None:
            continue
        if 0 <= left <= int(days):
            out.append(item)
    return out


def _list_expired_users(db_path: Path, limit: int = 100) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM subscribers
        WHERE status = 'approved' AND paid_until IS NOT NULL
        ORDER BY paid_until DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        item = dict(r)
        left = _subscription_days_left(item)
        if left is not None and left < 0:
            out.append(item)
    return out


def _set_user_status(db_path: Path, user_id: int, status: str, admin_id: Optional[int] = None, note: str = "") -> bool:
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT user_id FROM subscribers WHERE user_id = ?", (int(user_id),)).fetchone()
    if not row:
        conn.close()
        return False

    approved_at = now if status == "approved" else None
    approved_by = int(admin_id) if status == "approved" and admin_id is not None else None

    conn.execute(
        """
        UPDATE subscribers
        SET status = ?,
            request_note = COALESCE(NULLIF(?, ''), request_note),
            approved_at = COALESCE(?, approved_at),
            approved_by = COALESCE(?, approved_by),
            updated_at = ?
        WHERE user_id = ?
        """,
        (status, note, approved_at, approved_by, now, int(user_id)),
    )
    conn.commit()
    conn.close()
    return True


def _get_user(db_path: Path, user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM subscribers WHERE user_id = ?", (int(user_id),)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def _list_users(db_path: Path, status: str, limit: int = 50) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM subscribers WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
        (status, int(limit)),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _approved_user_ids(db_path: Path) -> List[int]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT user_id FROM subscribers WHERE status = 'approved'").fetchall()
    conn.close()
    return [int(r[0]) for r in rows]


def _admin_ids() -> List[int]:
    raw = os.getenv("TELEGRAM_ADMIN_IDS", "").strip()
    ids = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            ids.append(int(p))
        except ValueError:
            continue
    return ids


def _is_admin(user_id: int) -> bool:
    return int(user_id) in set(_admin_ids())


def _payment_fee() -> int:
    return _env_int("TELEGRAM_PAYMENT_FEE_TL", 500)


def _payment_text() -> str:
    custom = os.getenv("TELEGRAM_PAYMENT_TEXT", "").strip()
    if custom:
        return custom
    fee = _payment_fee()
    payment_dest = os.getenv("TELEGRAM_PAYMENT_DEST", "").strip()
    payment_line = f"Odeme bilgisi: {payment_dest}\n" if payment_dest else ""
    return (
        f"Uyelik ucreti: {fee} TL\n"
        f"{payment_line}"
        "Odeme yaptiktan sonra dekont ile bu bota ozelden yazin.\n"
        "Admin onayi sonrasi ozel gruba giris linki paylasilir."
    )


def _msg_template(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _welcome_message() -> str:
    default = (
        "Bet AI Odds VIP'e hos geldiniz.\n"
        "Kuponlar sadece odeme onayli uyelere aciktir.\n\n"
        f"{_payment_text()}\n\n"
        "Odeme yaptiktan sonra dekontu bu sohbete gonderin."
    )
    return _msg_template("TELEGRAM_MSG_WELCOME", default)


def _approved_message(invite_link: Optional[str] = None) -> str:
    default = "Odemeniz onaylandi. VIP erisim acildi."
    text = _msg_template("TELEGRAM_MSG_APPROVED", default)
    if invite_link:
        text = f"{text}\nOzel grup giris linkiniz: {invite_link}"
    return text


def _rejected_message(note: str = "") -> str:
    default = "Odeme talebiniz su an onaylanmadi. Detay icin adminle iletisime gecin."
    text = _msg_template("TELEGRAM_MSG_REJECTED", default)
    if note:
        text = f"{text}\nNot: {note}"
    return text


def _blocked_message() -> str:
    default = "Hesabiniz bloke edildi. Detay icin adminle iletisime gecin."
    return _msg_template("TELEGRAM_MSG_BLOCKED", default)


def _build_ad_text() -> str:
    fee = _payment_fee()
    brand = os.getenv("TELEGRAM_BRAND_NAME", "Bet AI Odds VIP").strip() or "Bet AI Odds VIP"
    contact = os.getenv("TELEGRAM_CONTACT", "Bu bota ozelden yaz").strip() or "Bu bota ozelden yaz"

    return (
        f"{brand} acildi.\n"
        "Her gun guncel kuponlar + oynanacak oranlar paylasiliyor.\n"
        f"Katilim ucreti: {fee} TL.\n"
        f"Katilmak icin: {contact}\n"
        "Odeme + dekont sonrasi onay verilir ve ozel gruba alinir."
    )


def _create_invite_link(token: str, group_chat_id: int) -> Optional[str]:
    expire_date = int((datetime.now(timezone.utc) + timedelta(days=2)).timestamp())
    try:
        payload = _telegram_api(
            token,
            "createChatInviteLink",
            data={
                "chat_id": str(group_chat_id),
                "member_limit": "1",
                "expire_date": str(expire_date),
                "name": "odeme-onayli-giris",
            },
        )
    except Exception as exc:
        _log(f"Davet linki olusturma hatasi: {exc}")
        return None

    result = payload.get("result", {})
    return result.get("invite_link")


def _run_coupon_generation() -> Dict:
    odds_key = os.getenv("ODDS_API_KEY", "").strip()
    football_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not odds_key:
        raise RuntimeError("ODDS_API_KEY gerekli.")

    region = os.getenv("REGION", "eu").strip() or "eu"
    leagues = _parse_leagues(os.getenv("TELEGRAM_LEAGUES", ""))
    if not leagues:
        raise RuntimeError("TELEGRAM_LEAGUES gecersiz.")

    days = _env_int("TELEGRAM_DAYS", 3)
    simulations = _env_int("TELEGRAM_SIMULATIONS", 10000)
    open_mode = _env_bool("TELEGRAM_OPEN_MODE", True)
    quality_level = os.getenv("TELEGRAM_QUALITY", "Maksimum").strip() or "Maksimum"
    depth_level = os.getenv("TELEGRAM_DEPTH", "Maksimum").strip() or "Maksimum"

    injury_adjustments = {}
    injury_url = os.getenv("INJURY_CSV_URL", "").strip()
    use_injury = _env_bool("TELEGRAM_USE_INJURY_URL", bool(injury_url))
    if use_injury and injury_url:
        injury_adjustments, err = fetch_injury_csv_from_url(injury_url)
        if err:
            _log(f"Sakatlik CSV uyarisi: {err}")
            injury_adjustments = {}

    db_path = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
    payload = run_coupon_engine(
        db_path=db_path,
        odds_key=odds_key,
        football_key=football_key,
        region=region,
        leagues=leagues,
        days=days,
        simulations=simulations,
        open_mode=open_mode,
        quality_level=quality_level,
        depth_level=depth_level,
        save_history=True,
        auto_report=True,
        injury_adjustments=injury_adjustments,
    )
    payload["_db_path"] = str(db_path)
    return payload


def _refresh_settlement_for_reports() -> int:
    football_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not football_key:
        return 0
    db_path = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
    try:
        # Broader settle pass for report consistency.
        return int(settle_open_legs(db_path, football_key, list(LEAGUES.keys())))
    except Exception as exc:
        _log(f"Settlement yenileme hatasi: {exc}")
        return 0


def run_once() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN gerekli.")

    target_group = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip()
    if not target_group:
        target_group = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not target_group:
        raise RuntimeError("TELEGRAM_PRIVATE_GROUP_ID (veya TELEGRAM_CHAT_ID) gerekli.")

    payload = _run_coupon_generation()
    cards = _build_coupon_cards(payload)
    for card in cards:
        _send_text(token, int(target_group), card)

    include_proof = _env_bool("TELEGRAM_INCLUDE_PROOF", True)
    include_scoreboard = _env_bool("TELEGRAM_INCLUDE_SCOREBOARD", True)
    db_raw = str(payload.get("_db_path", "")).strip()
    access_db = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
    _init_access_db(access_db)
    if include_proof and db_raw:
        settled_now = _refresh_settlement_for_reports()
        played_settled_now = _settle_user_played_legs(access_db, os.getenv("FOOTBALL_DATA_API_KEY", "").strip())
        if settled_now > 0:
            _send_text(token, int(target_group), f"🔄 <b>Settlement:</b> <code>{settled_now}</code> bacak yeni sonuclandi.")
        if played_settled_now > 0:
            _send_text(token, int(target_group), f"🎫 <b>Oynanan Settlement:</b> <code>{played_settled_now}</code> bacak yeni sonuclandi.")
        for card in _build_proof_cards(Path(db_raw)):
            _send_text(token, int(target_group), card)
        for card in _build_user_played_cards(access_db, limit=30):
            _send_text(token, int(target_group), card)
    if include_scoreboard and db_raw:
        for card in _build_scoreboard_cards(Path(db_raw), title_prefix="🏁 <b>GUNCEL SKOR USTALIK LIDERLIGI</b>"):
            _send_text(token, int(target_group), card)

    report_path = payload.get("report_path")
    if isinstance(report_path, Path):
        _send_document(token, int(target_group), report_path)

    _log("Gunluk kupon gonderimi tamamlandi.")


def _next_run(now: datetime, hhmm: str) -> datetime:
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError("TELEGRAM_SEND_TIME HH:MM formatinda olmali.")
    hour = int(parts[0])
    minute = int(parts[1])
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _resolve_timezone(tz_name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    _log(f"zoneinfo desteklenmiyor veya gecersiz TZ ({tz_name}), UTC+3 fallback kullaniliyor.")
    return timezone(timedelta(hours=3))


def run_daily_loop() -> None:
    tz_name = os.getenv("TELEGRAM_TZ", "Europe/Istanbul").strip() or "Europe/Istanbul"
    send_time = os.getenv("TELEGRAM_SEND_TIME", "10:00").strip() or "10:00"
    run_on_start = _env_bool("TELEGRAM_RUN_ON_START", False)
    tz = _resolve_timezone(tz_name)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    weekly_enabled = _env_bool("TELEGRAM_INCLUDE_WEEKLY", True)
    weekly_day = _env_int("TELEGRAM_WEEKLY_SUMMARY_DAY", 0)  # 0 = Monday
    target_group = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    access_db = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
    _init_access_db(access_db)

    if run_on_start:
        try:
            _log("Baslangicta tek seferlik gonderim basliyor...")
            run_once()
            if token:
                _notify_admins(token, "Gunluk kupon botu baslangicta bir kez gonderim yapti.")
        except Exception as exc:
            _log(f"Baslangic gonderimi hatasi: {exc}")
            if token:
                _notify_admins(token, f"Gunluk kupon botu baslangic hatasi: {exc}")

    while True:
        now = datetime.now(tz)
        target = _next_run(now, send_time)
        wait_seconds = max(1, int((target - now).total_seconds()))
        _log(f"Siradaki gonderim: {target.strftime('%Y-%m-%d %H:%M:%S %Z')} ({wait_seconds}s sonra)")
        while wait_seconds > 0:
            chunk = min(300, wait_seconds)
            time.sleep(chunk)
            wait_seconds -= chunk

        try:
            run_once()
            if weekly_enabled and token and target_group:
                now_local = datetime.now(tz)
                week_tag = now_local.strftime("%G-W%V")
                posted_week = _get_state(access_db, "weekly_posted_week", "")
                if now_local.weekday() == weekly_day and posted_week != week_tag:
                    db_path = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
                    for card in _build_weekly_success_cards(db_path):
                        _send_text(token, int(target_group), card)
                    _set_state(access_db, "weekly_posted_week", week_tag)
                    _notify_admins(token, f"Haftalik basari raporu gonderildi: {week_tag}")
            if token:
                _notify_admins(token, "Gunluk kupon gonderimi basarili.")
        except Exception as exc:
            _log(f"Gunluk gonderim hatasi: {exc}")
            if token:
                _notify_admins(token, f"Gunluk kupon gonderimi hatasi: {exc}")


def _notify_admins(token: str, text: str) -> None:
    for admin_id in _admin_ids():
        try:
            _send_text(token, admin_id, text)
        except Exception as exc:
            _log(f"Admin bildirim hatasi ({admin_id}): {exc}")


def _admin_help() -> str:
    return (
        "Admin komutlari:\n"
        "/pending\n"
        "/approved\n"
        "/approve <user_id> [not]\n"
        "/renew <user_id> [days]\n"
        "/expiring [days]\n"
        "/expired\n"
        "/reject <user_id> [not]\n"
        "/block <user_id> [not]\n"
        "/unblock <user_id>\n"
        "/who <user_id>\n"
        "/sendnow\n"
        "/weeklynow\n"
        "/scoreboard\n"
        "/scoreboardnow\n"
        "/health\n"
        "/settlenow\n"
        "/payinfo\n"
        "/playedstats\n"
        "/playedmenu (kullanici icin adim adim giris)\n"
        "/templates\n"
        "/ad"
    )


def _render_user_row(row: dict) -> str:
    uname = f"@{row.get('username')}" if row.get("username") else "-"
    full_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
    full_name = full_name or "-"
    left = _subscription_days_left(row)
    left_text = f" | kalan={left} gun" if left is not None else ""
    return f"id={row['user_id']} | {uname} | {full_name} | durum={row['status']}{left_text}"


def _handle_admin_command(token: str, db_path: Path, admin_id: int, text: str) -> None:
    parts = text.strip().split()
    cmd = parts[0].lower()

    if cmd in {"/help", "/admin"}:
        _send_text(token, admin_id, _admin_help())
        return

    if cmd == "/payinfo":
        _send_text(token, admin_id, _payment_text())
        return

    if cmd == "/sendnow":
        _send_text(token, admin_id, "Anlik kupon gonderimi baslatildi...")
        try:
            run_once()
            _send_text(token, admin_id, "Anlik kupon gonderimi tamamlandi.")
        except Exception as exc:
            _send_text(token, admin_id, f"Anlik gonderim hatasi: {exc}")
        return

    if cmd == "/weeklynow":
        target_group = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not target_group:
            _send_text(token, admin_id, "TELEGRAM_PRIVATE_GROUP_ID eksik.")
            return
        db_path = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
        settled_now = _refresh_settlement_for_reports()
        for card in _build_weekly_success_cards(db_path):
            _send_text(token, int(target_group), card)
        _send_text(token, admin_id, f"Haftalik rapor gruba gonderildi. Settlement: {settled_now}")
        return

    if cmd == "/scoreboard":
        db_main = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
        for card in _build_scoreboard_cards(db_main, title_prefix="🏁 <b>SKOR USTALIK LIDERLIGI</b>"):
            _send_text(token, admin_id, card)
        return

    if cmd == "/scoreboardnow":
        target_group = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not target_group:
            _send_text(token, admin_id, "TELEGRAM_PRIVATE_GROUP_ID eksik.")
            return
        db_main = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
        for card in _build_scoreboard_cards(db_main, title_prefix="🏁 <b>HAFTALIK SKOR USTALIK LIDERLIGI</b>"):
            _send_text(token, int(target_group), card)
        _send_text(token, admin_id, "Skor liderlik karti gruba gonderildi.")
        return

    if cmd == "/health":
        db_main = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
        access_db = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
        _init_access_db(access_db)
        football_key = bool(os.getenv("FOOTBALL_DATA_API_KEY", "").strip())
        odds_key = bool(os.getenv("ODDS_API_KEY", "").strip())
        group_id = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        conn = sqlite3.connect(access_db)
        subs = int(conn.execute("SELECT COUNT(*) FROM subscribers").fetchone()[0] or 0)
        active = int(conn.execute("SELECT COUNT(*) FROM subscribers WHERE status='approved'").fetchone()[0] or 0)
        played = int(conn.execute("SELECT COUNT(*) FROM user_played_legs").fetchone()[0] or 0)
        conn.close()
        msg = (
            "🩺 <b>SISTEM SAGLIK RAPORU</b>\n"
            f"• Main DB: <code>{'OK' if db_main.exists() else 'YOK'}</code>\n"
            f"• Access DB: <code>{'OK' if access_db.exists() else 'YOK'}</code>\n"
            f"• ODDS_API_KEY: <code>{'OK' if odds_key else 'EKSIK'}</code>\n"
            f"• FOOTBALL_DATA_API_KEY: <code>{'OK' if football_key else 'EKSIK'}</code>\n"
            f"• Grup ID: <code>{group_id or 'EKSIK'}</code>\n"
            f"• Uye: <code>{subs}</code> | Aktif: <code>{active}</code> | Oynanan kayit: <code>{played}</code>"
        )
        _send_text(token, admin_id, msg)
        return

    if cmd == "/settlenow":
        settled_now = _refresh_settlement_for_reports()
        db_path_local = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
        _init_access_db(db_path_local)
        played_now = _settle_user_played_legs(db_path_local, os.getenv("FOOTBALL_DATA_API_KEY", "").strip())
        _send_text(token, admin_id, f"Settlement tamamlandi. Sistem: {settled_now}, oynanan: {played_now}")
        return

    if cmd == "/playedstats":
        db_path_local = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
        _init_access_db(db_path_local)
        cards = _build_user_played_cards(db_path_local, limit=40)
        if not cards:
            _send_text(token, admin_id, "Gercek oynanan kupon sonucu henuz yok.")
        else:
            for c in cards:
                _send_text(token, admin_id, c)
        return

    if cmd == "/templates":
        msg = (
            "Mesaj sablonlari (.env):\n"
            "- TELEGRAM_MSG_WELCOME\n"
            "- TELEGRAM_MSG_APPROVED\n"
            "- TELEGRAM_MSG_REJECTED\n"
            "- TELEGRAM_MSG_BLOCKED\n"
            "Not: Bos birakirsan varsayilan metinler kullanilir."
        )
        _send_text(token, admin_id, msg)
        return

    if cmd == "/pending":
        rows = _list_users(db_path, "pending", limit=100)
        if not rows:
            _send_text(token, admin_id, "Bekleyen kullanici yok.")
            return
        msg = "Bekleyenler:\n" + "\n".join(_render_user_row(r) for r in rows)
        _send_text(token, admin_id, msg)
        return

    if cmd == "/approved":
        rows = _list_users(db_path, "approved", limit=100)
        if not rows:
            _send_text(token, admin_id, "Onayli kullanici yok.")
            return
        msg = "Onaylilar:\n" + "\n".join(_render_user_row(r) for r in rows)
        _send_text(token, admin_id, msg)
        return

    if cmd == "/expiring":
        days = _default_plan_days()
        if len(parts) >= 2:
            try:
                days = max(1, int(parts[1]))
            except ValueError:
                _send_text(token, admin_id, "Kullanim: /expiring [days]")
                return
        rows = _list_expiring_users(db_path, days=days, limit=200)
        if not rows:
            _send_text(token, admin_id, f"Onumuzde {days} gunde bitecek uye yok.")
            return
        msg = f"{days} gun icinde bitecek uyeler:\n" + "\n".join(_render_user_row(r) for r in rows)
        _send_text(token, admin_id, msg)
        return

    if cmd == "/expired":
        rows = _list_expired_users(db_path, limit=200)
        if not rows:
            _send_text(token, admin_id, "Suresi dolmus aktif uye yok.")
            return
        msg = "Suresi dolan uyeler:\n" + "\n".join(_render_user_row(r) for r in rows)
        _send_text(token, admin_id, msg)
        return

    if cmd == "/who":
        if len(parts) < 2:
            _send_text(token, admin_id, "Kullanim: /who <user_id>")
            return
        try:
            user_id = int(parts[1])
        except ValueError:
            _send_text(token, admin_id, "user_id sayi olmali.")
            return
        row = _get_user(db_path, user_id)
        if not row:
            _send_text(token, admin_id, "Kullanici bulunamadi.")
            return
        msg = (
            f"{_render_user_row(row)}\n"
            f"talep_notu: {row.get('request_note') or '-'}\n"
            f"talep_tarihi: {row.get('requested_at') or '-'}\n"
            f"onay_tarihi: {row.get('approved_at') or '-'}\n"
            f"onaylayan: {row.get('approved_by') or '-'}\n"
            f"uyelik_bitis: {row.get('paid_until') or '-'}"
        )
        _send_text(token, admin_id, msg)
        return

    if cmd == "/renew":
        if len(parts) < 2:
            _send_text(token, admin_id, "Kullanim: /renew <user_id> [days]")
            return
        try:
            user_id = int(parts[1])
        except ValueError:
            _send_text(token, admin_id, "user_id sayi olmali.")
            return
        days = _default_plan_days()
        if len(parts) >= 3:
            try:
                days = max(1, int(parts[2]))
            except ValueError:
                _send_text(token, admin_id, "days sayi olmali.")
                return
        ok = _set_subscription_days(db_path, user_id, days)
        if not ok:
            _send_text(token, admin_id, "Kullanici kaydi yok.")
            return
        _set_user_status(db_path, user_id, "approved", admin_id=admin_id, note=f"renew {days}d")
        row = _get_user(db_path, user_id) or {}
        left = _subscription_days_left(row)
        _send_text(token, user_id, f"Uyeliginiz yenilendi. Kalan sure: {left} gun.")
        _send_text(token, admin_id, f"Yenileme tamam: {user_id} | {days} gun")
        return

    if cmd in {"/approve", "/reject", "/block", "/unblock"}:
        if len(parts) < 2:
            _send_text(token, admin_id, f"Kullanim: {cmd} <user_id> [not]")
            return
        try:
            user_id = int(parts[1])
        except ValueError:
            _send_text(token, admin_id, "user_id sayi olmali.")
            return

        note = " ".join(parts[2:]).strip()
        plan_days = _default_plan_days()
        if cmd == "/approve" and len(parts) >= 3:
            try:
                plan_days = max(1, int(parts[2]))
                note = " ".join(parts[3:]).strip()
            except ValueError:
                pass
        status_map = {
            "/approve": "approved",
            "/reject": "rejected",
            "/block": "blocked",
            "/unblock": "pending",
        }
        ok = _set_user_status(db_path, user_id, status_map[cmd], admin_id=admin_id, note=note)
        if not ok:
            _send_text(token, admin_id, "Kullanici kaydi yok. Kullanici once bota yazmali.")
            return

        if cmd == "/approve":
            _set_subscription_days(db_path, user_id, plan_days)
            group_raw = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip()
            invite_link = None
            if group_raw and _env_bool("TELEGRAM_AUTO_INVITE_LINK", True):
                invite_link = _create_invite_link(token, int(group_raw))

            approve_msg = _approved_message(invite_link=invite_link)
            row = _get_user(db_path, user_id) or {}
            left = _subscription_days_left(row)
            if left is not None:
                approve_msg += f"\nUyelik suresi: {left} gun"
            if not invite_link:
                approve_msg += "\nGruba alinmak icin bu mesaja geri donus yapin."
            try:
                _send_text(token, user_id, approve_msg)
                _send_text(token, admin_id, f"Onaylandi: {user_id} | plan: {plan_days} gun")
            except Exception as exc:
                _send_text(
                    token,
                    admin_id,
                    f"Onay kaydi yapildi ama kullaniciya mesaj gidemedi: {user_id} | hata: {exc}",
                )
            return

        if cmd == "/reject":
            _send_text(token, user_id, _rejected_message(note=note))
            _send_text(token, admin_id, f"Reddedildi: {user_id}")
            return

        if cmd == "/block":
            _send_text(token, user_id, _blocked_message())
            _send_text(token, admin_id, f"Bloklandi: {user_id}")
            return

        _send_text(token, user_id, "Durumunuz guncellendi. Odeme dekontu ile tekrar yazabilirsiniz.")
        _send_text(token, admin_id, f"Kullanici tekrar beklemeye alindi: {user_id}")
        return

    if cmd == "/ad":
        _send_text(token, admin_id, _build_ad_text())
        return

    _send_text(token, admin_id, _admin_help())


def _handle_user_message(token: str, db_path: Path, user: dict, text: str) -> None:
    user_id = int(user["id"])
    row = _get_user(db_path, user_id)
    status = row["status"] if row else "none"
    draft = _get_played_draft(db_path, user_id)

    if text.startswith("/cancelplayed"):
        _clear_played_draft(db_path, user_id)
        _send_text(token, user_id, "Oynanan kupon giris menusu iptal edildi.")
        return

    if text.startswith("/playedmenu"):
        _set_played_draft(db_path, user_id, step="await_match")
        _send_text(
            token,
            user_id,
            "Oynanan kupon girisi basladi.\n"
            "1/4 Maci gir: Ornek\nChelsea - Leeds United\n"
            "Iptal icin: /cancelplayed",
        )
        return

    if draft is not None and not text.startswith("/"):
        step = str(draft.get("step") or "")
        if step == "await_match":
            if " - " in text:
                home, away = [x.strip() for x in text.split(" - ", 1)]
            elif "-" in text:
                home, away = [x.strip() for x in text.split("-", 1)]
            else:
                _send_text(token, user_id, "Format hatali. Ornek: Chelsea - Leeds United")
                return
            if not home or not away:
                _send_text(token, user_id, "Mac isimleri bos olamaz.")
                return
            _set_played_draft(db_path, user_id, step="await_market", home=home, away=away)
            _send_text(
                token,
                user_id,
                "2/4 Market sec:\n"
                "1) h2h (MS)\n2) totals (Alt/Ust)\n3) btts (KG)\n"
                "Cevap: h2h/totals/btts veya 1/2/3",
            )
            return

        if step == "await_market":
            market_raw = text.strip().lower()
            market_map = {"1": "h2h", "2": "totals", "3": "btts"}
            market_key = market_map.get(market_raw) or _normalize_market_key(market_raw)
            if market_key is None:
                _send_text(token, user_id, "Gecersiz market. h2h, totals, btts veya 1/2/3 yaz.")
                return
            _set_played_draft(
                db_path,
                user_id,
                step="await_bet",
                home=str(draft.get("home") or ""),
                away=str(draft.get("away") or ""),
                market_key=market_key,
            )
            if market_key == "h2h":
                hint = "3/4 Bahis: home / away / draw (veya MS1/MS2/MSX)"
            elif market_key == "totals":
                hint = "3/4 Bahis: Over 2.5 / Under 2.5 (veya Ust 2.5 / Alt 2.5)"
            else:
                hint = "3/4 Bahis: yes/no (veya var/yok)"
            _send_text(token, user_id, hint)
            return

        if step == "await_bet":
            bet_text = _normalize_bet_text_for_market(str(draft.get("market_key") or ""), text)
            _set_played_draft(
                db_path,
                user_id,
                step="await_odd",
                home=str(draft.get("home") or ""),
                away=str(draft.get("away") or ""),
                market_key=str(draft.get("market_key") or ""),
                bet_text=bet_text,
            )
            _send_text(token, user_id, "4/4 Orani gir (ornek: 2.10)")
            return

        if step == "await_odd":
            try:
                odd = float(text.replace(",", "."))
            except ValueError:
                _send_text(token, user_id, "Oran sayi olmali. Ornek: 2.10")
                return
            if odd <= 1.01:
                _send_text(token, user_id, "Oran 1.01'den buyuk olmali.")
                return
            rid = _save_user_played_leg(
                db_path,
                user_id=user_id,
                home=str(draft.get("home") or ""),
                away=str(draft.get("away") or ""),
                market_key=str(draft.get("market_key") or ""),
                bet_text=str(draft.get("bet_text") or ""),
                odd=float(odd),
            )
            _clear_played_draft(db_path, user_id)
            _send_text(
                token,
                user_id,
                f"Oynanan kupon kaydedildi ✅\nID: {rid}\n"
                f"{draft.get('home')} - {draft.get('away')} | {draft.get('market_key')} | {draft.get('bet_text')} @ {odd:.2f}",
            )
            _notify_admins(token, f"Yeni oynanan kupon kaydi: user={user_id} id={rid}")
            return

    if status == "blocked":
        _send_text(token, user_id, _blocked_message())
        return

    if text.startswith("/status"):
        if status == "approved":
            left = _subscription_days_left(row)
            if _is_subscription_active(row):
                _send_text(token, user_id, f"Durum: ONAYLI. VIP aktif. Kalan sure: {left} gun.")
            else:
                _send_text(token, user_id, "Durum: UYELIK SURESI DOLDU. Yenileme icin adminle iletisime gecin.")
        elif status == "pending":
            _send_text(token, user_id, "Durum: BEKLEMEDE. Odeme kontrolu sonrasi onay verilecek.")
        elif status == "rejected":
            _send_text(token, user_id, "Durum: RED. Detay icin adminle iletisime gecin.")
        else:
            _send_text(token, user_id, "Durum: KAYIT YOK. /start ile baslayabilirsiniz.")
        return

    if text.startswith("/mydays"):
        left = _subscription_days_left(row)
        if left is None:
            _send_text(token, user_id, "Uyelik suresi bilgisi bulunamadi.")
        else:
            _send_text(token, user_id, f"Kalan uyelik suresi: {left} gun")
        return

    if text.startswith("/played "):
        parsed = _parse_played_command(text)
        if not parsed:
            _send_text(
                token,
                user_id,
                "Format hatali. Ornek:\n"
                "/played Chelsea - Leeds United ; totals ; Over 2.5 ; 2.10\n"
                "market: h2h / totals / btts",
            )
            return
        rid = _save_user_played_leg(
            db_path,
            user_id=user_id,
            home=parsed["home"],
            away=parsed["away"],
            market_key=parsed["market_key"],
            bet_text=parsed["bet_text"],
            odd=float(parsed["odd"]),
        )
        _send_text(
            token,
            user_id,
            f"Oynanan kupon kaydedildi. ID: {rid}\n"
            f"{parsed['home']} - {parsed['away']} | {parsed['market_key']} | {parsed['bet_text']} @ {parsed['odd']:.2f}",
        )
        _notify_admins(token, f"Yeni oynanan kupon kaydi: user={user_id} id={rid}")
        return

    if text.startswith("/playedstats"):
        cards = _build_user_played_cards(db_path, limit=30)
        if not cards:
            _send_text(token, user_id, "Henuz sonuclanan oynanan kupon kaydin yok.")
        else:
            for c in cards:
                _send_text(token, user_id, c)
        return

    if text.startswith("/scoreboard"):
        db_main = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
        for card in _build_scoreboard_cards(db_main):
            _send_text(token, user_id, card)
        return

    if text.startswith("/payinfo"):
        _send_text(token, user_id, _payment_text())
        return

    if text.startswith("/help"):
        _send_text(
            token,
            user_id,
            "Komutlar:\n/start\n/status\n/mydays\n/payinfo\n"
            "/played <home-away ; market ; bet ; odd>\n/playedmenu\n/cancelplayed\n/playedstats\n/scoreboard\n/help",
        )
        return

    if text.startswith("/start"):
        _upsert_user(db_path, user, "pending", note="/start")
        welcome = _welcome_message()
        _send_text(token, user_id, welcome)
        _notify_admins(
            token,
            f"Yeni basvuru: id={user_id} @{user.get('username','-')}\n"
            f"Onay: /approve {user_id}\nReddet: /reject {user_id}",
        )
        return

    if status == "approved" and _is_subscription_active(row):
        _send_text(token, user_id, "Uyeliginiz aktif. Gunluk kuponlari ozel gruptan takip edin.")
        return
    if status == "approved":
        _send_text(token, user_id, "Uyelik sureniz dolmus gorunuyor. Yenileme icin adminle iletisime gecin.")
        return

    _upsert_user(db_path, user, "pending", note=text[:200])
    _send_text(token, user_id, "Mesajiniz alindi. Odeme kontrolunden sonra onay verilecektir.")
    _notify_admins(
        token,
        f"Odeme bildirimi: id={user_id} @{user.get('username','-')}\n"
        f"Mesaj: {text[:250]}\n"
        f"Onay: /approve {user_id}\nReddet: /reject {user_id}",
    )


def run_access_bot() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN gerekli.")
    if not _admin_ids():
        raise RuntimeError("TELEGRAM_ADMIN_IDS gerekli. Ornek: 12345,67890")

    access_db = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
    _init_access_db(access_db)

    last_update = int(_get_state(access_db, "last_update_id", "0") or "0")
    _log(f"Access bot basladi. son_update={last_update}")

    while True:
        try:
            resp = _telegram_api(
                token,
                "getUpdates",
                data={"offset": str(last_update + 1), "timeout": "30", "allowed_updates": '["message"]'},
            )
            updates = resp.get("result", [])
            for upd in updates:
                update_id = int(upd.get("update_id", 0))
                last_update = max(last_update, update_id)
                _process_telegram_update(token, access_db, upd)
            _set_state(access_db, "last_update_id", str(last_update))
        except Exception as exc:
            _log(f"Access bot dongu hatasi: {exc}")
            time.sleep(3)


def _process_telegram_update(token: str, access_db: Path, upd: dict) -> None:
    msg = upd.get("message")
    if not msg:
        return
    chat = msg.get("chat", {})
    if chat.get("type") != "private":
        return
    user = msg.get("from", {})
    if not user or user.get("is_bot"):
        return
    text = (msg.get("text") or "").strip()
    if not text:
        return
    uid = int(user.get("id"))
    cmd = text.split()[0].lower() if text else ""
    user_flow_cmds = {
        "/start",
        "/status",
        "/mydays",
        "/payinfo",
        "/played",
        "/playedmenu",
        "/cancelplayed",
        "/playedstats",
        "/help",
    }
    if _is_admin(uid) and text.startswith("/") and cmd not in user_flow_cmds:
        _handle_admin_command(token, access_db, uid, text)
    else:
        _handle_user_message(token, access_db, user, text)


def _target_group_id() -> int:
    group_raw = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip() or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not group_raw:
        raise RuntimeError("TELEGRAM_PRIVATE_GROUP_ID (veya TELEGRAM_CHAT_ID) gerekli.")
    return int(group_raw)


def _send_weekly_now(token: str) -> None:
    db_path = Path(os.getenv("DB_PATH", "data/coupon_history.db")).expanduser()
    gid = _target_group_id()
    for card in _build_weekly_success_cards(db_path):
        _send_text(token, gid, card)


def _check_internal_auth(path: str, headers) -> bool:
    internal_key = os.getenv("INTERNAL_API_KEY", "").strip()
    if not internal_key:
        return True
    parsed = urlparse(path)
    qs = parse_qs(parsed.query)
    key_q = (qs.get("key") or [""])[0]
    key_h = headers.get("X-Api-Key", "")
    return internal_key in {key_q, key_h}


def run_webhook_server() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN gerekli.")
    if not _admin_ids():
        raise RuntimeError("TELEGRAM_ADMIN_IDS gerekli.")

    access_db = Path(os.getenv("TELEGRAM_ACCESS_DB", "data/telegram_access.db")).expanduser()
    _init_access_db(access_db)
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    port = int(os.getenv("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: str = "ok") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/health"}:
                self._reply(200, "alive")
                return
            self._reply(404, "not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/telegram/webhook":
                if secret:
                    hdr = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
                    if hdr != secret:
                        self._reply(403, "forbidden")
                        return
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    upd = json.loads(raw.decode("utf-8"))
                    _process_telegram_update(token, access_db, upd)
                    self._reply(200, "ok")
                except Exception as exc:
                    _log(f"Webhook update hatasi: {exc}")
                    self._reply(500, "error")
                return

            if parsed.path == "/internal/sendnow":
                if not _check_internal_auth(self.path, self.headers):
                    self._reply(403, "forbidden")
                    return
                try:
                    run_once()
                    self._reply(200, "sent")
                except Exception as exc:
                    _log(f"internal sendnow hatasi: {exc}")
                    self._reply(500, f"error: {exc}")
                return

            if parsed.path == "/internal/weeklynow":
                if not _check_internal_auth(self.path, self.headers):
                    self._reply(403, "forbidden")
                    return
                try:
                    _send_weekly_now(token)
                    self._reply(200, "weekly-sent")
                except Exception as exc:
                    _log(f"internal weeklynow hatasi: {exc}")
                    self._reply(500, f"error: {exc}")
                return

            self._reply(404, "not found")

        def log_message(self, format, *args):
            return

    _log(f"Webhook server basladi: 0.0.0.0:{port}")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.serve_forever()


def set_webhook() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN gerekli.")
    if not url:
        raise RuntimeError("TELEGRAM_WEBHOOK_URL gerekli. Ornek: https://app.onrender.com/telegram/webhook")

    data = {"url": url}
    if secret:
        data["secret_token"] = secret
    _telegram_api(token, "setWebhook", data=data)
    _log(f"Webhook ayarlandi: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bet AI Odds Telegram bot")
    parser.add_argument(
        "--mode",
        choices=["once", "daily", "bot", "webhook", "setwebhook"],
        default="once",
        help="once: tek gonderim, daily: zamanli gonderim, bot: polling, webhook: HTTP server, setwebhook: Telegram webhook ayarla",
    )
    args = parser.parse_args()

    if args.mode == "daily":
        run_daily_loop()
    elif args.mode == "bot":
        run_access_bot()
    elif args.mode == "webhook":
        run_webhook_server()
    elif args.mode == "setwebhook":
        set_webhook()
    else:
        run_once()


if __name__ == "__main__":
    main()

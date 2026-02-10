import argparse
import html
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    format_kickoff,
    run_coupon_engine,
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
        "✅ <b>SISTEM KANIT PANELI | SONUC RAPORU</b>\n"
        f"📚 <b>Incelenen Kupon Bacagi:</b> <code>{int(stats['n'])}</code>\n"
        f"🏁 <b>Kazanan:</b> <code>{int(stats['wins'])}</code>\n"
        f"🎯 <b>Tutma Orani:</b> <code>%{stats['hit']*100:.2f}</code>\n"
        f"💰 <b>Kazanan Ortalama Oran:</b> <code>{stats['avg_odd_win']:.2f}</code>\n"
        f"{line}"
    )
    cards.append(head + f"\n{sep}")

    winners = [r for r in rows if int(r.get("won", 0) or 0) == 1][:8]
    if winners:
        lines = ["🏆 <b>SON KAZANANLAR</b>", line]
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
        cards.append("🏆 <b>SON KAZANANLAR</b>\nBu periyotta kazanan bacak yok.")

    high_odds_hits = [r for r in rows if int(r.get("won", 0) or 0) == 1 and float(r.get("odd_open") or 0) >= 2.20][:6]
    if high_odds_hits:
        lines = ["🚀 <b>YUKSEK ORAN ISABETLERI</b>", line]
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
        "Bu panel sonuçlanan bahis performansını (MS / Alt-Ust / KG) şeffaf şekilde gösterir."
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
    conn.commit()
    conn.close()


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
                request_note, requested_at, approved_at, approved_by, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (
                int(user["id"]),
                user.get("username", ""),
                user.get("first_name", ""),
                user.get("last_name", ""),
                status,
                note,
                requested_at or now,
                now,
            ),
        )
    conn.commit()
    conn.close()


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
    return (
        f"Uyelik ucreti: {fee} TL\n"
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
    db_raw = str(payload.get("_db_path", "")).strip()
    if include_proof and db_raw:
        for card in _build_proof_cards(Path(db_raw)):
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
        "/reject <user_id> [not]\n"
        "/block <user_id> [not]\n"
        "/unblock <user_id>\n"
        "/who <user_id>\n"
        "/sendnow\n"
        "/weeklynow\n"
        "/templates\n"
        "/ad"
    )


def _render_user_row(row: dict) -> str:
    uname = f"@{row.get('username')}" if row.get("username") else "-"
    full_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
    full_name = full_name or "-"
    return f"id={row['user_id']} | {uname} | {full_name} | durum={row['status']}"


def _handle_admin_command(token: str, db_path: Path, admin_id: int, text: str) -> None:
    parts = text.strip().split()
    cmd = parts[0].lower()

    if cmd in {"/help", "/admin"}:
        _send_text(token, admin_id, _admin_help())
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
        for card in _build_weekly_success_cards(db_path):
            _send_text(token, int(target_group), card)
        _send_text(token, admin_id, "Haftalik rapor gruba gonderildi.")
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
            f"onaylayan: {row.get('approved_by') or '-'}"
        )
        _send_text(token, admin_id, msg)
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
            group_raw = os.getenv("TELEGRAM_PRIVATE_GROUP_ID", "").strip()
            invite_link = None
            if group_raw and _env_bool("TELEGRAM_AUTO_INVITE_LINK", True):
                invite_link = _create_invite_link(token, int(group_raw))

            approve_msg = _approved_message(invite_link=invite_link)
            if not invite_link:
                approve_msg += "\nGruba alinmak icin bu mesaja geri donus yapin."
            _send_text(token, user_id, approve_msg)
            _send_text(token, admin_id, f"Onaylandi: {user_id}")
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

    if status == "blocked":
        _send_text(token, user_id, _blocked_message())
        return

    if text.startswith("/status"):
        if status == "approved":
            _send_text(token, user_id, "Durum: ONAYLI. VIP icerik erisiminiz aktif.")
        elif status == "pending":
            _send_text(token, user_id, "Durum: BEKLEMEDE. Odeme kontrolu sonrasi onay verilecek.")
        elif status == "rejected":
            _send_text(token, user_id, "Durum: RED. Detay icin adminle iletisime gecin.")
        else:
            _send_text(token, user_id, "Durum: KAYIT YOK. /start ile baslayabilirsiniz.")
        return

    if text.startswith("/help"):
        _send_text(token, user_id, "Komutlar:\n/start\n/status\n/help")
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

    if status == "approved":
        _send_text(token, user_id, "Uyeliginiz aktif. Gunluk kuponlari ozel gruptan takip edin.")
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
                msg = upd.get("message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                if chat.get("type") != "private":
                    continue
                user = msg.get("from", {})
                if not user or user.get("is_bot"):
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                uid = int(user.get("id"))
                cmd = text.split()[0].lower() if text else ""
                user_flow_cmds = {"/start", "/status", "/help"}
                if _is_admin(uid) and text.startswith("/") and cmd not in user_flow_cmds:
                    _handle_admin_command(token, access_db, uid, text)
                else:
                    _handle_user_message(token, access_db, user, text)

            _set_state(access_db, "last_update_id", str(last_update))
        except Exception as exc:
            _log(f"Access bot dongu hatasi: {exc}")
            time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bet AI Odds Telegram bot")
    parser.add_argument(
        "--mode",
        choices=["once", "daily", "bot"],
        default="once",
        help="once: kuponu bir kere gonder, daily: her gun gonder, bot: odeme/onay botu",
    )
    args = parser.parse_args()

    if args.mode == "daily":
        run_daily_loop()
    elif args.mode == "bot":
        run_access_bot()
    else:
        run_once()


if __name__ == "__main__":
    main()

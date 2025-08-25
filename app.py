# -*- coding: utf-8 -*-
# file: app.py
"""PullMyBallsLotto - Lottery Stats Explorer (Educational)

Powerball-only build. Includes:
- Power Play + EV (uniform/weighted/fixed), 10x cap logic
- Pick Builder UI + CSV import/export
- History-weighted Auto-pick (hot/cold, optional recency window)
- Best-effort jackpot scrape + latest draw detail with BeautifulSoup fallback
- Pyodide-safe timezone bootstrap and fixed-offset fallback (-05:00) if tzdata missing
- CLI demo fallback when Streamlit isn't available

This app does NOT predict results. Educational use only. Not affiliated with any lottery commission.
"""

from __future__ import annotations

# allow `import app` to resolve to this module when run as a single file/REPL
import sys as _app_sys
_app_sys.modules.setdefault("app", _app_sys.modules[__name__])

import itertools
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Optional matplotlib (app still runs if missing)
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
    MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    MPL_AVAILABLE = False

# Optional BeautifulSoup (fallback available)
try:  # pragma: no cover
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False

# Streamlit optional at import time
try:  # pragma: no cover
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st = None  # type: ignore

# -------- Timezone bootstrap (Pyodide/tzdata safe) --------

def _ensure_tzdata_loaded_for_pyodide() -> None:
    """Load tzdata when running under Pyodide so ZoneInfo works (best-effort)."""
    try:
        import tzdata  # noqa: F401
        return
    except Exception:
        pass
    try:
        import sys
        if sys.platform == "emscripten":
            import asyncio  # type: ignore
            import js  # type: ignore

            async def _load():
                await js.pyodide.loadPackage("tzdata")

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if not loop.is_running():
                loop.run_until_complete(_load())
    except Exception:
        pass


def _get_et_zone():
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        _ensure_tzdata_loaded_for_pyodide()
        try:
            return ZoneInfo("America/New_York")
        except Exception:
            # Fallback: fixed -05:00 (standard time), avoids crashes in restricted envs
            return timezone(timedelta(hours=-5))


# ---------------------- Globals ----------------------
ET = _get_et_zone()
DRAW_DAYS_PB = {0, 2, 5}  # Mon, Wed, Sat
DRAW_HOUR, DRAW_MIN = 22, 59

MATRIX_CUTOFF_PB = pd.to_datetime("2015-10-04").date()

POWERBALL_HOME = "https://www.powerball.com/"
PB_PREV_RESULTS = "https://www.powerball.com/previous-results"
TX_PB_CSV = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"
NY_PB_CSV = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
FL_PB = "https://floridalottery.com/games/draw-games/powerball"

# Optional Gumroad secrets
PRODUCT_IDS_RAW = ""
PRODUCT_IDS: List[str] = []
DEMO_ONLY = False
if st is not None:  # pragma: no cover
    try:
        PRODUCT_IDS_RAW = (st.secrets.get("GUMROAD_PRODUCT_IDS", "") or "").strip()
        PRODUCT_IDS = [p.strip() for p in PRODUCT_IDS_RAW.split(",") if p.strip()]
        DEMO_ONLY = (st.secrets.get("DEMO_ONLY", "false") or "false").lower() == "true"
    except Exception:
        pass

# HTTP session
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
})
HTTP_TIMEOUT = 15

# -------- Minimal HTML->text fallback when BeautifulSoup isn't available --------

def _html_to_text(html: str) -> str:
    try:
        # cheap tag stripper; keeps line breaks to help regex parsing
        return re.sub(r'<[^>]+>', '\\n', html)  # literal backslash+n
    except Exception:
        return html


# ---------------------- License ----------------------

def verify_gumroad_license(license_key: str) -> bool:
    if not PRODUCT_IDS or not license_key:
        return False
    try:
        for pid in PRODUCT_IDS:
            resp = requests.post(
                "https://api.gumroad.com/v2/licenses/verify",
                data={"product_id": pid, "license_key": license_key.strip()},
                timeout=15,
            )
            js = resp.json()
            if bool(js.get("success")) and not js.get("purchase", {}).get("refunded", False):
                return True
    except Exception:
        return False
    return False


# ---------------------- Prizes ----------------------
POWERBALL_PRIZES: Dict[Tuple[int, bool], int | str] = {
    (5, True): "JACKPOT",
    (5, False): 1_000_000,
    (4, True): 50_000,
    (4, False): 100,
    (3, True): 100,
    (3, False): 7,
    (2, True): 7,
    (1, True): 4,
    (0, True): 4,
}


def prize_for_result(match_white: int, pb_matched: bool, jackpot_estimate: Optional[int]) -> int:
    val = POWERBALL_PRIZES.get((match_white, pb_matched), 0)
    if val == "JACKPOT":
        return int(jackpot_estimate or 0)
    return int(val)


# ---------------------- Helpers ----------------------

def to_table(counter: Counter, domain_range: Iterable[int]) -> pd.DataFrame:
    total = sum(counter.get(n, 0) for n in domain_range)
    rows = []
    for n in domain_range:
        c = int(counter.get(n, 0))
        pct = (100.0 * c / total) if total else 0.0
        rows.append({"number": n, "count": c, "percent": round(pct, 3)})
    return pd.DataFrame(rows)


def freq_counts(df: pd.DataFrame) -> Tuple[Counter, Counter]:
    whites = list(itertools.chain.from_iterable(df.get("W", []).tolist())) if not df.empty else []
    reds = df.get("R", pd.Series([], dtype=int)).tolist() if not df.empty else []
    return Counter(whites), Counter(reds)


def pairs_triplets(df: pd.DataFrame) -> Tuple[Counter, Counter]:
    pair_counts, trip_counts = Counter(), Counter()
    for ws in df.get("W", []):
        s = sorted(ws)
        for a, b in itertools.combinations(s, 2):
            pair_counts[(a, b)] += 1
        for a, b, c in itertools.combinations(s, 3):
            trip_counts[(a, b, c)] += 1
    return pair_counts, trip_counts


def pair_matrix(pair_counts: Counter, max_white: int) -> np.ndarray:
    mat = np.zeros((max_white, max_white), dtype=int)
    for (a, b), cnt in pair_counts.items():
        if 1 <= a <= max_white and 1 <= b <= max_white:
            mat[a - 1, b - 1] = cnt
            mat[b - 1, a - 1] = cnt
    return mat


# ---------------------- Time ----------------------

def next_powerball_draw(now: Optional[datetime] = None) -> datetime:
    if now is None:
        now = datetime.now(ET)
    d = now
    for _ in range(8):
        draw_dt = datetime(d.year, d.month, d.day, DRAW_HOUR, DRAW_MIN, tzinfo=ET)
        if d.weekday() in DRAW_DAYS_PB and now <= draw_dt:
            return draw_dt
        d += timedelta(days=1)
    fallback = now + timedelta(days=2)
    return fallback.replace(hour=DRAW_HOUR, minute=DRAW_MIN, second=0, microsecond=0)


# ---------------------- Scrape ----------------------

def get_powerball_jackpot_estimate() -> Optional[int]:
    try:
        html = SESSION.get(POWERBALL_HOME, timeout=HTTP_TIMEOUT).text
        m = re.search(r"Estimated Jackpot[^$]*\$\s*([\d,.]+)\s*(Million|Billion)", html, re.I)
        if not m:
            return None
        num = float(m.group(1).replace(",", ""))
        mult = 1_000_000_000 if m.group(2).lower().startswith("b") else 1_000_000
        return int(num * mult)
    except Exception:
        return None


@dataclass
class LatestDraw:
    date_text: Optional[str]
    est_jackpot_str: Optional[str]
    cash_value_str: Optional[str]
    states_match5: List[str]
    tiers: List[Dict[str, object]]
    detail_url: Optional[str]


def get_latest_powerball_draw_detail() -> Optional[LatestDraw]:
    try:
        listing_html = SESSION.get(PB_PREV_RESULTS, timeout=HTTP_TIMEOUT).text
    except Exception:
        return None

    detail_url: Optional[str] = None
    try:
        m = re.search(r'href="(/draw-result\?date=\d{4}-\d{2}-\d{2}&gc=powerball)"', listing_html)
        if m:
            detail_url = "https://www.powerball.com" + m.group(1)
    except Exception:
        detail_url = None

    if not detail_url:
        return None

    try:
        dhtml = SESSION.get(detail_url, timeout=HTTP_TIMEOUT).text
        if BeautifulSoup is not None:
            soup = BeautifulSoup(dhtml, "html.parser")
            page_txt = soup.get_text("\n", strip=True)
        else:
            soup = None
            page_txt = _html_to_text(dhtml)
    except Exception:
        return None

    date_text: Optional[str] = None
    try:
        date_pat = re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\b")
        if 'soup' in locals() and soup is not None:
            for tag in ("h1", "h2", "h3", "h4", "h5"):
                h = soup.find(tag)
                if h:
                    ht = h.get_text(" ", strip=True)
                    mdate = date_pat.search(ht)
                    if mdate:
                        date_text = mdate.group(0)
                        break
        if not date_text:
            mdate = date_pat.search(page_txt)
            if mdate:
                date_text = mdate.group(0)
    except Exception:
        date_text = None

    est_jackpot_text = None
    cash_value_text = None
    try:
        mj = re.search(r"Estimated Jackpot:\s*\$[^\n]+", page_txt, re.I)
        if mj:
            est_jackpot_text = mj.group(0).split(":", 1)[1].strip()
        mc = re.search(r"Cash Value:\s*\$[^\n]+", page_txt, re.I)
        if mc:
            cash_value_text = mc.group(0).split(":", 1)[1].strip()
    except Exception:
        pass

    states_match5: List[str] = []
    try:
        sm = re.search(r"Match\s*5\s*\$?1\s*Million Winners\s*([A-Z ,]+)", page_txt, re.I)
        if sm:
            states_match5 = [s.strip() for s in sm.group(1).split(",") if s.strip()]
    except Exception:
        states_match5 = []

    tiers: List[Dict[str, object]] = []
    try:
        gp = re.search(r"\b(\d+)\s+Grand Prize\b", page_txt, re.I)
        if gp:
            tiers.append({
                "label": "5+PB",
                "pb_winners": int(gp.group(1)),
                "pb_prize": "Jackpot",
                "pp_winners": None,
                "pp_prize": None,
            })
        for m2 in re.finditer(r"\n\s*(\d+)\s+\$([\d,]+)\s+(\d+)\s+\$([\d,]+)", page_txt):
            pbw, pbp, ppw, ppp = m2.groups()
            tiers.append({
                "label": "(tier)",
                "pb_winners": int(pbw),
                "pb_prize": f"${pbp}",
                "pp_winners": int(ppw),
                "pp_prize": f"${ppp}",
            })
    except Exception:
        tiers = []

    return LatestDraw(
        date_text=date_text,
        est_jackpot_str=est_jackpot_text,
        cash_value_str=cash_value_text,
        states_match5=states_match5,
        tiers=tiers,
        detail_url=detail_url,
    )


# ---------------------- Data loaders ----------------------

if st is not None:  # pragma: no cover
    cache_data = st.cache_data
else:  # testing/CLI fallback
    def cache_data(*dargs, **dkwargs):
        def deco(fn):
            return fn
        return deco


def _parse_whites_red_from_text(s: str) -> Optional[Tuple[List[int], int]]:
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) < 6:
        return None
    whites, red = nums[:5], int(nums[5])
    return whites, red


def _normalize_df(df: pd.DataFrame, white_max: int, red_max: int, cutoff_date: datetime.date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["draw_date"] = pd.to_datetime(df["date"]).dt.date
    df["W"] = df[["w1", "w2", "w3", "w4", "w5"]].values.tolist()
    df["R"] = df["r"].astype(int)
    df = df[df["draw_date"] >= cutoff_date]
    df = df[
        (df["W"].apply(lambda ws: len(ws) == 5 and all(1 <= x <= white_max for x in ws)))
        & (df["R"].between(1, red_max))
    ]
    return df.sort_values("draw_date").reset_index(drop=True)


@cache_data(ttl=600, show_spinner=False)
def load_powerball() -> pd.DataFrame:
    """Load Powerball draws (5 whites 1..69, PB 1..26). TX CSV -> NY Open Data fallback."""
    rows: List[Dict[str, int | datetime.date]] = []

    # Try Texas CSV (schema may vary)
    try:
        tx = pd.read_csv(TX_PB_CSV, dtype=str, engine="python", on_bad_lines="skip")
        date_col = None
        num_col = None
        for c in tx.columns:
            if re.search(r"draw\s*date", str(c), re.I):
                date_col = c
            if re.search(r"winning\s*numbers", str(c), re.I):
                num_col = c
        if date_col is not None and num_col is not None:
            for _, rec in tx.iterrows():
                try:
                    d = pd.to_datetime(str(rec[date_col])).date()
                    parsed = _parse_whites_red_from_text(str(rec[num_col]))
                    if not parsed:
                        continue
                    whites, red = parsed
                    rows.append({
                        "date": d,
                        "w1": whites[0], "w2": whites[1], "w3": whites[2], "w4": whites[3], "w5": whites[4],
                        "r": red
                    })
                except Exception:
                    continue
    except Exception:
        pass

    # Fallback NY Open Data
    if not rows:
        try:
            ny = pd.read_csv(NY_PB_CSV)
            for _, rec in ny.iterrows():
                try:
                    d = pd.to_datetime(str(rec.get("Draw Date") or rec.get("draw_date"))).date()
                except Exception:
                    continue
                wn = str(rec.get("Winning Numbers") or rec.get("winning_numbers") or "")
                parsed = _parse_whites_red_from_text(wn)
                if not parsed:
                    continue
                whites, red = parsed
                rows.append({
                    "date": d,
                    "w1": whites[0], "w2": whites[1], "w3": whites[2], "w4": whites[3], "w5": whites[4],
                    "r": red
                })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    return _normalize_df(df, white_max=69, red_max=26, cutoff_date=MATRIX_CUTOFF_PB)


# ---------------------- Weights (from history) ----------------------

def make_weight_arrays(
    df: pd.DataFrame,
    white_max: int = 69,
    red_max: int = 26,
    smoothing: float = 1.0,
    *,
    cold: bool = False,
    recency_n: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build probability arrays from historical frequencies with Laplace smoothing.
    If `recency_n > 0`, only the last N draws are used. If `cold=True`, invert
    frequencies to favor least-drawn numbers. Falls back to uniform.
    """
    src = df
    if not df.empty and recency_n and recency_n > 0:
        src = df.tail(int(recency_n))
    w = np.ones(white_max, dtype=float) * smoothing
    r = np.ones(red_max, dtype=float) * smoothing
    if not src.empty and "W" in src.columns and "R" in src.columns:
        wcnt, rcnt = freq_counts(src)
        w_raw = np.zeros(white_max, dtype=float)
        r_raw = np.zeros(red_max, dtype=float)
        for i in range(white_max):
            w_raw[i] = float(wcnt.get(i + 1, 0))
        for i in range(red_max):
            r_raw[i] = float(rcnt.get(i + 1, 0))
        if cold:
            w_max = float(w_raw.max()) if w_raw.size else 0.0
            r_max = float(r_raw.max()) if r_raw.size else 0.0
            w += (w_max - w_raw)
            r += (r_max - r_raw)
        else:
            w += w_raw
            r += r_raw
    ws = w.sum()
    rs = r.sum()
    w = w / ws if ws > 0 else np.full(white_max, 1.0 / white_max)
    r = r / rs if rs > 0 else np.full(red_max, 1.0 / red_max)
    return w, r


# ---------------------- Simulation ----------------------

@dataclass(frozen=True)
class Pick:
    whites: Tuple[int, int, int, int, int]
    red: int

    def normalized(self) -> "Pick":
        return Pick(tuple(sorted(self.whites)), int(self.red))


# Official-ish Power Play weights (approx.)
POWER_PLAY_10X_CAP = 150_000_000  # 10x available only when jackpot <= cap
PP_WEIGHTS_WITH_10X = {2: 0.557, 3: 0.303, 4: 0.070, 5: 0.047, 10: 0.023}
PP_WEIGHTS_NO_10X   = {2: 0.571, 3: 0.309, 4: 0.069, 5: 0.051}


def _score_pick(draw_w: Sequence[int], draw_r: int, pick: Pick, jackpot_estimate: Optional[int]) -> Tuple[Tuple[int, bool], int]:
    w_matches = len(set(draw_w).intersection(pick.whites))
    r_match = draw_r == pick.red
    prize = prize_for_result(w_matches, r_match, jackpot_estimate)
    return (w_matches, r_match), prize


def _effective_allow_10x(pp_allow_10x: bool, pp_auto_10x_cap: bool, jackpot_estimate: Optional[int], cap: int) -> bool:
    if not pp_allow_10x:
        return False
    if not pp_auto_10x_cap:
        return True
    return (jackpot_estimate or 0) <= cap


def _draw_power_play_multiplier(
    rng: np.random.Generator,
    *,
    allow_10x: bool,
    mode: str,
    fixed: int,
    weights_with_10x: Dict[int, float],
    weights_no_10x: Dict[int, float],
) -> int:
    options = [2, 3, 4, 5] + ([10] if allow_10x else [])
    if mode in {"uniform", "random"}:
        return int(rng.choice(options))
    if mode == "fixed":
        return int(max(2, min(10, fixed)))
    # weighted
    weights = dict(weights_with_10x if allow_10x else weights_no_10x)
    ps = [weights[o] for o in options]
    total = float(sum(ps))
    if total <= 0:
        return 2
    ps = [p / total for p in ps]
    return int(rng.choice(options, p=ps))


def _apply_power_play(prize: int, tier: Tuple[int, bool], multiplier: int) -> int:
    w_matches, r_match = tier
    if w_matches == 5 and not r_match:
        return 2_000_000
    if w_matches == 5 and r_match:
        return prize
    return int(prize * multiplier)


def _generate_pick_uniform(rng: np.random.Generator, white_max: int, red_max: int) -> Tuple[List[int], int]:
    whites = sorted(rng.choice(np.arange(1, white_max + 1), size=5, replace=False).tolist())
    red = int(rng.integers(1, red_max + 1))
    return whites, red


def _generate_pick_weighted(
    rng: np.random.Generator,
    white_max: int,
    red_max: int,
    w_p: Optional[np.ndarray],
    r_p: Optional[np.ndarray],
) -> Tuple[List[int], int]:
    if w_p is None or len(w_p) != white_max:
        whites = sorted(rng.choice(np.arange(1, white_max + 1), size=5, replace=False).tolist())
    else:
        whites = sorted(rng.choice(np.arange(1, white_max + 1), size=5, replace=False, p=w_p).tolist())
    if r_p is None or len(r_p) != red_max:
        red = int(rng.integers(1, red_max + 1))
    else:
        red = int(rng.choice(np.arange(1, red_max + 1), p=r_p))
    return whites, red


def simulate_strategy(
    picks: Sequence[Tuple[Sequence[int], int]] | Sequence[Pick],
    *,
    draws: int = 10_000,
    white_max: int = 69,
    red_max: int = 26,
    jackpot_estimate: Optional[int] = None,
    seed: Optional[int] = None,
    power_play: bool = False,
    pp_mode: str = "uniform",  # "uniform" | "weighted" | "fixed"
    pp_fixed_multiplier: int = 2,
    pp_allow_10x: bool = True,
    pp_auto_10x_cap: bool = True,
    pp_10x_cap_amount: int = POWER_PLAY_10X_CAP,
    ticket_price: int = 2,
    ticket_price_power_play: int = 3,
    # auto-pick options
    auto_pick: bool = False,
    auto_pick_n: int = 1,
    auto_pick_weighting: str = "uniform",  # "uniform" | "historical"
    auto_pick_each_draw: bool = False,
    white_weights: Optional[np.ndarray] = None,
    red_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if draws <= 0:
        return {"draws": 0, "by_pick": [], "overall": {"tier_counts": {}, "total_prize": 0}}

    rng = np.random.default_rng(seed)

    # Resolve picks list
    norm_picks: List[Pick] = []
    if auto_pick:
        if auto_pick_each_draw:
            norm_picks = []  # generated per draw
        else:
            for _ in range(max(auto_pick_n, 1)):
                if auto_pick_weighting == "historical":
                    whites, red = _generate_pick_weighted(rng, white_max, red_max, white_weights, red_weights)
                else:
                    whites, red = _generate_pick_uniform(rng, white_max, red_max)
                norm_picks.append(Pick(tuple(sorted(int(x) for x in whites)), int(red)))
    else:
        for p in picks:
            if isinstance(p, Pick):
                norm_picks.append(p.normalized())
            else:
                whites, red = p
                w = tuple(sorted(int(x) for x in whites))
                if len(w) != 5 or len(set(w)) != 5 or not all(1 <= x <= white_max for x in w):
                    raise ValueError("Each pick must have 5 unique white balls within range.")
                r = int(red)
                if not 1 <= r <= red_max:
                    raise ValueError("Red ball out of range.")
                norm_picks.append(Pick(w, r))

    overall_tier_counts: Dict[Tuple[int, bool], int] = {}
    overall_total_prize = 0

    eff_allow_10x = _effective_allow_10x(pp_allow_10x, pp_auto_10x_cap, jackpot_estimate, pp_10x_cap_amount)
    eff_mode = "uniform" if pp_mode == "random" else pp_mode

    for _ in range(draws):
        # Generate draw
        draw_w = tuple(sorted(rng.choice(np.arange(1, white_max + 1), size=5, replace=False).tolist()))
        draw_r = int(rng.integers(1, red_max + 1))

        # Generate auto picks per draw if requested
        if auto_pick and auto_pick_each_draw:
            current_picks: List[Pick] = []
            for _n in range(max(auto_pick_n, 1)):
                if auto_pick_weighting == "historical":
                    whites, red = _generate_pick_weighted(rng, white_max, red_max, white_weights, red_weights)
                else:
                    whites, red = _generate_pick_uniform(rng, white_max, red_max)
                current_picks.append(Pick(tuple(sorted(int(x) for x in whites)), int(red)))
            picks_iter = current_picks
        else:
            picks_iter = norm_picks

        # Determine PP multiplier for this draw
        if power_play:
            draw_multiplier = _draw_power_play_multiplier(
                rng,
                allow_10x=eff_allow_10x,
                mode=eff_mode,
                fixed=pp_fixed_multiplier,
                weights_with_10x=PP_WEIGHTS_WITH_10X,
                weights_no_10x=PP_WEIGHTS_NO_10X,
            )
        else:
            draw_multiplier = 1

        for pick in picks_iter:
            tier, prize = _score_pick(draw_w, draw_r, pick, jackpot_estimate)
            if power_play:
                prize = _apply_power_play(prize, tier, draw_multiplier)
            overall_tier_counts[tier] = overall_tier_counts.get(tier, 0) + 1
            overall_total_prize += prize

    num_picks = max((auto_pick_n if auto_pick and auto_pick_each_draw else len(norm_picks)) or 1, 1)
    gross_ev_per_draw = float(overall_total_prize) / float(draws * num_picks)
    cost_per_play = float(ticket_price_power_play if power_play else ticket_price)
    net_ev_per_draw = gross_ev_per_draw - cost_per_play

    by_pick_payload = (
        [] if (auto_pick and auto_pick_each_draw) else [{"pick": {"W": list(p.whites), "R": p.red}} for p in norm_picks]
    )

    result = {
        "draws": int(draws),
        "by_pick": by_pick_payload,
        "overall": {
            "tier_counts": {f"{k[0]}+{'PB' if k[1] else 'noPB'}": v for k, v in overall_tier_counts.items()},
            "total_prize": int(overall_total_prize),
            "gross_ev_per_draw": gross_ev_per_draw,
            "cost_per_play": cost_per_play,
            "net_ev_per_draw": net_ev_per_draw,
            "power_play_used": bool(power_play),
            "pp_mode": eff_mode,
            "pp_allow_10x": eff_allow_10x,
            "auto_pick": bool(auto_pick),
            "auto_pick_each_draw": bool(auto_pick_each_draw),
            "auto_pick_n": int(auto_pick_n),
            "auto_pick_weighting": auto_pick_weighting,
        },
    }
    return result


# ---------------------- CSV helpers ----------------------

def simulation_to_csvs(res: Dict[str, object], *, draws: int, num_picks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tiers = res.get("overall", {}).get("tier_counts", {})  # type: ignore[assignment]
    rows = []
    denom = max(draws * max(num_picks, 1), 1)
    for tier, cnt in sorted(tiers.items()):
        p = float(cnt) / float(denom)
        rows.append({"tier": tier, "count": int(cnt), "probability_per_draw": p})
    tiers_df = pd.DataFrame(rows)

    ov = res.get("overall", {})  # type: ignore[assignment]
    metrics_df = pd.DataFrame([{
        "gross_ev_per_draw": ov.get("gross_ev_per_draw", 0.0),
        "cost_per_play": ov.get("cost_per_play", 0.0),
        "net_ev_per_draw": ov.get("net_ev_per_draw", 0.0),
        "power_play_used": ov.get("power_play_used", False),
        "pp_mode": ov.get("pp_mode", ""),
        "pp_allow_10x": ov.get("pp_allow_10x", False),
        "draws": draws,
        "num_picks": num_picks,
    }])
    return tiers_df, metrics_df


# ---------------------- UI helpers ----------------------

def _parse_user_picks(text: str) -> List[Tuple[List[int], int]]:
    picks: List[Tuple[List[int], int]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.replace("|", " ").replace(",", " ")
        nums: List[int] = []
        buf: List[str] = []
        for ch in s:
            if ch.isdigit():
                buf.append(ch)
            else:
                if buf:
                    nums.append(int("".join(buf)))
                    buf = []
        if buf:
            nums.append(int("".join(buf)))
        if len(nums) < 6:
            continue
        whites, red = nums[:5], nums[5]
        picks.append((whites, red))
    return picks


def _format_timedelta(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 0:
        total = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _picks_to_csv_rows(picks: List[Tuple[List[int], int]]) -> pd.DataFrame:
    rows = []
    for whites, r in picks:
        rows.append({"w1": whites[0], "w2": whites[1], "w3": whites[2], "w4": whites[3], "w5": whites[4], "r": r})
    return pd.DataFrame(rows)


def _parse_picks_csv(file) -> List[Tuple[List[int], int]]:
    try:
        df = pd.read_csv(file)
    except Exception:
        return []
    cols = {c.lower(): c for c in df.columns}
    out: List[Tuple[List[int], int]] = []
    if all(k in cols for k in ["w1", "w2", "w3", "w4", "w5", "r"]):
        for _, rec in df.iterrows():
            try:
                whites = [int(rec[cols[f"w{i}"]]) for i in range(1, 6)]
                r = int(rec[cols["r"]])
                out.append((whites, r))
            except Exception:
                continue
    elif "w" in cols and "r" in cols:
        for _, rec in df.iterrows():
            try:
                s = str(rec[cols["w"]])
                nums: List[int] = []
                buf: List[str] = []
                for ch in s:
                    if ch.isdigit():
                        buf.append(ch)
                    else:
                        if buf:
                            nums.append(int("".join(buf)))
                            buf = []
                if buf:
                    nums.append(int("".join(buf)))
                whites = nums[:5]
                r = int(rec[cols["r"]])
                if len(whites) == 5:
                    out.append((whites, r))
            except Exception:
                continue
    elif "pick" in cols:
        for _, rec in df.iterrows():
            out.extend(_parse_user_picks(str(rec[cols["pick"]])))
    return out


def _tz_mode_str() -> str:
    return "IANA America/New_York" if isinstance(ET, ZoneInfo) else "Fixed -05:00 fallback"


# ---------------------- Streamlit App ----------------------

def main() -> None:  # pragma: no cover - UI only
    if st is None:
        # CLI fallback: run a tiny demo instead of raising in non-Streamlit envs
        print("[PullMyBallsLotto] Non-Streamlit environment detected. Running CLI demo…")
        try:
            picks = [([1, 2, 3, 4, 5], 6)]
            res = simulate_strategy(picks, draws=1000, seed=0)
            print({
                "draws": res["draws"],
                "gross_ev_per_draw": res["overall"]["gross_ev_per_draw"],
                "cost_per_play": res["overall"]["cost_per_play"],
                "net_ev_per_draw": res["overall"]["net_ev_per_draw"],
            })
        except Exception as e:
            print("CLI demo failed:", e)
        return

    st.set_page_config(page_title="PullMyBallsLotto - Lottery Stats Explorer", layout="wide")
    st.title("PullMyBallsLotto - Lottery Stats Explorer")
    st.caption("Educational only. Not affiliated with any lottery commission.")

    # Optional license gate (soft)
    colL, colR = st.columns([1, 2])
    with colL:
        license_key = st.text_input("License key (Gumroad)", type="password", help="Optional; unlock Pro if configured.")
        licensed = verify_gumroad_license(license_key) if license_key else False
        if DEMO_ONLY and not licensed:
            st.info("Demo mode: some features may be limited.")
        elif licensed:
            st.success("License verified.")

    with colR:
        nxt = next_powerball_draw()
        remaining = nxt - datetime.now(ET)
        st.metric("Next Powerball draw (ET)", nxt.strftime("%a, %b %d %Y %I:%M %p"), _format_timedelta(remaining))
        try:
            st_autorefresh(interval=60 * 1000, key="autorefresh_minutely")
        except Exception:
            pass

    # Jackpot estimate (best-effort)
    jp = get_powerball_jackpot_estimate()
    if jp:
        st.info(f"Estimated Jackpot: ${jp:,}")

    # Data
    with st.spinner("Loading datasets…"):
        df_pb = load_powerball()
    st.write(f"Powerball draws: **{len(df_pb)}**")

    tab_overview, tab_stats, tab_sim = st.tabs(["Overview", "Stats", "Simulator"])

    with tab_overview:
        ld = get_latest_powerball_draw_detail()
        if ld:
            st.subheader("Latest Powerball Detail (best-effort)")
            st.write({
                "date": ld.date_text,
                "est_jackpot": ld.est_jackpot_str,
                "cash_value": ld.cash_value_str,
                "match5_states": ", ".join(ld.states_match5) if ld.states_match5 else None,
                "tiers": ld.tiers[:5],
                "url": ld.detail_url,
            })
            if ld.detail_url:
                st.markdown(f"[Open detail page]({ld.detail_url})")
        else:
            st.warning("Could not retrieve the latest draw detail right now.")
        st.caption(f"HTML parser: {'BeautifulSoup' if BS4_AVAILABLE else 'regex fallback'} · Timezone: {_tz_mode_str()}")
        st.markdown(f"Ticket cutoff varies by state. Example: [Florida info]({FL_PB}).")

    with tab_stats:
        if df_pb.empty:
            st.warning("No Powerball data available.")
        else:
            wcnt, rcnt = freq_counts(df_pb)
            whites_tbl = to_table(wcnt, range(1, 70))
            reds_tbl = to_table(rcnt, range(1, 27))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("White ball frequency")
                st.dataframe(whites_tbl, use_container_width=True)
                st.write("Top 10 hot:", whites_tbl.sort_values("count", ascending=False).head(10)[["number", "count"]])
            with col2:
                st.subheader("Red ball frequency")
                st.dataframe(reds_tbl, use_container_width=True)
                st.write("Cold 10:", reds_tbl.sort_values("count", ascending=True).head(10)[["number", "count"]])

    with tab_sim:
        st.subheader("EZ-pick simulation")
        # --- Pick Builder ---
        st.markdown("**Pick Builder (optional)**")
        if "built_picks" not in st.session_state:
            st.session_state["built_picks"] = []  # list[(whites, r)]
        colb1, colb2, colb3 = st.columns([3, 2, 2])
        with colb1:
            sel_whites = st.multiselect("Select 5 white balls (1–69)", options=list(range(1, 70)), max_selections=5)
        with colb2:
            sel_red = st.number_input("Powerball (1–26)", min_value=1, max_value=26, value=6, step=1)
            if st.button("Add pick") and len(sel_whites) == 5:
                pick = (sorted(sel_whites), int(sel_red))
                if pick not in st.session_state["built_picks"]:
                    st.session_state["built_picks"].append(pick)
        with colb3:
            up = st.file_uploader("CSV import (w1..w5,r or W,R)", type=["csv"], accept_multiple_files=False)
            if up is not None:
                imported = _parse_picks_csv(up)
                for p in imported:
                    if p not in st.session_state["built_picks"]:
                        st.session_state["built_picks"].append(p)
            if st.button("Clear picks"):
                st.session_state["built_picks"] = []

        if st.session_state["built_picks"]:
            dfbp = _picks_to_csv_rows([(w, r) for (w, r) in st.session_state["built_picks"]])
            st.dataframe(dfbp, use_container_width=True, hide_index=True)
            st.download_button(
                "Download picks CSV",
                data=dfbp.to_csv(index=False).encode("utf-8"),
                file_name="picks.csv",
                mime="text/csv",
            )

        st.markdown("**Or paste picks**")
        example = "1 2 3 4 5 | 6\n7, 8, 9, 10, 11 12"
        txt = st.text_area("Text picks (one per line)", value=example, height=90)
        picks_text = _parse_user_picks(txt)

        # Unified picks (built + text, deduped)
        unified: List[Tuple[List[int], int]] = []
        seen = set()
        for src in (st.session_state["built_picks"], picks_text):
            for whites, r in src:
                key = (tuple(sorted(whites)), int(r))
                if key not in seen:
                    seen.add(key)
                    unified.append((list(key[0]), key[1]))

        st.write(f"Total picks: **{len(unified)}**")

        # --- Auto-pick controls ---
        st.markdown("**Auto-pick (Quick Pick strategy)**")
        colap1, colap2, colap3, colap4 = st.columns([1, 1, 2, 2])
        with colap1:
            use_auto_pick = st.checkbox("Auto-pick tickets", value=False)
        with colap2:
            auto_pick_n = st.number_input("Tickets to auto-pick", min_value=1, max_value=100, value=5, step=1, disabled=not use_auto_pick)
        with colap3:
            auto_weight_ui = st.radio(
                "Weighting",
                ["Uniform", "Historical (hot-weighted)", "Historical (cold-weighted)"],
                horizontal=False,
                index=0,
                disabled=not use_auto_pick,
            )
        with colap4:
            auto_each_draw = st.checkbox("Re-pick every draw (advanced)", value=False, disabled=not use_auto_pick)

        recency_max = int(len(df_pb)) if isinstance(df_pb, pd.DataFrame) else 0
        recency_enabled = use_auto_pick and auto_weight_ui.startswith("Historical") and recency_max > 0
        recency_n = st.slider(
            "Recency window (last N draws; 0 = all history)",
            min_value=0,
            max_value=recency_max,
            value=0,
            step=10 if recency_max > 200 else 1,
            disabled=not recency_enabled,
        )

        if recency_enabled:
            try:
                cold_flag = auto_weight_ui.startswith("Historical (cold")
                w_arr, r_arr = make_weight_arrays(
                    df_pb, white_max=69, red_max=26, smoothing=1.0, cold=cold_flag, recency_n=recency_n
                )
                if MPL_AVAILABLE:
                    idx_w = np.argsort(w_arr)[::-1][:20]
                    fig1 = plt.figure()
                    plt.bar([int(i + 1) for i in idx_w], w_arr[idx_w])
                    plt.title("White weights (top 20)")
                    plt.xlabel("Number")
                    plt.ylabel("Probability")
                    st.pyplot(fig1, clear_figure=True)

                    idx_r = np.argsort(r_arr)[::-1][:10]
                    fig2 = plt.figure()
                    plt.bar([int(i + 1) for i in idx_r], r_arr[idx_r])
                    plt.title("Red weights (top 10)")
                    plt.xlabel("Number")
                    plt.ylabel("Probability")
                    st.pyplot(fig2, clear_figure=True)
                else:
                    st.caption("Charts disabled: matplotlib not installed")
                    w_df = pd.DataFrame({"number": np.arange(1, 70), "prob": w_arr})
                    r_df = pd.DataFrame({"number": np.arange(1, 27), "prob": r_arr})
                    st.write("White weights (top 20):")
                    st.dataframe(w_df.sort_values("prob", ascending=False).head(20), use_container_width=True)
                    st.write("Red weights (top 10):")
                    st.dataframe(r_df.sort_values("prob", ascending=False).head(10), use_container_width=True)
            except Exception:
                pass

        # Preview auto-picked tickets when not re-picking each draw
        if use_auto_pick and not auto_each_draw:
            try:
                rng_prev = np.random.default_rng(seed or None)
                cold_flag = auto_weight_ui.startswith("Historical (cold")
                if auto_weight_ui.startswith("Historical"):
                    w_arr, r_arr = make_weight_arrays(
                        df_pb, white_max=69, red_max=26, smoothing=1.0, cold=cold_flag, recency_n=recency_n
                    )
                else:
                    w_arr, r_arr = None, None
                preview = []
                for _p in range(int(auto_pick_n)):
                    if auto_weight_ui.startswith("Historical"):
                        ws, rr = _generate_pick_weighted(rng_prev, 69, 26, w_arr, r_arr)
                    else:
                        ws, rr = _generate_pick_uniform(rng_prev, 69, 26)
                    preview.append({"w1": ws[0], "w2": ws[1], "w3": ws[2], "w4": ws[3], "w5": ws[4], "r": rr})
                if preview:
                    dfprev = pd.DataFrame(preview)
                    st.dataframe(dfprev, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download auto-picked CSV",
                        data=dfprev.to_csv(index=False).encode("utf-8"),
                        file_name="auto_picks.csv",
                        mime="text/csv",
                    )
            except Exception:
                pass

        # --- Power Play & EV controls ---
        colpp1, colpp2, colpp3 = st.columns([1, 1, 2])
        with colpp1:
            enable_pp = st.checkbox("Enable Power Play (+$1/play)", value=False)
        with colpp2:
            pp_mode_ui = st.radio("PP mode", ["Uniform", "Weighted (official-ish)", "Fixed"], index=0, disabled=not enable_pp)
        with colpp3:
            pp_fixed = st.number_input("Fixed multiplier", min_value=2, max_value=10, value=2, step=1, disabled=not (enable_pp and pp_mode_ui == "Fixed"))

        colpp4, colpp5, colpp6 = st.columns([1, 1, 2])
        with colpp4:
            pp_allow_10x = st.checkbox("Allow 10×", value=True, disabled=not enable_pp)
        with colpp5:
            auto_cap = st.checkbox("Apply 10× cap (≤ $150M)", value=True, disabled=not enable_pp)
        with colpp6:
            draws = st.slider("Simulated draws", 100, 200_000, 10_000, step=100)

        seed = st.number_input("Seed (optional)", value=0, min_value=0, step=1)
        jp_override = st.number_input("Jackpot estimate (override)", value=int(jp or 0), min_value=0, step=1)

        if st.button("Run simulation", type="primary"):
            if not unified and not use_auto_pick:
                st.error("Please enter at least one valid pick or enable Auto-pick.")
            else:
                try:
                    mode_map = {"Uniform": "uniform", "Weighted (official-ish)": "weighted", "Fixed": "fixed"}
                    auto_w = None
                    auto_r = None
                    if use_auto_pick and auto_weight_ui.startswith("Historical"):
                        try:
                            cold_flag = auto_weight_ui.startswith("Historical (cold")
                            auto_w, auto_r = make_weight_arrays(
                                df_pb, white_max=69, red_max=26, smoothing=1.0, cold=cold_flag, recency_n=recency_n
                            )
                        except Exception:
                            auto_w, auto_r = None, None
                    res = simulate_strategy(
                        ([] if use_auto_pick else unified),
                        draws=draws if (licensed or not DEMO_ONLY) else min(draws, 5000),
                        jackpot_estimate=jp_override or None,
                        seed=seed or None,
                        power_play=enable_pp,
                        pp_mode=mode_map.get(pp_mode_ui, "uniform"),
                        pp_fixed_multiplier=pp_fixed,
                        pp_allow_10x=pp_allow_10x,
                        pp_auto_10x_cap=auto_cap,
                        auto_pick=use_auto_pick,
                        auto_pick_n=auto_pick_n,
                        auto_pick_weighting=("historical" if auto_weight_ui.startswith("Historical") else "uniform"),
                        auto_pick_each_draw=auto_each_draw,
                        white_weights=auto_w,
                        red_weights=auto_r,
                    )

                    met1, met2, met3 = st.columns(3)
                    with met1:
                        st.metric("Gross EV per draw", f"${res['overall']['gross_ev_per_draw']:.4f}")
                    with met2:
                        st.metric("Ticket cost per play", f"${res['overall']['cost_per_play']:.2f}")
                    with met3:
                        st.metric("Net EV per draw", f"${res['overall']['net_ev_per_draw']:.4f}")

                    st.json(res)

                    if licensed:
                        tiers_df, metrics_df = simulation_to_csvs(res, draws=res["draws"], num_picks=(auto_pick_n if use_auto_pick and auto_each_draw else max(len(unified), auto_pick_n)))
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "Download tiers CSV",
                                data=tiers_df.to_csv(index=False).encode("utf-8"),
                                file_name="sim_tiers.csv",
                                mime="text/csv",
                            )
                        with c2:
                            st.download_button(
                                "Download metrics CSV",
                                data=metrics_df.to_csv(index=False).encode("utf-8"),
                                file_name="sim_metrics.csv",
                                mime="text/csv",
                            )
                except Exception as e:
                    st.exception(e)


if __name__ == "__main__":
    if st is not None:
        main()
    else:
        # Standalone execution without Streamlit
        try:
            picks = [([1, 2, 3, 4, 5], 6)]
            out = simulate_strategy(picks, draws=1000, seed=0)
            print("CLI demo:", {"draws": out["draws"], "gross_ev_per_draw": out["overall"]["gross_ev_per_draw"]})
        except Exception as e:
            print("CLI demo failed:", e)

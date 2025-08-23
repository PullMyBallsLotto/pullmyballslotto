# PullMyBallsLotto • Lottery Stats Explorer (Educational)
# Features: hot/cold, pairs/triplets, pair heatmap, EZ-pick simulation,
# countdown to next Powerball draw, auto-refresh near draws, official prize mapping,
# latest winners & payouts (official), state links, Gumroad license unlock (Pro),
# credits model (safe; not gambling), CSV downloads.
# IMPORTANT: Does NOT predict results. Educational/entertainment only.

import re, itertools
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
from bs4 import BeautifulSoup

st.set_page_config(page_title="PullMyBallsLotto • Lottery Stats Explorer", layout="wide")
ET = ZoneInfo("America/New_York")

# ---------- Official/public sources ----------
POWERBALL_WATCH = "https://www.powerball.com/watch-drawing"             # 10:59 pm ET, Mon/Wed/Sat
POWERBALL_PRIZES_URL = "https://www.powerball.com/powerball-prize-chart" # Prize tiers & odds
POWERBALL_HOME = "https://www.powerball.com/"                            # Estimated jackpot scrape
PB_PREV_RESULTS = "https://www.powerball.com/previous-results"           # Latest draw detail links

# Historical data endpoints
TX_PB_CSV = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"
NY_PB_CSV = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
NY_MM_CSV = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"

# State examples
FL_PB = "https://floridalottery.com/games/draw-games/powerball"  # Tickets until 10:00 pm ET

# Matrices since current formats
MATRIX_CUTOFF_PB = pd.to_datetime("2015-10-04").date()  # PB 69/26 since Oct 2015
MATRIX_CUTOFF_MM = pd.to_datetime("2017-10-31").date()  # MM 70/25 since Oct 2017

# ---------- Gumroad secrets (set in Streamlit → Settings → Advanced → Secrets) ----------
# Supports multiple products (e.g., monthly, yearly), comma-separated
PRODUCT_IDS_RAW = st.secrets.get("GUMROAD_PRODUCT_IDS", "").strip()
PRODUCT_IDS = [p.strip() for p in PRODUCT_IDS_RAW.split(",") if p.strip()]
DEMO_ONLY  = st.secrets.get("DEMO_ONLY", "false").lower() == "true"

# ---------- Licensing ----------
def verify_gumroad_license(license_key: str) -> bool:
    """Verify license against one or more Gumroad product_ids (monthly/yearly)."""
    if not PRODUCT_IDS or not license_key:
        return False
    try:
        for pid in PRODUCT_IDS:
            r = requests.post(
                "https://api.gumroad.com/v2/licenses/verify",
                data={"product_id": pid, "license_key": license_key.strip()},
                timeout=20,
            )
            js = r.json()
            ok = bool(js.get("success")) and not js.get("purchase", {}).get("refunded", False)
            # Optional: treat paused/canceled subscriptions as invalid if present
            if ok:
                return True
    except Exception:
        pass
    return False

# ---------- Utilities ----------
def to_table(counter, domain):
    total = sum(counter.get(n,0) for n in domain)
    rows = []
    for n in domain:
        c = int(counter.get(n,0)); pct = (100*c/total) if total else 0
        rows.append({"number": n, "count": c, "percent": round(pct,3)})
    return pd.DataFrame(rows)

def freq_counts(df):
    whites = list(itertools.chain.from_iterable(df["W"].tolist()))
    reds = df["R"].tolist()
    return Counter(whites), Counter(reds)

def pairs_triplets(df):
    pair_counts, trip_counts = Counter(), Counter()
    for ws in df["W"]:
        s = sorted(ws)
        for a,b in itertools.combinations(s,2): pair_counts[(a,b)] += 1
        for a,b,c in itertools.combinations(s,3): trip_counts[(a,b,c)] += 1
    return pair_counts, trip_counts

def pair_matrix(pair_counts, max_white):
    mat = np.zeros((max_white, max_white), dtype=int)
    for (a,b),cnt in pair_counts.items():
        mat[a-1,b-1] = cnt
        mat[b-1,a-1] = cnt
    return mat

# ---------- Next draw & jackpot ----------
def next_powerball_draw(now=None):
    """Next Powerball draw time in ET: Mon/Wed/Sat 10:59 pm ET."""
    if now is None: now = datetime.now(ET)
    draw_days = {0,2,5}  # Mon, Wed, Sat
    d = now
    while True:
        dt_draw = datetime(d.year, d.month, d.day, 22, 59, tzinfo=ET)
        if d.weekday() in draw_days and now < dt_draw:
            return dt_draw
        d += timedelta(days=1)

def get_powerball_jackpot_estimate():
    """Scrape estimated jackpot from Powerball home; fallback to None if not found."""
    try:
        html = requests.get(POWERBALL_HOME, timeout=15).text
        m = re.search(r"Estimated Jackpot[^$]*\$\s*([\d,.]+)\s*(Million|Billion)", html, re.I)
        if m:
            num = float(m.group(1).replace(",", ""))
            mult = m.group(2).lower()
            return int(num * (1_000_000_000 if "billion" in mult else 1_000_000))
    except Exception:
        pass
    return None

# ---------- Latest winners & payouts (official) ----------
def get_latest_powerball_draw_detail():
    """
    Parse latest draw detail from powerball.com/previous-results → first result's detail page.
    Returns dict with: date, est_jackpot_str, cash_value_str, states_match5 (list),
    tiers: list of dicts {label, pb_winners, pb_prize, pp_winners, pp_prize}, detail_url.
    """
    try:
        listing_html = requests.get(PB_PREV_RESULTS, timeout=20).text  # page with latest draw links
        # Find a draw-result link pattern; fallback to first matching 'draw-result' href
        m = re.search(r'href="(/draw-result\?date=\d{4}-\d{2}-\d{2}&gc=powerball)"', listing_html)
        if not m:
            return None
        detail_url = "https://www.powerball.com" + m.group(1)
        dhtml = requests.get(detail_url, timeout=20).text
soup = BeautifulSoup(dhtml, "html.parser")


        # Draw date (headline often present)
        date_text = None
        h = soup.find(["h1","h2","h3","h4","h5"], string=re.compile(r"\w{3}, \w{3} \d{1,2}, \d{4}"))
        if h: date_text = h.text.strip()

        # Estimated Jackpot / Cash Value
        est_jackpot_text = None; cash_value_text = None
        for label in soup.find_all(text=re.compile(r"Estimated Jackpot", re.I)):
            parent = label.parent
            if parent: est_jackpot_text = parent.get_text(strip=True).split(":")[-1]
        for label in soup.find_all(text=re.compile(r"Cash Value", re.I)):
            parent = label.parent
            if parent: cash_value_text = parent.get_text(strip=True).split(":")[-1]

        # States for Match 5 ($1M) winners (when listed)
        states_match5 = []
        lab = soup.find(text=re.compile(r"Match\s*5\s*\$?1\s*Million Winners", re.I))
        if lab:
            line = lab.parent.get_text(" ", strip=True)
            # Extract trailing state codes after the label
            s = re.split(r"(?i)Match\s*5\s*\$?1\s*Million Winners", line)[-1]
            states_match5 = [t.strip(" ,") for t in s.split(",") if t.strip()]

        # Winners table: try reading rows with numbers and dollar amounts
        tiers = []
        # Grand prize line may appear as "0 Grand Prize"
        gp = soup.find(string=re.compile(r"Grand Prize", re.I))
        if gp:
            # Look back for a number in same row
            row = gp.find_parent()
            num = re.search(r"\b\d+\b", row.get_text(" ", strip=True)) if row else None
            tiers.append({"label":"5+PB", "pb_winners": int(num.group(0)) if num else 0, "pb_prize": "Jackpot",
                          "pp_winners": None, "pp_prize": None})
        # Other tiers often list "X  $YY,YYY  |  Power Play: Z  $YY,YYY"
        text = soup.get_text("\n", strip=True)
        for m2 in re.finditer(r"\n\s*(\d+)\s+\$([\d,]+)\s+(\d+)\s+\$([\d,]+)", text):
            pbw, pbp, ppw, ppp = m2.groups()
            tiers.append({"label":"(tier)", "pb_winners": int(pbw), "pb_prize": f"${pbp}",
                          "pp_winners": int(ppw), "pp_prize": f"${ppp}"})
        return {
            "date": date_text,
            "est_jackpot_str": est_jackpot_text,
            "cash_value_str": cash_value_text,
            "states_match5": states_match5,
            "tiers": tiers,
            "detail_url": detail_url
        }
    except Exception:
        return None

# ---------- Loaders (history) ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_powerball():
    rows=[]
    try:  # Texas official CSV
        txt = requests.get(TX_PB_CSV, timeout=30).text.strip().splitlines()
        for line in txt:
            p = [q.strip() for q in line.split(",")]
            if len(p) < 11 or "powerball" not in p[0].lower(): continue
            mm, dd, yy = int(p[1]), int(p[2]), int(p[3])
            whites = list(map(int, p[4:9])); red = int(p[9])
            rows.append({"date": datetime(yy,mm,dd).date(),
                         "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":red})
    except Exception:
        pass
    if not rows:
        try:  # NY Open Data fallback
            csv = pd.read_csv(NY_PB_CSV)
            for _,rec in csv.iterrows():
                try:
                    d = pd.to_datetime(str(rec.get("Draw Date") or rec.get("draw_date"))).date()
                except Exception:
                    continue
                wn = str(rec.get("Winning Numbers") or "").replace(",", " ")
                nums = [int(x) for x in wn.split() if x.isdigit()]
                if len(nums) < 6: continue
                whites, red = nums[:5], nums[5]
                rows.append({"date": d, "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":int(red)})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["draw_date"] = df["date"]
    df["W"] = df[["w1","w2","w3","w4","w5"]].values.tolist()
    df["R"] = df["r"].astype(int)
    df = df[df["draw_date"] >= MATRIX_CUTOFF_PB]
    df = df[(df["W"].apply(lambda ws: len(ws)==5 and all(1<=x<=69 for x in ws))) & (df["R"].between(1,26))]
    return df.sort_values("draw_date").reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def load_megamillions():
    try:
        csv = pd.read_csv(NY_MM_CSV)
        rows=[]
        for _,rec in csv.iterrows():
            try:
                d = pd.to_datetime(str(rec.get("Draw Date") or rec.get("draw_date"))).date()
            except Exception:
                continue
            wn = str(rec.get("Winning Numbers") or "").replace(",", " ")
            nums = [int(x) for x in wn.split() if x.isdigit()]
            if len(nums) < 6: continue
            whites, mega = nums[:5], nums[5]
            rows.append({"date": d, "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":int(mega)})
        df = pd.DataFrame(rows)
    except Exception:
        df = pd.DataFrame()
    if df.empty: return df
    df["draw_date"] = df["date"]
    df["W"] = df[["w1","w2","w3","w4","w5"]].values.tolist()
    df["R"] = df["r"].astype(int)
    df = df[df["draw_date"] >= MATRIX_CUTOFF_MM]
    df = df[(df["W"].apply(lambda ws: len(ws)==5 and all(1<=x<=70 for x in ws))) & (df["R"].between(1,25))]
    return df.sort_values("draw_date").reset_index(drop=True)

# ---------- Prize mapping (official tiers; no Power Play) ----------
POWERBALL_PRIZES = {
    (5, True):  "JACKPOT",
    (5, False): 1_000_000,
    (4, True):  50_000,
    (4, False): 100,
    (3, True):  100,
    (3, False): 7,
    (2, True):  7,
    (1, True):  4,
    (0, True):  4
}

def prize_for_result(match_white, pb_matched, jackpot_estimate):
    val = POWERBALL_PRIZES.get((match_white, pb_matched), 0)
    return jackpot_estimate if val == "JACKPOT" else val

def simulate_strategy(picks, draws=10000, white_max=69, red_max=26, jackpot_estimate=500_000_000, seed=42):
    """Simulate random official draws vs your tickets; return total winnings by tier."""
    rng = np.random.default_rng(seed)
    total = 0
    buckets = defaultdict(int)
    for _ in range(draws):
        win_whites = set(rng.choice(np.arange(1, white_max+1), size=5, replace=False))
        win_red = int(rng.integers(1, red_max+1))
        for whites, red in picks:
            mw = len(set(whites) & win_whites)
            pb = (red == win_red)
            amt = prize_for_result(mw, pb, jackpot_estimate)
            total += amt
            buckets[(mw, pb)] += 1
    return total, dict(buckets)

def quick_pick_sim(n_tickets, white_max, red_max, seed=2025):
    rng = np.random.default_rng(seed)
    wc, rc = Counter(), Counter()
    for _ in range(n_tickets):
        whites = rng.choice(np.arange(1, white_max+1), size=5, replace=False)
        red = int(rng.integers(1, red_max+1))
        for w in whites: wc[int(w)] += 1
        rc[red] += 1
    return wc, rc

# ---------- Sidebar (license, credits, lottery) ----------
st.sidebar.title("Access")
st.sidebar.markdown("[Upgrade to Pro (Monthly)](https://YOUR_GUMROAD_MONTHLY_LINK)")
st.sidebar.markdown("[Upgrade to Pro (Yearly)](https://YOUR_GUMROAD_YEARLY_LINK)")

license_key = st.sidebar.text_input("Enter Pro License Key", type="password", help="Paste your Gumroad license key here.")
licensed = verify_gumroad_license(license_key)
if DEMO_ONLY: licensed = False
st.sidebar.write("License:", "✅ Valid" if licensed else "🔓 Demo mode")

st.sidebar.title("Credits")
if "credits" not in st.session_state: st.session_state.credits = 60  # free starter credits
st.sidebar.write(f"Available credits: {st.session_state.credits}")
def spend_credits(n=1):
    if licensed: return True     # Pro ignores credits
    if st.session_state.credits >= n:
        st.session_state.credits -= n
        return True
    st.warning("Not enough credits. Upgrade to Pro for more.")
    return False

st.sidebar.title("Lottery edition")
lottery = st.sidebar.selectbox("Choose", ["Powerball (US)", "Mega Millions (US)", "Upload CSV (any game)"])

# ---------- Title & disclaimer ----------
st.title("PullMyBallsLotto • Lottery Stats Explorer")
with st.expander("Read me first (disclaimer)"):
    st.markdown("""
**DISCLAIMER:** This app is for **educational and entertainment** purposes only.  
It **does not predict** or improve your chances of winning any lottery.  
Lotteries are games of chance. Use at your own risk.  
Not affiliated with Powerball, Mega Millions, or any lottery commission.
""")

# ---------- Countdown (Powerball) + auto-refresh ----------
st.subheader("Next Powerball draw")
npd = next_powerball_draw()
left = npd - datetime.now(ET)
interval_ms = 30000 if left.total_seconds() > 900 else 10000
st_autorefresh(interval=interval_ms, key="auto_refresh")
c1,c2 = st.columns(2)
with c1: st.metric("Draw time (ET)", npd.strftime("%a %b %d, %Y 10:59 pm ET"))
with c2:
    secs = int(max(left.total_seconds(), 0)); h=secs//3600; m=(secs%3600)//60; s=secs%60
    st.metric("Countdown", f"{h:02d}:{m:02d}:{s:02d}")
st.caption("Official drawings Mon/Wed/Sat at 10:59 pm ET; livestream available on Powerball.com.")

# ---------- Load data ----------
if lottery.startswith("Powerball"): df = load_powerball(); white_max, red_max = 69, 26
elif lottery.startswith("Mega"):    df = load_megamillions(); white_max, red_max = 70, 25
else:
    st.subheader("Upload your history CSV")
    st.write("**CSV** = simple spreadsheet text file (Comma‑Separated Values). Required columns: `Date, White1..White5, BonusBall` (6th optional).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up)
        rows=[]
        for _,r in raw.iterrows():
            try:
                d = pd.to_datetime(str(r["Date"])).date()
                whites = [int(r["White1"]),int(r["White2"]),int(r["White3"]),int(r["White4"]),int(r["White5"])]
                bonus = int(r.get("BonusBall", 0))
                rows.append({"draw_date": d, "W": sorted(whites), "R": bonus})
            except Exception:
                pass
        df = pd.DataFrame(rows).sort_values("draw_date").reset_index(drop=True)
        white_max = int(max(itertools.chain.from_iterable(df["W"]))) if not df.empty else 69
        red_max   = int(max(df["R"])) if not df.empty else 26
    else:
        df = pd.DataFrame(columns=["draw_date","W","R"])
        white_max, red_max = 69, 26

if df.empty:
    st.error("Could not load historical data. Try again later or use the Upload option.")
    st.stop()

# Demo limit
df_show = df.copy()
if not licensed:
    df_show = df.tail(100)
    st.info("Demo mode: last 100 draws only. Enter a valid license key to unlock full history, downloads, and premium charts.")

# Manual refresh
if st.button("🔁 Refresh now"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------- Headline cards ----------
a,b,c = st.columns(3)
with a: st.metric("Draws (shown)", len(df_show))
with b: st.metric("Date range", f"{df_show['draw_date'].min()} → {df_show['draw_date'].max()}")
with c: st.metric("Edition", lottery)

# ---------- Frequencies ----------
wc_all, rc_all = freq_counts(df_show)
white_tbl = to_table(wc_all, range(1, white_max+1)).sort_values(["count","number"], ascending=[False,True])
red_tbl   = to_table(rc_all, range(1, red_max+1)).sort_values(["count","number"], ascending=[False,True])

st.subheader("Overall frequencies")
x1,x2 = st.columns([2,1])
with x1:
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(x="number", y="count", data=white_tbl.sort_values("number"), ax=ax, palette="viridis")
    ax.set_title("White-ball counts"); ax.set_xlabel("Number"); ax.set_ylabel("Count")
    st.pyplot(fig)
with x2:
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x="number", y="count", data=red_tbl.sort_values("number"), ax=ax2, palette="rocket")
    ax2.set_title("Bonus-ball counts"); ax2.set_xlabel("Bonus"); ax2.set_ylabel("Count")
    st.pyplot(fig2)

# ---------- Recent hot/cold ----------
tail_n = st.sidebar.select_slider("Recent window (hot/cold)", options=[50,100,250,500], value=100)
wc_tail, rc_tail = freq_counts(df_show.tail(min(tail_n, len(df_show))))
st.subheader(f"Recent hot/cold (last {min(tail_n, len(df_show))} draws)")
y1,y2 = st.columns(2)
with y1:
    st.write("**Hot whites (top 10)**")
    st.dataframe(to_table(wc_tail, range(1, white_max+1)).sort_values("count", ascending=False).head(10), use_container_width=True)
with y2:
    st.write("**Cold whites (bottom 10)**")
    st.dataframe(to_table(wc_tail, range(1, white_max+1)).sort_values("count", ascending=True).head(10), use_container_width=True)

# ---------- Pairs & Triplets + heatmap (credits-gated unless Pro) ----------
st.subheader("Common pairs & triplets (whites)")
if st.button("Generate pairs/triplets + heatmap (cost: 1 credit)"):
    if spend_credits(1):
        pair_counts, trip_counts = pairs_triplets(df_show)
        z1,z2 = st.columns(2)
        with z1:
            st.dataframe(pd.DataFrame(pair_counts.most_common(15), columns=["pair","count"]), use_container_width=True)
        with z2:
            st.dataframe(pd.DataFrame(trip_counts.most_common(15), columns=["triplet","count"]), use_container_width=True)
        mat = pair_matrix(pair_counts, white_max)
        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(mat, cmap="YlOrRd", cbar=True, ax=ax3)
        ax3.set_title("Pair frequency heatmap"); ax3.set_xlabel("White ball"); ax3.set_ylabel("White ball")
        st.pyplot(fig3)

# ---------- EZ-pick simulation ----------
sim_n = st.sidebar.select_slider("EZ-pick simulation tickets", options=[1000,5000,10000,20000], value=10000)
st.subheader(f"EZ-pick simulation — {sim_n:,} random tickets")
sim_wc, sim_rc = quick_pick_sim(sim_n, white_max, red_max)
sim_tbl = to_table(sim_wc, range(1, white_max+1)).sort_values("number")
fig4, ax4 = plt.subplots(figsize=(12,4))
sns.barplot(x="number", y="count", data=sim_tbl, ax=ax4, palette="coolwarm")
ax4.set_title("Simulated white-ball pick counts"); ax4.set_xlabel("Number"); ax4.set_ylabel("Count")
st.pyplot(fig4)

# ---------- Latest winners & payouts (official) ----------
st.subheader("Latest winners & payouts (official)")
detail = get_latest_powerball_draw_detail()
if detail:
    st.write(f"**Draw:** {detail['date']}")
    if detail['est_jackpot_str'] or detail['cash_value_str']:
        st.write(f"**Estimated Jackpot:** {detail.get('est_jackpot_str','—')} • **Cash Value:** {detail.get('cash_value_str','—')}")
    if detail["states_match5"]:
        st.write(f"**Match 5 ($1M) winners came from:** {', '.join(detail['states_match5'])}")
    if detail["tiers"]:
        st.write("**National winners (by tier):**")
        st.dataframe(pd.DataFrame(detail["tiers"]), use_container_width=True)
    st.caption(f"Source: official Powerball draw detail page. {detail['detail_url']}")
else:
    st.info("Latest winners & payouts not available right now from powerball.com; try again later.")

# ---------- Prize mapping & Simulation Arena (Powerball only) ----------
if lottery.startswith("Powerball"):
    st.subheader("Simulated prize mapping (official tiers)")
    jackpot_est = get_powerball_jackpot_estimate() or 500_000_000
    jackpot_est = st.number_input("Estimated Jackpot (USD)", min_value=20_000_000, value=int(jackpot_est), step=5_000_000)
    st.caption("Prize tiers per official chart; jackpot is an estimate (no Power Play in v1).")

    # Build up to 5 tickets (Pro ignores credits; Demo costs credits)
    picks = []
    for i in range(1, 4 if not licensed else 6):
        cols = st.columns(6)
        with cols[0]:
            use = st.checkbox(f"Ticket {i}", value=(i==1))
        whites=[]
        for j in range(1,6):
            with cols[j if j<5 else 4]:
                n = st.number_input(f"W{j} (T{i})", min_value=1, max_value=white_max, value=min(j*10, white_max))
                whites.append(int(n))
        with cols[5]:
            red = st.number_input(f"PB (T{i})", min_value=1, max_value=red_max, value=min(16, red_max))
        if use: picks.append((sorted(list(set(whites)))[:5], int(red)))

    if st.button("Run Simulation Arena (10,000 random draws)"):
        if spend_credits(3):
            total, buckets = simulate_strategy(picks, draws=10_000, white_max=white_max, red_max=red_max, jackpot_estimate=jackpot_est)
            st.success(f"Total simulated winnings across 10,000 draws: ${total:,.0f}")
            bucket_tbl = pd.DataFrame(
                [{"matches":k[0], "PB?":k[1], "count": v, "prize_each": prize_for_result(k[0],k[1],jackpot_est),
                  "total_prize": v*(jackpot_est if prize_for_result(k[0],k[1],jackpot_est)=="JACKPOT" else prize_for_result(k[0],k[1],jackpot_est))}
                 for k,v in buckets.items()]
            ).sort_values(["total_prize","matches"], ascending=[False,False])
            st.dataframe(bucket_tbl, use_container_width=True)

# ---------- State rules & links (informational) ----------
st.subheader("State rules & purchase cut-offs (informational)")
state = st.selectbox("Your state", sorted([
    "Alabama (N/A)","Arizona","California","Florida","Michigan","New York","Texas","Washington DC","Puerto Rico","US Virgin Islands"
]))
st.write("• Official Powerball site (draw times, livestream, prize chart).")
st.write(POWERBALL_WATCH)
if state=="Florida":
    st.write("• Florida: Draw 10:59 pm ET; ticket sales cut-off 10:00 pm ET (official).")
    st.write(FL_PB)
st.info("Sales cut-off times vary by jurisdiction. Always check your state lottery’s official site.")

# ---------- Downloads (Pro) ----------
if licensed:
    st.subheader("Download tables (CSV)")
    st.download_button("White frequencies CSV", white_tbl.to_csv(index=False), "whites.csv", "text/csv")
    st.download_button("Bonus frequencies CSV", red_tbl.to_csv(index=False), "bonus.csv", "text/csv")
    out_df = df_show.copy()
    out_df["W1"]=[w[0] for w in out_df["W"]]; out_df["W2"]=[w[1] for w in out_df["W"]]; out_df["W3"]=[w[2] for w in out_df["W"]]
    out_df["W4"]=[w[3] for w in out_df["W"]]; out_df["W5"]=[w[4] for w in out_df["W"]]
    st.download_button("Draw history (shown) CSV", out_df.drop(columns=["W"]).to_csv(index=False), "history.csv", "text/csv")

st.caption("Data: Texas Lottery CSV; NY Open Data; official Powerball pages for draw details. PullMyBallsLotto is not affiliated with any lottery. Educational/entertainment only.")

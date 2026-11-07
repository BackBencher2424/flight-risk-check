"""Flight Risk Check — Team 5, Questrom MSBA Chart Champ 2026.

Gives the calibrated chance that a U.S. flight is severely disrupted
(cancelled, diverted or more than 60 minutes late) or arrives 15+ minutes late,
using gradient-boosted tree models trained on all 2025 BTS flights for ten airlines.
The trees are evaluated in pure Python, so the app needs no compiled ML libraries.
"""
import gzip
import json
import math
from datetime import date, time, timedelta
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import live

st.set_page_config(page_title="Flight Risk Check", page_icon="✈️", layout="wide",
                   initial_sidebar_state="expanded")

HERE = Path(__file__).parent
AIRLINES = {
    "AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta",
    "MQ": "Envoy (American Eagle)", "OH": "PSA (American Eagle)",
    "OO": "SkyWest (regional)", "UA": "United",
    "WN": "Southwest", "YX": "Republic (regional)",
}
AVG_SEVERE, AVG_LATE = 0.096, 0.235  # all 2025 flights in the data
GREEN, BLUE, AMBER, RED = "#5BC79A", "#7FB2E5", "#F5B841", "#EF6A55"
PLANE = ("M32 2C34.5 2 36 4 36 7V24L62 38V44L36 37V52L43 57V62L32 59L21 62V57L28 52V37"
         "L2 44V38L28 24V7C28 4 29.5 2 32 2Z")


def secret(name):
    try:
        v = st.secrets.get(name)
        return str(v).strip() if v else None
    except Exception:  # no secrets file configured
        return None


AV_KEY, AERO_KEY = secret("AVIATIONSTACK_KEY"), secret("AEROAPI_KEY")


@st.cache_data(ttl=300, show_spinner=False)  # 5-minute cache protects the free monthly quota
def live_status(flight):
    return live.aeroapi(AERO_KEY, flight) if AERO_KEY else live.aviationstack(AV_KEY, flight)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the 2025 flight model…")
def load():
    data = json.loads((HERE / "model" / "calc_data.json").read_text())
    cities = json.loads((HERE / "model" / "cities.json").read_text())
    with gzip.open(HERE / "model" / "schedule.json.gz", "rt") as f:
        sched = json.load(f)  # "ORIG|DEST" -> [airline, flight no, sched dep HHMM, times flown, % severe, month×weekday bitmask]
    for m in data["models"].values():
        m["iso_x"] = np.array(m["iso_x"])
        m["iso_y"] = np.array(m["iso_y"])
    route_cars = {}
    for c, origins in data["routes"].items():
        for o, dests in origins.items():
            for d in dests:
                route_cars.setdefault((o, d), []).append(c)
    return data, cities, sched, route_cars


def predict(model, x):
    s = 0.0
    for F, T, L, R, V in model["trees"]:
        i = 0
        while F[i] >= 0:
            i = L[i] if x[F[i]] <= T[i] else R[i]
        s += V[i]
    p = 1.0 / (1.0 + math.exp(-s))
    return float(np.interp(p, model["iso_x"], model["iso_y"]))  # calibrated


def later_flights(hours, h):
    return sum(v for k, v in hours.items() if int(k) > h)


def features(D, tgt, car, o, de, month, dow, dep_hr, prev):
    dist, el, rf, hours = D["routes"][car][o][de]
    h = int(dep_hr)
    ov = D["OV"].get(o, {}).get(str(h), 0)
    ah = int((dep_hr * 60 + el) // 60) % 24
    dv = D["DV"].get(de, {}).get(str(ah), 0)
    TE = D["TE"][tgt]

    def te(cc, key):
        return TE[cc].get(key, TE["prior"])

    x = [month, dow, dep_hr, dist, el, rf, ov, dv, later_flights(hours, h),
         te("car", car), te("o", o), te("de", de), te("route", f"{car}|{o}|{de}")]
    if prev is not None:
        x.append(prev)
    return x


def risk(D, tgt, car, o, de, month, dow, dep_hr, prev=None):
    key = f"{tgt}_{'booking' if prev is None else 'dayof'}"
    return predict(D["models"][key], features(D, tgt, car, o, de, month, dow, dep_hr, prev))


def backups(sched, o, de, month, dow, dep_hr):
    """Real 2025 flights on this route that ran on this weekday in this month, departing after dep_hr."""
    bit = 1 << ((month - 1) * 7 + (dow - 1))
    after = int(dep_hr) * 100 + round((dep_hr % 1) * 60)
    best = {}
    for c, fn, dep, n, sev, mask in sched.get(f"{o}|{de}", []):
        if dep > after and int(mask) & bit:
            k = (c, fn)
            if k not in best or n > best[k][3]:
                best[k] = (c, fn, dep, n, sev, mask)
    return sorted(best.values(), key=lambda r: r[2])


# ---------------------------------------------------------------- formatting
def pct(p):
    return "<1%" if p < 0.01 else (f"{p*100:.1f}%" if p < 0.1 else f"{round(p*100)}%")


def one_in(p):
    return max(2, round(1 / p)) if p > 0 else 100


def clock(h):
    hh, mm = int(h), int(round((h - int(h)) * 60))
    t = time(hh % 24, mm)
    return f"{(t.hour % 12) or 12}{':%02d' % t.minute if t.minute else ''} {'AM' if t.hour < 12 else 'PM'}"


def hhmm(dep):
    return clock(dep // 100 + (dep % 100) / 60)


def verdict(p):
    if p < 0.06:
        return GREEN, "Low risk", "Looks smooth. This is a low-risk flight."
    if p < 0.12:
        return BLUE, "Average risk", "Typical risk for a U.S. flight."
    if p < 0.20:
        return AMBER, "Elevated risk", "Noticeably riskier than the average flight."
    return RED, "High risk", "One of the riskier ways to make this trip."


def short(c):
    return AIRLINES.get(c, c).split(" (")[0]


def arc(cx, cy, r, t0, t1):
    a0, a1 = math.pi * (1 - t0), math.pi * (1 - t1)
    x0, y0 = cx + r * math.cos(a0), cy - r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
    return f"M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}"


def gauge(p, color):
    top = 0.40
    t = min(p, top) / top
    bands = [(0, .06, GREEN), (.06, .12, BLUE), (.12, .20, AMBER), (.20, .40, RED)]
    segs = "".join(f'<path d="{arc(110, 110, 86, a/top, b/top)}" stroke="{c}" stroke-opacity=".22" stroke-width="18" fill="none"/>'
                   for a, b, c in bands)
    val = f'<path d="{arc(110, 110, 86, 0, max(t, .005))}" stroke="{color}" stroke-width="18" fill="none" stroke-linecap="round"/>'
    ang = math.pi * (1 - t)
    nx, ny = 110 + 62 * math.cos(ang), 110 - 62 * math.sin(ang)
    needle = (f'<line x1="110" y1="110" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#F2EEE4" stroke-width="4" stroke-linecap="round"/>'
              f'<circle cx="110" cy="110" r="7" fill="#F2EEE4"/>')
    avg_t = AVG_SEVERE / top
    aa = math.pi * (1 - avg_t)
    tick = (f'<line x1="{110+74*math.cos(aa):.1f}" y1="{110-74*math.sin(aa):.1f}" x2="{110+100*math.cos(aa):.1f}" '
            f'y2="{110-100*math.sin(aa):.1f}" stroke="#8E9AA8" stroke-width="2" stroke-dasharray="3 3"/>')
    return (f'<svg viewBox="0 0 220 124" width="100%" style="max-width:260px" role="img" '
            f'aria-label="Gauge showing {pct(p)} chance of serious disruption">{segs}{val}{tick}{needle}</svg>')


# ---------------------------------------------------------------- styles
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap">
<style>
:root{--board:#0F1720;--panel:#141E29;--tile:#1B2633;--line:#243040;--ink:#F2EEE4;--text:#D7DEE6;--muted:#8E9AA8;--amber:#F5B841}
html, body, [class*="css"], .stMarkdown, .stText{font-family:'Inter',sans-serif}
.block-container{padding-top:3.2rem;max-width:1240px}
section[data-testid="stSidebar"]{background:#0B121A;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .brand{font-family:'JetBrains Mono',monospace;color:var(--amber);letter-spacing:.18em;font-size:.78rem}
.mono{font-family:'JetBrains Mono',monospace}
.eyebrow{font-family:'JetBrains Mono',monospace;letter-spacing:.2em;color:var(--amber);font-size:.75rem;text-transform:uppercase}
.pass{display:flex;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:20px;overflow:hidden;margin:.4rem 0 1.2rem}
.pass-main{flex:3 1 520px;padding:22px 28px;min-width:0}
.pass-stub{flex:1 1 230px;padding:22px 26px;border-left:2px dashed var(--line);position:relative;display:flex;flex-direction:column;justify-content:center;gap:6px}
.pass-stub:before,.pass-stub:after{content:"";position:absolute;left:-13px;width:24px;height:24px;border-radius:50%;background:var(--board)}
.pass-stub:before{top:-12px}.pass-stub:after{bottom:-12px}
.pass-top{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.78rem;letter-spacing:.16em;color:var(--muted);text-transform:uppercase}
.route{display:flex;align-items:center;gap:14px;margin:14px 0 18px}
.ap{min-width:0}.ap.r{text-align:right}
.code{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:clamp(2.4rem,5vw,3.6rem);line-height:1;color:var(--ink)}
.city{color:var(--muted);font-size:.85rem;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px}
.path{flex:1;min-width:80px}
.fields{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;border-top:1px solid var(--line);padding-top:14px}
@media (max-width:700px){.fields{grid-template-columns:repeat(2,minmax(0,1fr))}}
.f .k{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
.f .v{color:var(--ink);font-weight:600;font-size:1.02rem;margin-top:2px}
.chip{display:inline-block;align-self:flex-start;font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.14em;padding:4px 10px;border-radius:999px;border:1px solid currentColor;text-transform:uppercase}
.stub-big{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:2.8rem;line-height:1}
.stub-sub{color:var(--text);font-size:.92rem}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:1.1rem}
@media (max-width:900px){.tiles{grid-template-columns:minmax(0,1fr)}}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;min-width:0;display:flex;flex-direction:column;gap:8px}
.tile .k{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;display:flex;align-items:center;gap:8px}
.tile .big{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:2.4rem;line-height:1.05;color:var(--ink)}
.tile .note{color:var(--muted);font-size:.85rem}
.bar{height:12px;border-radius:6px;background:var(--tile);overflow:hidden}.bar i{display:block;height:100%;border-radius:6px}
.barrow{display:grid;grid-template-columns:92px 1fr 44px;gap:10px;align-items:center;font-size:.85rem;color:var(--text)}
.dep{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.88rem;color:var(--text);border-top:1px solid var(--line);padding-top:6px}
.tips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:.2rem 0 1.2rem}
@media (max-width:900px){.tips{grid-template-columns:minmax(0,1fr)}}
.tip{background:var(--tile);border-radius:14px;padding:16px 18px;display:flex;gap:12px;align-items:flex-start;min-width:0}
.tip .tag{font-family:'JetBrains Mono',monospace;font-size:.7rem;background:var(--amber);color:#0F1720;border-radius:5px;padding:2px 7px;white-space:nowrap;margin-top:2px}
.tip .t{color:var(--text);font-size:.92rem;line-height:1.45}.tip b{color:var(--ink)}
.status{display:flex;flex-wrap:wrap;gap:10px 26px;align-items:center;background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--amber);border-radius:14px;padding:14px 18px;margin-bottom:1rem}
.status .k{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
.status .v{font-family:'JetBrains Mono',monospace;color:var(--ink);font-weight:700}
.foot{color:var(--muted);font-size:.8rem;font-family:'JetBrains Mono',monospace;letter-spacing:.06em;margin-top:1.5rem;border-top:1px solid var(--line);padding-top:12px}
</style>""", unsafe_allow_html=True)

D, CITIES, SCHED, ROUTE_CARS = load()
city = lambda c: CITIES.get(c, c)

# ---------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.markdown('<div class="brand">✈ FLIGHT RISK CHECK</div>', unsafe_allow_html=True)
    st.markdown("#### Your flight")
    origins = sorted({o for (o, _) in ROUTE_CARS})
    o = st.selectbox("From", origins, index=origins.index("DFW"), format_func=lambda a: f"{a} · {city(a)}", key="from")
    dests = sorted({d for (oo, d) in ROUTE_CARS if oo == o})
    de = st.selectbox("To", dests, index=dests.index("LGA") if "LGA" in dests else 0,
                      format_func=lambda a: f"{a} · {city(a)}", key="to")
    cars = [c for c in AIRLINES if c in ROUTE_CARS[(o, de)]]
    car = st.selectbox("Airline", cars, index=cars.index("AA") if "AA" in cars else 0,
                       format_func=lambda c: f"{AIRLINES[c]} ({c})", key="airline")
    dist, el, rf, hours = D["routes"][car][o][de]
    flown = sorted(int(h) for h, v in hours.items() if v >= 0.05)
    default_day = date.today() + timedelta(days=(6 - date.today().weekday()) % 7 + 7)
    day = st.date_input("Travel date", value=default_day, key="date")
    slots = [h + m / 60 for h in range(5, 24) for m in (0, 30)]
    default_slot = 18.5 if 18 in flown else (flown[-1] + 0.0 if flown else 9.0)
    dep_hr = st.selectbox("Departure time", slots, index=slots.index(default_slot) if default_slot in slots else 0,
                          format_func=lambda h: clock(h) + ("" if int(h) in flown else "  · no flights this hour"), key="dep")
    month, dow = day.month, day.isoweekday()

    st.divider()
    st.markdown("#### On the day of travel")
    MODES = ["Not known yet", "First flight of the day for this plane", "Incoming plane is running late"]
    st.session_state.setdefault("ib_mode", MODES[0])
    st.session_state.setdefault("ib_delay", 45)
    if AV_KEY or AERO_KEY:
        flight_no = st.text_input("Flight number for live status", placeholder=f"e.g. {car}2117", key="live_flight")
        if st.button("Get live status", type="primary", width="stretch"):
            with st.spinner("Checking live status…"):
                res = live_status(live.clean_flight(flight_no))
            st.session_state["live_res"] = res
            if not res.get("error"):
                mins, _ = live.delay_for_model(res)
                if mins is not None and mins > 0:
                    st.session_state["ib_mode"] = MODES[2]
                    st.session_state["ib_delay"] = int(min(180, 5 * round(mins / 5)))
    else:
        st.caption("Live status is off. Add a free AviationStack key in the app’s Secrets to turn it on.")
    mode = st.radio("Incoming plane", MODES, key="ib_mode")
    prev = None
    if mode == MODES[1]:
        prev = -999
    elif mode == MODES[2]:
        prev = st.slider("Minutes late leaving its previous airport", 0, 180, step=5, key="ib_delay")
    st.divider()
    st.caption("Team 5 · Ahrar Karim · Hamza Tariq · Vishesh Goyal\n\nQuestrom MSBA Chart Champ 2026")

# ---------------------------------------------------------------- compute
ps = risk(D, "sev", car, o, de, month, dow, dep_hr, prev)
pl = risk(D, "late15", car, o, de, month, dow, dep_hr, prev)
col, head, line = verdict(ps)
bk = backups(SCHED, o, de, month, dow, dep_hr)
same = [b for b in bk if b[0] == car]
n_backup = len(same) if f"{o}|{de}" in SCHED else round(later_flights(hours, int(dep_hr)))
base = ps if prev is None else risk(D, "sev", car, o, de, month, dow, dep_hr)
best = min(((h, risk(D, "sev", car, o, de, month, dow, h + 0.5)) for h in flown), key=lambda t: t[1], default=None)

# ---------------------------------------------------------------- header
st.markdown('<div class="eyebrow">Departures · risk check · BTS 2025 model</div>', unsafe_allow_html=True)
st.markdown("## How likely is your flight to go wrong?")

# live status strip
res = st.session_state.get("live_res")
if res:
    if res.get("error"):
        st.warning(res["error"])
    else:
        _, how = live.delay_for_model(res)
        st.markdown(f"""<div class="status">
          <div><div class="k">Live · {escape(res['source'])}</div><div class="v">{escape(str(res['flight']))}</div></div>
          <div><div class="k">Status</div><div class="v">{escape(res['status'].title())}</div></div>
          <div><div class="k">Departure delay</div><div class="v">{'–' if res['dep_delay'] is None else str(res['dep_delay']) + ' min'}</div></div>
          <div><div class="k">Route</div><div class="v">{escape(str(res.get('origin') or '?'))} → {escape(str(res.get('dest') or '?'))}</div></div>
          <div><div class="k">Gate</div><div class="v">{escape(str(res.get('gate') or '–'))}</div></div>
          <div style="flex-basis:100%;color:#8E9AA8;font-size:.85rem">{escape(how)}</div></div>""", unsafe_allow_html=True)
        if "cancel" in res["status"]:
            st.error("The airline shows this flight as cancelled. Open the Backup flights tab below.")
        if res.get("origin") and res.get("dest") and (res["origin"], res["dest"]) != (o, de):
            st.info(f"That flight runs {res['origin']} → {res['dest']}. Change From and To in the sidebar to match it.")

# boarding pass
path_svg = (f'<svg viewBox="0 0 300 60" width="100%" height="60" aria-hidden="true">'
            f'<path d="M6 50 Q150 -10 294 50" fill="none" stroke="#3D4D60" stroke-width="2.5" stroke-dasharray="6 7"/>'
            f'<circle cx="6" cy="50" r="4" fill="#3D4D60"/><circle cx="294" cy="50" r="4" fill="{AMBER}"/>'
            f'<g transform="translate(134 4) scale(.5) rotate(90 32 32)"><path d="{PLANE}" fill="{AMBER}"/></g></svg>')
st.markdown(f"""<div class="pass">
  <div class="pass-main">
    <div class="pass-top"><span>{escape(AIRLINES[car])} · {car}</span><span>{'Day-of estimate' if prev is not None else 'Booking-time estimate'}</span></div>
    <div class="route">
      <div class="ap"><div class="code">{o}</div><div class="city">{escape(city(o))}</div></div>
      <div class="path">{path_svg}</div>
      <div class="ap r"><div class="code">{de}</div><div class="city">{escape(city(de))}</div></div>
    </div>
    <div class="fields">
      <div class="f"><div class="k">Date</div><div class="v">{day:%a %b} {day.day}</div></div>
      <div class="f"><div class="k">Departs</div><div class="v">{clock(dep_hr)}</div></div>
      <div class="f"><div class="k">Distance</div><div class="v">{dist:,} mi</div></div>
      <div class="f"><div class="k">Flights / day</div><div class="v">{rf:.1f}</div></div>
    </div>
  </div>
  <div class="pass-stub">
    <span class="chip" style="color:{col}">{head}</span>
    <div class="stub-big" style="color:{col}">{pct(ps)}</div>
    <div class="stub-sub">About <b>1 in {one_in(ps)}</b> flights like this is cancelled, diverted or 60+ min late.</div>
  </div>
</div>""", unsafe_allow_html=True)

# tiles
late_w, avg_w = min(100, pl * 100), AVG_LATE * 100
next_deps = "".join(f'<div class="dep"><span>{hhmm(b[2])}</span><span>{b[0]} {b[1]}</span></div>' for b in same[:3])
st.markdown(f"""<div class="tiles">
  <div class="tile"><div class="k">Serious disruption</div><div class="big" style="color:{col}">{pct(ps)}</div>{gauge(ps, col)}
    <div class="note">{line} Dashed tick: average flight (9.6%).</div></div>
  <div class="tile"><div class="k">Arrives 15+ min late or cancelled</div>
    <div class="big">{pct(pl)}</div>
    <div class="barrow"><span>This flight</span><div class="bar"><i style="width:{late_w:.0f}%;background:{col}"></i></div><span>{round(pl*100)}%</span></div>
    <div class="barrow"><span>Average</span><div class="bar"><i style="width:{avg_w:.0f}%;background:#5C6E84"></i></div><span>{round(AVG_LATE*100)}%</span></div>
    <div class="note">The standard “delayed” definition used by the U.S. DOT.</div></div>
  <div class="tile"><div class="k">Backup flights, same airline</div>
    <div class="big" style="color:{RED if n_backup < 1 else (AMBER if n_backup == 1 else GREEN)}">{'None' if n_backup < 1 else n_backup}</div>
    {next_deps or '<div class="note">No later flight on this airline that day. A cancellation likely means staying overnight.</div>'}
    <div class="note">From 2025 schedules for {day:%A}s in {day:%B}.</div></div>
</div>""", unsafe_allow_html=True)

# tips
tips = []
if best and best[1] < base - 0.01:
    tips.append(("Rule 1", f"<b>Fly earlier.</b> Around {clock(best[0])} the risk on {escape(short(car))} drops to <b>{pct(best[1])}</b>."))
elif best:
    tips.append(("Rule 1", "<b>Good timing.</b> This is about the safest hour this airline flies the route."))
if n_backup < 1:
    tips.append(("Rule 2", "<b>No backup flight.</b> For an important trip, fly earlier or the day before."))
elif n_backup == 1:
    tips.append(("Rule 2", "<b>Only one backup</b> later that day. A cancellation could still strand you."))
else:
    tips.append(("Rule 2", f"<b>{n_backup} backups</b> later that day on {escape(short(car))} if this flight is cancelled."))
if prev is None:
    tips.append(("Rule 3", "<b>On the day,</b> check the incoming plane about 3 hours before departure; rebook if it is 45+ min late."))
elif prev >= 45:
    tips.append(("Rule 3", "<b>Incoming plane is 45+ min late.</b> Half of these flights end badly: look at rebooking now."))
else:
    tips.append(("Rule 3", "<b>Incoming plane looks fine.</b> Recheck closer to departure."))
st.markdown("#### How to lower your risk")
st.markdown('<div class="tips">' + "".join(f'<div class="tip"><span class="tag">{t}</span><span class="t">{x}</span></div>'
                                          for t, x in tips) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
t1, t2, t3, t4 = st.tabs([":material/schedule: Best time to fly", ":material/luggage: Backup flights",
                          ":material/flight: Other airlines", ":material/info: How it works"])

with t1:
    rows = []
    for h in range(5, 23):
        p = risk(D, "sev", car, o, de, month, dow, h + 0.5)
        status = "Your departure" if h == int(dep_hr) else ("Airline flies this hour" if h in flown else "No scheduled flight")
        rows.append({"Hour": clock(h), "order": h, "Risk": p, "Status": status})
    cdf = pd.DataFrame(rows)
    chart = (alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
             .encode(x=alt.X("Hour:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("Risk:Q", axis=alt.Axis(format="%", grid=True, gridColor="#243040"), title="Serious disruption"),
                     color=alt.Color("Status:N", scale=alt.Scale(domain=["Your departure", "Airline flies this hour", "No scheduled flight"],
                                                                  range=[AMBER, "#5C6E84", "#2A3645"]),
                                     legend=alt.Legend(orient="bottom", title=None)),
                     tooltip=[alt.Tooltip("Hour:N"), alt.Tooltip("Risk:Q", format=".1%"), alt.Tooltip("Status:N")])
             .properties(height=300))
    rule = alt.Chart(pd.DataFrame({"y": [AVG_SEVERE]})).mark_rule(strokeDash=[6, 4], color="#8E9AA8").encode(y="y:Q")
    st.altair_chart((chart + rule).configure_view(strokeWidth=0), width="stretch")
    st.caption(f"Booking-time risk for {short(car)} {o} → {de} on {day:%A}s in {day:%B}, by departure hour. "
               "Dashed line: the average 2025 flight (9.6%).")

with t2:
    if not bk:
        st.warning(f"No later flights on {o} → {de} ran on {day:%A}s in {day:%B} 2025 on any of the ten airlines. "
                   "If this flight is cancelled you are likely stuck until the next day.")
    else:
        brows = []
        for c, fn, dep, n, sev_hist, _ in bk:
            h = dep // 100 + (dep % 100) / 60
            has_model = D["routes"].get(c, {}).get(o, {}).get(de) is not None
            brows.append({"Departs": hhmm(dep), "Flight": f"{c} {fn}", "Airline": short(c),
                          "Same airline": "Yes" if c == car else "No",
                          "Chance of serious disruption": 100 * risk(D, "sev", c, o, de, month, dow, h) if has_model else None,
                          "Severe in 2025": float(sev_hist), "Flown in 2025": n})
        bdf = pd.DataFrame(brows)
        st.dataframe(bdf, hide_index=True, width="stretch",
                     column_config={
                         "Chance of serious disruption": st.column_config.ProgressColumn(
                             format="%.0f%%", min_value=0, max_value=max(30.0, float(bdf["Chance of serious disruption"].max() or 0))),
                         "Severe in 2025": st.column_config.NumberColumn(
                             format="%.0f%%", help="Share of this flight’s 2025 operations that were cancelled, diverted or 60+ min late."),
                         "Flown in 2025": st.column_config.NumberColumn(format="%d times")})
        st.caption(f"{len(same)} on {short(car)} and {len(bk) - len(same)} on other airlines departed {o} → {de} after "
                   f"{clock(dep_hr)} on {day:%A}s in {day:%B} 2025. Same-airline flights are usually the easiest free rebooking. "
                   "Schedules change, so confirm on the airline’s site.")

with t3:
    others = []
    for c in cars:
        r = D["routes"][c][o][de]
        others.append({"Airline": f"{AIRLINES[c]} ({c})",
                       "Serious disruption": 100 * risk(D, "sev", c, o, de, month, dow, dep_hr),
                       "15+ min late": 100 * risk(D, "late15", c, o, de, month, dow, dep_hr),
                       "Flights a day": r[2]})
    odf = pd.DataFrame(others).sort_values("Serious disruption")
    st.dataframe(odf, hide_index=True, width="stretch",
                 column_config={"Serious disruption": st.column_config.ProgressColumn(
                                    format="%.0f%%", min_value=0, max_value=max(30.0, float(odf["Serious disruption"].max()))),
                                "15+ min late": st.column_config.NumberColumn(format="%.0f%%"),
                                "Flights a day": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"Every airline in the data that flies {o} → {de}, at {clock(dep_hr)} on {day:%A, %B} {day.day} (booking-time estimate).")

with t4:
    a, b, c = st.columns(3)
    a.metric("Flights it learned from", "6.4M", help="All 2025 flights by the ten airlines in the BTS on-time data.")
    b.metric("Accuracy at booking (AUC)", "0.69", help="0.79 once the incoming plane’s delay is known.")
    c.metric("Calibration", "±1.5 pts", help="When it says 10%, about 10% of those flights were disrupted, on held-out weeks.")
    st.markdown("""
- **Model:** gradient-boosted decision trees (LightGBM, 400 trees). Inputs: airline, airports, route, month, weekday, departure
  time, distance, how busy the airports are at that hour, and later flights on the route. The day-of version adds how late
  the incoming plane left its previous airport.
- **Tested** on 1.2 million flights from weeks the model never saw; probabilities calibrated on a separate set of weeks.
- **Backup flights** come from 2025 schedules: flights that ran on this route on the same weekday in the same month.
- **Limits:** it cannot see the weather or air-traffic problems on your actual day, fares or seat availability, and it reflects
  2025 patterns. Treat it as a risk band, not a promise.
""")

st.markdown('<div class="foot">TEAM 5 · AHRAR KARIM · HAMZA TARIQ · VISHESH GOYAL · QUESTROM MSBA CHART CHAMP 2026 · DATA: BTS ON-TIME PERFORMANCE 2025</div>',
            unsafe_allow_html=True)

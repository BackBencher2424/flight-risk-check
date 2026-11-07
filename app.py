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
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Risk Check", page_icon="✈️", layout="wide")

HERE = Path(__file__).parent
AIRLINES = {
    "AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta",
    "MQ": "Envoy (flies as American Eagle)", "OH": "PSA (flies as American Eagle)",
    "OO": "SkyWest (flies for United, Delta, American, Alaska)", "UA": "United",
    "WN": "Southwest", "YX": "Republic (flies for American, Delta, United)",
}
AVG_SEVERE, AVG_LATE = 0.096, 0.235  # all 2025 flights in the data


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
    return data, cities, sched


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


# ---------------------------------------------------------------- helpers
def pct(p):
    return "<1%" if p < 0.01 else (f"{p*100:.1f}%" if p < 0.1 else f"{round(p*100)}%")


def one_in(p):
    return max(2, round(1 / p)) if p > 0 else 100


def clock(h):
    hh, mm = int(h), int(round((h - int(h)) * 60))
    return time(hh % 24, mm).strftime("%-I:%M %p").replace(":00", "")


def verdict(p):
    if p < 0.06:
        return "success", "Looks smooth", "This is a low-risk flight."
    if p < 0.12:
        return "info", "About average", "Typical risk for a U.S. flight."
    if p < 0.20:
        return "warning", "Risky", "Noticeably riskier than the average flight."
    return "error", "High risk", "One of the riskier ways to make this trip."


# ---------------------------------------------------------------- page
D, CITIES, SCHED = load()
city = lambda c: CITIES.get(c, c)

st.markdown(
    """<style>
    .block-container{padding-top:4rem;max-width:1200px}
    div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace}
    .eyebrow{font-family:'JetBrains Mono',monospace;letter-spacing:.2em;color:#F5B841;font-size:.8rem}
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">DEPARTURES · RISK CHECK · BTS 2025 MODEL</div>', unsafe_allow_html=True)
st.title("How likely is your flight to go wrong?")
st.caption("Enter a flight the way you would book it. You get the chance it is cancelled, diverted or more "
           "than an hour late, and what would make the trip safer. Built by Team 5 for the Questrom Chart Champ datathon.")

# --- step 1: the flight
st.subheader("1 · Your flight")
c1, c2, c3 = st.columns(3)
carriers = [c for c in AIRLINES if c in D["routes"]]
car = c1.selectbox("Airline", carriers, index=carriers.index("AA"),
                   format_func=lambda c: f"{AIRLINES[c]} ({c})")
origins = sorted(D["routes"][car])
o = c2.selectbox("From", origins, index=origins.index("DFW") if "DFW" in origins else 0,
                 format_func=lambda a: f"{a} · {city(a)}")
dests = sorted(D["routes"][car][o])
de = c3.selectbox("To", dests, index=dests.index("LGA") if "LGA" in dests else 0,
                  format_func=lambda a: f"{a} · {city(a)}")

dist, el, rf, hours = D["routes"][car][o][de]
flown = sorted(int(h) for h, v in hours.items() if v >= 0.05)
c4, c5 = st.columns(2)
default_day = date.today() + timedelta(days=(6 - date.today().weekday()) % 7 + 7)  # a Sunday ~2 weeks out
day = c4.date_input("Travel date", value=default_day)
slots = [h + m / 60 for h in range(5, 24) for m in (0, 30)]
default_slot = 18.5 if 18 in flown else (flown[-1] if flown else 9)
dep_hr = c5.selectbox("Scheduled departure", slots, index=slots.index(default_slot) if default_slot in slots else 0,
                      format_func=clock)
month, dow = day.month, day.isoweekday()  # BTS weekday: Monday = 1 ... Sunday = 7
if flown:
    st.caption(f"{AIRLINES[car].split(' (')[0]} flies {o} → {de} about {rf:.1f} times a day, "
               f"usually between {clock(flown[0])} and {clock(flown[-1] + 0.99).replace(':59', '')}. "
               f"{dist:,} miles, about {round(el/60, 1)} hours.")

# --- step 2: day of travel
st.subheader("2 · On the day of travel (optional)")
mode = st.radio("Do you know how the incoming plane is doing?",
                ["Not yet", "My flight is the plane’s first of the day", "The incoming plane is running late"],
                horizontal=True)
prev = None
if mode.startswith("My flight"):
    prev = -999
elif mode.startswith("The incoming"):
    prev = st.slider("How late did it leave its previous airport? (minutes)", 0, 180, 45, 5)

# --- results
ps = risk(D, "sev", car, o, de, month, dow, dep_hr, prev)
pl = risk(D, "late15", car, o, de, month, dow, dep_hr, prev)
kind, head, line = verdict(ps)
st.subheader("3 · Your risk")
getattr(st, kind)(f"**{head}.** About **1 in {one_in(ps)}** flights like this one is cancelled, diverted "
                  f"or more than an hour late. {line}")
m1, m2, m3 = st.columns(3)
m1.metric("Serious disruption", pct(ps), f"{(ps-AVG_SEVERE)*100:+.1f} pts vs average flight", delta_color="inverse",
          help="Cancelled, diverted or arriving more than 60 minutes late. Average 2025 flight: 9.6%.")
m2.metric("Arrives 15+ min late or cancelled", pct(pl), f"{(pl-AVG_LATE)*100:+.1f} pts vs average flight",
          delta_color="inverse", help="The standard 'delayed' definition. Average 2025 flight: 23.5%.")
bk = backups(SCHED, o, de, month, dow, dep_hr)
same_airline = [b for b in bk if b[0] == car]
later = len(same_airline) if f"{o}|{de}" in SCHED else round(later_flights(hours, int(dep_hr)))
m3.metric("Backup flights later that day", "None" if later < 1 else str(later),
          help="Flights on the same airline and route that departed later the same day in 2025 schedules: "
               "your free rebooking options if this flight is cancelled.")
st.caption("Booking-time estimate from 2025 patterns." if prev is None else
           "Day-of estimate: includes how the incoming plane is doing.")

# --- step 4: backup flights
st.subheader("4 · Backup flights if yours is cancelled")
if st.toggle("Show backup flights", value=True, key="show_backups"):
    if not bk:
        st.warning(f"No later flights on {o} → {de} ran on {day:%A}s in {day:%B} 2025, on any of the ten airlines. "
                   "If this flight is cancelled you are likely stuck until the next day. Consider an earlier flight "
                   "or a nearby airport.")
    else:
        rows = []
        for c, fn, dep, n, sev_hist, _ in bk:
            h = dep // 100 + (dep % 100) / 60
            has_model = D["routes"].get(c, {}).get(o, {}).get(de) is not None
            rows.append({
                "Departs": clock(h),
                "Airline": AIRLINES.get(c, c).split(" (")[0] + f" ({c})",
                "Flight": f"{c} {fn}",
                "Same airline": "Yes" if c == car else "No",
                "Chance of serious disruption": 100 * risk(D, "sev", c, o, de, month, dow, h) if has_model else None,
                "Severe in 2025": float(sev_hist),
                "Flown in 2025": n,
            })
        bdf = pd.DataFrame(rows)
        st.dataframe(bdf, hide_index=True, width="stretch",
                     column_config={
                         "Chance of serious disruption": st.column_config.ProgressColumn(
                             format="%.0f%%", min_value=0, max_value=max(30.0, float(bdf["Chance of serious disruption"].max() or 0))),
                         "Severe in 2025": st.column_config.NumberColumn(
                             format="%.0f%%", help="Share of this flight’s 2025 operations that were cancelled, diverted or 60+ min late."),
                         "Flown in 2025": st.column_config.NumberColumn(format="%d times")})
        st.caption(f"{len(same_airline)} on {AIRLINES[car].split(' (')[0]} and {len(bk) - len(same_airline)} on other airlines "
                   f"departed {o} → {de} after {clock(dep_hr)} on {day:%A}s in {day:%B} 2025. Same-airline flights are usually "
                   "the easiest free rebooking; other airlines need a new ticket or the airline’s agreement. "
                   "Schedules change, so confirm on the airline’s site.")

# --- step 5: how to lower the risk
st.subheader("5 · How to lower your risk")
tips = []
base = ps if prev is None else risk(D, "sev", car, o, de, month, dow, dep_hr)
best = min(((h, risk(D, "sev", car, o, de, month, dow, h + 0.5)) for h in flown), key=lambda t: t[1], default=None)
if best and best[1] < base - 0.01:
    tips.append(f"**Fly earlier.** Around {clock(best[0])} on this airline the risk drops to **{pct(best[1])}** "
                f"(from {pct(base)}).")
elif best:
    tips.append("**Timing is already good.** This is about the safest hour this airline flies the route.")
if later < 1:
    tips.append("**No backup flight.** This looks like the airline’s last flight of the day on this route. "
                "If it is cancelled you are likely stuck overnight, so for an important trip fly earlier or the day before.")
elif later == 1:
    tips.append("**Only one backup flight** later that day. A cancellation could still strand you overnight.")
if prev is None:
    tips.append("**On the day,** check the airline’s flight-status page about three hours before departure. "
                "If the incoming plane is 45+ minutes late, rebook while there are seats.")
elif prev >= 45:
    tips.append("**The incoming plane is 45+ minutes late.** Half of flights in this situation end badly: look at rebooking now.")
for t in tips:
    st.markdown(f"- {t}")

others = []
for c in carriers:
    r = D["routes"].get(c, {}).get(o, {}).get(de)
    if r:
        others.append({"Airline": AIRLINES[c].split(" (")[0] + f" ({c})",
                       "Serious disruption": 100 * risk(D, "sev", c, o, de, month, dow, dep_hr),
                       "15+ min late": 100 * risk(D, "late15", c, o, de, month, dow, dep_hr),
                       "Flights a day": r[2]})
if len(others) > 1:
    st.markdown(f"**Other airlines on {o} → {de}** at the same time and date (booking-time estimate):")
    odf = pd.DataFrame(others).sort_values("Serious disruption")
    st.dataframe(odf, hide_index=True, width="stretch",
                 column_config={"Serious disruption": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=max(30.0, float(odf["Serious disruption"].max()))),
                                "15+ min late": st.column_config.NumberColumn(format="%.0f%%"),
                                "Flights a day": st.column_config.NumberColumn(format="%.1f")})

# --- chart: risk through the day
st.markdown(f"**Risk through the day for {AIRLINES[car].split(' (')[0]} {o} → {de}** on {day:%A, %B %-d}")
rows = []
for h in range(5, 23):
    p = risk(D, "sev", car, o, de, month, dow, h + 0.5)
    status = "Your departure" if h == int(dep_hr) else ("Airline flies this hour" if h in flown else "No scheduled flight")
    rows.append({"Hour": clock(h), "order": h, "Risk": p, "Status": status})
cdf = pd.DataFrame(rows)
chart = (alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
         .encode(x=alt.X("Hour:N", sort=alt.SortField("order"), title="Scheduled departure"),
                 y=alt.Y("Risk:Q", axis=alt.Axis(format="%"), title="Serious disruption"),
                 color=alt.Color("Status:N", scale=alt.Scale(domain=["Your departure", "Airline flies this hour", "No scheduled flight"],
                                                              range=["#F5B841", "#5C6E84", "#2A3645"]),
                                 legend=alt.Legend(orient="bottom", title=None)),
                 tooltip=[alt.Tooltip("Hour:N"), alt.Tooltip("Risk:Q", format=".1%"), alt.Tooltip("Status:N")])
         .properties(height=280))
rule = alt.Chart(pd.DataFrame({"y": [AVG_SEVERE]})).mark_rule(strokeDash=[6, 4], color="#8E9AA8").encode(y="y:Q")
st.altair_chart(chart + rule, width="stretch")
st.caption("Dashed line: the average 2025 flight (9.6%).")

# --- about
with st.expander("How this works, and its limits"):
    st.markdown(
        """
- **Data:** all 6.4 million 2025 flights by American, Delta, United, Southwest, Alaska, JetBlue, SkyWest, Republic, Envoy and PSA (U.S. DOT / BTS on-time data).
- **Model:** gradient-boosted decision trees (LightGBM, 400 trees). Inputs: airline, airports, route, month, weekday, departure time, distance, how busy the airports are at that hour, and how many later flights the airline has on the route. The day-of version adds how late the incoming plane left its previous airport.
- **Tested on 1.2 million flights from weeks the model never saw.** The percentages are calibrated: when it says 10%, about 10% of those flights really were disrupted (within 1.5 points). Booking-time accuracy (AUC) 0.69; with the incoming plane 0.79.
- **Limits:** it cannot see the weather or air-traffic problems on your actual day, fares or seat availability, and it reflects 2025 patterns. Treat it as a risk band, not a promise.
        """
    )
st.caption("Team 5 · Ahrar Karim · Hamza Tariq · Vishesh Goyal · Questrom MSBA Chart Champ 2026")

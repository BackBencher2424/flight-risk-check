"""Live flight status lookups for Flight Risk Check.

Two optional providers, chosen by which key is present in Streamlit secrets:
  AVIATIONSTACK_KEY  free plan, 100 requests a month. Gives the flight's own live status and
                     departure delay; the incoming-plane delay is approximated by that delay.
  AEROAPI_KEY        FlightAware AeroAPI (Personal tier, $5 free usage a month, payment details
                     required). Also returns the inbound aircraft's flight, so its real delay is used.
Every function returns a plain dict; on failure the dict has an "error" message instead of raising.
"""
from datetime import date, datetime, timezone

import requests

TIMEOUT = 12


def clean_flight(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def _iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    except ValueError:
        return None


def aviationstack(key: str, flight: str) -> dict:
    flight = clean_flight(flight)
    if not flight:
        return {"error": "Enter a flight number such as AA2117."}
    last = "no response"
    for scheme in ("https", "http"):  # some free accounts only allow http
        try:
            r = requests.get(f"{scheme}://api.aviationstack.com/v1/flights",
                             params={"access_key": key, "flight_iata": flight}, timeout=TIMEOUT)
            j = r.json()
        except (requests.RequestException, ValueError) as e:
            last = str(e)[:120]
            continue
        if isinstance(j, dict) and j.get("error"):
            err = j["error"]
            code = str(err.get("code", ""))
            if "https" in code and scheme == "https":
                continue
            return {"error": f"AviationStack: {err.get('message') or code}"}
        data = (j or {}).get("data") or []
        if not data:
            return {"error": f"No live data for {flight} right now. Check the flight number, or try closer to departure."}
        today = date.today().isoformat()
        pick = next((d for d in data if d.get("flight_date") == today and d.get("flight_status") != "landed"),
                    next((d for d in data if d.get("flight_date") == today), data[0]))
        dep, arr = pick.get("departure") or {}, pick.get("arrival") or {}
        return {
            "source": "AviationStack", "flight": (pick.get("flight") or {}).get("iata") or flight,
            "date": pick.get("flight_date"), "status": (pick.get("flight_status") or "unknown").lower(),
            "origin": dep.get("iata"), "dest": arr.get("iata"),
            "scheduled": dep.get("scheduled"), "estimated": dep.get("estimated"), "actual": dep.get("actual"),
            "gate": dep.get("gate"), "dep_delay": dep.get("delay"),
            "inbound_delay": None, "inbound_flight": None,
        }
    return {"error": f"Could not reach AviationStack ({last})."}


def aeroapi(key: str, flight: str) -> dict:
    flight = clean_flight(flight)
    if not flight:
        return {"error": "Enter a flight number such as AA2117."}
    base, h = "https://aeroapi.flightaware.com/aeroapi", {"x-apikey": key}
    try:
        r = requests.get(f"{base}/flights/{flight}", headers=h, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"error": f"Could not reach FlightAware ({str(e)[:120]})."}
    if r.status_code != 200:
        return {"error": f"FlightAware returned {r.status_code}. Check the key and flight number."}
    flights = (r.json() or {}).get("flights") or []
    if not flights:
        return {"error": f"No live data for {flight} right now."}
    now = datetime.now(timezone.utc)
    live = [f for f in flights if not f.get("actual_in")] or flights
    pick = min(live, key=lambda f: abs((_iso(f.get("scheduled_out")) or now).timestamp() - now.timestamp()))
    secs = pick.get("departure_delay")
    out = {
        "source": "FlightAware", "flight": pick.get("ident_iata") or pick.get("ident") or flight,
        "date": (pick.get("scheduled_out") or "")[:10],
        "status": "cancelled" if pick.get("cancelled") else (pick.get("status") or "unknown").lower(),
        "origin": (pick.get("origin") or {}).get("code_iata"), "dest": (pick.get("destination") or {}).get("code_iata"),
        "scheduled": pick.get("scheduled_out"), "estimated": pick.get("estimated_out"), "actual": pick.get("actual_out"),
        "gate": pick.get("gate_origin"), "dep_delay": None if secs is None else round(secs / 60),
        "inbound_delay": None, "inbound_flight": None,
    }
    ib = pick.get("inbound_fa_flight_id")
    if ib:
        try:
            r2 = requests.get(f"{base}/flights/{ib}", headers=h, timeout=TIMEOUT)
            prev = ((r2.json() or {}).get("flights") or [None])[0] if r2.status_code == 200 else None
        except (requests.RequestException, ValueError):
            prev = None
        if prev:
            s2 = prev.get("departure_delay")
            out["inbound_delay"] = None if s2 is None else round(s2 / 60)
            out["inbound_flight"] = f"{prev.get('ident_iata') or prev.get('ident')} from {(prev.get('origin') or {}).get('code_iata')}"
    return out


def delay_for_model(res: dict):
    """Minutes to feed the day-of model, and a sentence saying where it came from."""
    if res.get("inbound_delay") is not None:
        return max(0, res["inbound_delay"]), f"Incoming plane {res['inbound_flight']} left {res['inbound_delay']} min late."
    if res.get("dep_delay") is not None:
        return max(0, res["dep_delay"]), (f"The airline expects your flight to leave {res['dep_delay']} min late; "
                                          "used as the incoming plane’s delay.")
    return None, "No delay reported yet."

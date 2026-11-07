# Flight Risk Check

Team 5 (Ahrar Karim, Hamza Tariq, Vishesh Goyal), Questrom MSBA Chart Champ 2026.

Pick From and To, then the airline, travel date and departure time in the sidebar. The app gives the calibrated chance the flight is
cancelled, diverted or more than an hour late (and the chance it arrives 15+ minutes late), suggests a safer
departure time, compares other airlines on the route, and lists real backup flights later that day (from 2025 schedules for that month and weekday). On the day
of travel you can add how late the incoming plane is.

## Run it on your laptop

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

## Host it on Streamlit Community Cloud (free, about 5 minutes)

1. Create a new GitHub repository and upload everything in this folder, keeping the structure:
   `app.py`, `live.py`, `requirements.txt`, `README.md`, `model/calc_data.json`, `model/cities.json`, `model/schedule.json.gz`, `.streamlit/config.toml`.
   (On GitHub's web upload, drag the whole folder in; the `.streamlit` folder is hidden on macOS, so press
   Cmd+Shift+. in Finder to show it.)
2. Go to share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. The first build takes a couple of minutes; you then get a public link to share.

## What is inside

- `model/calc_data.json`: four gradient-boosted tree models (LightGBM, 400 trees each) exported as plain
  arrays, their isotonic calibration curves, and route and airport lookups. The app evaluates the trees in pure
  Python, so it needs no machine-learning libraries to run.
- `model/schedule.json.gz`: every flight that ran at least 4 times in 2025 (304,567 flights on 5,534 routes) with its scheduled time, how often it flew, its own severe-disruption rate, and which month-and-weekday combinations it operated. Used for the backup-flight list.
- Models were trained on 2025 BTS on-time data for 10 airlines (6.4 million flights) and tested on 1.2 million
  flights from held-out weeks: booking-time AUC 0.69, day-of AUC 0.79, probabilities within 1.5 points of
  observed rates.
- Limits: the model cannot see weather or air-traffic problems on the actual day, fares or seat availability,
  and it reflects 2025 patterns.

## Live flight status (optional)

The app can check a flight's real-time status and fill in the day-of delay automatically. Without a key,
everything else works and the live box just says the lookup is off.

**Free option: AviationStack** (100 lookups a month on the free plan). It gives the flight's live status,
gate and departure delay; that delay is used as the incoming plane's delay.

1. Go to aviationstack.com and click **Get Free API Key**. Pick the **Free** plan and create an account.
2. Confirm your email, log in, and copy **Your API Access Key** from the dashboard.
3. On Streamlit Community Cloud: open your app, click the **⋮** menu (or **Manage app**), then
   **Settings → Secrets**, paste the line below with your key, and click **Save**. The app restarts by itself.

   ```
   AVIATIONSTACK_KEY = "paste-your-key-here"
   ```

4. Running on your laptop instead: create the file `.streamlit/secrets.toml` with the same line.
   Never upload `secrets.toml` to GitHub.

Each click on **Get live status** uses one lookup (results are cached for 5 minutes), so 100 a month is
plenty for a demo; avoid clicking repeatedly.

**Paid option: FlightAware AeroAPI** (Personal tier: up to $5 of usage free a month, but payment details are
required at signup). It also reports the inbound aircraft's own delay, which is exactly what the day-of model
was trained on. Add it the same way as `AEROAPI_KEY = "..."`. If both keys are present, FlightAware is used.

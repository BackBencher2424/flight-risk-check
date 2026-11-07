# Flight Risk Check

Team 5 (Ahrar Karim, Hamza Tariq, Vishesh Goyal), Questrom MSBA Chart Champ 2026.

Enter an airline, route, travel date and departure time. The app gives the calibrated chance the flight is
cancelled, diverted or more than an hour late (and the chance it arrives 15+ minutes late), suggests a safer
departure time, compares other airlines on the route, and counts backup flights later that day. On the day
of travel you can add how late the incoming plane is.

## Run it on your laptop

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

## Host it on Streamlit Community Cloud (free, about 5 minutes)

1. Create a new GitHub repository and upload everything in this folder, keeping the structure:
   `app.py`, `requirements.txt`, `README.md`, `model/calc_data.json`, `model/cities.json`, `.streamlit/config.toml`.
   (On GitHub's web upload, drag the whole folder in; the `.streamlit` folder is hidden on macOS, so press
   Cmd+Shift+. in Finder to show it.)
2. Go to share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. The first build takes a couple of minutes; you then get a public link to share.

## What is inside

- `model/calc_data.json`: four gradient-boosted tree models (LightGBM, 400 trees each) exported as plain
  arrays, their isotonic calibration curves, and route and airport lookups. The app evaluates the trees in pure
  Python, so it needs no machine-learning libraries to run.
- Models were trained on 2025 BTS on-time data for 10 airlines (6.4 million flights) and tested on 1.2 million
  flights from held-out weeks: booking-time AUC 0.69, day-of AUC 0.79, probabilities within 1.5 points of
  observed rates.
- Limits: the model cannot see weather or air-traffic problems on the actual day, fares or seat availability,
  and it reflects 2025 patterns.

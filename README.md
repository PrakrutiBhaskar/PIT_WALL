# 🏎️ Pit Wall — F1 Race Outcome Predictor

A machine learning pipeline that predicts Formula 1 race finishing 
positions lap by lap using tire degradation and pit stop strategy.The website link is https://pitwall-f1.streamlit.app

## Results
- **MAE: 1.56 positions** on the complete 2025 season (held-out test set)
- **Top 5 Finish Accuracy: 94.1%**
- Trained on 76,000+ laps across 4 seasons (2021–2024)
- Tested on 25,000+ laps from the 2025 season — never seen during training

## How It Works
1. FastF1 pulls lap-by-lap telemetry for every race
2. Feature engineering creates 14 signals including tire degradation 
   rate, position momentum, relative tyre age, and pit stop duration
3. LightGBM is trained on historical seasons to predict final position
4. A Streamlit dashboard visualises predictions lap by lap with 
   a race replay slider

## Features Engineered
- Tire degradation rate — how much slower vs fresh tyre pace
- Position momentum — rolling 5-lap average of position change
- Relative tyre age — fresher or older than the field average
- Gap to leader pace — competitive context vs P1
- Stint number, laps since pit, race completion percentage

## Tech Stack
FastF1 · LightGBM · SHAP · Pandas · Streamlit · Plotly

## Known Limitations
- DNF drivers are recorded at their last known position
- Cross-season generalization is limited by regulation changes
- Does not currently support live timing (planned future addition)

## How to Run
```bash
pip install -r requirements.txt
python src/data_pipeline.py
python src/features.py
python src/model.py
streamlit run src/app.py
```

## Project Structure
```
PIT_WALL/
├── src/
│   ├── data_pipeline.py   # FastF1 data extraction
│   ├── features.py        # Feature engineering
│   ├── model.py           # LightGBM training
│   └── app.py             # Streamlit dashboard
├── data/                  # Generated CSVs
├── models/                # Saved model + charts
└── requirements.txt
```

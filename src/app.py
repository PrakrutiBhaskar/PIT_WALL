import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------

st.set_page_config(
    page_title="Pit Wall — F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# -------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load('models/lgbm_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/featured_laps.csv')

model = load_model()
df    = load_data()

FEATURES = [
    'LapNumber', 'Position', 'TyreLife', 'CompoundEncoded',
    'DegradationRate', 'PositionMomentum', 'RaceCompletion',
    'PitThisLap', 'PitStopDuration', 'StintNumber',
    'GapToLeaderPace', 'RelativeTyreAge', 'LapsSincePit',
    'GridPosition'
]

COMPOUND_COLORS = {
    'SOFT':         '#FF3333',
    'MEDIUM':       '#FFD700',
    'HARD':         '#EEEEEE',
    'INTERMEDIATE': '#39B54A',
    'WET':          '#0067FF'
}

# -------------------------------------------
# HEADER
# -------------------------------------------

st.title("🏎️ Pit Wall — F1 Race Outcome Predictor")
st.markdown(
    "Predicting final race positions lap by lap using "
    "tire degradation and pit stop strategy."
)
st.divider()

# -------------------------------------------
# SIDEBAR
# -------------------------------------------

st.sidebar.header("Select Race & Driver")

year_options    = sorted(df['Year'].unique(), reverse=True)
selected_year   = st.sidebar.selectbox("Season", year_options)

race_options    = sorted(
    df[df['Year'] == selected_year]['Race'].unique()
)
selected_race   = st.sidebar.selectbox("Grand Prix", race_options)

driver_options  = sorted(
    df[
        (df['Year'] == selected_year) &
        (df['Race'] == selected_race)
    ]['Driver'].unique()
)
selected_driver = st.sidebar.selectbox("Driver", driver_options)

st.sidebar.divider()
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("MAE",            "1.56 positions")
st.sidebar.metric("Top 5 Accuracy", "94.1%")
st.sidebar.metric("Training Races", "57")
st.sidebar.metric("Total Laps",     "76,186")
st.sidebar.divider()
st.sidebar.markdown(
    "**Data:** 2021–2025 F1 Seasons  \n"
    "**Model:** LightGBM  \n"
    "**Test Set:** Full 2025 Season"
)

# -------------------------------------------
# FILTER DATA FOR SELECTED DRIVER + RACE
# -------------------------------------------

race_df = df[
    (df['Year']   == selected_year) &
    (df['Race']   == selected_race) &
    (df['Driver'] == selected_driver)
].copy().sort_values('LapNumber')

if race_df.empty:
    st.error("No data found for this selection.")
    st.stop()

race_df['PredictedFinalPosition'] = np.round(
    model.predict(race_df[FEATURES])
).astype(int).clip(1, 20)

actual_final    = int(race_df['FinalPosition'].iloc[-1])
predicted_final = int(race_df['PredictedFinalPosition'].iloc[-1])
error           = abs(actual_final - predicted_final)

# -------------------------------------------
# TOP METRICS ROW
# -------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Driver",         selected_driver)
col2.metric("Actual Finish",  f"P{actual_final}")
col3.metric(
    "Predicted Finish",
    f"P{predicted_final}",
    delta=f"{'+' if predicted_final > actual_final else ''}"
          f"{predicted_final - actual_final} positions"
)
col4.metric(
    "Prediction Error",
    f"{error} position{'s' if error != 1 else ''}"
)

st.divider()

# -------------------------------------------
# CHART 1 — LAP BY LAP PREDICTION
# -------------------------------------------

st.subheader(
    f"📈 Predicted vs Actual Final Position — "
    f"{selected_driver} at {selected_race} {selected_year}"
)

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=race_df['LapNumber'],
    y=race_df['PredictedFinalPosition'],
    name='Predicted Final Position',
    line=dict(color='#00D2FF', width=2),
    mode='lines'
))

fig1.add_hline(
    y=actual_final,
    line_dash='dash',
    line_color='#FF6B6B',
    annotation_text=f"Actual Finish: P{actual_final}",
    annotation_position="right"
)

pit_laps = race_df[race_df['PitThisLap'] == 1]
for _, pit in pit_laps.iterrows():
    fig1.add_vline(
        x=pit['LapNumber'],
        line_color='#FFD700',
        line_dash='dot',
        line_width=1.5,
        annotation_text="PIT",
        annotation_font_size=10,
        annotation_font_color='#FFD700'
    )

fig1.update_layout(
    xaxis_title="Lap Number",
    yaxis_title="Predicted Final Position",
    yaxis=dict(autorange='reversed', dtick=1),
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='white',
    legend=dict(bgcolor='#0E1117'),
    height=400
)

st.plotly_chart(fig1, use_container_width=False)

# -------------------------------------------
# CHART 2 — LAP TIMES WITH TIRE COMPOUNDS
# -------------------------------------------

st.subheader("🔄 Lap Times & Tire Strategy")

fig2 = go.Figure()

for compound in race_df['Compound'].unique():
    compound_laps = race_df[race_df['Compound'] == compound]
    color         = COMPOUND_COLORS.get(compound, '#FFFFFF')

    fig2.add_trace(go.Scatter(
        x=compound_laps['LapNumber'],
        y=compound_laps['LapTimeSeconds'],
        name=compound,
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color)
    ))

fig2.update_layout(
    xaxis_title="Lap Number",
    yaxis_title="Lap Time (seconds)",
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='white',
    legend=dict(bgcolor='#0E1117'),
    height=350
)

st.plotly_chart(fig2, use_container_width=False)

# -------------------------------------------
# CHART 3 — TIRE DEGRADATION RATE
# -------------------------------------------

st.subheader("📉 Tire Degradation Rate")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=race_df['LapNumber'],
    y=race_df['DegradationRate'],
    fill='tozeroy',
    line=dict(color='#FF6B6B', width=2),
    fillcolor='rgba(255, 107, 107, 0.2)',
    name='Degradation Rate'
))

fig3.add_hline(
    y=0,
    line_color='white',
    line_width=1,
    line_dash='dot'
)

fig3.update_layout(
    xaxis_title="Lap Number",
    yaxis_title="Seconds Slower Than Fresh Tire Pace",
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='white',
    height=300
)

st.plotly_chart(fig3, use_container_width=False)

# -------------------------------------------
# RACE REPLAY SLIDER
# -------------------------------------------

st.divider()
st.subheader("⏱️ Race Replay — Simulate Race Progress Lap by Lap")
st.markdown(
    "Drag the slider to any lap and see what the model "
    "was predicting at that moment in the race."
)

replay_lap = st.slider(
    "Select Lap",
    min_value=int(race_df['LapNumber'].min()),
    max_value=int(race_df['LapNumber'].max()),
    value=int(race_df['LapNumber'].min()),
    step=1
)

replay_df = race_df[
    race_df['LapNumber'] <= replay_lap
].copy()

current_pred = int(np.round(
    model.predict(replay_df[FEATURES].tail(1))[0]
).clip(1, 20))

current_pos  = int(replay_df['Position'].iloc[-1])
current_tyre = replay_df['Compound'].iloc[-1]
current_deg  = round(float(replay_df['DegradationRate'].iloc[-1]), 2)
laps_left    = int(race_df['LapNumber'].max()) - replay_lap

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Lap",               f"{replay_lap}")
col2.metric("Laps Remaining",    f"{laps_left}")
col3.metric("Current Position",  f"P{current_pos}")
col4.metric("Predicted Finish",  f"P{current_pred}")
col5.metric("Tyre Degradation",  f"+{current_deg}s")

fig_replay = go.Figure()

fig_replay.add_trace(go.Scatter(
    x=replay_df['LapNumber'],
    y=replay_df['PredictedFinalPosition'],
    name='Predicted Final Position',
    line=dict(color='#00D2FF', width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 210, 255, 0.1)'
))

fig_replay.add_hline(
    y=actual_final,
    line_dash='dash',
    line_color='#FF6B6B',
    annotation_text=f"Actual Finish: P{actual_final}",
    annotation_position="right"
)

# mark pit stops up to current lap
for _, pit in pit_laps[
    pit_laps['LapNumber'] <= replay_lap
].iterrows():
    fig_replay.add_vline(
        x=pit['LapNumber'],
        line_color='#FFD700',
        line_dash='dot',
        line_width=1.5,
        annotation_text="PIT",
        annotation_font_size=10,
        annotation_font_color='#FFD700'
    )

fig_replay.update_layout(
    xaxis_title="Lap Number",
    yaxis_title="Predicted Final Position",
    yaxis=dict(
        autorange='reversed',
        range=[20, 1],
        dtick=1
    ),
    xaxis=dict(
        range=[1, int(race_df['LapNumber'].max())]
    ),
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='white',
    height=350
)

st.plotly_chart(fig_replay, use_container_width=False)

# -------------------------------------------
# FULL RACE STANDINGS TABLE
# -------------------------------------------

st.divider()
st.subheader(
    f"🏁 Full Race Predicted Standings — "
    f"{selected_race} {selected_year}"
)

all_drivers_df = df[
    (df['Year'] == selected_year) &
    (df['Race'] == selected_race)
].copy()

last_laps = all_drivers_df.sort_values(
    'LapNumber'
).groupby('Driver').last().reset_index()

last_laps['PredictedFinalPosition'] = np.round(
    model.predict(last_laps[FEATURES])
).astype(int).clip(1, 20)

standings = last_laps[[
    'Driver', 'PredictedFinalPosition',
    'FinalPosition', 'Compound', 'TyreLife'
]].copy()

standings['Error'] = abs(
    standings['FinalPosition'] -
    standings['PredictedFinalPosition']
)
standings = standings.sort_values('PredictedFinalPosition')
standings.columns = [
    'Driver', 'Predicted Position',
    'Actual Position', 'Final Compound',
    'Final Tyre Age', 'Error (positions)'
]

st.dataframe(standings, hide_index=True)

# -------------------------------------------
# FOOTER
# -------------------------------------------

st.divider()
st.markdown(
    "Built with FastF1 · LightGBM · Streamlit &nbsp;|&nbsp; "
    "Data: 2021–2025 F1 Seasons &nbsp;|&nbsp; "
    "Trained on 76,186 laps &nbsp;|&nbsp; "
    "MAE: 1.56 positions on 2025 season"
)
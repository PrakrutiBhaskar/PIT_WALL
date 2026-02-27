import pandas as pd

def add_stint_number(df):
    """
    Every time a driver pits, their stint number goes up by 1.
    Stint 0 = first stint, Stint 1 = after first pit, etc.
    """
    df = df.sort_values(['Year', 'Race', 'Driver', 'LapNumber'])
    
    df['StintNumber'] = df.groupby(
        ['Year', 'Race', 'Driver']
    )['PitThisLap'].cumsum()
    
    return df


def add_degradation_rate(df):
    """
    How much slower is this driver getting compared to 
    their best lap in the current stint?
    Higher number = tires are dying.
    """
    df = df.sort_values(['Year', 'Race', 'Driver', 'LapNumber'])
    
    # best lap time per driver per stint = their fresh tire baseline
    stint_baseline = df.groupby(
        ['Year', 'Race', 'Driver', 'StintNumber']
    )['LapTimeSeconds'].transform('min')
    
    df['DegradationDelta'] = df['LapTimeSeconds'] - stint_baseline
    
    # smooth over 3 laps to reduce lap-to-lap noise
    df['DegradationRate'] = df.groupby(
        ['Year', 'Race', 'Driver']
    )['DegradationDelta'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    return df


def add_position_momentum(df):
    """
    Rolling 5-lap average of position change.
    Negative = moving forward. Positive = falling back.
    """
    df['PositionChange'] = df.groupby(
        ['Year', 'Race', 'Driver']
    )['Position'].diff()
    
    df['PositionMomentum'] = df.groupby(
        ['Year', 'Race', 'Driver']
    )['PositionChange'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    return df


def add_race_completion(df):
    """
    How far through the race are we — 0.0 to 1.0.
    0.5 means we're at the halfway point.
    """
    df['RaceCompletion'] = df['LapNumber'] / df.groupby(
        ['Year', 'Race']
    )['LapNumber'].transform('max')
    
    return df


def add_compound_encoding(df):
    """
    Convert tire compound text into an ordered number.
    Soft degrades fastest (0), Hard slowest (2).
    """
    compound_map = {
        'SOFT': 0, 
        'MEDIUM': 1, 
        'HARD': 2,
        'INTERMEDIATE': 3, 
        'WET': 4
    }
    df['CompoundEncoded'] = df['Compound'].map(compound_map)
    
    # fill any unmapped compounds as MEDIUM (safest assumption)
    df['CompoundEncoded'] = df['CompoundEncoded'].fillna(1)
    
    return df

def add_threat_behind(df):
    """
    How fresh are the tyres of the driver directly behind?
    A driver with much fresher tyres behind is a threat.
    """
    df = df.sort_values(['Year', 'Race', 'LapNumber', 'Position'])
    
    df['TyreLifeBehind'] = df.groupby(
        ['Year', 'Race', 'LapNumber']
    )['TyreLife'].shift(-1)
    
    df['TyreAgeDeltaBehind'] = df['TyreLife'] - df['TyreLifeBehind']
    # positive = you have older tyres than the car behind = vulnerable
    
    return df


def add_threat_ahead(df):
    """
    How old are the tyres of the driver directly ahead?
    A driver ahead on very old tyres is a sitting duck.
    """
    df = df.sort_values(['Year', 'Race', 'LapNumber', 'Position'])
    
    df['TyreLifeAhead'] = df.groupby(
        ['Year', 'Race', 'LapNumber']
    )['TyreLife'].shift(1)
    
    df['TyreAgeDeltaAhead'] = df['TyreLifeAhead'] - df['TyreLife']
    # positive = car ahead has older tyres = overtaking opportunity
    
    return df


def add_pit_stop_count(df):
    """
    Total pit stops taken so far in the race.
    A driver who has pitted once fewer than the median
    is under strategic pressure to pit soon.
    """
    df['TotalPitsTaken'] = df.groupby(
        ['Year', 'Race', 'Driver']
    )['PitThisLap'].cumsum()
    
    median_pits = df.groupby(
        ['Year', 'Race', 'LapNumber']
    )['TotalPitsTaken'].transform('median')
    
    df['PitStopDelta'] = df['TotalPitsTaken'] - median_pits
    # negative = fewer stops than average = likely to pit soon
    
    return df


def add_position_vs_grid(df):
    """
    How many positions gained or lost since the start.
    Captures whether a driver is recovering or fading.
    """
    df['PositionsGained'] = df['GridPosition'] - df['Position']
    # positive = gained positions, negative = lost positions
    
    return df

def add_gap_to_leader(df):
    """
    How far back is this driver from P1 in terms of position.
    Simple but powerful context feature.
    """
    df = df.sort_values(['Year', 'Race', 'LapNumber', 'Position'])
    
    leader_lap_time = df.groupby(
        ['Year', 'Race', 'LapNumber']
    )['LapTimeSeconds'].transform('min')
    
    df['GapToLeaderPace'] = df['LapTimeSeconds'] - leader_lap_time
    
    return df


def add_relative_tyre_age(df):
    """
    How does this driver's tyre age compare to the average 
    tyre age of all drivers on the same lap?
    Positive = older tyres than average = potential vulnerability
    Negative = fresher tyres than average = potential advantage
    """
    avg_tyre_age = df.groupby(
        ['Year', 'Race', 'LapNumber']
    )['TyreLife'].transform('mean')
    
    df['RelativeTyreAge'] = df['TyreLife'] - avg_tyre_age
    
    return df


def add_laps_since_pit(df):
    """
    How many laps has it been since this driver's last pit stop?
    Drivers who haven't pitted in a long time are prime 
    candidates for a position change soon.
    """
    df = df.sort_values(['Year', 'Race', 'Driver', 'LapNumber'])
    
    df['LapsSincePit'] = df.groupby(
        ['Year', 'Race', 'Driver', 'StintNumber']
    )['LapNumber'].transform(
        lambda x: x - x.min()
    )
    
    return df

def add_circuit_type(df):
    """
    Categorise circuits by overtaking difficulty.
    This helps the model understand strategic context.
    """
    street_circuits = [
        'Monaco', 'Singapore', 'Azerbaijan', 
        'Saudi Arabia', 'Las Vegas', 'Miami'
    ]
    high_deg_circuits = [
        'Spain', 'Britain', 'Hungary', 
        'Bahrain', 'Abu Dhabi'
    ]
    
    df['IsStreetCircuit'] = df['Race'].isin(
        street_circuits
    ).astype(int)
    
    df['IsHighDegCircuit'] = df['Race'].isin(
        high_deg_circuits
    ).astype(int)
    
    return df

def add_laps_remaining(df):
    """
    How many laps are left in the race.
    Critical for understanding strategic urgency.
    """
    total_laps = df.groupby(
        ['Year', 'Race']
    )['LapNumber'].transform('max')
    
    df['LapsRemaining'] = total_laps - df['LapNumber']
    
    return df

def build_all_features(df):
    print("Building features...")
    df = add_stint_number(df)
    df = add_degradation_rate(df)
    df = add_position_momentum(df)
    df = add_race_completion(df)
    df = add_compound_encoding(df)
    df = add_gap_to_leader(df)
    df = add_relative_tyre_age(df)
    df = add_laps_since_pit(df)
    df = df.drop(columns=['DegradationDelta', 'PositionChange'])
    feature_cols = [
        'DegradationRate', 'PositionMomentum', 'RaceCompletion',
        'CompoundEncoded', 'StintNumber', 'GapToLeaderPace',
        'RelativeTyreAge', 'LapsSincePit'
    ]
    df = df.dropna(subset=feature_cols)
    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    
    # read the raw data that data_pipeline.py already saved
    df = pd.read_csv('data/raw_laps.csv')
    print(f"Loaded raw data: {df.shape}")
    
    # build all features
    df = build_all_features(df)
    
    # save the featured dataset
    df.to_csv('data/featured_laps.csv', index=False)
    print("\nSaved to data/featured_laps.csv")
    
    # sanity check
    print("\nColumns in featured dataset:")
    print(df.columns.tolist())
    
    print("\nSample row:")
    print(df.iloc[100].to_string())
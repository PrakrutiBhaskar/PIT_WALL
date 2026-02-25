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


def build_all_features(df):
    """
    Master function — runs all feature engineering steps in order.
    Call this with your raw dataframe and get back a featured one.
    """
    print("Building features...")
    
    df = add_stint_number(df)
    print("  -> Stint number done")
    
    df = add_degradation_rate(df)
    print("  -> Degradation rate done")
    
    df = add_position_momentum(df)
    print("  -> Position momentum done")
    
    df = add_race_completion(df)
    print("  -> Race completion done")
    
    df = add_compound_encoding(df)
    print("  -> Compound encoding done")
    
    # drop intermediate columns we don't need in the model
    df = df.drop(columns=['DegradationDelta', 'PositionChange'])
    
    # drop any rows that have NaN in feature columns
    feature_cols = ['DegradationRate', 'PositionMomentum', 
                    'RaceCompletion', 'CompoundEncoded', 'StintNumber']
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
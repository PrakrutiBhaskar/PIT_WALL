import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('data/f1_cache')

def load_race(year, race_name):
    """
    Loads a single race session and returns cleaned lap data.
    """
    print(f"Loading {race_name} {year}...")
    
    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
        laps = session.laps.copy()
         
        key_cols = ['Driver', 'LapNumber', 'LapTime',
                    'Compound', 'TyreLife', 'Position',
                    'PitInTime', 'PitOutTime']
        
        laps = laps[key_cols].copy()
         
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
         
        laps['PitThisLap'] = laps['PitInTime'].notna().astype(int)
         
        
        laps['PitStopDuration'] = (
            laps['PitOutTime'] - laps['PitInTime']
        ).dt.total_seconds()

        
        laps['PitStopDuration'] = laps['PitStopDuration'].fillna(0)

        
        laps = laps.drop(columns=['LapTime', 'PitInTime', 'PitOutTime'])
                
        laps['Year'] = year
        laps['Race'] = race_name
         
        laps = laps.dropna(subset=['LapTimeSeconds', 'Position', 'Compound'])
        
        laps = laps[laps['LapTimeSeconds'].between(70, 150)]
        
        laps['TyreLife'] = laps.groupby(
            ['Driver']
        )['TyreLife'].ffill().bfill()
        
        print(f"  -> {len(laps)} laps loaded, "
              f"{laps['Driver'].nunique()} drivers")
        
        return laps
    
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return None



def load_all_races(race_list):
    """
    Loads a list of (year, race_name) tuples and 
    stacks them into one dataframe.
    """
    all_laps = []
    
    for year, race in race_list:
        df = load_race(year, race)
        if df is not None:
            all_laps.append(df)
    
    if not all_laps:
        raise ValueError("No races loaded successfully")
    
    combined = pd.concat(all_laps, ignore_index=True)
    print(f"\nTotal dataset: {len(combined)} laps across "
          f"{combined['Race'].nunique()} races")
    
    return combined



def add_final_position(df):
    """
    For each driver in each race, find their final recorded position
    and add it as a column. This is what we will predict.
    """
    final_pos = df.groupby(
        ['Year', 'Race', 'Driver']
    )['Position'].last().reset_index()
    
    final_pos.columns = ['Year', 'Race', 'Driver', 'FinalPosition']
    
    df = df.merge(final_pos, on=['Year', 'Race', 'Driver'], how='left')
    
    return df

 
if __name__ == "__main__":
    
    race_list = [
        (2023, 'Bahrain'),
        (2023, 'Saudi Arabia'),
        (2023, 'Australia'),
        (2022, 'Bahrain'),
        (2022, 'Australia'),
        (2022, 'Monaco'),
    ]
    
    df = load_all_races(race_list)
    
    df = add_final_position(df)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_laps.csv', index=False)
    print("\nSaved to data/raw_laps.csv")
    
    print("\nSample of saved data:")
    print(df.head(10).to_string())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
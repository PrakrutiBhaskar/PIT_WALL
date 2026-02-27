import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('data/f1_cache')

def get_grid_positions(session, year, race_name):
    try:
        results = session.results[['Abbreviation', 'GridPosition']]
        results.columns = ['Driver', 'GridPosition']
        results['Year'] = year
        results['Race'] = race_name
        return results
    except Exception as e:
        print(f"  -> Could not get grid positions: {e}")
        return None


def load_race(year, race_name):
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
        
        # add grid positions
        grid = get_grid_positions(session, year, race_name)
        if grid is not None:
            laps = laps.merge(grid, on=['Driver', 'Year', 'Race'], how='left')
        
        # mark DNF drivers
        try:
            results = session.results[['Abbreviation', 'Status']]
            results.columns = ['Driver', 'Status']
            
            dnf_drivers = results[
                ~results['Status'].str.contains('Finished|Lap', na=False)
            ]['Driver'].tolist()
            
            laps['DNF'] = 0
            laps.loc[laps['Driver'].isin(dnf_drivers), 'DNF'] = 1
            
        except Exception as e:
            print(f"  -> Could not get DNF status: {e}")
            laps['DNF'] = 0
        
        print(f"  -> {len(laps)} laps loaded, "
              f"{laps['Driver'].nunique()} drivers")
        
        return laps
    
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return None


def load_all_races(race_list):
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
    final_pos = df.groupby(
        ['Year', 'Race', 'Driver']
    )['Position'].last().reset_index()
    
    final_pos.columns = ['Year', 'Race', 'Driver', 'FinalPosition']
    df = df.merge(final_pos, on=['Year', 'Race', 'Driver'], how='left')
    
    return df


if __name__ == "__main__":
    
    race_list = [
        # 2021 season
        (2021, 'Bahrain'),
        (2021, 'Emilia Romagna'),
        (2021, 'Portugal'),
        (2021, 'Spain'),
        (2021, 'Monaco'),
        (2021, 'Azerbaijan'),
        (2021, 'France'),
        (2021, 'Styria'),
        (2021, 'Austria'),
        (2021, 'Britain'),
        (2021, 'Hungary'),
        (2021, 'Belgium'),
        (2021, 'Netherlands'),
        (2021, 'Italy'),
        (2021, 'Russia'),
        (2021, 'Turkey'),
        (2021, 'United States'),
        (2021, 'Mexico City'),
        (2021, 'São Paulo'),
        (2021, 'Qatar'),
        (2021, 'Saudi Arabia'),
        (2021, 'Abu Dhabi'),

        # 2022 season
        (2022, 'Bahrain'),
        (2022, 'Saudi Arabia'),
        (2022, 'Australia'),
        (2022, 'Emilia Romagna'),
        (2022, 'Monaco'),
        (2022, 'Spain'),
        (2022, 'Azerbaijan'),
        (2022, 'Canada'),
        (2022, 'Britain'),
        (2022, 'Hungary'),
        (2022, 'Belgium'),
        (2022, 'Netherlands'),
        (2022, 'Italy'),
        (2022, 'Singapore'),
        (2022, 'Japan'),
        (2022, 'United States'),
        (2022, 'Mexico City'),
        (2022, 'São Paulo'),
        (2022, 'Abu Dhabi'),

        # 2023 season
        (2023, 'Bahrain'),
        (2023, 'Saudi Arabia'),
        (2023, 'Australia'),
        (2023, 'Azerbaijan'),
        (2023, 'Monaco'),
        (2023, 'Spain'),
        (2023, 'Canada'),
        (2023, 'Britain'),
        (2023, 'Hungary'),
        (2023, 'Belgium'),
        (2023, 'Netherlands'),
        (2023, 'Italy'),
        (2023, 'Singapore'),
        (2023, 'Japan'),
        (2023, 'United States'),
        (2023, 'Mexico City'),
        (2023, 'São Paulo'),
        (2023, 'Abu Dhabi'),

        # 2024 season
        (2024, 'Bahrain'),
        (2024, 'Saudi Arabia'),
        (2024, 'Australia'),
        (2024, 'Japan'),
        (2024, 'China'),
        (2024, 'Monaco'),
        (2024, 'Spain'),
        (2024, 'Canada'),
        (2024, 'Britain'),
        (2024, 'Hungary'),
        (2024, 'Belgium'),
        (2024, 'Netherlands'),
        (2024, 'Italy'),
        (2024, 'Singapore'),
        (2024, 'United States'),
        (2024, 'Mexico City'),
        (2024, 'São Paulo'),
        (2024, 'Abu Dhabi'),

        # 2025 season — true holdout test set
        (2025, 'Australia'),
        (2025, 'China'),
        (2025, 'Japan'),
        (2025, 'Bahrain'),
        (2025, 'Saudi Arabia'),
        (2025, 'Miami'),
        (2025, 'Emilia Romagna'),
        (2025, 'Monaco'),
        (2025, 'Spain'),
        (2025, 'Canada'),
        (2025, 'Austria'),
        (2025, 'Britain'),
        (2025, 'Belgium'),
        (2025, 'Hungary'),
        (2025, 'Netherlands'),
        (2025, 'Italy'),
        (2025, 'Azerbaijan'),
        (2025, 'Singapore'),
        (2025, 'United States'),
        (2025, 'Mexico City'),
        (2025, 'São Paulo'),
        (2025, 'Las Vegas'),
        (2025, 'Qatar'),
        (2025, 'Abu Dhabi'),
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
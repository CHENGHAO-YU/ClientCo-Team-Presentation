import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def load_data():
    df_b = pd.read_csv('barley_yield_from_1982(in).csv', sep=';')
    df_b.drop(columns='Unnamed: 0', inplace=True)
    # Calculating missing yield values
    df_b['yield'] = df_b['yield'].fillna(df_b['production'] / df_b['area'])
    df_climatic = pd.read_parquet('climate_data_from_1982.parquet')
    return df_b, df_climatic

def transform_climatic_data(df_climatic):
    # REMOVED: [df_climatic['scenario'] == 'historical']
    df_climatic_wide = df_climatic.pivot_table(
        index=['nom_dep', 'year', 'time', 'scenario'], # Added scenario to index
        columns='metric', 
        values='value'
    ).reset_index()

    df_climatic_wide = df_climatic_wide.rename(columns={
        'nom_dep': 'department',
        'near_surface_air_temperature': 'temp_avg',
        'daily_maximum_near_surface_air_temperature': 'temp_max',
        'precipitation': 'precip'
    })

    df_climatic_wide['temp_max'] -= 273.15
    df_climatic_wide['temp_avg'] -= 273.15

    return df_climatic_wide

def transform_climatic_yearly(df_climatic_wide):
    # 1. Daily Flags & Agronomic Year (No changes needed here)
    df_climatic_wide['is_spring'] = df_climatic_wide['time'].dt.month.isin([3, 4, 5])
    df_climatic_wide['gdd_day'] = (df_climatic_wide['temp_avg'] - 5).clip(lower=0)
    df_climatic_wide['is_heat_stress'] = (df_climatic_wide['temp_max'] > 33).astype(int)
    df_climatic_wide['daily_heat_peak'] = df_climatic_wide['temp_max'].where(df_climatic_wide['is_heat_stress'] == 1)
    df_climatic_wide['is_freezing'] = (df_climatic_wide['temp_avg'] < 0).astype(int)
    df_climatic_wide['is_dry_spring'] = ((df_climatic_wide['precip'] < 1e-7) & (df_climatic_wide['is_spring'])).astype(int)
    df_climatic_wide['is_heavy_rain'] = (df_climatic_wide['precip'] > 20).astype(int)

    df_climatic_wide['harvest_year'] = df_climatic_wide['year']
    df_climatic_wide.loc[df_climatic_wide['time'].dt.month >= 10, 'harvest_year'] += 1

    # 2. Aggregation - Must include 'scenario' in groupby
    df_climatic_yearly = df_climatic_wide.groupby(['department', 'harvest_year', 'scenario']).agg(
        avg_temp=('temp_avg', 'mean'),
        total_precip=('precip', 'sum'),
        heat_stress_days=('is_heat_stress', 'sum'),
        total_gdd=('gdd_day', 'sum'),
        heavy_rain_days=('is_heavy_rain', 'sum'),
        winter_cold_days=('is_freezing', 'sum'),
        spring_dry_days=('is_dry_spring', 'sum'),
        peak_heat_intensity=('daily_heat_peak', 'mean')
    ).reset_index()

    df_climatic_yearly['peak_heat_intensity'] = df_climatic_yearly['peak_heat_intensity'].fillna(33)
    df_climatic_yearly = df_climatic_yearly.rename(columns={'harvest_year': 'year'})

    # 3. Multi-Scenario Anomalies
    # We sort by year within each (dept, scenario) to ensure the rolling window is chronological
    df_climatic_yearly = df_climatic_yearly.sort_values(['department', 'scenario', 'year'])
    
    group = df_climatic_yearly.groupby(['department', 'scenario'])
    
    df_climatic_yearly['temp_5yr_norm'] = group['avg_temp'].transform(
        lambda x: x.rolling(window=5, min_periods=1, closed='left').mean()
    )
    df_climatic_yearly['precip_5yr_norm'] = group['total_precip'].transform(
        lambda x: x.rolling(window=5, min_periods=1, closed='left').mean()
    )

    df_climatic_yearly['temp_anomaly'] = df_climatic_yearly['avg_temp'] - df_climatic_yearly['temp_5yr_norm']
    df_climatic_yearly['precip_anomaly'] = df_climatic_yearly['total_precip'] - df_climatic_yearly['precip_5yr_norm']

    df_climatic_yearly[['temp_anomaly', 'precip_anomaly']] = df_climatic_yearly[['temp_anomaly', 'precip_anomaly']].fillna(0)
    df_climatic_yearly.drop(columns=['temp_5yr_norm', 'precip_5yr_norm'], inplace=True)

    return df_climatic_yearly

def transform_barley_data(df_b):
    # 1. Ensure the data is sorted by year to make "forward filling" chronological
    df_b = df_b.sort_values(by=['department', 'year'])

    # 2. Fill yield based on production/area first (using existing data where possible)
    # This is more accurate than using last year's yield
    mask = df_b['yield'].isna()
    df_b.loc[mask, 'yield'] = df_b.loc[mask, 'production'] / df_b.loc[mask, 'area']

    # 3. Fill remaining N/A values using the previous year's value within each department
    # We group by department so we don't bleed data across different regions
    cols_to_fix = ['yield', 'area', 'production']
    df_b[cols_to_fix] = df_b.groupby('department')[cols_to_fix].ffill()
    
    # 4. Handle cases where the first year of a department is N/A (no previous year to pull from)
    # We use bfill (backfill) as a secondary fallback for those specific edges
    df_b[cols_to_fix] = df_b.groupby('department')[cols_to_fix].bfill()

    return df_b

def merge_climatic_yearly(df_climatic_historical, df_b):
    df_combined = pd.merge(
        df_b, 
        df_climatic_historical, 
        on=['department', 'year'], 
        how='inner'
    )
    df_combined = df_combined.fillna(df_combined.mean(numeric_only=True))
    return df_combined

params = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 5,
    'subsample': 0.8
}

def train_xgb(df_combined, department_name, hyperparams=params):
    # 1. Filter for the specific department
    df_dept = df_combined[df_combined['department'] == department_name].copy()
    
    # 2. Define Features (X) and Target (y)
    # We remove 'department' as it's now a constant, and 'year' to focus on climate drivers
    exclude = ['yield', 'production', 'area', 'department', 'year']
    features = [col for col in df_dept.columns if col not in exclude]
    
    X = df_dept[features]
    y = df_dept['yield']
    
    # 3. Internal Split
    # Since we have fewer samples per department, we keep a standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train XGBoost
    # We unpack the hyperparams dictionary (e.g., n_estimators, max_depth)
    model = xgb.XGBRegressor(random_state=42, **hyperparams)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 5. Quick Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    # print(f"Model for {department_name} trained. Test MAE: {mae:.3f} t/ha")
    
    return model


hyperparams_rf = {
    'n_estimators': 500,     # RF usually needs fewer trees than XGB to converge
    'max_depth': 10,         # RF can handle deeper trees than XGB
    'max_features': 'sqrt',  # Standard practice for RF
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

def train_rf(df_combined, department_name, hyperparams=hyperparams_rf):
    # 1. Filter for the specific department
    df_dept = df_combined[df_combined['department'] == department_name].copy()
    
    # 2. Define Features (X) and Target (y)
    exclude = ['yield', 'production', 'area', 'department']
    features = [col for col in df_dept.columns if col not in exclude]
    
    X = df_dept[features]
    y = df_dept['yield']
    
    # 3. Internal Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train Random Forest
    # No 'eval_set' or 'learning_rate' here as RF is not a boosting model
    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)
    
    # 5. Quick Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    # print(f"RF Model for {department_name} trained. Test MAE: {mae:.3f} t/ha")
    
    return model


def predict_yield(model, df_combined, department_name, year):
    # 1. Filter for the specific department
    df_dept = df_combined[df_combined['department'] == department_name].copy()
    
    # 2. Define Features (X) and Target (y)
    # We remove 'department' as it's now a constant, and 'year' to focus on climate drivers
    exclude = ['yield', 'production', 'area', 'department']
    features = [col for col in df_dept.columns if col not in exclude]
    
    X = df_dept[features]
    y = df_dept['yield']
    
    # 3. Internal Split
    # Since we have fewer samples per department, we keep a standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train Gradient Boosting Regressor
    # We unpack the hyperparams dictionary (e.g., n_estimators, max_depth)
    model = GradientBoostingRegressor(random_state=42, **hyperparams)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 5. Quick Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Model for {department_name} trained. Test MAE: {mae:.3f} t/ha")
    
    return model


def predict_scenario_yields(models_dict, df_scenario):
    all_preds = []
    
    # Identify the features used during training (exclude metadata)
    # We use the first model's feature names as the reference
    example_model = list(models_dict.values())[0]
    features = example_model.get_booster().feature_names

    for dept, model in models_dict.items():
        # Filter the scenario data for this specific department
        df_dept_future = df_scenario[df_scenario['department'] == dept].copy()
        
        if not df_dept_future.empty:
            # Predict yield using the department-specific model
            df_dept_future['predicted_yield'] = model.predict(df_dept_future[features])
            all_preds.append(df_dept_future)
            
    return pd.concat(all_preds)

def predict_scenario_yields_rf(models_dict, df_scenario):
    all_preds = []
    
    # Identify the features used during training for Random Forest
    example_model = list(models_dict.values())[0]
    features = example_model.feature_names_in_
    
    for dept, model in models_dict.items():
        # Filter the scenario data for this specific department
        df_dept_future = df_scenario[df_scenario['department'] == dept].copy()
        
        if not df_dept_future.empty:
            # Reorder/Select columns to match what the model saw during training
            X_future = df_dept_future[features]
            
            # Predict yield
            df_dept_future['predicted_yield'] = model.predict(X_future)
            all_preds.append(df_dept_future)
            
    return pd.concat(all_preds)

def combine_scenarios_weighted(df_opt, df_med, df_pess, weights={'opt': 0.2, 'med': 0.5, 'pess': 0.3}):
    # 1. Standardize columns and set index for alignment
    # We rename 'predicted_yield' to 'yield' to match your request
    cols_to_keep = ['department', 'year', 'predicted_yield', 'heat_stress_days', 
                    'winter_cold_days', 'heavy_rain_days', 'spring_dry_days']
    
    df_opt = df_opt[cols_to_keep].set_index(['department', 'year'])
    df_med = df_med[cols_to_keep].set_index(['department', 'year'])
    df_pess = df_pess[cols_to_keep].set_index(['department', 'year'])

    # 2. Perform weighted average
    # The logic: Result = (Opt * W_opt) + (Med * W_med) + (Pess * W_pess)
    df_weighted = (
        df_opt * weights['opt'] + 
        df_med * weights['med'] + 
        df_pess * weights['pess']
    )

    # 3. Cleanup: Reset index and rename yield
    df_weighted = df_weighted.reset_index().rename(columns={'predicted_yield': 'yield'})
    
    return df_weighted
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from google.colab import files

# =====================================================================
# 1. AUTOMATED FILE UPLOAD & FLEXIBLE DATA PREPARATION
# =====================================================================
print("Please click the 'Choose Files' button below to select your Excel (.xlsx) file:")
uploaded = files.upload()

# Automatically capture the selected filename
file_name = list(uploaded.keys())[0]
print(f"File uploaded successfully: {file_name}")

print("\nProcessing historical timeline and cleaning operational features...")

# Read the Excel file directly
df = pd.read_excel(file_name)

# Normalize column names to lowercase and strip hidden spaces to prevent KeyErrors
df.columns = df.columns.str.strip().str.lower()

# Safety Check: Verify if 'date' and 'complaints' exist after normalization
if 'date' not in df.columns:
    # Fallback if the column is named differently (e.g. taking the first column as date)
    print(f"Warning: 'date' column not found. Available columns are: {list(df.columns)}")
    print("Attempting to automatically assign the first column as the baseline date...")
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)

if 'complaints' not in df.columns:
    # Fallback to search for common synonyms if 'complaints' is named differently
    for col in df.columns:
        if 'complaint' in col or 'volume' in col:
            df.rename(columns={col: 'complaints'}, inplace=True)
            break

# Ensure strict chronological order for time-series evaluation
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Smooth out the 10 missing data gaps in complaints using linear interpolation
df['complaints'] = df['complaints'].interpolate(method='linear')

# Safely handle missing values in operational driver features
df['staffing_level_fte'] = df['staffing_level_fte'].ffill() if 'staffing_level_fte' in df.columns else pd.Series(100, index=df.index)
df['media_mentions'] = df['media_mentions'].fillna(0) if 'media_mentions' in df.columns else pd.Series(0, index=df.index)
df['bank_holiday_flag'] = df['bank_holiday_flag'].fillna(0) if 'bank_holiday_flag' in df.columns else pd.Series(0, index=df.index)

if 'channel_mix_index' in df.columns:
    df['channel_mix_index'] = df['channel_mix_index'].fillna(df['channel_mix_index'].median())

# Mitigate data leakage: Engineer a rolling, trailing 7-day historical moving average
df['historical_7d_mean'] = df['complaints'].shift(1).rolling(window=7, min_periods=1).mean()
df['historical_7d_mean'] = df['historical_7d_mean'].fillna(df['complaints'].iloc[0])


# =====================================================================
# 2. STATISTICAL FORECASTING MODEL SETUP (SARIMAX)
# =====================================================================
print("Configuring and training the SARIMAX model architecture...")
horizon_days = 90
y = df['complaints']

# Available exogenous features
exog_features = []
for feature in ['is_weekend', 'bank_holiday_flag', 'staffing_level_fte', 'media_mentions', 'historical_7d_mean']:
    if feature in df.columns:
        exog_features.append(feature)
    elif feature == 'is_weekend':
        df['is_weekend'] = df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        exog_features.append('is_weekend')

X = df[exog_features]

# Fit model parameters accounting for deterministic trend and 7-day weekly cycles
model = SARIMAX(
    y, 
    exog=X, 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 7), 
    enforce_stationarity=False, 
    enforce_invertibility=False
)
results = model.fit(disp=False)
print("-> Statistical model training completed successfully!")


# =====================================================================
# 3. GENERATE FUTURE OPERATIONAL BASELINE (90-DAY HORIZON)
# =====================================================================
print("Projecting future features and generating out-of-sample predictions...")
last_date = df['date'].max()
future_dates = pd.date_range(start=last_date + pd.offsets.Day(1), periods=horizon_days, freq='D')
future_df = pd.DataFrame({'date': future_dates})

# Compute future calendar variables and steady-state operational baselines
future_df['is_weekend'] = future_df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

if 'bank_holiday_flag' in exog_features:
    future_df['bank_holiday_flag'] = 0 
if 'staffing_level_fte' in exog_features:
    future_df['staffing_level_fte'] = df['staffing_level_fte'].iloc[-30:].mean()
if 'media_mentions' in exog_features:
    future_df['media_mentions'] = df['media_mentions'].iloc[-30:].mean()
if 'historical_7d_mean' in exog_features:
    future_df['historical_7d_mean'] = df['historical_7d_mean'].iloc[-1] 

# Run forward forecast computation
X_future = future_df[exog_features]
forecast = results.get_forecast(steps=horizon_days, exog=X_future)

# Enforce non-negativity operational constraints
future_df['predicted_complaints'] = np.maximum(0, forecast.predicted_mean.values)


# =====================================================================
# 4. EXPORT PREDICTIONS AND VISUALIZE TRENDS
# =====================================================================
# Export structural predictions data table to CSV format
future_df[['date', 'predicted_complaints']].to_csv('complaints_90_day_forecast.csv', index=False)
print("-> Success: Exported prediction matrix to 'complaints_90_day_forecast.csv'")

# Generate publication-quality forecast graph
plt.figure(figsize=(12, 6))
plt.plot(df['date'].iloc[-180:], df['complaints'].iloc[-180:], label='Historical Complaints (Last 6 Mos)', color='teal')
plt.plot(future_df['date'], future_df['predicted_complaints'], label='90-Day Future Forecast', color='orange', linestyle='--')
plt.title('Daily Complaints Volume: Historical vs Future Operational Forecast', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Complaints', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('forecast_chart.png', dpi=300)
plt.show()

print("-> Success: Generated and saved visualization to 'forecast_chart.png'")
print("\nPipeline execution complete! Both data structures and plots are finalized.")
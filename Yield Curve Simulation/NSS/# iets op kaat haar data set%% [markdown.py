# iets op kaat haar data set%% [markdown]
# 

# %%


# %%
import pandas as pd
import numpy as np

# ----------------- 1. CONFIGURATION -----------------
# File path for the historical yield curve data
FILE_PATH = r"C:/Users/calvi/Thesis ALM/Thesis-ALM/Yield Curve Simulation/NSS/yield-curve-rates-1990-2024.csv"

# Select the fixed set of maturities (T) in years
selected_maturities_yrs = [1, 2, 5, 10, 20, 30]

# Define the date range for the analysis
start_date = '2019-02-01'
end_date = '2024-12-31' # The code will use data up to the last available date within this range

# ----------------- 2. LOAD AND PREPARE DATA -----------------
try:
    # Load the entire dataset, parsing 'Date' as dates and setting it as the index
    df = pd.read_csv(FILE_PATH, parse_dates=['Date'], index_col='Date')
    
    # Sort chronologically (Oldest -> Newest) is essential for time series operations
    df.sort_index(inplace=True)

    # --- NEW: FILTER DATA BY DATE RANGE ---
    # Use .loc to select all rows from the start_date to the end_date
    df_filtered = df.loc[start_date:end_date]
    print(f"Data successfully loaded. Filtering for dates between {start_date} and {end_date}.")
    print(f"Original total rows: {len(df)}, Rows after filtering: {len(df_filtered)}")
    
    # Map integer maturities to column names (e.g., 1 -> '1 Yr')
    col_map = {T: f'{T} Yr' for T in selected_maturities_yrs}
    selected_columns = list(col_map.values())

    # Check if all required columns exist in the file
    missing_cols = [c for c in selected_columns if c not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}. Please check CSV headers.")

    # Create the spot rates DataFrame with only the selected maturities
    spot_rates_df = df_filtered[selected_columns].copy()

    # Drop any rows within our date range that have missing values for the selected tenors
    original_count = len(spot_rates_df)
    spot_rates_df.dropna(inplace=True)
    cleaned_count = len(spot_rates_df)
    
    if cleaned_count < original_count:
        print(f"Note: Dropped {original_count - cleaned_count} rows from the selected period due to missing data.")

    # Convert rates from percentages to decimals (e.g., 4.5 -> 0.045)
    spot_rates_df = spot_rates_df / 100

    # ----------------- 3. DERIVE FORWARD RATES -----------------
    forward_rates_df = pd.DataFrame(index=spot_rates_df.index)

    # Loop through maturities to calculate forward rates between each consecutive pair (T1, T2)
    for i in range(len(selected_maturities_yrs) - 1):
        T1 = selected_maturities_yrs[i]
        T2 = selected_maturities_yrs[i+1]
        
        col_T1 = col_map[T1]
        col_T2 = col_map[T2]
        
        R1 = spot_rates_df[col_T1]
        R2 = spot_rates_df[col_T2]
        
        # Formula: f(T1, T2) = (T2 * R(T2) - T1 * R(T1)) / (T2 - T1)
        forward_rate = (T2 * R2 - T1 * R1) / (T2 - T1)
        
        # Use a descriptive column name for the forward rate
        col_name = f'f_{T1}y_{T2}y'
        forward_rates_df[col_name] = forward_rate

    # ----------------- 4. CALCULATE DAILY CHANGES -----------------
    # Use the .diff() method to calculate the simple daily change
    daily_changes_df = forward_rates_df.diff()

    # The first row after .diff() is always NaN, so drop it
    final_matrix = daily_changes_df.dropna()

    # ----------------- 5. VIEW AND SAVE RESULTS -----------------
    print("\n--- Sample of Spot Rates (Filtered and in Decimals) ---")
    print(spot_rates_df.head()) # Show the beginning of the filtered range
    print("...")
    print(spot_rates_df.tail()) # Show the end

    print("\n--- Sample of Calculated Forward Rates ---")
    print(forward_rates_df.tail())

    print("\n--- Sample of Daily Changes Matrix (Final Output) ---")
    print(final_matrix.tail())

    # OPTIONAL: Save the processed matrix to a new, descriptively named CSV
    output_path = FILE_PATH.replace(".csv", f"_forward_changes_matrix_{start_date}_to_{end_date}.csv")
    final_matrix.to_csv(output_path)
    print(f"\nProcessed matrix saved to: {output_path}")

except FileNotFoundError:
    print(f"Error: The file was not found at {FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")

# %%
import pandas as pd
import numpy as np

# ----------------- 1. CONFIGURATION -----------------
# File path for the historical yield curve data
FILE_PATH = r"C:/Users/calvi/Thesis ALM/Thesis-ALM/Yield Curve Simulation/NSS/yield-curve-rates-1990-2024.csv"

# --- NEW: Define maturities with mixed units (Months and Years) ---
# Each tuple contains (value, unit). This list must be in increasing order of maturity.
selected_maturities = [ (1, 'Mo'),
    (3, 'Mo'),
    (6, 'Mo'),
    (1, 'Yr'),
    (2, 'Yr'),
    (5, 'Yr'),
    (10, 'Yr'),
    (20, 'Yr'),
    (30, 'Yr')
]

# Define the date range for the analysis
start_date = '2019-02-01'
end_date = '2024-12-31'

# ----------------- 2. LOAD AND PREPARE DATA -----------------

# --- Helper structures to process mixed-unit maturities ---
selected_columns = []      # e.g., ['3 Mo', '6 Mo', '1 Yr']
maturities_in_years = {}   # e.g., {'3 Mo': 0.25, '1 Yr': 1.0}
short_labels = []          # e.g., ['3m', '6m', '1y']

for value, unit in selected_maturities:
    # Create the column name that matches the CSV header
    column_name = f'{value} {unit}'
    selected_columns.append(column_name)
    
    # Create a short label for naming forward rate columns
    short_label = f'{value}m' if unit == 'Mo' else f'{value}y'
    short_labels.append(short_label)
    
    # CRITICAL: Convert all maturities to a consistent unit (years) for the formula
    year_value = value / 12 if unit == 'Mo' else float(value)
    maturities_in_years[column_name] = year_value

try:
    # Load the entire dataset
    df = pd.read_csv(FILE_PATH, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    # Filter data by the specified date range
    df_filtered = df.loc[start_date:end_date]
    print(f"Data successfully loaded. Filtering for dates between {start_date} and {end_date}.")
    print(f"Original total rows: {len(df)}, Rows after filtering: {len(df_filtered)}")
    
    # Check if all required columns exist in the file
    missing_cols = [c for c in selected_columns if c not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}. Please check CSV headers.")

    # Create the spot rates DataFrame
    spot_rates_df = df_filtered[selected_columns].copy()

    # Drop rows with any missing values for the selected tenors
    original_count = len(spot_rates_df)
    spot_rates_df.dropna(inplace=True)
    cleaned_count = len(spot_rates_df)
    
    if cleaned_count < original_count:
        print(f"Note: Dropped {original_count - cleaned_count} rows from the selected period due to missing data.")

    # Convert rates from percentages to decimals
    spot_rates_df = spot_rates_df / 100

    # ----------------- 3. DERIVE FORWARD RATES -----------------
    forward_rates_df = pd.DataFrame(index=spot_rates_df.index)

    # Loop through maturities to calculate forward rates between each consecutive pair
    for i in range(len(selected_columns) - 1):
        col_T1 = selected_columns[i]
        col_T2 = selected_columns[i+1]
        
        # Get the maturity values IN YEARS from our helper dictionary
        T1 = maturities_in_years[col_T1]
        T2 = maturities_in_years[col_T2]
        
        # Get the spot rate series
        R1 = spot_rates_df[col_T1]
        R2 = spot_rates_df[col_T2]
        
        # Apply the forward rate formula using the converted year values
        forward_rate = (T2 * R2 - T1 * R1) / (T2 - T1)
        
        # Use the short labels for a clean column name (e.g., 'f_3m_6m', 'f_6m_1y')
        label1 = short_labels[i]
        label2 = short_labels[i+1]
        col_name = f'f_{label1}_{label2}'
        forward_rates_df[col_name] = forward_rate

    # ----------------- 4. CALCULATE DAILY CHANGES -----------------
    daily_changes_df = forward_rates_df.diff()
    final_matrix = daily_changes_df.dropna()

    # ----------------- 5. VIEW AND SAVE RESULTS -----------------
    print("\n--- Sample of Selected Spot Rates (Filtered and in Decimals) ---")
    print(spot_rates_df.tail())

    print("\n--- Sample of Calculated Forward Rates ---")
    print(forward_rates_df.tail())

    print("\n--- Sample of Daily Changes Matrix (Final Output) ---")
    print(final_matrix.tail())

    # Save the processed matrix to a new CSV
    output_path = FILE_PATH.replace(".csv", f"1_forward_changes_matrix_{start_date}_to_{end_date}_with_months.csv")
    final_matrix.to_csv(output_path)
    print(f"\nProcessed matrix saved to: {output_path}")

except FileNotFoundError:
    print(f"Error: The file was not found at {FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")

# %%
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class YieldCurveSimulator:
    def __init__(self, n_components=3):
        """
        Initializes the simulator using PCA.
        
        Args:
            n_components: Number of PCA components to use. 
                          3 is standard (Level, Slope, Curvature).
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.hist_pc_shocks = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, data_df):
        """
        Fits the PCA model to the historical rate CHANGES.
        
        Args:
            data_df: DataFrame containing historical daily CHANGES in rates.
                     (Index should be Date, columns are tenors)
        """
        self.feature_names = data_df.columns
        
        # 1. Standardize the data (Mean=0, Var=1)
        # This is crucial so high-variance rates don't dominate the PCA
        scaled_data = self.scaler.fit_transform(data_df)
        
        # 2. Fit PCA
        self.pca.fit(scaled_data)
        
        # 3. Transform historical data into Principal Components (PC space)
        # These are the "historical shocks" we will sample from later
        self.hist_pc_shocks = self.pca.transform(scaled_data)
        
        self.is_fitted = True
        
        # Analysis printout
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Model Fitted. Top {self.n_components} components explain {explained_var:.2%} of variance.")
        print(f"PC1 (Level): {self.pca.explained_variance_ratio_[0]:.2%}")
        print(f"PC2 (Slope): {self.pca.explained_variance_ratio_[1]:.2%}")
        if self.n_components > 2:
            print(f"PC3 (Curvature): {self.pca.explained_variance_ratio_[2]:.2%}")

    def simulate_scenarios(self, n_scenarios, n_steps, initial_curve=None, seed=None):
        """
        Generates N scenarios of future yield curves.
        
        Args:
            n_scenarios: Number of independent paths to generate (e.g., 1000)
            n_steps: Number of days to simulate per path (e.g., 252 for 1 year)
            initial_curve: (Optional) Array of starting rates. If None, starts at 0.0.
            seed: Random seed for reproducibility.
            
        Returns:
            scenarios: 3D Array of shape (n_scenarios, n_steps + 1, n_rates)
                       containing ABSOLUTE rates.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")
            
        if seed is not None:
            np.random.seed(seed)

        # --- 1. Bootstrap Resampling ---
        # We need (n_scenarios * n_steps) total daily shocks.
        # We randomly sample indices from our historical PC shocks.
        # This preserves the "fat tails" of the real market data.
        n_historical_days = self.hist_pc_shocks.shape[0]
        random_indices = np.random.randint(0, n_historical_days, size=(n_scenarios, n_steps))
        
        # Gather the PC shocks corresponding to those indices
        # Shape: (n_scenarios, n_steps, n_components)
        simulated_pc_shocks = self.hist_pc_shocks[random_indices]

        # --- 2. Inverse Transform (Back to Rate Space) ---
        # We process this in 2D for efficiency, then reshape back
        flat_pc_shocks = simulated_pc_shocks.reshape(-1, self.n_components)
        
        # Inverse PCA: PC Space -> Scaled Rate Space
        flat_scaled_shocks = self.pca.inverse_transform(flat_pc_shocks)
        
        # Inverse Scaling: Scaled Rate Space -> Actual Rate Changes
        flat_real_shocks = self.scaler.inverse_transform(flat_scaled_shocks)
        
        # Reshape back to (n_scenarios, n_steps, n_rates)
        daily_shocks = flat_real_shocks.reshape(n_scenarios, n_steps, len(self.feature_names))

        # --- 3. Construct Absolute Curves (Cumulative Sum) ---
        # If no initial curve provided, assume starting at 0 (or use last historical point)
        if initial_curve is None:
            initial_curve = np.zeros(len(self.feature_names))
        else:
            initial_curve = np.array(initial_curve)

        # Create container including t=0
        scenarios = np.zeros((n_scenarios, n_steps + 1, len(self.feature_names)))
        
        # Set t=0
        scenarios[:, 0, :] = initial_curve
        
        # Calculate cumulative path
        # path[t] = path[t-1] + shock[t]
        cumulative_shocks = np.cumsum(daily_shocks, axis=1)
        
        # Add cumulative shocks to initial curve
        scenarios[:, 1:, :] = cumulative_shocks + initial_curve
        
        return scenarios

# %%
#####
#try simpel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# ==========================================
# 1. LOAD DATA
# ==========================================
# I am using the snippet you provided. 
# To use your actual file, uncomment the line below:
# df = pd.read_csv('yield-curve-rates-1990-20241_forward_changes_matrix_2019-02-01_to_2024-12-31_with_months.csv')


print("1. Data Loaded. Shape:", df.shape)

# ==========================================
# 2. PCA MODELING (Training the Generator)
# ==========================================
# Goal: Find the 3 main "drivers" (Principal Components) of yield curve changes.
# Why: The 7 rates are highly correlated. We want to model the underlying
#      drivers (Level, Slope, Curvature) instead of the 7 rates directly.

# Step A: Normalize the data (PCA requires this)
# Scales data so mean=0 and variance=1
scaler = StandardScaler()
scaled_changes = scaler.fit_transform(df)

# Step B: Fit PCA
# We use 3 components because they typically explain >95% of yield curve moves.
pca = PCA(n_components=3)
pca.fit(scaled_changes)

# Step C: Get Historical Shocks in PC Space
# This transforms our matrix of [Rates] into a matrix of [Drivers]
historical_pc_shocks = pca.transform(scaled_changes)

print("\n2. PCA Model Fitted.")
print(f"   - Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
print("   - We have compressed 7 rates into 3 drivers (Level, Slope, Curvature).")

# ==========================================
# 3. SIMULATION (Generating Scenarios)
# ==========================================
# Goal: Create thousands of NEW, plausible future paths for the yield curve.
# Method: "Bootstrapping" - We pick random days from history and string them together.

def simulate_yield_curve(n_scenarios, n_days, start_curve):
    """
    Generates simulated yield curves for ALM training.
    
    Args:
        n_scenarios: How many different futures to simulate (e.g., 1000)
        n_days: Length of simulation in days (e.g., 252 for 1 year)
        start_curve: The yield curve at t=0 (array of 7 rates)
        
    Returns:
        scenarios: 3D array [Scenario, Day, Rate] with ABSOLUTE rates.
    """
    
    # 1. Sample random indices from our history
    # We need (Scenarios * Days) total random picks
    max_idx = len(historical_pc_shocks)
    random_indices = np.random.randint(0, max_idx, size=(n_scenarios, n_days))
    
    # 2. Retrieve the PC shocks for those random days
    # Shape: (n_scenarios, n_days, 3)
    sim_pc_shocks = historical_pc_shocks[random_indices]
    
    # 3. Convert PC shocks back to Rate shocks (Inverse PCA)
    # Reshape to 2D for the library function, then back to 3D
    flat_pc_shocks = sim_pc_shocks.reshape(-1, 3)
    flat_scaled_rate_shocks = pca.inverse_transform(flat_pc_shocks)
    flat_real_rate_shocks = scaler.inverse_transform(flat_scaled_rate_shocks)
    
    # Shape: (n_scenarios, n_days, 7)
    sim_rate_changes = flat_real_rate_shocks.reshape(n_scenarios, n_days, 7)
    
    # 4. Construct Absolute Rates (Cumulative Sum)
    # We start with the initial curve and add the daily changes
    scenarios = np.zeros((n_scenarios, n_days + 1, 7))
    scenarios[:, 0, :] = start_curve # Set Day 0
    
    # Calculate cumulative sum of changes
    cumulative_changes = np.cumsum(sim_rate_changes, axis=1)
    
    # Add to start curve
    scenarios[:, 1:, :] = cumulative_changes + start_curve
    
    return scenarios

# Define parameters
N_SCENARIOS = 50     # Generate 50 potential futures
N_DAYS = 252         # 1 Year of trading days
# Define a starting yield curve (e.g., current market rates)
# [3m, 6m, 1y, 2y, 5y, 10y, 20y] - roughly 3% to 4%
current_yield_curve = np.array([0.030, 0.032, 0.034, 0.036, 0.038, 0.042, 0.045])

# Run Simulation
simulated_data = simulate_yield_curve(N_SCENARIOS, N_DAYS, current_yield_curve)

print(f"\n3. Simulation Complete.")
print(f"   - Generated Tensor Shape: {simulated_data.shape} -> (Scenarios, Days, Rates)")

# ==========================================
# 4. VISUALIZATION
# ==========================================
# Plotting the 10-Year Rate (Index 5) across different scenarios
plt.figure(figsize=(10, 6))

# Plot the first 10 scenarios
for i in range(10):
    plt.plot(simulated_data[i, :, 5], lw=1, alpha=0.7)

plt.title("Simulated 10-Year Interest Rate Scenarios (1 Year)")
plt.xlabel("Trading Days")
plt.ylabel("Interest Rate")
plt.grid(True, alpha=0.3)
plt.show()

# %%




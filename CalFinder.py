import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse

#----------------------------------------------------------------------------------------

def load_data(file_path):
    """Load calibration curve data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Clean input data."""
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    if 'sample_type' in data.columns:
        data = data[data['sample_type'].str.lower() == 'standard']
    
    data = data.dropna(subset=['total_area'])
    return data[['molecule', 'analyte_concentration', 'total_area']]

def weighted_linear_regression(x, y, weights):
    """Perform a weighted linear regression."""
    x = sm.add_constant(x)
    model = sm.WLS(y, x, weights=weights)
    return model.fit()

def calculate_r_squared(y_true, y_pred):
    """Calculate the R-squared value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.round(1 - (ss_res / ss_tot), 5)

def calculate_accuracy(y_true, y_pred):
    """Calculate the accuracy in percentage for each level."""
    return np.round(100 * (y_pred / y_true), 1)

def determine_loq(accuracies, analyte_concentrations, max_loq_bias=30):
    """Determine the LOQ as the analyte concentration one level higher than the level with a ±30% accuracy."""
    index_result = [index for index, accuracy in enumerate(accuracies) if accuracy < (100 - max_loq_bias) or accuracy > (100 + max_loq_bias)]
    
    if not index_result:
        max_value = -1
    elif len(analyte_concentrations) == (max(index_result) + 1):
        max_value = max(index_result)
    else:
        max_value = max(index_result)
        
    max_value = min(max_value + 1, len(analyte_concentrations) - 1)
    return analyte_concentrations.iloc[max_value]

def process_data(file_path, max_loq_bias):
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    
    molecules = cleaned_data['molecule'].unique()
    all_results = []

    for molecule in molecules:
        molecule_data = cleaned_data[cleaned_data['molecule'] == molecule].sort_values(by='analyte_concentration')
        x = molecule_data['analyte_concentration']
        y = molecule_data['total_area']
        weights = 1 / x
        
        for n_levels in range(len(x), 4, -1):
            current_x = x.iloc[:n_levels]
            current_y = y.iloc[:n_levels]
            current_weights = weights.iloc[:n_levels]
            
            model = weighted_linear_regression(current_x, current_y, current_weights)
            y_pred = model.predict(sm.add_constant(current_x))
            r_squared = calculate_r_squared(current_y, y_pred)
            accuracies = calculate_accuracy(current_y, y_pred)
            loq = determine_loq(accuracies, current_x, max_loq_bias)
        
            result = {
                'molecule': molecule,
                'num_levels': n_levels,
                'levels': list(current_x),
                'slope': model.params[1],
                'intercept': model.params[0],
                'r_squared': r_squared,
                'upper_level': max(list(current_x)),
                'loq': loq,
                'accuracies': list(accuracies)
            }
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('01_all_results.csv', index=False)
    return results_df

def filter_results(df, r_squared=0.99):
    # print(f"Filtering with r_squared: {r_squared}")  # Debug print to check passed argument
        
    # Group by 'molecule' column
    grouped_df = df.groupby('molecule')
    # Filter rows where 'loq' has the lowest value within each group
    lowest_loq_df = grouped_df.apply(lambda x: x[x['loq'] == x['loq'].min()]).reset_index(drop=True)
    # Filter rows where 'r_squared' is larger than defined (default: 0.99) and save results to csv file
    filtered_df = lowest_loq_df[lowest_loq_df['r_squared'] > r_squared]
    filtered_df.to_csv('02_filtered_results.csv', index=False)
    # Keep only the first entry per 'molecule' and save results to csv file
    first_entry_df = filtered_df.groupby('molecule').first().reset_index()
    first_entry_df.to_csv('03_first_entry_results.csv', index=False)
    # Identify and save molecules that failed the filter
    failed_molecules = set(df['molecule']) - set(filtered_df['molecule'])
    failed_df = df[df['molecule'].isin(failed_molecules)]
    failed_df.to_csv('04_failed_molecules.csv', index=False)
    
    return first_entry_df

# -------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CalFinder Script')
                            
    # Required positional argument: file_path
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    
    # Optional arguments with default values
    parser.add_argument('--max_loq_bias', type=int, default=30, help='Maximum LOQ bias in % (default: 30)')
    parser.add_argument('--r_squared', type=float, default=0.99, help='Minimum R-squared value (default: 0.99)')
    
    args = parser.parse_args()

    # Debug prints to verify argument parsing
    # print(f"Parsed max_loq_bias: {args.max_loq_bias}")
    # print(f"Parsed r_squared: {args.r_squared}")
    
    # Script logic using the parsed arguments
    results_df = process_data(args.file_path, args.max_loq_bias)
    first_entry_df = filter_results(results_df, args.r_squared)
    
    print(f'File path: {args.file_path}')
    print(f'Maximum LOQ bias in %: {args.max_loq_bias}')
    print(f'Minimum R-squared value: {args.r_squared}')
    print('\nDone!')


# python CalFinder.py path/to/your/input.csv --max_loq_bias 30 --r_squared 0.99
# python CalFinder.py example_data/input.csv --max_loq_bias 30 --r_squared 0.99
# python CalFinder.py example_data/input.csv
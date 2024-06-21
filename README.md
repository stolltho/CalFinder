# CalFinder

Imagine you have hundred(s) of calibration curves e.g. from a targeted metabolomics experiment and you have to manually exclude calibration levels to obtain acceptable R_squared and Limit of Quantification (LOQ) values. The following code helps to automate the exclusion process of calibration levels e.g. from Skyline calibration data, so that linearity and LOQ are optimised. 

In [Skyline](https://www.skyline.ms/project/home/begin.view) the LOQ values are calculated based on external standard replicates. The LOQ value is the lowest standard concentration in the calibration curve that satisfies the 'Maximum LOQ bias' criteria, which is the maximum allowed difference (in %) between the actual analyte concentration and the calculated value from the calibration curve. The other criteria 'Maximum LOQ CV' - which is not included here - is the maximum allowed %CV of standard replicates.

Whether you prefer the interactive environment of a Jupyter Notebook or the streamlined execution of the Python script, CalFinder provides a versatile solution to pick the most appropriate calibration curve levels.

## Input
- CSV file (e.g. Skyline report using template file SkylineReport_CalFinder.skyr). Check example data for required header format. Minimum required columns are 'molecule', 'analyte_concentration' and 'total_area'.

## Output
Four CSV files will be generated:
- 01_all_results.csv (unfiltered results: file containing all LOQ and R_squared value calculations. This file can be used to check results more broadly and select other LOQ and R_squared combinations as required)
- 02_filtered_results.csv (results filtered for lowest LOQ and respective defined R_squared value(s). This file can also be used to select other LOQ and R_squared combinations as required)
- 03_first_entry_results.csv (returning final calibration curve metrices to be used in subsequent sample calculations)
- 04_failed_molecules.csv (list of molecules which failed the filtering process. Manual inspection and selection is required)

## Installation

#### Clone Repository
```bash
git clone https://github.com/stolltho/CalFinder.git
```
#### Navigate to Repository
```bash
cd repo
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
```bash
python CalFinder.py example_data/input.csv
````

### Required Argument
- `file_path` : Path to your CSV file (default: example_data/input.csv)

### Optional Arguments

- `--max_loq_bias` : Maximum LOQ bias in % (default: 30)

- `--r_squared` : Minimum R-squared value (default: 0.99)


## Example Usages
```bash
# Basic usage employing the example data provided
python CalFinder.py example_data/input.csv

# Custom parameters
python CalFinder.py example_data/input.csv --max_loq_bias 20 --r_squared 0.999
````
    

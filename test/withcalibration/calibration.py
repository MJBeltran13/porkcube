import pandas as pd
import numpy as np

# Function to generate random uncalibrated L* values and corresponding calibrated values
def generate_calibration_dataset(num_samples):
    uncalibrated_values = np.random.uniform(low=0, high=100, size=num_samples)
    # Example calibration function: calibrated_value = uncalibrated_value * 0.2 + 20
    calibrated_values = uncalibrated_values * 0.2 + 20
    return pd.DataFrame({'UnCalibrated L* Value': uncalibrated_values, 'Calibrated L* Value': calibrated_values})

def main():
    num_samples = 100  # Number of samples in the calibration dataset
    calibration_dataset = generate_calibration_dataset(num_samples)
    calibration_dataset.to_excel('withcalibration/calibration_dataset.xlsx', index=False)
    print("Calibration dataset generated and saved successfully.")

if __name__ == "__main__":
    main()

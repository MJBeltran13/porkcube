import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

# Read calibration values from Excel file
def read_calibration_values(filename):
    try:
        df = pd.read_excel(filename)
        calibration_values = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return calibration_values
    except Exception as e:
        print(f"Error reading calibration values: {e}")
        return None

# Find the closest value in a list
def find_closest_value(value, value_list):
    return min(value_list, key=lambda x: abs(x - value))

# Calibration values
calibration_filename = 'withcalibration\calibration_values.xlsx'

def get_L_star(image):
    # Convert the image to L*a*b* color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split L*a*b* channels
    L, _, _ = cv2.split(lab_image)
    
    # Calculate the average L* value
    L_mean = np.mean(L)
    
    return L_mean

def main():
    # Read pork image
    frame = cv2.imread('pork.png')
    if frame is None:
        print("Error: Unable to read image pork.png")
        return
    
    # Read calibration values
    calibration_values = read_calibration_values(calibration_filename)
    if calibration_values is None:
        return
    
    # Get L* value
    L_star = get_L_star(frame)
    print("Original L* value:", L_star)
    
    # Find the closest initial value in the calibration table
    closest_initial_value = find_closest_value(L_star, list(calibration_values.keys()))
    
    # Calibrate L* value
    calibrated_L_star = calibration_values.get(closest_initial_value)
    if calibrated_L_star is None:
        print("Calibrated L* value not found for the original value:", L_star)
        return
    print("Calibrated L* value:", calibrated_L_star)
    
    # Display image and calibrated L* value using Tkinter
    root = tk.Tk()
    root.title("Pork Image")
    
    # Convert the image from OpenCV format to Tkinter format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    
    # Display the image
    panel = tk.Label(root, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    
    # Display the calibrated L* value
    l_star_label = tk.Label(root, text=f"Calibrated L* value: {calibrated_L_star:.2f}", font=("Helvetica", 16))
    l_star_label.pack(side="bottom")
    
    root.mainloop()

if __name__ == "__main__":
    main()

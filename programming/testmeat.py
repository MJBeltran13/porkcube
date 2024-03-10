import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Calibration values
target_L_star = 43

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
    
    # Get L* value
    L_star = get_L_star(frame)
    print("Original L* value:", L_star)
    
    # Calibrate L* value
    calibrated_L_star = L_star - (L_star - target_L_star)
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

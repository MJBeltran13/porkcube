import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def get_L_star(image):
    # converting the pork image to L*a*b* color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # splitting L*a*b* channels
    L, a, b = cv2.split(lab_image)
    
    # calculating the average L* value
    L_mean = np.mean(L)
    
    return L_mean

def main():
    # Load the image
    image = cv2.imread('pork2.png')
    
    # Create Tkinter window
    root = tk.Tk()
    root.title("L* Value")
    
    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Display image on Tkinter window
    label = tk.Label(root, image=imgtk)
    label.pack()
    
    # Get L* value
    L_star = get_L_star(image)
    l_label = tk.Label(root, text="L* value: {:.2f}".format(L_star))
    l_label.pack()
    
    # Close the window when a key is pressed
    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()

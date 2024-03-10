import cv2
import numpy as np

def get_L_star(image):
    # converting the pork image to L*a*b* color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # splitting L*a*b* channels
    L, a, b = cv2.split(lab_image)
    
    # calculating the average L* value
    L_mean = np.mean(L)
    
    return L_mean

def main():
    # initializing the webcam (secondary)
    cap = cv2.VideoCapture(1)
    
    print("Press any key to capture the frame...")
    cv2.waitKey(0)
    
    # capturing a single frame and getting L*
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    L_star = get_L_star(frame)
    print("L* value:", L_star)
    
    # closing all windows after 10 seconds
    cv2.waitKey(10000)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

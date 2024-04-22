import os
import cv2
from tkinter import Tk, filedialog

# Initialize Tkinter
root = Tk()
root.withdraw()  # Hide the main window

# Ask the user to choose an image from the Altered directory
print("Please select an image from the Altered directory.")
file_path = filedialog.askopenfilename(initialdir=r"C:\Users\DELL\OneDrive\Desktop\Python project gp11\Finger print analysis\archive\SOCOFing\SOCOFing\Altered\Altered-Hard", title="Select an Image", filetypes=(("Image files", "*.BMP;*.bmp;*.tif;*.TIF"), ("All files", "*.*")))

if not file_path:
    print("No file selected.")
    exit()

# Read the sample image
sample = cv2.imread(file_path)

# Check if the sample image is loaded correctly
if sample is None or sample.size == 0:
    print(f"Error: Unable to read or empty image at {file_path}")
else:
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the sample image
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

    best_score = 0
    filename = None
    image = None
    kp1, kp2, mp = None, None, None 

    # Loop through the Real directory to find the best match
    real_dir = r"C:\Users\DELL\OneDrive\Desktop\Python project gp11\Finger print analysis\archive\SOCOFing\SOCOFing\Real"
    for file in os.listdir(real_dir):
        fingerprint_image = cv2.imread(os.path.join(real_dir, file))

        # Detect keypoints and compute descriptors for the fingerprint image
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        # Match keypoints using FLANN
        matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
        
        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)
        
        keypoints = min(len(keypoints_1), len(keypoints_2))

        if len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points) / keypoints * 100
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    if filename:
        print("BEST MATCH : " + str(filename))
        print("Score : " + str(best_score))

        result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
        result = cv2.resize(result , None , fx = 5 , fy =5)

        cv2.imshow("Result", result)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows
    else:
        print("No match found in the Real directory.")

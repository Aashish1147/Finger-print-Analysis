import os
import cv2

def choose_image(directory):
    print("Choose an image:")
    counter = 1
    for file in os.listdir(directory):
        print(f"{counter}. {file}")
        counter += 1

    choice = int(input("Enter the number of the image: "))
    chosen_file = os.listdir(directory)[choice - 1]
    image_path = os.path.join(directory, chosen_file)
    return cv2.imread(image_path), chosen_file

# Directory paths for sample and fingerprint images
sample_directory = r"C:\Users\DELL\OneDrive\Desktop\Python project gp11\Finger print analysis"
fingerprint_directory = r"C:\Users\DELL\OneDrive\Desktop\Python project gp11\Finger print analysis"

# Choose the sample image
sample, sample_file = choose_image(sample_directory)

# Check if the sample image is loaded correctly
if sample is None:
    print("Error: Unable to read sample image")
else:
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the sample image
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

    # Choose the fingerprint image
    fingerprint_image, fingerprint_file = choose_image(fingerprint_directory)

    # Check if the chosen fingerprint image is loaded correctly
    if fingerprint_image is not None:
        # Convert image to correct depth (if needed)
        fingerprint_image = cv2.convertScaleAbs(fingerprint_image)

        # Detect keypoints and compute descriptors for the fingerprint image
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        # Match keypoints using FLANN
        matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
        
        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)
        
        keypoints = min(len(keypoints_1), len(keypoints_2))

        best_score = len(match_points) / keypoints * 100
        print(f"BEST MATCH : {sample_file} with {fingerprint_file}")
        print(f"Score : {best_score}")

        result = cv2.drawMatches(sample, keypoints_1, fingerprint_image, keypoints_2, match_points, None)

        cv2.imshow("Result", result)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows
    else:
        print(f"Error: Unable to read chosen fingerprint image '{fingerprint_file}'")

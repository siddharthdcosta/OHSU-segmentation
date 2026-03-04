import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation(image_path, count):
    # Load the image
    image = cv2.imread(image_path)

    # Check Loading
    if image is None:
        print(f"Error: Could not load the image at '{image_path}'. Please check the file path/integrity.")
        return

    # Perform Resizing
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (15392, 8465))

    # Gaussian Blur
    blurred = cv2.GaussianBlur(resized_image, (7, 7), 7)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    # Opening Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Smoothing and Final Thresholding
    blurred = cv2.GaussianBlur(opened, (51, 51), 0)
    blurred = cv2.medianBlur(blurred, 51)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 2801, 4)
    final = cv2.bitwise_not(binary)
    resized_final = cv2.resize(final, (2000, 1100))
    
    cv2.imshow("image", resized_final)
    #cv2.imwrite(r'c:\Users\siddc\OHSU\JCT and SC Modeling\Segmented Images\\' + image_path[image_path.rfind("\\") + 1:], resized_final, params=(cv2.IMWRITE_BITS_PER_CHANNEL, 1)) #Writes file to specified locaation
    cv2.waitKey(0)

if __name__ == "__main__":
    with open(r"c:\Users\siddc\OHSU\JCT and SC Modeling\ImageList.txt", "r") as imageList:
        try:
            image_path = imageList.readline()
            image_path = image_path.strip()
            count = 1
            while image_path:
                print("Processing image #" + str(count) + ": ", image_path)
                segmentation(image_path, count)
                image_path = imageList.readline()  # Read the next line
                image_path = image_path.strip()
                count += 1
                print("Processed.")
        except FileNotFoundError:
            print("File not found.")
        except IOError:
            print("Error occurred while reading the file.")


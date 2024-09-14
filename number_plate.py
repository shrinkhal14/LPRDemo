import cv2
import pytesseract
import re

# Path to the Haar Cascade file for license plate recognition
haarcascade = "model/haarcascade_russian_plate_number.xml"

# Path to Tesseract-OCR executable (required for pytesseract)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Start video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Set the width and height of the video frame
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum area of detected plates
min_area = 500
count = 0

# Create a named window to ensure it displays properly
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

# Infinite loop for real-time plate detection
while True:
    success, img = cap.read()  # Read the current frame

    if not success:
        print("Failed to capture image")
        break

    # Load the Haar Cascade classifier for plate detection
    plate_cascade = cv2.CascadeClassifier(haarcascade)

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterate over each detected plate
    for (x, y, w, h) in plates:
        area = w * h

        # Check if the detected plate area is above the minimum threshold
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the text 'Number Plate' above the detected rectangle
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Crop the detected plate from the image
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

            # Apply OCR to extract text from the cropped license plate
            plate_text = pytesseract.image_to_string(img_roi, config='--psm 8')  # Use PSM 8 for single word/line

            # Filter out only alphanumeric characters using regex
            alphanumeric_texts = re.findall(r'\b[A-Za-z0-9]+\b', plate_text)

            if alphanumeric_texts:
                # Find the longest text which is usually the most relevant one
                most_relevant_text = max(alphanumeric_texts, key=len)

                # Save the recognized license plate text to plates.txt
                with open("plates.txt", "a") as f:
                    f.write(f"License Plate {count}: {most_relevant_text}\n")

                print(f"License Plate {count}: {most_relevant_text}")

    # Show the result in a window
    cv2.imshow("Result", img)

    # Break the loop when 's' is pressed and save the image with the detected plate
    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite("plates/scanned_img" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

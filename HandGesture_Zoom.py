import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def zoom_image(image):
    """ Function to apply zoom effect on a static image using hand gestures """
    h, w, _ = image.shape
    # Open webcam to detect gestures
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        # Default zoom level
        zoom_factor = 1.0  

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Calculate zoom level based on finger distance
                distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
                zoom_factor = max(1.0, min(3.5, distance / 75))  # Adjusted for sensitivity

        # Apply zoom effect to the image
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        # Ensure the crop region is within bounds
        x_offset = max(0, min(x_offset, w - new_w))
        y_offset = max(0, min(y_offset, h - new_h))

        # Crop & Resize
        zoomed_img = image[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        zoomed_img_resized = cv2.resize(zoomed_img, (w, h))

        cv2.imshow("Image Zoom", zoomed_img_resized)

        #  Exit when 'b' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            break

    cap.release()
    cv2.destroyAllWindows()

def zoom_camera():
    """ Function to apply zoom effect on live webcam feed using hand gestures """
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        zoom_factor = 1.0  # Default zoom level

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Calculate zoom level based on finger distance
                distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
                zoom_factor = max(1.0, min(3.5, distance / 75))  # Adjusted for sensitivity

        # Apply zoom effect to the frame
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        # Ensure the crop region is within bounds
        x_offset = max(0, min(x_offset, w - new_w))
        y_offset = max(0, min(y_offset, h - new_h))

        # Crop & Resize
        zoomed_img = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        zoomed_img_resized = cv2.resize(zoomed_img, (w, h))

        cv2.imshow("Live Camera Zoom", zoomed_img_resized)

        #  Exit when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu for user choice
print("Select an option:")
print("1 - Live Camera Zoom")
print("2 - Image Zoom (JPG)")

choice = input("Enter 1 or 2: ")

if choice == '1':
    zoom_camera()
elif choice == '2':
    image_path = input("Enter the path of the image (JPG file): ")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image. Check the file path.")
    else:
        zoom_image(image)
else:
    print("Invalid choice. Exiting...")


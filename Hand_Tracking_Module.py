import mediapipe as mp
import numpy as np
import cv2
import math

cap = cv2.VideoCapture(0)

def draw_broken_line(img, start_point, end_point, segment_length, gap_size, line_color, line_thickness):
    line_vector = np.array(end_point) - np.array(start_point)
    line_length = np.linalg.norm(line_vector)
    num_segments = int(line_length / (segment_length + gap_size))
    step_size = line_vector / num_segments
    current_point = np.array(start_point)
    for _ in range(num_segments):
        next_point = current_point + (step_size / np.linalg.norm(step_size)) * segment_length
        cv2.line(img, tuple(current_point.astype(int)), tuple(next_point.astype(int)), line_color, line_thickness)
        current_point = next_point + (step_size / np.linalg.norm(step_size)) * gap_size
    return img

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
    return distance

def resize(img,scale_size, width, height):
    re_width = width * scale_size
    re_height = height * scale_size
    img = cv2.resize(img, (int(re_width), int(re_height)))
    return img, re_width, re_height

def hand(img, max_hands=1, line_color=(0, 0, 255), line_thickness=2, point_color=(0, 255, 0), point_size=5):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=max_hands)
    mpDraw = mp.solutions.drawing_utils

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hand_landmarks = []
    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            if idx == max_hands:
                break

            # Draw landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=point_color, thickness=point_size),
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=line_color, thickness=line_thickness))

            # Store hand landmarks in a list
            hand_landmarks.append(handLms)

    return hand_landmarks, img



red = (0, 0, 255)
black = (0, 0, 0)
green = (0, 255, 0)
gold = (0, 215, 255)
gray = (128, 128, 128)
yellow = (0, 255, 255)
white = (255, 255, 255)
light_blue = (230, 216, 173)

scale_size = 2.5
gap_size = 20
segment_length = 5
line_color = yellow

while True:
    success, img = cap.read()
    if not success:
        break
    height = img.shape[0]
    width = img.shape[1]
    img, re_width, re_height = resize(img, scale_size, width, height)
    img = draw_broken_line(img, (re_width / 2, 0), (re_width / 2, re_height), segment_length, gap_size, line_color, 10)
    img = hand(img)[1]
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
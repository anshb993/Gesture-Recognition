import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
activeGesture = None

BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, "images")

monkeAha = cv2.imread(os.path.join(IMG_DIR, "aha.png"))
monke = cv2.imread(os.path.join(IMG_DIR, "suprise.webp"))
monkeThinking = cv2.imread(os.path.join(IMG_DIR, "thinking.png"))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        
        currentGesture = None
        _, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        def calcAngle(a,b,c): #a=shoulder, b=elbow, c=wrist
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #takes y,x
            angle = np.abs(radians*180/np.pi)

            if angle > 180.0:
                angle = 360-angle

            return angle
        
        def distance(point1, point2, threshold):
            return np.linalg.norm(np.array(point1)-np.array(point2)) < threshold
        
        try:
            landmarks = results.pose_landmarks.landmark
            landmarksHands = results.left_hand_landmarks.landmark
            leftShoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            rightShoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            leftElbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            rightElbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            
            leftWrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            rightWrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

            leftMouth = [landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value].y]
            rightMouth = [landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value].y]
            
            leftHandIndex = [landmarksHands[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].x, landmarksHands[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y]
            leftEye = [landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value].y]
            mouthX = (leftMouth[0] + rightMouth[0]) / 2
            mouthY = (leftMouth[1] + rightMouth[1]) / 2
            mouth = [mouthX, mouthY]
            
            leftElbowAngle = calcAngle(leftShoulder, leftElbow, leftWrist)
            rightElbowAngle = calcAngle(rightShoulder, rightElbow, rightWrist)
            leftShoulderAngle = calcAngle(leftElbow, leftShoulder, rightShoulder)
            rightShoulderAngle = calcAngle(leftShoulder, rightShoulder, rightElbow)
            leftfingertipMouth = distance(leftHandIndex, mouth, 0.05)

            #hands on top of head gesture:
            def in_range(val, low, high):
                return low <= val <= high
            
            leftWristAboveHead = leftWrist[1] < leftShoulder[1] - 0.08
            rightWristAboveHead = rightWrist[1] < rightShoulder[1] - 0.08
            leftIndexAboveEyes = leftHandIndex[1] < leftEye[1] - 0.08
            leftWristAboveEyes = leftWrist[1] < leftEye[1] - 0.03
            HandsOnHead = (
                in_range(leftElbowAngle, 60, 110) and
                in_range(rightElbowAngle, 60, 110) and
                in_range(leftShoulderAngle, 110, 140) and
                in_range(rightShoulderAngle, 110, 140) and
                leftWristAboveHead and
                rightWristAboveHead
            )

            Thinking = (
                leftfingertipMouth
            )
            Aha = (
                leftIndexAboveEyes and
                not leftWristAboveEyes
            )

            if HandsOnHead:
                currentGesture = "monke"
            elif Thinking:
                currentGesture = "thinking"
            elif Aha:
                currentGesture = "aha"

            if currentGesture != activeGesture:
                # close previous
                if activeGesture == "monke":
                    cv2.destroyWindow("monke")
                elif activeGesture == "thinking":
                    cv2.destroyWindow("thinking")
                elif activeGesture == "aha":
                    cv2.destroyWindow("aha")

                # open new
                if currentGesture == "monke":
                    cv2.imshow("monke", monke)
                elif currentGesture == "thinking":
                    cv2.imshow("thinking", monkeThinking)
                elif currentGesture == "aha":
                    cv2.imshow("aha", monkeAha)

                activeGesture = currentGesture


            cv2.putText(image, f"{leftElbowAngle:.6f}",
                        tuple(np.multiply(leftElbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f"{rightElbowAngle:.6f}",
                        tuple(np.multiply(rightElbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f"{leftShoulderAngle:.6f}",
                        tuple(np.multiply(leftShoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f"{rightShoulderAngle:.6f}",
                        tuple(np.multiply(rightShoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f"{rightShoulderAngle:.2f}",
                        tuple(np.multiply(leftHandIndex, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )

        except:
            pass

        #drawing part
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("webcam", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

cap.release()
cv2.destroyAllWindows()
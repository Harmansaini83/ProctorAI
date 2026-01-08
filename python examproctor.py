import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from datetime import datetime

class ExamProctor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.SUSPICIOUS_MOVEMENT_THRESH = 50
        self.counter = 0
        self.alerts = []

        self.cap = cv2.VideoCapture(0)
        self.prev_position = None
        self.suspicious_count = 0

    def eye_aspect_ratio(self, eye_landmarks, landmarks, image_shape):
        h, w = image_shape[:2]
        eye_coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_landmarks]
        A = dist.euclidean(eye_coords[1], eye_coords[5])
        B = dist.euclidean(eye_coords[2], eye_coords[4])
        C = dist.euclidean(eye_coords[0], eye_coords[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_suspicious_behavior(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        is_suspicious = False

        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                self.alerts.append(f"[{datetime.now()}] Multiple faces detected")
                cv2.putText(frame, "MULTIPLE FACES DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame

            face_landmarks = results.multi_face_landmarks[0].landmark

            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            leftEAR = self.eye_aspect_ratio(LEFT_EYE, face_landmarks, frame.shape)
            rightEAR = self.eye_aspect_ratio(RIGHT_EYE, face_landmarks, frame.shape)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.alerts.append(f"[{datetime.now()}] Potential cheating: Eyes closed")
                    cv2.putText(frame, "EYES CLOSED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    is_suspicious = True
            else:
                self.counter = 0

            nose_tip = (int(face_landmarks[1].x * frame.shape[1]), int(face_landmarks[1].y * frame.shape[0]))
            if self.prev_position is not None:
                movement = dist.euclidean(self.prev_position, nose_tip)
                if movement > self.SUSPICIOUS_MOVEMENT_THRESH:
                    self.suspicious_count += 1
                    if self.suspicious_count > 5:
                        self.alerts.append(f"[{datetime.now()}] Suspicious head movement detected")
                        cv2.putText(frame, "SUSPICIOUS MOVEMENT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        is_suspicious = True
                else:
                    self.suspicious_count = max(0, self.suspicious_count - 1)
            self.prev_position = nose_tip

            for idx in LEFT_EYE + RIGHT_EYE:
                pt = (int(face_landmarks[idx].x * frame.shape[1]), int(face_landmarks[idx].y * frame.shape[0]))
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in face_landmarks]) * w
            y_min = min([landmark.y for landmark in face_landmarks]) * h
            x_max = max([landmark.x for landmark in face_landmarks]) * w
            y_max = max([landmark.y for landmark in face_landmarks]) * h

            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            nose = face_landmarks[1]

            if left_eye.x < 0.3:
                cv2.putText(frame, "TURNING LEFT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                is_suspicious = True
            elif right_eye.x > 0.7:
                cv2.putText(frame, "TURNING RIGHT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                is_suspicious = True

            if nose.y < face_landmarks[33].y:
                cv2.putText(frame, "TILTING HEAD UP!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                is_suspicious = True
            elif nose.y > face_landmarks[152].y:
                cv2.putText(frame, "TILTING HEAD DOWN!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                is_suspicious = True

            box_color = (0, 255, 0)
            if is_suspicious:
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 2)

        else:
            self.alerts.append(f"[{datetime.now()}] No face detected")
            cv2.putText(frame, "NO FACE DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        print("Starting exam proctoring... Press 'q' to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture video")
                break
            frame = self.detect_suspicious_behavior(frame)
            cv2.imshow("Exam Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        with open("proctoring_log.txt", "w") as f:
            f.write("\n".join(self.alerts))
        print("Proctoring ended. Alerts saved to proctoring_log.txt")

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    proctor = ExamProctor()
    try:
        proctor.run()
    except KeyboardInterrupt:
        proctor.cleanup()

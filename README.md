# 🛡️ ExamProctor
### Real-Time AI-Based Exam Proctoring System using Computer Vision

ExamProctor is a real-time automated exam invigilation system built using OpenCV, MediaPipe Face Mesh, and Python.  
It monitors a student through a webcam and detects suspicious behaviors such as abnormal head movements, prolonged eye closure, multiple faces, and absence of face during an examination.

---

## 🚀 Features

- Live webcam-based monitoring
- Eye Aspect Ratio (EAR) based eye-closure detection
- Head movement tracking using facial landmarks
- Multiple face detection
- Head direction analysis (left, right, up, down)
- Real-time alerts displayed on video feed
- Automatic logging of suspicious activities with timestamps
- Green bounding box for normal behavior
- Red bounding box for suspicious behavior

---

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- SciPy
- Webcam (Real-time video input)

---

## ⚙️ Working Methodology

### Face Detection
- Uses MediaPipe Face Mesh
- Detects up to 2 faces
- Flags suspicious behavior when:
  - No face is detected
  - Multiple faces are detected

### Eye Closure Detection
- Uses Eye Aspect Ratio (EAR)
- EAR threshold: `0.25`
- Eyes closed for `20` consecutive frames triggers alert

### Head Movement Detection
- Tracks nose tip landmark position
- Calculates Euclidean distance between consecutive frames
- Movement beyond threshold (`50 pixels`) is considered suspicious

### Head Direction Detection
- Detects:
  - Turning Left
  - Turning Right
  - Tilting Up
  - Tilting Down

---

## 🚨 Suspicious Behaviors Detected

- No face detected
- Multiple faces detected
- Eyes closed for long duration
- Sudden or repeated head movement
- Head turning left or right
- Head tilting up or down

---

## 📊 Output

- Real-time alerts shown on the video feed
- Bounding box color changes:
  - Green → Normal behavior
  - Red → Suspicious behavior
    
## 📸 Screenshots

### Normal Monitoring
![Normal Detection](<img width="1331" height="823" alt="Screenshot 2025-04-20 104542" src="https://github.com/user-attachments/assets/77ec3692-468c-422a-bcb6-ee18b886bf17" />
)

### Suspicious Head Movement Detected
![Suspicious Movement](<img width="1360" height="785" alt="Screenshot 2025-04-20 105127" src="https://github.com/user-attachments/assets/52421135-6a14-41c7-ae53-35b9b25bf77f" />
)

- All alerts are saved in a log file


---

## ⚠️ Limitations

- Requires good lighting conditions
- Works best with frontal face orientation
- Designed for single-student monitoring
- Does not detect mobile phones or hand gestures

---

## 🚀 Future Enhancements

- Mobile phone and object detection (YOLO / Faster R-CNN)
- Hand gesture recognition
- Eye gaze tracking
- Multi-student classroom support
- Cloud-based monitoring dashboard

---

## 🎓 Academic Use

- PBL / Mini Project
- Computer Vision & AI Demonstration
- Automated Examination Monitoring Prototype

---

## 👨‍💻 Author

Harman Saini  
B.Tech CSE (AIML)  
Symbiosis Institute of Technology, Nagpur

---

## 📄 License

This project is intended for educational and research purposes only.
Commercial use requires prior permission.



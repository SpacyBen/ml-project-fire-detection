
# 🔥 Fire & Face Detector (Triple YOLO)

Real-time fire & face detection using YOLO models with GUI support.

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/SpacyBen/ml-project-fire-detection.git
cd ml-project-fire-detection


python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate


pip install -r requirements.txt


NOTE:
- This project uses YOLO for **face detection**, not recognition.
- You don’t need dlib or face_recognition to run it.
- If you want to add face recognition:
    1. Install Visual Studio (Desktop C++ workload)
    2. Install CMake
    3. Install dlib in your venv:
        pip install dlib
- Current setup just detects faces and labels manually.


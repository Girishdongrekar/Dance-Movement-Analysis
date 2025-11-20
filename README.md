# 🕺 Dance Movement Analysis — AI/ML Cloud App

This project analyzes dance movements from short videos using **MediaPipe**, **OpenCV**, and **FastAPI**, and provides a real-time skeleton overlay video for visualization.  
It also includes a **Streamlit UI** for easy upload and visualization.

---

## 🚀 Features

- Detects body keypoints using **MediaPipe Pose**
- Generates a skeleton-overlay output video
- JSON output of keypoints per frame
- REST API endpoint for video upload (`/analyze`)
- **Streamlit UI** for uploading and viewing processed videos
- **Dockerized** for cross-platform and cloud deployment

---

## 🧩 Project Structure

├── app/
│ ├── main.py # FastAPI backend
│ ├── dance.py # Core video processing logic
│
├── streamlit_app.py # Streamlit front-end
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
├── README.md
└── .gitignore


---

## ⚙️ Run Locally

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Start FastAPI backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 3️⃣ Start Streamlit app
streamlit run streamlit_app.py
Then open:

Streamlit UI → http://localhost:8501

FastAPI Docs → http://localhost:8000/docs

🐳 Docker Usage

# Build the image
docker build -t dance-ui-api .

# Run the container (expose both FastAPI and Streamlit)
docker run -p 8501:8501 -p 8000:8000 dance-ui-api
Once running, open:

Streamlit UI → http://localhost:8501

FastAPI Docs → http://localhost:8000/docs


🧠 Tech Stack

Python 3.11

FastAPI — REST API for video analysis

Streamlit — UI for interaction

MediaPipe + OpenCV — Pose detection and skeleton overlay

Docker — Containerization for deployment

🧪 Example Usage

Upload a short dance video (≤ 20 seconds)

The backend detects poses frame-by-frame

Output includes:

output.mp4 → Video with skeleton overlay

keypoints.json → JSON with all body landmarks

🧠 Author
Girish Dongrekar
🎓 B.Tech in Computer Science (AI/ML)
💡 Projects: Computer Vision, FastAPI, YOLO, ML Deployment
🌐 GitHub


🧾 License
This project is open-source under the MIT License.

---

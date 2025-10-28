import streamlit as st
import requests
import os
import json
from urllib.parse import urljoin

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
# Use environment variable to set backend host
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = "http://127.0.0.1:8000/analyze"  # FastAPI endpoint inside same container
DOWNLOAD_URL = "http://127.0.0.1:8000/download"  # For download

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="💃 Dance Movement Analysis", layout="centered")
st.title("💃 Dance Movement Analysis")
st.write("Upload a short dance video to analyze body movement with skeleton overlay.")

# ---------------------------------------
# VIDEO UPLOAD
# ---------------------------------------
uploaded_file = st.file_uploader("📤 Upload Dance Video", type=["mp4", "mov", "avi"])
trails = st.checkbox("Show Motion Trails", value=True)

if uploaded_file is not None:
    st.video(uploaded_file)
    os.makedirs("temp", exist_ok=True)
    tmp_input_path = os.path.join("temp", uploaded_file.name)

    with open(tmp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------------------------------
    # CALL FASTAPI BACKEND
    # ---------------------------------------
    if st.button("🚀 Analyze"):
        with st.spinner("Processing video..."):
            try:
                with open(tmp_input_path, "rb") as f:
                    files = {"file": (uploaded_file.name, f, "video/mp4")}
                    data = {"trails": trails}
                    response = requests.post(API_URL, files=files, data=data)

                if response.status_code == 200:
                    res_json = response.json()
                    st.success("✅ Video processed successfully!")

                    st.write(f"**Frames with keypoints:** {res_json.get('keypoints_count_frames')}")

                    # ---------------------------------------
                    # SHOW PROCESSED VIDEO & DOWNLOAD LINKS
                    # ---------------------------------------
                    processed_filename = os.path.basename(res_json.get("temp_video_file", ""))

                    if processed_filename:
                        video_url = f"{DOWNLOAD_URL}/{processed_filename}"
                        st.video(video_url)

                        # Download processed video
                        try:
                            video_bytes = requests.get(video_url).content
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name="processed_dance.mp4",
                                mime="video/mp4"
                            )
                        except Exception as e:
                            st.warning(f"Could not fetch processed video: {e}")

                        # Download JSON
                        json_data = json.dumps(res_json, indent=4)
                        st.download_button(
                            label="📄 Download Analysis JSON",
                            data=json_data,
                            file_name="dance_analysis.json",
                            mime="application/json"
                        )
                    else:
                        st.warning("⚠️ Processed video not found in response.")

                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

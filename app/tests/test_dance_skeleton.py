import os
import json
import pytest
from app.dance import process_video
import warnings


TEST_INPUT = "dance.mp4"
OUT_VIDEO = "output/out_test.mp4"
OUT_JSON = "output/out_test.json"

@pytest.mark.parametrize("trails", [True, False])
def test_process_video(trails):
    process_video(
        input_path=TEST_INPUT,
        output_path=OUT_VIDEO,
        save_json_path=OUT_JSON,
        trails=trails,
        trail_len=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Ensure files created
    assert os.path.exists(OUT_VIDEO), "Output video not generated"
    assert os.path.exists(OUT_JSON), "Output JSON not generated"

    # Validate JSON structure
    with open(OUT_JSON, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), "JSON output should be a list"
    assert "frame_idx" in data[0], "Frame index missing in JSON"


warnings.filterwarnings("ignore", category=UserWarning)
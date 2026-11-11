import json
import queue
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.dataset import get_split, save_split, held_out_indices, load_dataset
from src.sensor_sync import image_for_frame


def test_temporal_split_has_embargo():
    train, val, test = get_split(1000, gap=30)
    assert train[-1] + 31 == val[0]
    assert val[-1] + 31 == test[0]
    assert not set(train) & set(test)
    assert np.array_equal(train, get_split(1000, seed=7)[0])


def test_recordings_never_cross_partitions():
    sessions = np.repeat(["one", "two", "three", "four"], 100)
    parts = get_split(len(sessions), sessions=sessions)
    for a, b in ((0,1), (0,2), (1,2)):
        assert not set(sessions[parts[a]]) & set(sessions[parts[b]])


def test_manifest_rejects_changed_data_and_overlap(tmp_path):
    path = tmp_path / "splits.json"
    save_split(path, get_split(1000), "original", False, 30, pd.DataFrame({"image":range(1000)}))
    assert len(held_out_indices(path, "original", 1000)) == 120
    with pytest.raises(ValueError, match="differs"):
        held_out_indices(path, "modified", 1000)
    manifest = json.loads(path.read_text())
    manifest["test"].append(manifest["train"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlapping"):
        held_out_indices(path, "original", 1000)


def test_missing_depth_is_actionable(tmp_path):
    import cv2
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "1.png"), np.zeros((48,64,3), np.uint8))
    pd.DataFrame({"image":["1.png"], "steering":[0.0]}).to_csv(tmp_path / "telemetry.csv", index=False)
    with pytest.raises(ValueError, match="Missing depth_image"):
        load_dataset(tmp_path, use_depth=True)


def test_sensor_pairing_discards_stale_frames_and_rejects_future():
    images = queue.Queue()
    for frame in [1,2,3]:
        images.put(SimpleNamespace(frame=frame))
    assert image_for_frame(images, 3).frame == 3
    assert image_for_frame(images, 4, timeout=.001) is None
    images.put(SimpleNamespace(frame=5))
    with pytest.raises(RuntimeError, match="another client"):
        image_for_frame(images, 4)

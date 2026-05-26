import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class ObjectDetection(object):
    def __init__(self):
        self._n_box = 6
        self._len_dist = 36
        self._thresh_conf = 0.10
        self._model = YOLO("yolov8n.pt")  # lädt automatisch herunter

    def img2box(self, img_in):
        results = self._model(img_in, verbose=False)
        box_out = np.zeros((self._n_box, 4))
        i = 0
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0 and float(box.conf) > self._thresh_conf:  # 0 = person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_out[i, 0] = x1 / 360
                    box_out[i, 1] = y1 / 180
                    box_out[i, 2] = x2 / 360
                    box_out[i, 3] = y2 / 180
                    i += 1
                    if i >= self._n_box:
                        break
        return box_out

    def _func_Gauss(self, x, mu, sigma):
        return np.exp(-(x - mu)**2 / sigma**2)

    def box2dist(self, box_in):
        points = np.arange(0, self._len_dist + 1, 1)
        dist_azi = np.zeros((self._n_box, self._len_dist + 1))
        dist_ele = np.zeros((self._n_box, self._len_dist + 1))
        for i, each_box in enumerate(box_in):
            if np.sum(each_box) > 0:
                center_azi = (each_box[0] + each_box[2]) / 2
                center_ele = (each_box[1] + each_box[3]) / 2
                len_azi = each_box[2] - each_box[0]
                len_ele = each_box[3] - each_box[1]
                dist_azi[i] = self._func_Gauss(points, center_azi * self._len_dist, len_azi / 2 * self._len_dist)
                dist_ele[i] = self._func_Gauss(points, center_ele * self._len_dist, len_ele / 2 * self._len_dist)
        dist_out = np.stack((dist_azi, dist_ele), axis=0)
        return dist_out


def precompute(video_root, output_root):
    od = ObjectDetection()
    mp4_paths = list(Path(video_root).rglob("*.mp4"))
    print(f"Gefundene Videos: {len(mp4_paths)}")

    for idx, mp4_path in enumerate(mp4_paths):
        relative = mp4_path.relative_to(video_root)
        npy_path = Path(output_root) / relative.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        if npy_path.exists():
            print(f"[{idx+1}/{len(mp4_paths)}] Übersprungen: {npy_path.name}")
            continue

        print(f"[{idx+1}/{len(mp4_paths)}] Verarbeite: {mp4_path.name}")
        cap = cv2.VideoCapture(str(mp4_path))
        features = []

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = frame[:, :, [2, 1, 0]]
            box = od.img2box(frame_rgb)
            dist = od.box2dist(box)
            features.append(dist)

        cap.release()
        features_array = np.array(features, dtype=np.float32)
        np.save(str(npy_path), features_array)
        print(f"   → Shape: {features_array.shape} | {npy_path.name}")

    print("\n✅ Fertig!")


if __name__ == "__main__":
    precompute(
        video_root="/app/data/video_dev",
        output_root="/app/data/visual_features"
    )

"""Camera GUI: sketch a clear smiling face, or an imported picture.

The left pane is live while scanning. Detection runs five times per second.
A frame is kept only when it is sharp, contains one obvious face, and that
face has open eyes and a smile. The left pane then freezes on that frame until
the user shoots again. An imported picture skips those checks. Either original
is saved under input/, the background is removed, and the pretrained conv
sketch network runs.
"""
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import filedialog, ttk

from sketch_net import WEIGHTS, PortraitSketchNet, default_device, sketch_bgr


ROOT = Path(__file__).resolve().parent
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
SEG_MODEL = ROOT / "models" / "selfie_segmenter.tflite"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

DETECT_INTERVAL = 0.2
STABLE_HITS = 3
# Lid gap divided by eye width. Clearly shut eyes fall well below this.
APERTURE_MIN = 0.48
BLINK_MAX = 0.20
SMILE_MIN = 0.003
# outer, inner, upper lid, lower lid
RIGHT_EYE = (33, 133, 159, 145)
LEFT_EYE = (263, 362, 386, 374)
FACE_MIN_SIDE = 0.18
SHARPNESS_MIN = 60.0
BRIGHTNESS_MIN = 50.0
BRIGHTNESS_MAX = 220.0

CAMERA_HELP = (
    "无法打开摄像头。请在「系统设置 → 隐私与安全性 → 摄像头」中允许当前应用，"
    "然后重新打开本程序。"
)


@dataclass
class Check:
    ok: bool
    message: str
    box: tuple | None


def mp_image(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def downscale(bgr, max_side):
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return bgr
    scale = max_side / longest
    return cv2.resize(
        bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )


def face_box(landmarks, width, height):
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    x0 = int(np.clip(min(xs), 0, 1) * width)
    y0 = int(np.clip(min(ys), 0, 1) * height)
    x1 = int(np.clip(max(xs), 0, 1) * width)
    y1 = int(np.clip(max(ys), 0, 1) * height)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1), max(xs) - min(xs), max(ys) - min(ys)


def _landmark_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def eye_aperture(landmarks, indices):
    outer, inner, upper, lower = (landmarks[i] for i in indices)
    opening = _landmark_distance(upper, lower)
    width = _landmark_distance(outer, inner)
    return opening / (width + 1e-6)


def evaluate_frame(landmarker, bgr):
    height, width = bgr.shape[:2]
    result = landmarker.detect(mp_image(downscale(bgr, 640)))
    faces = result.face_landmarks
    if not faces:
        return Check(False, "未检测到人脸，请正对摄像头", None)
    if len(faces) > 1:
        return Check(False, "检测到多张人脸，请保证画面里只有一个人", None)

    landmarks = faces[0]
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    x0, y0, x1, y1, norm_w, norm_h = face_box(landmarks, width, height)
    box = (x0, y0, x1, y1)
    if min(norm_w, norm_h) < FACE_MIN_SIDE:
        return Check(False, "人脸不够明显，请靠近一些", box)
    if min(xs) < -0.01 or max(xs) > 1.01 or min(ys) < -0.01 or max(ys) > 1.01:
        return Check(False, "请让整张脸进入画面", box)

    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if gray.size < 25:
        return Check(False, "人脸不够明显，请靠近一些", box)
    brightness = float(gray.mean())
    if brightness < BRIGHTNESS_MIN:
        return Check(False, "画面太暗，请增加光线", box)
    if brightness > BRIGHTNESS_MAX:
        return Check(False, "画面过亮，请避开强光", box)
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < SHARPNESS_MIN:
        return Check(False, "图像不够清晰，请保持稳定", box)

    right_open = eye_aperture(landmarks, RIGHT_EYE)
    left_open = eye_aperture(landmarks, LEFT_EYE)
    scores = {item.category_name: item.score for item in result.face_blendshapes[0]}
    blink = max(scores.get("eyeBlinkLeft", 1.0), scores.get("eyeBlinkRight", 1.0))
    if min(left_open, right_open) < APERTURE_MIN or blink > BLINK_MAX:
        return Check(False, "请把眼睛完全睁开", box)

    smile = max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0))
    if smile < SMILE_MIN:
        return Check(False, "请微笑", box)
    return Check(True, "画面合格，请保持", box)


def _smooth_loop(values, window):
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def _smooth_silhouette(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return binary
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(pts) < 30:
        return binary
    # A light pass removes shadow pits without turning the hair into a dome.
    mild = np.stack([
        _smooth_loop(pts[:, 0], max(9, len(pts) // 36)),
        _smooth_loop(pts[:, 1], max(9, len(pts) // 36)),
    ], axis=1)
    canvas = np.zeros_like(binary)
    cv2.fillPoly(canvas, [np.round(mild).astype(np.int32)], 1)
    blurred = cv2.GaussianBlur(canvas.astype(np.float32), (9, 9), 0)
    return np.clip((blurred - 0.45) / 0.15, 0.0, 1.0)


def person_mask(segmenter, bgr):
    result = segmenter.segment(mp_image(bgr))
    if not result.confidence_masks:
        raise RuntimeError("背景分割没有返回结果")
    mask = np.array(result.confidence_masks[0].numpy_view(), dtype=np.float32).squeeze()
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    binary = (mask >= 0.45).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
    return _smooth_silhouette(binary)


def remove_background(bgr, mask):
    foreground = mask[..., None]
    white = np.full_like(bgr, 255)
    merged = bgr.astype(np.float32) * foreground + white.astype(np.float32) * (1.0 - foreground)
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    merged[mask < 0.2] = 255
    ys, xs = np.where(mask > 0.5)
    if xs.size < 100:
        raise RuntimeError("没有分离出人物")
    pad_x = int(0.08 * (xs.max() - xs.min() + 1))
    pad_y = int(0.08 * (ys.max() - ys.min() + 1))
    x0 = max(0, int(xs.min()) - pad_x)
    y0 = max(0, int(ys.min()) - pad_y)
    x1 = min(bgr.shape[1], int(xs.max()) + pad_x + 1)
    y1 = min(bgr.shape[0], int(ys.max()) + pad_y + 1)
    return merged[y0:y1, x0:x1]


def draw_box(bgr, box, ok):
    if box is None:
        return bgr
    out = bgr.copy()
    color = (70, 170, 60) if ok else (50, 140, 240)
    cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
    return out


def open_camera():
    backends = [cv2.CAP_AVFOUNDATION] if sys.platform == "darwin" else [cv2.CAP_ANY]
    for backend in backends:
        for index in (0, 1):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
    return None


def to_photo(bgr, max_side):
    height, width = bgr.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    view = bgr
    if scale < 1:
        view = cv2.resize(
            bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = np.ascontiguousarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
    rows, cols = rgb.shape[:2]
    data = f"P6 {cols} {rows} 255\n".encode() + rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class SketchApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("人脸素描系统")
        self.root.geometry("1080x740")
        self.root.minsize(920, 640)
        self.root.configure(bg="#f3efe6")

        self.lock = threading.Lock()
        self.stop = False
        self.phase = "scan"
        self.hits = 0
        self.status = ""
        self.live = None
        self.preview = None
        self.sketch = None
        self.frozen = None
        self.last_box = None
        self.last_ok = False
        self.saved_path = None
        self.view_source = "camera"
        self.render_id = 0
        self.segmenter = None
        self.net = None
        self.device = None
        self.cap = None
        self.camera_ready = False
        self.camera_thread = None
        self.worker = None

        self.preview_photo = None
        self.sketch_photo = None
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.refresh)

    def _build(self):
        tk.Label(
            self.root,
            text="人脸素描系统",
            font=("PingFang SC", 22),
            bg="#f3efe6",
            fg="#2c2416",
        ).pack(pady=(16, 12))
        panels = tk.Frame(self.root, bg="#f3efe6")
        panels.pack(fill="both", expand=True, padx=20)
        preview_frame, self.preview_label = self._panel(panels, "原图")
        sketch_frame, self.sketch_label = self._panel(panels, "素描")
        preview_frame.pack(side="left", expand=True, fill="both", padx=(0, 8))
        sketch_frame.pack(side="left", expand=True, fill="both", padx=(8, 0))

        self.status_var = tk.StringVar(value=self.status)
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("PingFang SC", 15),
            bg="#f3efe6",
            fg="#2c2416",
            wraplength=980,
            justify="center",
        )
        status.pack(pady=(14, 8))

        buttons = tk.Frame(self.root, bg="#f3efe6")
        buttons.pack(pady=(0, 18))
        ttk.Button(buttons, text="导入图片", command=self.import_image).pack(side="left", padx=8)
        ttk.Button(buttons, text="重新拍摄", command=self.retake).pack(side="left", padx=8)
        self.save_btn = ttk.Button(buttons, text="另存为", command=self.save_as)
        self.save_btn.state(["disabled"])
        self.save_btn.pack(side="left", padx=8)

    def _panel(self, parent, caption):
        frame = tk.Frame(parent, bg="#f3efe6")
        tk.Label(
            frame,
            text=caption,
            font=("PingFang SC", 14),
            bg="#f3efe6",
            fg="#2c2416",
        ).pack()
        holder = tk.Frame(frame, width=500, height=380, bg="#e7e1d6")
        holder.pack(pady=6)
        holder.pack_propagate(False)
        label = tk.Label(
            holder,
            text="",
            font=("PingFang SC", 13),
            bg="#e7e1d6",
            fg="#6b5e49",
        )
        label.pack(expand=True, fill="both")
        return frame, label

    def start_worker(self):
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _set_status(self, text):
        with self.lock:
            self.status = text

    def _worker(self):
        try:
            self.cap = open_camera()
            if self.cap is None:
                self._set_status(CAMERA_HELP)
                return
            with self.lock:
                self.camera_ready = True
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            landmarker, segmenter, net, device = self._load()
            with self.lock:
                self.segmenter = segmenter
                self.net = net
                self.device = device
            self._detect_loop(landmarker)
        except Exception as exc:
            self._set_status(f"启动失败：{exc}")
        finally:
            if self.stop:
                if self.camera_thread is not None:
                    self.camera_thread.join(timeout=1)
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

    def _camera_loop(self):
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._set_status("摄像头中断")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.live = frame
                if self.phase != "scan":
                    continue
                box = self.last_box
                good = self.last_ok
            shown = draw_box(frame, box, good)
            with self.lock:
                if self.phase == "scan":
                    self.preview = shown

    def _load(self):
        missing = [
            path.name
            for path in (FACE_MODEL, SEG_MODEL, WEIGHTS)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("缺少模型文件：" + "、".join(missing))
        cpu = python.BaseOptions.Delegate.CPU
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL), delegate=cpu),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                output_face_blendshapes=True,
            )
        )
        segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(model_asset_path=str(SEG_MODEL), delegate=cpu),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
            )
        )
        device = default_device()
        net = PortraitSketchNet().to(device).eval()
        with torch.no_grad():
            net(torch.full((1, 3, 64, 64), 0.5, device=device))
        return landmarker, segmenter, net, device

    def _detect_loop(self, landmarker):
        last_detect = 0.0
        while not self.stop:
            with self.lock:
                phase = self.phase
                frame = None if self.live is None else self.live.copy()
            if phase != "scan" or frame is None:
                time.sleep(0.03)
                continue
            wait = DETECT_INTERVAL - (time.monotonic() - last_detect)
            if wait > 0:
                time.sleep(wait)
            with self.lock:
                if self.phase != "scan":
                    continue
                frame = None if self.live is None else self.live.copy()
            if frame is None:
                continue
            last_detect = time.monotonic()
            check = evaluate_frame(landmarker, frame)
            start_render = False
            with self.lock:
                if self.phase != "scan":
                    continue
                self.last_box = check.box
                self.last_ok = check.ok
                if check.ok:
                    self.hits += 1
                else:
                    self.hits = 0
                if self.hits >= STABLE_HITS:
                    self.frozen = frame.copy()
                    self.preview = self.frozen.copy()
                    self.view_source = "camera"
                    self.render_id += 1
                    self.phase = "render"
                    self.last_box = None
                    self.status = ""
                    start_render = True
                elif check.ok:
                    self.status = "请保持"
                else:
                    self.status = check.message
            if start_render:
                threading.Thread(target=self._render, daemon=True).start()

    def _render(self):
        with self.lock:
            token = self.render_id
            source = self.view_source
            frozen = None if self.frozen is None else self.frozen.copy()
            segmenter = self.segmenter
            net = self.net
            device = self.device
        if frozen is None or segmenter is None or net is None:
            with self.lock:
                if self.phase == "render" and self.render_id == token:
                    self.phase = "scan"
                    self.hits = 0
            return
        try:
            cut = remove_background(frozen, person_mask(segmenter, frozen))
            sketch = sketch_bgr(net, cut, device)
        except Exception as exc:
            with self.lock:
                if self.phase != "render" or self.render_id != token:
                    return
                if source == "import":
                    self.phase = "done"
                    self.status = f"生成失败：{exc}"
                else:
                    self.phase = "scan"
                    self.hits = 0
                    self.view_source = "camera"
                    self.status = f"生成失败：{exc}"
            return
        _, saved = self._write_pair(frozen, sketch, source)
        with self.lock:
            if self.phase != "render" or self.render_id != token:
                return
            self.sketch = sketch
            self.preview = frozen
            self.saved_path = saved
            self.phase = "done"
            self.last_box = None
            self.status = ""

    def _write_pair(self, original, sketch, source):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        prefix = "import" if source == "import" else "capture"
        src = INPUT_DIR / f"{prefix}_{stamp}.png"
        saved = OUTPUT_DIR / f"sketch_{stamp}.png"
        if not cv2.imwrite(str(src), original):
            raise RuntimeError(f"无法保存 {src.name}")
        if not cv2.imwrite(str(saved), sketch):
            raise RuntimeError(f"无法保存 {saved.name}")
        return src, saved

    def refresh(self):
        with self.lock:
            preview = None if self.preview is None else self.preview.copy()
            sketch = None if self.sketch is None else self.sketch.copy()
            status = self.status
            phase = self.phase
        if preview is not None:
            self.preview_photo = to_photo(preview, 500)
            self.preview_label.configure(image=self.preview_photo, text="")
        if sketch is not None:
            self.sketch_photo = to_photo(sketch, 500)
            self.sketch_label.configure(image=self.sketch_photo, text="")
            self.save_btn.state(["!disabled"])
        else:
            waiting = "生成中" if phase == "render" else ""
            if self.sketch_photo is not None or str(self.sketch_label.cget("text")) != waiting:
                self.sketch_label.configure(image="", text=waiting)
                self.sketch_photo = None
            self.save_btn.state(["disabled"])
        self.status_var.set(status)
        if not self.stop:
            self.root.after(30, self.refresh)

    def retake(self):
        with self.lock:
            ready = self.camera_ready
            self.render_id += 1
            self.phase = "scan"
            self.view_source = "camera"
            self.hits = 0
            self.sketch = None
            self.frozen = None
            self.last_box = None
            self.last_ok = False
            self.saved_path = None
            if self.live is not None:
                self.preview = self.live.copy()
            self.status = "" if ready else CAMERA_HELP
        self.sketch_label.configure(image="", text="")
        self.sketch_photo = None

    def import_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="导入图片",
            initialdir=str(INPUT_DIR if INPUT_DIR.is_dir() else Path.home()),
            filetypes=[
                ("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            self._set_status("无法读取图片")
            return
        bgr = downscale(bgr, 1920)
        with self.lock:
            if self.segmenter is None or self.net is None or self.phase == "render":
                self.status = "请稍候"
                return
            self.frozen = bgr
            self.preview = bgr.copy()
            self.sketch = None
            self.saved_path = None
            self.view_source = "import"
            self.hits = 0
            self.last_box = None
            self.last_ok = False
            self.render_id += 1
            self.phase = "render"
            self.status = ""
        self.sketch_label.configure(image="", text="生成中")
        self.sketch_photo = None
        self.save_btn.state(["disabled"])
        threading.Thread(target=self._render, daemon=True).start()

    def save_as(self):
        with self.lock:
            sketch = None if self.sketch is None else self.sketch.copy()
        if sketch is None:
            return
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="保存素描",
            initialdir=str(OUTPUT_DIR),
            initialfile=f"sketch_{time.strftime('%Y%m%d_%H%M%S')}.png",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")],
        )
        if not path:
            return
        cv2.imwrite(path, sketch)
        self._set_status("已保存")

    def close(self):
        self.stop = True
        if self.worker is not None:
            self.worker.join(timeout=2)
        self.root.destroy()


def main():
    app = SketchApp()
    app.root.after(100, app.start_worker)
    app.root.mainloop()


if __name__ == "__main__":
    main()

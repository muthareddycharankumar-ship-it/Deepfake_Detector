import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
import subprocess
import tempfile
from model_loader import device

import mediapipe as mp
_mp_face  = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def _detect_faces_mp(frame_bgr):
    H, W     = frame_bgr.shape[:2]
    min_size = int(W * 0.05)
    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results  = _detector.process(rgb)
    boxes    = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x  = max(0, int(bb.xmin * W))
            y  = max(0, int(bb.ymin * H))
            w  = min(int(bb.width  * W), W - x)
            h  = min(int(bb.height * H), H - y)
            if w >= min_size and h >= min_size:
                boxes.append((x, y, w, h, bb.xmin, bb.ymin))
    return boxes

def _crop_face(frame_bgr, x, y, w, h, pad=0.10):
    H, W   = frame_bgr.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(W, x + w + px), min(H, y + h + py)
    return Image.fromarray(cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

def _resize_frame(frame_bgr, max_dim=224):
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    return frame_bgr

def convert_video_with_ffmpeg(input_path):
    temp_dir    = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"converted_{os.path.basename(input_path)}.mp4")
    if os.path.exists(output_path):
        return output_path
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-preset", "fast",
           "-c:a", "aac", "-movflags", "+faststart", output_path]
    print("Converting video with ffmpeg...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}", file=sys.stderr)
        return None
    return output_path

def _score_image(pil_img, model, processor, fake_index=1):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        return probs[fake_index].item()
    return torch.sigmoid(outputs).squeeze().item()

def analyze_video(video_path, model, processor, sample_every=10):
    from model_loader import load_ai_model
    ai_model, ai_processor = load_ai_model()

    print(f"Opening video: {video_path}", file=sys.stderr)
    cap = cv2.VideoCapture(video_path)
    ok, test_frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()

    if not ok or test_frame is None:
        converted = convert_video_with_ffmpeg(video_path)
        if converted and os.path.exists(converted):
            video_path = converted
        else:
            raise ValueError(f"Cannot decode video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    W           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {W}x{H} {fps:.2f}fps {frame_count} frames", file=sys.stderr)

    face_scores    = []
    ai_scores      = []
    face_positions = []
    face_crop_count = 0
    total_sampled   = 0
    frame_id        = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % sample_every != 0: continue
        total_sampled += 1

        small    = _resize_frame(frame.copy())
        pil_full = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ai_score = _score_image(pil_full, ai_model, ai_processor)
        ai_scores.append(ai_score)

        faces = _detect_faces_mp(frame)
        for (x, y, w, h, xrel, yrel) in faces:
            pil_face   = _crop_face(frame, x, y, w, h, pad=0.10)
            face_score = _score_image(pil_face, model, processor, fake_index=0)
            face_scores.append(face_score)
            face_positions.append((xrel, yrel))
            face_crop_count += 1

    cap.release()

    if face_crop_count > 0:
        xs     = [p[0] for p in face_positions]
        ys     = [p[1] for p in face_positions]
        motion = max(float(np.std(xs)), float(np.std(ys)))
        face_coverage = face_crop_count / max(total_sampled, 1)
        print(f"Face motion={motion:.4f} coverage={face_coverage:.2f}", file=sys.stderr)

        if face_coverage < 0.40:
            # Faces detected in less than 40% of frames — likely CGI/animation
            # where character turns away. Use ai_vs_real for reliable detection.
            print(f"PATH C2: low face coverage ({face_coverage:.0%}) → ai_vs_real", file=sys.stderr)
            final_scores = ai_scores
        elif motion > 0.03:
            print("PATH A: moving face → face scores", file=sys.stderr)
            final_scores = face_scores
        elif motion < 0.01:
            print("PATH B: static face → ai_vs_real scores", file=sys.stderr)
            final_scores = ai_scores
        else:
            print(f"PATH borderline → face scores", file=sys.stderr)
            final_scores = face_scores
    else:
        print("PATH C: no faces → ai_vs_real scores", file=sys.stderr)
        final_scores = ai_scores

    print(f"Sampled {total_sampled} | crops: {face_crop_count} | final: {len(final_scores)}", file=sys.stderr)

    if not final_scores:
        return 0.0, [], face_crop_count if face_crop_count > 0 else -1, total_sampled

    avg       = float(np.mean(final_scores))
    faces_out = face_crop_count if face_crop_count > 0 else -1
    return avg, final_scores, faces_out, total_sampled

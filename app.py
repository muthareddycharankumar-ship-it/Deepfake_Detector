import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import streamlit as st
import tempfile
import time
import uuid
from PIL import Image as PILImage

from model_loader import load_model, load_ai_model, device
from video_processor import analyze_video

st.cache_data.clear()
st.set_page_config(page_title="DeepScan · AI Detector", page_icon="🕵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a0f; color: #e8e8f0; }
.stApp { background-color: #0a0a0f; }
.ds-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.ds-wordmark { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ds-tagline { font-size: 0.85rem; letter-spacing: 0.18em; text-transform: uppercase; color: #555570; margin-top: 0.4rem; }
.ds-divider { border: none; border-top: 1px solid #1e1e2e; margin: 1rem 0 2rem; }
.ds-card { background: #12121e; border: 1px solid #1e1e2e; border-radius: 16px; padding: 2rem; margin: 1.5rem 0; }
.ds-verdict { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.5rem; }
.verdict-real { color: #00f5a0; }
.verdict-ai   { color: #ff4f6d; }
.verdict-grey { color: #888899; }
.ds-score-row { display: flex; align-items: center; gap: 0.75rem; margin: 1rem 0; }
.ds-score-label { font-size: 0.8rem; color: #888899; min-width: 90px; }
.ds-progress-bg { flex: 1; height: 8px; background: #1e1e2e; border-radius: 4px; overflow: hidden; }
.ds-progress-fill { height: 100%; border-radius: 4px; }
.ds-score-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; min-width: 60px; text-align: right; }
.ds-stats { display: flex; gap: 1.5rem; margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid #1e1e2e; }
.ds-stat { flex: 1; text-align: center; }
.ds-stat-value { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #e8e8f0; }
.ds-stat-label { font-size: 0.7rem; color: #555570; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.ds-footer { text-align: center; font-size: 0.72rem; color: #white; padding: 2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ds-header">
    <div class="ds-wordmark">DEEPSCAN</div>
    <div class="ds-tagline">AI-Generated Media Detection</div>
</div>
<hr class="ds-divider"/>
""", unsafe_allow_html=True)

THRESHOLD = 0.50

def render_result(score, faces, frames, scores_count, media_type="VIDEO"):
    pct = round(score * 100, 1)
    if score >= THRESHOLD:
        verdict_label = f"AI GENERATED {media_type}"
        verdict_css   = "verdict-ai"
        verdict_color = "#ff4f6d"
    else:
        verdict_label = f"REAL {media_type}"
        verdict_css   = "verdict-real"
        verdict_color = "#00f5a0"

    faces_display = str(abs(faces)) if faces >= 0 else "N/A"
    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-verdict {verdict_css}">{verdict_label}</div>
        <div class="ds-score-row">
            <span class="ds-score-label">AI probability</span>
            <div class="ds-progress-bg">
                <div class="ds-progress-fill" style="width:{pct}%;background:{verdict_color};"></div>
            </div>
            <span class="ds-score-value" style="color:{verdict_color};">{pct}%</span>
        </div>
        <div class="ds-stats">
            <div class="ds-stat"><div class="ds-stat-value">{faces_display}</div><div class="ds-stat-label">Faces Scored</div></div>
            <div class="ds-stat"><div class="ds-stat-value">{frames}</div><div class="ds-stat-label">Frames</div></div>
            <div class="ds-stat"><div class="ds-stat-value">{scores_count}</div><div class="ds-stat-label">Scores</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Single uploader for both video and image ──────────────────────────────────
uploaded = st.file_uploader(
    "Drop a video or image to analyse",
    type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="visible",
)

IMAGE_TYPES = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_TYPES = {"mp4", "avi", "mov", "mkv"}

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    is_image = ext in IMAGE_TYPES
    is_video = ext in VIDEO_TYPES

    # Preview
    if is_image:
        pil_img = PILImage.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True)

    if st.button("⟶  Analyse"):

        # ── IMAGE ─────────────────────────────────────────────────────────────
        if is_image:
            import torch, numpy as np, cv2, mediapipe as mp
            from video_processor import _resize_frame

            with st.spinner("Loading models…"):
                face_model, face_proc = load_model()
                ai_model,   ai_proc   = load_ai_model()

            img_cv   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            H, W     = img_cv.shape[:2]
            min_size = int(max(W, H) * 0.02)  # 2% for images — group photos have small faces

            # Use both model_selection=0 (close) and 1 (far) to catch all faces
            all_detections = []
            for ms in [0, 1]:
                fd = mp.solutions.face_detection.FaceDetection(model_selection=ms, min_detection_confidence=0.5)
                r  = fd.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                if r.detections:
                    all_detections.extend(r.detections)

            # Deduplicate by IoU — remove overlapping boxes
            def iou(a, b):
                ax,ay,aw,ah = a; bx,by,bw,bh = b
                ix = max(0, min(ax+aw,bx+bw) - max(ax,bx))
                iy = max(0, min(ay+ah,by+bh) - max(ay,by))
                inter = ix*iy
                union = aw*ah + bw*bh - inter
                return inter/union if union > 0 else 0

            kept = []
            for det in all_detections:
                bb = det.location_data.relative_bounding_box
                box = (bb.xmin, bb.ymin, bb.width, bb.height)
                if all(iou(box, (k.location_data.relative_bounding_box.xmin,
                                 k.location_data.relative_bounding_box.ymin,
                                 k.location_data.relative_bounding_box.width,
                                 k.location_data.relative_bounding_box.height)) < 0.4 for k in kept):
                    kept.append(det)

            face_scores = []
            face_count  = 0

            for det in kept:
                    bb = det.location_data.relative_bounding_box
                    x  = max(0, int(bb.xmin * W))
                    y  = max(0, int(bb.ymin * H))
                    fw = min(int(bb.width  * W), W - x)
                    fh = min(int(bb.height * H), H - y)
                    if fw < min_size: continue
                    px, py   = int(fw * 0.10), int(fh * 0.10)
                    crop     = img_cv[max(0,y-py):min(H,y+fh+py), max(0,x-px):min(W,x+fw+px)]
                    pil_crop = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inp      = face_proc(images=pil_crop, return_tensors="pt")
                    inp      = {k: v.to(device) for k, v in inp.items()}
                    with torch.no_grad():
                        out  = face_model(**inp)
                    probs = torch.softmax(out.logits, dim=1).squeeze()
                    face_scores.append(probs[0].item())
                    face_count += 1

            small     = _resize_frame(img_cv.copy())
            pil_small = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            inp2      = ai_proc(images=pil_small, return_tensors="pt")
            inp2      = {k: v.to(device) for k, v in inp2.items()}
            with torch.no_grad():
                out2   = ai_model(**inp2)
            ai_score = torch.softmax(out2.logits, dim=1).squeeze()[1].item()
            avg_face = 0.0

            if face_scores:
                avg_face = float(np.mean(face_scores))
                if face_count >= 3:
                    # Group photo: many real faces → trust face model only
                    # ai_vs_real gets confused by group/indoor scenes
                    final_score = avg_face
                elif avg_face > 0.40:
                    # Face model already says fake → trust it
                    final_score = avg_face
                else:
                    # Single face, face model unsure → weighted combo
                    # ai_vs_real gets 60% weight for single-face AI portraits
                    final_score = (avg_face * 0.4) + (ai_score * 0.6)
            else:
                final_score = ai_score

            print(f"Image: {W}x{H} | Faces={face_count} | face_avg={avg_face if face_scores else 0:.2f} | ai={ai_score:.2f} | final={final_score*100:.1f}%")
            render_result(final_score, face_count, 1, 1, "IMAGE")

        # ── VIDEO ─────────────────────────────────────────────────────────────
        elif is_video:
            suffix = f"_{uuid.uuid4().hex}.{ext}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                video_path = tmp.name

            with st.spinner("Loading models…"):
                model, processor = load_model()

            bar = st.progress(0, text="Scanning frames…")
            for p in range(0, 80, 10):
                time.sleep(0.05)
                bar.progress(p, text="Scanning frames…")

            score, frame_scores, faces_detected, total_frames = analyze_video(video_path, model, processor)

            for p in range(80, 101, 5):
                time.sleep(0.03)
                bar.progress(p, text="Finalising…")
            bar.empty()

            render_result(score, faces_detected, total_frames, len(frame_scores), "VIDEO")

            try: os.unlink(video_path)
            except: pass

st.markdown(
    '<div class="ds-footer">Powered by Okulr Techminds · SigLIP + ViT · Results are probabilistic</div>',
    unsafe_allow_html=True,
)
import logging
import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse

from app.routes import upload, preprocess, features, predict, heuristic, fusion, explain, result
from app.schemas.detection_schema import FinalDetectionResponse
from app.services.audio_processing import extract_audio
from app.services.explanation_service import generate_explanation
from app.services.feature_extraction import extract_features
from app.services.fusion_service import fuse_scores
from app.services.heuristic_service import heuristic_analysis
from app.services.model_service import predict_model
from app.services.video_processing import extract_frames, detect_and_crop_faces
from app.utils.config import (
    FAKE_THRESHOLD,
    HEURISTIC_WEIGHT,
    MODEL_WEIGHT,
    UPLOAD_DIR,
    VIDEO_MODEL_WEIGHT,
    AUDIO_MODEL_WEIGHT,
    AUDIO_LIBROSA_WEIGHT,
    AUDIO_LR_WEIGHT,
    AUDIO_LIPSYNC_WEIGHT,
    FUSION_NONLINEAR_POWER,
    FUSION_MODEL_WEIGHT,
    FUSION_LIPSYNC_WEIGHT,
    FUSION_DEEPFACE_WEIGHT,
    VIDEO_CNN_WEIGHT,
    VIDEO_TEMPORAL_WEIGHT,
    VIDEO_FFT_WEIGHT,
    VIDEO_DEEPFACE_WEIGHT,
    VIDEO_LANDMARK_WEIGHT,
    VIDEO_RF_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Detection Pipeline",
        "description": "Upload a video to run the full multimodal deepfake detection pipeline.",
    }
]

app = FastAPI(
    title="Authentix - Deepfake Detection API",
    description=(
        "## Welcome to the Authentix API Testing Interface\n\n"
        "Use this documentation page to test the deepfake detection system.\n\n"
        "### Instructions:\n"
        "1. Expand the **`POST /detect`** endpoint below.\n"
        "2. Click **`Try it out`**.\n"
        "3. Choose a `.mp4` file.\n"
        "4. Click **`Execute`** to run the analysis.\n\n"
        "---\n"
        "*The system uses a hybrid multimodal pipeline: 6 video AI models + 3 audio AI models "
        "fused with heuristic analysis for the final verdict.*"
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
)

app.include_router(upload.router)
app.include_router(preprocess.router)
app.include_router(features.router)
app.include_router(predict.router)
app.include_router(heuristic.router)
app.include_router(fusion.router)
app.include_router(explain.router)
app.include_router(result.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/detect",
    summary="Full end-to-end deepfake detection (v2 — advanced hybrid)",
    tags=["Detection Pipeline"],
    response_model=FinalDetectionResponse,
)
async def detect(video_file: UploadFile = File(...)):
    """
    Complete deepfake detection pipeline in a single HTTP call.

    Pipeline v2:
        A. Upload → save video to disk
        B. OpenCV → extract 10–15 frames, face detection + crop
        C. FFmpeg → extract .wav audio track
        D. Features → video: [variance, flicker, blur]
                       audio: [mfcc_mean, pitch_var, energy]
        E. Video Models → CNN + DeepFace + Temporal + FFT + Landmark + RF
        F. Audio Models → Librosa + LR + Lip-Sync
        G. Heuristics → blur/flicker/variance + pitch/energy rules
        H. Fusion → 0.85 * model + 0.15 * heuristic
        I. Decision → FAKE >= 0.5 | REAL < 0.5
    """
    video_id = str(uuid.uuid4())

    print("\n" + "=" * 60)
    print("  ===== AUTHENTIX v2 DEBUG START =====")
    print(f"  video_id: {video_id}")
    print("=" * 60)

    # ── [A] Upload ─────────────────────────────────────────────────────────
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    try:
        contents = await video_file.read()
        with open(video_path, "wb") as handle:
            handle.write(contents)
        file_size_kb = os.path.getsize(video_path) / 1024
        print("\n[A] INPUT")
        print(f"    Video Path  : {os.path.abspath(video_path)}")
        print(f"    File Size   : {file_size_kb:.1f} KB")
        print(f"    Original    : {video_file.filename}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    # ── [B] Frame Extraction + Face Detection ──────────────────────────────
    try:
        frames = extract_frames(video_path)
        print("\n[B] VIDEO PROCESSING (OpenCV)")
        print(f"    Frames extracted : {len(frames)}")
        if frames:
            print(f"    Frame shape      : {frames[0].shape}")
        else:
            raise HTTPException(status_code=422, detail="No frames could be read from video.")

        # Face detection and cropping
        face_crops, original_frames = detect_and_crop_faces(frames)
        print(f"    Face crops       : {len(face_crops)}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {exc}")

    # ── [C] Audio Extraction ───────────────────────────────────────────────
    audio_path = None
    audio_missing = False
    try:
        audio_path = extract_audio(video_path)
        audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        print("\n[C] AUDIO EXTRACTION (FFmpeg)")
        print(f"    Audio Path  : {audio_path}")
        print(f"    Exists      : {os.path.exists(audio_path)}")
        print(f"    File Size   : {audio_size / 1024:.1f} KB")
        if audio_size == 0:
            logger.warning("Audio fallback: extracted audio is empty")
            print("    ⚠ Audio is empty — using fallback")
            audio_path = None
            audio_missing = True
    except RuntimeError as exc:
        logger.warning("Audio fallback: extraction failed (%s)", exc)
        print(f"\n[C] AUDIO EXTRACTION fallback: {exc}")
        audio_path = None
        audio_missing = True

    # ── [D] Feature Extraction ─────────────────────────────────────────────
    features_dict = extract_features(frames, audio_path or "", face_crops=face_crops)
    vf = features_dict["video"]
    af = features_dict["audio"]
    print("\n[D] FEATURE EXTRACTION")
    print(f"    Video features  : variance={vf['variance']:.4f}  flicker={vf['flicker']:.4f}  blur={vf['blur']:.4f}")
    print(f"    Audio features  : mfcc={af['mfcc_mean']:.4f}  pitch_var={af['pitch_var']:.4f}  energy={af['energy']:.6f}")

    # ── [E] Model Prediction (ALL AI models) ───────────────────────────────
    model_scores = predict_model(features_dict)

    # Extract all scores
    cnn_score       = model_scores["cnn_score"]
    deepface_score  = model_scores["deepface_score"]
    temporal_score  = model_scores["temporal_score"]
    fft_score       = model_scores["fft_score"]
    landmark_score  = model_scores["landmark_score"]
    rf_score        = model_scores["rf_score"]
    texture_score   = model_scores["texture_score"]
    librosa_score   = model_scores["librosa_score"]
    lr_score        = model_scores["lr_score"]
    lip_sync_score  = model_scores["lip_sync_score"]

    video_model_score = model_scores["video_model_score"]
    audio_model_score = model_scores["audio_model_score"]
    model_score       = model_scores["model_score"]

    print("\n[E] MODEL PREDICTION")
    print(f"    Video model score  : {video_model_score:.4f}")
    print(f"      CNN       ({VIDEO_CNN_WEIGHT:.2f}) : {cnn_score:.4f}")
    print(f"      Temporal  ({VIDEO_TEMPORAL_WEIGHT:.2f}) : {temporal_score:.4f}")
    print(f"      FFT       ({VIDEO_FFT_WEIGHT:.2f}) : {fft_score:.4f}")
    print(f"      DeepFace  ({VIDEO_DEEPFACE_WEIGHT:.2f}) : {deepface_score:.4f}")
    print(f"      Landmark  ({VIDEO_LANDMARK_WEIGHT:.2f}) : {landmark_score:.4f}")
    print(f"      RF        ({VIDEO_RF_WEIGHT:.2f}) : {rf_score:.4f}")
    print(f"    Audio model score  : {audio_model_score:.4f}")
    print(f"      Librosa   ({AUDIO_LIBROSA_WEIGHT:.2f}) : {librosa_score:.4f}")
    print(f"      LR        ({AUDIO_LR_WEIGHT:.2f}) : {lr_score:.4f}")
    print(f"      LipSync   ({AUDIO_LIPSYNC_WEIGHT:.2f}) : {lip_sync_score:.4f}")
    print(f"    model_score        : {VIDEO_MODEL_WEIGHT:.2f} * {video_model_score:.4f} + {AUDIO_MODEL_WEIGHT:.2f} * {audio_model_score:.4f} = {model_score:.4f}")

    # ── [F] Heuristic Analysis (logged only — not used in final score) ────────
    heuristic_scores = heuristic_analysis(features_dict)
    video_heuristic_score = heuristic_scores["video_heuristic_score"]
    audio_heuristic_score = heuristic_scores["audio_heuristic_score"]
    heuristic_score       = heuristic_scores["heuristic_score"]
    print("\n[F] HEURISTIC ANALYSIS (info only)")
    print(f"    Video heuristic : {video_heuristic_score:.4f}")
    print(f"    Audio heuristic : {audio_heuristic_score:.4f}")
    print(f"    heuristic_score : {heuristic_score:.4f}  [not used in final score]")
    for issue in heuristic_scores["issues"]:
        print(f"    ⚠ {issue}")

    # ── [G] Final Fusion — 3-step direct injection ────────────────────────
    # Step 1: model_score = 0.50 * video + 0.50 * audio  (done in model_loader)
    # Step 2: non-linear boost  (x^0.85 amplifies mid-range scores)
    boosted_model = model_score ** FUSION_NONLINEAR_POWER
    # Step 3: direct signal injection of strongest fake signals
    final_score = (
        FUSION_MODEL_WEIGHT    * boosted_model  +
        FUSION_LIPSYNC_WEIGHT  * lip_sync_score +
        FUSION_DEEPFACE_WEIGHT * deepface_score
    )
    final_score = max(0.0, min(1.0, final_score))

    print("\n[G] FINAL FUSION")
    print(f"    boosted_model = {model_score:.4f} ^ {FUSION_NONLINEAR_POWER} = {boosted_model:.4f}")
    print(f"    final_score   = {FUSION_MODEL_WEIGHT} * {boosted_model:.4f}")
    print(f"                  + {FUSION_LIPSYNC_WEIGHT} * {lip_sync_score:.4f}  [lip_sync]")
    print(f"                  + {FUSION_DEEPFACE_WEIGHT} * {deepface_score:.4f}  [deepface]")
    print(f"                  = {final_score:.4f}")

    # ── [H] Decision Engine ───────────────────────────────────────────────
    label = "FAKE" if final_score >= FAKE_THRESHOLD else "REAL"

    print("\n[H] DECISION ENGINE")
    print(f"    final_score : {final_score:.4f}")
    print(f"    label       : {label}  (FAKE >= {FAKE_THRESHOLD:.2f} | REAL < {FAKE_THRESHOLD:.2f})")

    # ── [I] Fusion service (populate structured result) ────────────────────
    fusion_result = fuse_scores(model_scores, heuristic_scores)

    # ── [J] Explanation ────────────────────────────────────────────────────
    try:
        explanation_result = generate_explanation(features_dict, fusion_result)
    except Exception as exc:
        logger.warning("Explanation fallback: %s", exc)
        explanation_result = {
            "explanation": ["Detection completed, but detailed explanation generation fell back."],
            "attribution": "Attribution unavailable",
        }

    print("\n[J] EXPLANATION")
    for line in explanation_result["explanation"]:
        print(f"    - {line}")
    print(f"    Attribution : {explanation_result['attribution']}")

    print("\n" + "=" * 60)
    print("  ===== AUTHENTIX v2 DEBUG END =====")
    print("=" * 60 + "\n")

    # ── Build response ─────────────────────────────────────────────────────
    return {
        "video_id":   video_id,
        "label":      label,
        "confidence": round(final_score, 4),

        # ── Top-level scores ───────────────────────────────────────────────
        "video_model_score": round(video_model_score, 4),
        "audio_model_score": round(audio_model_score, 4),
        "model_score":       round(model_score, 4),
        "heuristic_score":   round(heuristic_score, 4),
        "final_score":       round(final_score, 4),

        # ── Signal breakdown ───────────────────────────────────────────────
        "breakdown": {
            "cnn":       round(cnn_score, 4),
            "temporal":  round(temporal_score, 4),
            "fft":       round(fft_score, 4),
            "deepface":  round(deepface_score, 4),
            "landmark":  round(landmark_score, 4),
            "lip_sync":  round(lip_sync_score, 4),
        },

        # ── Explanation ────────────────────────────────────────────────────
        "explanation": explanation_result["explanation"],
        "attribution": explanation_result["attribution"],

        # ── Required nested blocks ─────────────────────────────────────────
        "model_prediction": {
            "video_model_score": round(video_model_score, 4),
            "audio_model_score": round(audio_model_score, 4),
            "model_score":       round(model_score, 4),
        },
        "heuristic_analysis": {
            "video_heur":      round(video_heuristic_score, 4),
            "audio_heur":      round(audio_heuristic_score, 4),
            "heuristic_score": round(heuristic_score, 4),
        },
        "fusion": {
            "final_score": round(final_score, 4),
        },

        # ── Legacy flat fields (backward compat) ───────────────────────────
        "scores": {
            "video_model_score":    round(video_model_score, 4),
            "audio_model_score":    round(audio_model_score, 4),
            "model_score":          round(model_score, 4),
            "video_heuristic_score": round(video_heuristic_score, 4),
            "audio_heuristic_score": round(audio_heuristic_score, 4),
            "heuristic_score":      round(heuristic_score, 4),
            "final_score":          round(final_score, 4),
        },
        "issues":  heuristic_scores["issues"],
        "features": {
            "video": features_dict["video"],
            "audio": features_dict["audio"],
            "pretrained_video_score": round(deepface_score, 4),
            "pretrained_audio_score": round(librosa_score, 4),
        },
    }

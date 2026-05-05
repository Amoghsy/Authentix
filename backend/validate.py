"""
validate.py - Authentix Pipeline Validation Script
==================================================
Run from backend/ directory:
    python validate.py
    python validate.py path/to/video.mp4

Validation checklist:
  A. Input save
  B. Audio extraction
  C. Video processing
  D. Feature extraction
  E. Hybrid model inference
  F. Heuristic analysis
  G. Score aggregation
  H. Final fusion
  I. Decision engine
  J. Final response schema
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.config import (  # noqa: E402
    CUSTOM_WEIGHT,
    FAKE_THRESHOLD,
    PRETRAINED_WEIGHT,
)

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    OK = lambda s: f"{Fore.GREEN}[PASS]{Style.RESET_ALL} {s}"
    ERR = lambda s: f"{Fore.RED}[FAIL]{Style.RESET_ALL} {s}"
    WARN = lambda s: f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {s}"
    HDR = lambda s: f"{Fore.CYAN}{s}{Style.RESET_ALL}"
except ImportError:
    OK = lambda s: f"[PASS] {s}"
    ERR = lambda s: f"[FAIL] {s}"
    WARN = lambda s: f"[WARN] {s}"
    HDR = lambda s: s

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, "app", "models")
DATA_DIR = os.path.join(THIS_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VIDEO_REAL = os.path.join(DATA_DIR, "video", "real")

os.makedirs(UPLOAD_DIR, exist_ok=True)

results = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {OK(label)}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  {ERR(label)}" + (f"  [{detail}]" if detail else ""))
    results.append((label, condition))
    return condition


def section(title: str):
    print(f"\n{HDR('-' * 60)}")
    print(HDR(f"  {title}"))
    print(HDR("-" * 60))


def main():
    print("\n" + "=" * 60)
    print("  AUTHENTIX - COMPLETE PIPELINE VALIDATION")
    print("=" * 60)

    if len(sys.argv) > 1:
        video_path = os.path.abspath(sys.argv[1])
    else:
        real_videos = [name for name in os.listdir(VIDEO_REAL) if name.endswith(".mp4")]
        if not real_videos:
            print(ERR("No .mp4 files found in data/video/real/"))
            sys.exit(1)
        video_path = os.path.join(VIDEO_REAL, real_videos[0])

    print(f"\n  Test video : {video_path}")
    check("Test video exists", os.path.exists(video_path), video_path)

    section("A. INPUT MODULE")
    import shutil
    import uuid

    video_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    shutil.copy2(video_path, upload_path)

    check("File saved to data/uploads/", os.path.exists(upload_path))
    file_size = os.path.getsize(upload_path)
    check("File size > 0", file_size > 0, f"{file_size:,} bytes")
    print(f"  Video Path : {upload_path}")

    section("B. AUDIO EXTRACTION (FFmpeg)")
    audio_path = None
    try:
        from app.services.audio_processing import extract_audio

        t0 = time.time()
        audio_path = extract_audio(upload_path)
        elapsed = time.time() - t0
        audio_exists = os.path.exists(audio_path)
        audio_size = os.path.getsize(audio_path) if audio_exists else 0
        check(".wav file created", audio_exists, audio_path)
        check("Audio size > 0", audio_size > 0, f"{audio_size:,} bytes")
        check("Extraction < 10s", elapsed < 10, f"{elapsed:.2f}s")
        print(f"  Audio Path  : {audio_path}")
        print(f"  Exists      : {audio_exists}")
        print(f"  File Size   : {audio_size / 1024:.1f} KB")
    except RuntimeError as exc:
        print(f"  {WARN('FFmpeg failed - continuing with audio fallback')}")
        print(f"  Error: {exc}")
        audio_path = None

    section("C. VIDEO PROCESSING (OpenCV)")
    frames = []
    try:
        from app.services.video_processing import extract_frames

        t0 = time.time()
        frames = extract_frames(upload_path)
        elapsed = time.time() - t0
        check("Frames extracted >= 1", len(frames) >= 1, f"{len(frames)} frames")
        check("Frames extracted >= 3", len(frames) >= 3, f"{len(frames)} frames")
        if frames and hasattr(frames[0], "shape"):
            height, width, channels = frames[0].shape
            check("Frame resized 224x224", height == 224 and width == 224, f"{width}x{height}x{channels}")
        check("Extraction < 10s", elapsed < 10, f"{elapsed:.2f}s")
        print(f"  Frames extracted : {len(frames)}")
        if frames and hasattr(frames[0], "shape"):
            print(f"  Frame shape      : {frames[0].shape}")
    except Exception as exc:
        print(f"  {ERR(f'OpenCV failed: {exc}')}")

    if not frames:
        print(ERR("Cannot continue without frames."))
        sys.exit(1)

    section("D. FEATURE EXTRACTION")
    from app.services.feature_extraction import (  # noqa: E402
        AUDIO_FEATURE_NAMES,
        VIDEO_FEATURE_NAMES,
        extract_features,
    )

    features_dict = extract_features(frames, audio_path or "")
    vf = features_dict["video"]
    af = features_dict["audio"]

    check("Video features: variance", "variance" in vf, f"{vf.get('variance', 0.0):.4f}")
    check("Video features: flicker", "flicker" in vf, f"{vf.get('flicker', 0.0):.4f}")
    check("Video features: blur", "blur" in vf, f"{vf.get('blur', 0.0):.4f}")
    check("Audio features: mfcc_mean", "mfcc_mean" in af, f"{af.get('mfcc_mean', 0.0):.4f}")
    check("Audio features: pitch_var", "pitch_var" in af, f"{af.get('pitch_var', 0.0):.4f}")
    check("Audio features: energy", "energy" in af, f"{af.get('energy', 0.0):.6f}")
    check("Pretrained video score present", "pretrained_video_score" in features_dict)
    check("Pretrained audio score present", "pretrained_audio_score" in features_dict)

    video_feature_vector = [vf[key] for key in VIDEO_FEATURE_NAMES]
    audio_feature_vector = [af[key] for key in AUDIO_FEATURE_NAMES]
    check("Video feature vector len = 3", len(video_feature_vector) == 3, str(video_feature_vector))
    check("Audio feature vector len = 3", len(audio_feature_vector) == 3, str(audio_feature_vector))

    print(f"\n  Video features  : variance={vf['variance']:.4f}  flicker={vf['flicker']:.4f}  blur={vf['blur']:.4f}")
    print(f"  Audio features  : mfcc={af['mfcc_mean']:.4f}  pitch_var={af['pitch_var']:.4f}  energy={af['energy']:.6f}")

    section("E. MODEL LOADING + INFERENCE")
    for filename in ["audio_model.pkl", "video_model.pkl", "scaler_audio.pkl", "scaler_video.pkl"]:
        path = os.path.join(MODEL_DIR, filename)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        check(f"{filename} exists", os.path.exists(path), f"{size:,} bytes")

    from app.services.model_service import predict_model  # noqa: E402

    model_scores = predict_model(features_dict)
    pretrained_video_score = model_scores["pretrained_video_score"]
    custom_video_score = model_scores["custom_video_score"]
    video_model_score = model_scores["video_model_score"]
    pretrained_audio_score = model_scores["pretrained_audio_score"]
    custom_audio_score = model_scores["custom_audio_score"]
    audio_model_score = model_scores["audio_model_score"]
    model_score = model_scores["model_score"]

    check("Pretrained video score in [0,1]", 0 <= pretrained_video_score <= 1, f"{pretrained_video_score:.4f}")
    check("Custom video score in [0,1]", 0 <= custom_video_score <= 1, f"{custom_video_score:.4f}")
    check("Video model score in [0,1]", 0 <= video_model_score <= 1, f"{video_model_score:.4f}")
    check("Pretrained audio score in [0,1]", 0 <= pretrained_audio_score <= 1, f"{pretrained_audio_score:.4f}")
    check("Custom audio score in [0,1]", 0 <= custom_audio_score <= 1, f"{custom_audio_score:.4f}")
    check("Audio model score in [0,1]", 0 <= audio_model_score <= 1, f"{audio_model_score:.4f}")
    check(
        "video_model_score = 0.7 pretrained + 0.3 custom",
        abs(video_model_score - (PRETRAINED_WEIGHT * pretrained_video_score + CUSTOM_WEIGHT * custom_video_score)) < 1e-6,
        f"{video_model_score:.4f}",
    )
    check(
        "audio_model_score = 0.7 pretrained + 0.3 custom",
        abs(audio_model_score - (PRETRAINED_WEIGHT * pretrained_audio_score + CUSTOM_WEIGHT * custom_audio_score)) < 1e-6,
        f"{audio_model_score:.4f}",
    )
    check(
        "model_score = avg of audio/video model scores",
        abs(model_score - (audio_model_score + video_model_score) / 2) < 1e-6,
        f"{model_score:.4f}",
    )

    print(f"\n  Video model score  : {video_model_score:.4f}  ({PRETRAINED_WEIGHT:.1f} pretrained + {CUSTOM_WEIGHT:.1f} custom)")
    print(f"    Video breakdown  : pretrained={pretrained_video_score:.4f}  custom={custom_video_score:.4f}")
    print(f"  Audio model score  : {audio_model_score:.4f}  ({PRETRAINED_WEIGHT:.1f} pretrained + {CUSTOM_WEIGHT:.1f} custom)")
    print(f"    Audio breakdown  : pretrained={pretrained_audio_score:.4f}  custom={custom_audio_score:.4f}")
    print(f"  model_score        : {model_score:.4f}")

    section("F. HEURISTIC MODULE")
    from app.services.heuristic_service import heuristic_analysis  # noqa: E402

    heuristic_scores = heuristic_analysis(features_dict)
    video_heuristic_score = heuristic_scores["video_heuristic_score"]
    audio_heuristic_score = heuristic_scores["audio_heuristic_score"]
    heuristic_score = heuristic_scores["heuristic_score"]

    check("Video heuristic score in [0,1]", 0 <= video_heuristic_score <= 1, f"{video_heuristic_score:.4f}")
    check("Audio heuristic score in [0,1]", 0 <= audio_heuristic_score <= 1, f"{audio_heuristic_score:.4f}")
    check(
        "heuristic_score = avg of video/audio heuristics",
        abs(heuristic_score - (video_heuristic_score + audio_heuristic_score) / 2) < 1e-4,
        f"{heuristic_score:.4f}",
    )
    check("'issues' key present", "issues" in heuristic_scores)

    print(f"\n  Video heuristic    : {video_heuristic_score:.4f}  (blur + flicker + variance)")
    print(f"  Audio heuristic    : {audio_heuristic_score:.4f}  (pitch + energy patterns)")
    print(f"  heuristic_score    : {heuristic_score:.4f}")

    section("G. SCORE AGGREGATION")
    expected_model_score = (audio_model_score + video_model_score) / 2
    expected_heuristic_score = (video_heuristic_score + audio_heuristic_score) / 2

    print(f"  model_score      = ({audio_model_score:.4f} + {video_model_score:.4f}) / 2 = {model_score:.4f}")
    print(
        f"  heuristic_score  = ({video_heuristic_score:.4f} + {audio_heuristic_score:.4f}) / 2 = {heuristic_score:.4f}"
    )

    check("model_score formula correct", abs(model_score - expected_model_score) < 1e-6)
    check("heuristic_score formula correct", abs(heuristic_score - expected_heuristic_score) < 1e-4)

    section("H. FUSION LOGIC")
    from app.services.fusion_service import fuse_scores  # noqa: E402

    from app.utils.config import (  # noqa: E402
        FUSION_NONLINEAR_POWER, FUSION_MODEL_WEIGHT,
        FUSION_LIPSYNC_WEIGHT, FUSION_DEEPFACE_WEIGHT,
    )
    fusion_result = fuse_scores(model_scores, heuristic_scores)
    final_score   = fusion_result["final_score"]

    # Re-derive expected value using the same 3-step formula
    _model_s   = model_scores.get("model_score", 0.0)
    _boosted   = _model_s ** FUSION_NONLINEAR_POWER
    _lip_sync  = model_scores.get("lip_sync_score", 0.0)
    _deepface  = model_scores.get("deepface_score", 0.0)
    expected_final = round(
        FUSION_MODEL_WEIGHT    * _boosted  +
        FUSION_LIPSYNC_WEIGHT  * _lip_sync +
        FUSION_DEEPFACE_WEIGHT * _deepface,
        4,
    )

    print(f"  boosted_model = {_model_s:.4f} ^ {FUSION_NONLINEAR_POWER} = {_boosted:.4f}")
    print(f"  final_score   = {FUSION_MODEL_WEIGHT} * {_boosted:.4f}")
    print(f"                + {FUSION_LIPSYNC_WEIGHT} * {_lip_sync:.4f}  [lip_sync]")
    print(f"                + {FUSION_DEEPFACE_WEIGHT} * {_deepface:.4f}  [deepface]")
    print(f"                = {final_score:.4f}  (expected: {expected_final:.4f})")

    check(
        "Fusion formula: 0.65*boost + 0.25*lip_sync + 0.10*deepface",
        abs(final_score - expected_final) < 1e-4,
        f"{final_score:.4f}",
    )
    check("final_score in [0,1]", 0 <= final_score <= 1, f"{final_score:.4f}")

    section("I. DECISION ENGINE")
    status = fusion_result["status"]
    print(f"  Final Score : {final_score:.4f}")
    print(f"  Label       : {status}")

    expected_status = "FAKE" if final_score >= FAKE_THRESHOLD else "REAL"
    check("Decision threshold correct", status == expected_status, f"got={status}  expected={expected_status}")
    check("Status is one of FAKE/REAL", status in {"FAKE", "REAL"})

    section("J. FINAL API RESPONSE SCHEMA")
    from app.services.explanation_service import generate_explanation  # noqa: E402

    explanation_result = generate_explanation(features_dict, fusion_result)
    final_response = {
        "video_id": video_id,
        "status": status,
        "confidence": final_score,
        "scores": {
            "pretrained_video_score": pretrained_video_score,
            "custom_video_score": custom_video_score,
            "video_model_score": video_model_score,
            "pretrained_audio_score": pretrained_audio_score,
            "custom_audio_score": custom_audio_score,
            "audio_model_score": audio_model_score,
            "model_score": model_score,
            "video_heuristic_score": video_heuristic_score,
            "audio_heuristic_score": audio_heuristic_score,
            "heuristic_score": heuristic_score,
            "final_score": final_score,
        },
        "explanation": explanation_result["explanation"],
        "attribution": explanation_result["attribution"],
        "issues": heuristic_scores["issues"],
        "features": {
            "video": vf,
            "audio": af,
            "pretrained_video_score": pretrained_video_score,
            "pretrained_audio_score": pretrained_audio_score,
        },
    }

    required_keys = ["video_id", "status", "confidence", "scores", "explanation", "attribution", "issues", "features"]
    required_score_keys = [
        "pretrained_video_score",
        "custom_video_score",
        "video_model_score",
        "pretrained_audio_score",
        "custom_audio_score",
        "audio_model_score",
        "model_score",
        "video_heuristic_score",
        "audio_heuristic_score",
        "heuristic_score",
        "final_score",
    ]

    for key in required_keys:
        check(f"Response has key: '{key}'", key in final_response)
    for key in required_score_keys:
        check(f"scores.{key} present", key in final_response["scores"])

    check("explanation is a list", isinstance(final_response["explanation"], list))
    check("attribution is a string", isinstance(final_response["attribution"], str))

    import json

    print("\n  Final JSON Response:")
    print("  " + json.dumps(final_response, indent=4).replace("\n", "\n  "))

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    total = len(results)

    for label, result in results:
        icon = "PASS" if result else "FAIL"
        print(f"  {icon}  {label}")

    print(f"\n  Result: {passed}/{total} checks passed")
    if failed == 0:
        print("  ALL CHECKS PASSED - Pipeline is valid and hackathon-ready!")
    else:
        print(f"  {failed} check(s) FAILED - review the output above")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

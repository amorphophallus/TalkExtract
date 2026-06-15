#!/usr/bin/env python3
"""
Transcribe a Bilibili interview audio with speaker diarization.
Uses faster-whisper (large-v3 GPU) + pyannote.audio for speaker labels.

Prerequisites:
    export HF_TOKEN="your_huggingface_token"
    # Must accept user conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import os
import sys
from pathlib import Path
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment

# --- Auto-detect and add nvidia CUDA library paths ---
import site
_nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
if os.path.isdir(_nvidia_base):
    _lib_dirs = []
    for _root, _dirs, _files in os.walk(_nvidia_base):
        if os.path.basename(_root) == "lib":
            _lib_dirs.append(_root)
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + (":" + _existing_ld if _existing_ld else "")

# --- Clear potentially broken HF_TOKEN early (will restore for pyannote) ---
_broken_token = os.environ.pop("HF_TOKEN", None)  # Remove "你的token" placeholder

# --- Use HF mirror for model downloads in China ---
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# --- Config ---
AUDIO_PATH = Path("/home/huyue/projects/talk-extract/从旷视到原力灵机_唐文斌的第二次创业_晚点聊 LateTalk_.m4a")
OUTPUT_DIR = Path("/home/huyue/projects/talk-extract")
MODEL_SIZE = "large-v3"
LANGUAGE = "zh"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # or "int8_float16" for mixed precision

# --- Step 1: Transcription with faster-whisper ---
print("=" * 60)
print("Step 1: Loading faster-whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"Model loaded: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")

print(f"\nTranscribing: {AUDIO_PATH}")
print("This may take a few minutes for a 2-hour audio...")

segments, info = model.transcribe(
    str(AUDIO_PATH),
    language=LANGUAGE,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,
    ),
    beam_size=5,
)

print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
print(f"Duration: {info.duration:.0f}s ({info.duration/60:.1f} min)")

# Collect all segments and save intermediate results
all_segments = list(segments)
print(f"Transcription complete: {len(all_segments)} segments")

# Save intermediate whisper results (so we don't lose them)
import json
_intermediate = {
    "language": info.language,
    "duration": info.duration,
    "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in all_segments],
}
_intermediate_path = OUTPUT_DIR / "whisper_segments.json"
with open(_intermediate_path, "w", encoding="utf-8") as f:
    json.dump(_intermediate, f, ensure_ascii=False, indent=2)
print(f"Whisper results saved to: {_intermediate_path}")

# --- Step 2: Speaker Diarization ---
print("\n" + "=" * 60)
print("Step 2: Loading speaker diarization pipeline...")

# Get HF token for pyannote (extract directly from bashrc to bypass non-interactive guard)
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token or len(hf_token) < 30:
    try:
        with open(os.path.expanduser("~/.bashrc")) as f:
            for line in f:
                if "export HF_TOKEN=" in line and len(line.strip()) > 25:
                    hf_token = line.strip().split('"')[1] if '"' in line else line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass

if not hf_token or len(hf_token) < 30:
    print("ERROR: Valid HF_TOKEN not found. Please set a real HuggingFace token in ~/.bashrc.")
    print("  export HF_TOKEN=\"hf_...\"")
    print("And accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1")
    sys.exit(1)

# Unset HF_TOKEN globally to avoid sending invalid token to public model downloads
hf_token_saved = hf_token
os.environ.pop("HF_TOKEN", None)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token_saved,
)
pipeline.to(torch.device(DEVICE))
print("Diarization model loaded.")

print(f"\nRunning speaker diarization on: {AUDIO_PATH}")
print("This may take 10-30 minutes for a 2-hour audio...")

diarization = pipeline(str(AUDIO_PATH))
print(f"Diarization complete: {len(diarization)} speaker turns")

# --- Step 3: Merge ---
print("\n" + "=" * 60)
print("Step 3: Merging transcription with speaker labels...")

def get_speaker(turn_start, turn_end):
    """Find the dominant speaker in the given time window."""
    best_speaker = "SPK_UNK"
    best_overlap = 0
    for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(turn_start, speech_turn.start)
        overlap_end = min(turn_end, speech_turn.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker

def srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Map speaker labels to readable names
speaker_map = {}
speaker_counter = [0]

def label_speaker(raw_label):
    if raw_label not in speaker_map:
        speaker_map[raw_label] = f"发言人{speaker_counter[0] + 1}"
        speaker_counter[0] += 1
    return speaker_map[raw_label]

# Merge and write outputs
safe_title = "从旷视到原力灵机_唐文斌的第二次创业_晚点聊LateTalk"
srt_path = OUTPUT_DIR / f"{safe_title}.srt"
txt_path = OUTPUT_DIR / f"{safe_title}.txt"
raw_path = OUTPUT_DIR / f"{safe_title}_raw.txt"

with open(srt_path, "w", encoding="utf-8") as srt_f, \
     open(txt_path, "w", encoding="utf-8") as txt_f, \
     open(raw_path, "w", encoding="utf-8") as raw_f:

    txt_f.write("从旷视到原力灵机，唐文斌的第二次创业\n")
    txt_f.write("【晚点聊 LateTalk】\n")
    txt_f.write("=" * 60 + "\n\n")

    for i, seg in enumerate(all_segments, start=1):
        speaker_raw = get_speaker(seg.start, seg.end)
        speaker_name = label_speaker(speaker_raw)

        text = seg.text.strip()
        if not text:
            continue

        # SRT output
        srt_f.write(
            f"{i}\n"
            f"{srt_time(seg.start)} --> {srt_time(seg.end)}\n"
            f"[{speaker_name}] {text}\n\n"
        )

        # Formatted text output
        txt_f.write(f"[{speaker_name}] {text}\n\n")

        # Raw text
        raw_f.write(f"{text}\n")

print(f"\nSpeaker mapping:")
for raw, name in speaker_map.items():
    turns = sum(1 for t, _, s in diarization.itertracks(yield_label=True) if s == raw)
    total = sum(t.end - t.start for t, _, s in diarization.itertracks(yield_label=True) if s == raw)
    print(f"  {name} ({raw}): {turns} turns, {total/60:.1f} min total")

# --- Step 4: Summary ---
print("\n" + "=" * 60)
print("Output files:")
print(f"  SRT subtitles: {srt_path}")
print(f"  Formatted text: {txt_path}")
print(f"  Raw text: {raw_path}")
print(f"\nTotal segments: {len(all_segments)}")
print("Done!")

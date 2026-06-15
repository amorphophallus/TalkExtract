#!/usr/bin/env python3
"""
Diarization-only pass: takes saved whisper segments + audio, runs pyannote, merges results.
Run after HF_TOKEN is properly set in ~/.bashrc.
"""
import os, sys, json
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

# --- Auto-detect nvidia CUDA library paths ---
import site
_nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
if os.path.isdir(_nvidia_base):
    _lib_dirs = []
    for _root, _dirs, _files in os.walk(_nvidia_base):
        if os.path.basename(_root) == "lib":
            _lib_dirs.append(_root)
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + (":" + _existing_ld if _existing_ld else "")

# Use HF_ENDPOINT only if set; default to direct for gated repos
if not os.environ.get("HF_ENDPOINT"):
    pass  # Use direct connection for gated model access

OUTPUT_DIR = Path("/home/huyue/projects/talk-extract")
AUDIO_PATH = OUTPUT_DIR / "从旷视到原力灵机_唐文斌的第二次创业_晚点聊 LateTalk_.m4a"
SEGMENTS_PATH = OUTPUT_DIR / "whisper_segments.json"
DEVICE = "cuda"

# --- Step 1: Load saved whisper segments ---
print("Loading saved whisper transcription...")
with open(SEGMENTS_PATH, encoding="utf-8") as f:
    data = json.load(f)
all_segments = data["segments"]
print(f"Loaded {len(all_segments)} segments from whisper")

# --- Step 2: Extract real HF token from bashrc ---
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
    print(f"ERROR: Valid HF_TOKEN not found. Got: {hf_token[:20] if hf_token else '(empty)'}")
    sys.exit(1)

print(f"HF token found (length={len(hf_token)})")
os.environ.pop("HF_TOKEN", None)  # Don't leak to public model downloads

# --- Step 3: Load pyannote pipeline ---
print("Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,
)
pipeline.to(torch.device(DEVICE))
print("Diarization model loaded.")

# --- Step 4: Run diarization ---
print(f"Running diarization on: {AUDIO_PATH}")
print("This may take 15-30 minutes for 2-hour audio...")
result = pipeline(str(AUDIO_PATH))
# pyannote 4.x returns DiarizeOutput; use exclusive_speaker_diarization
diarization = result.exclusive_speaker_diarization
num_turns = sum(1 for _ in diarization.itertracks(yield_label=True))
print(f"Diarization complete: {num_turns} speaker turns")

# --- Step 5: Merge with whisper segments ---
print("Merging speaker labels with whisper segments...")

def get_speaker(turn_start, turn_end):
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

def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

speaker_map = {}
counter = [0]
def label_speaker(raw):
    if raw not in speaker_map:
        speaker_map[raw] = f"Speaker_{counter[0] + 1}"
        counter[0] += 1
    return speaker_map[raw]

safe_title = "从旷视到原力灵机_唐文斌的第二次创业_晚点聊LateTalk"
srt_path = OUTPUT_DIR / f"{safe_title}.srt"
txt_path = OUTPUT_DIR / f"{safe_title}_labeled.txt"

with open(srt_path, "w", encoding="utf-8") as srt_f, \
     open(txt_path, "w", encoding="utf-8") as txt_f:

    txt_f.write("从旷视到原力灵机，唐文斌的第二次创业\n")
    txt_f.write("【晚点聊 LateTalk】\n")
    txt_f.write("=" * 60 + "\n")
    txt_f.write(f"带说话人标签 | {len(all_segments)} 段落\n")
    txt_f.write("=" * 60 + "\n\n")

    idx = 0
    for seg in all_segments:
        text = seg["text"].strip()
        if not text:
            continue
        idx += 1
        speaker_raw = get_speaker(seg["start"], seg["end"])
        speaker_name = label_speaker(speaker_raw)

        srt_f.write(
            f"{idx}\n"
            f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n"
            f"[{speaker_name}] {text}\n\n"
        )
        txt_f.write(f"[{speaker_name}] {text}\n\n")

# --- Step 6: Report ---
print("\nSpeaker statistics:")
for raw, name in speaker_map.items():
    turns = sum(1 for t, _, s in diarization.itertracks(yield_label=True) if s == raw)
    total = sum(t.end - t.start for t, _, s in diarization.itertracks(yield_label=True) if s == raw)
    print(f"  {name} ({raw}): {turns} turns, {total/60:.1f} min total")

print(f"\nOutput files:")
print(f"  SRT: {srt_path}")
print(f"  Labeled text: {txt_path}")
print("Done!")

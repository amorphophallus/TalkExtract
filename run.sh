#!/bin/bash
# Wrapper: extract HF_TOKEN from bashrc (works in non-interactive mode)
HF_TOKEN=$(grep -m1 'export HF_TOKEN=' ~/.bashrc | sed 's/.*"\(.*\)".*/\1/')
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
exec /home/huyue/miniconda3/envs/talk-extract/bin/python3 /home/huyue/projects/talk-extract/transcribe.py

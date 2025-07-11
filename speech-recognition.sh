#!/usr/bin/env bash
sleep 30
cd /home/pi/voice-ad/
source .venv/bin/activate
python silero-vad-test.py

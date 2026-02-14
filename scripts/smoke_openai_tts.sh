#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:9898}"
MODELS_URL="${BASE_URL}/v1/models"
SPEECH_URL="${BASE_URL}/v1/audio/speech"

OUT_WAV="${OUT_WAV:-out.wav}"
OUT_MP3="${OUT_MP3:-out.mp3}"

echo "Smoke: GET ${MODELS_URL}"
curl -fsS "${MODELS_URL}" | tee /tmp/openai_tts_models.json
echo

echo "Smoke: POST ${SPEECH_URL} -> ${OUT_WAV}"
curl -fsS -X POST "${SPEECH_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "xenia",
    "input": "This is a smoke test for OpenAI compatible TTS.",
    "response_format": "wav"
  }' \
  --output "${OUT_WAV}"

echo "Smoke: POST ${SPEECH_URL} -> ${OUT_MP3}"
curl -fsS -X POST "${SPEECH_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "xenia",
    "input": "This is an mp3 smoke test for OpenAI compatible TTS.",
    "response_format": "mp3",
    "bitrate": 128
  }' \
  --output "${OUT_MP3}"

if command -v file >/dev/null 2>&1; then
  echo "Inspect with file:"
  file "${OUT_WAV}" "${OUT_MP3}"
fi

if command -v ffprobe >/dev/null 2>&1; then
  echo "Inspect with ffprobe:"
  ffprobe -hide_banner -loglevel error -show_format -show_streams "${OUT_WAV}" || true
  ffprobe -hide_banner -loglevel error -show_format -show_streams "${OUT_MP3}" || true
fi

echo "Smoke test finished: ${OUT_WAV}, ${OUT_MP3}"

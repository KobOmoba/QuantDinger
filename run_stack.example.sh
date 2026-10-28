#!/usr/bin/env bash
# Copy to run_stack.sh (git-ignored), paste your Groq key below, then run it.
# Requires docker-compose.override.yml (see docker-compose.override.example.yml).
set -euo pipefail
cd "$(dirname "$0")"

export CUSTOM_API_KEY='YOUR_KEY_HERE'

if [ "$CUSTOM_API_KEY" = 'YOUR_KEY_HERE' ] || [ -z "$CUSTOM_API_KEY" ]; then
  echo "Edit run_stack.sh and replace YOUR_KEY_HERE with your Groq API key." >&2
  exit 1
fi

docker compose up -d --build

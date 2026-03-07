#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_HOST="${PI_HOST:-sam@10.10.10.101}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/sam/shutter_deck}"
SPACING_SEC="${SPACING_SEC:-0.5}"
ARRIVAL_BASENAMES=()

if [ "$#" -gt 0 ]; then
  IMAGES=("$@")
else
  IMAGES=(
    "$ROOT_DIR/data/imagenet-r/n02113799/sketch_0.jpg"
    "$ROOT_DIR/data/imagenet-r/n02113799/misc_1.jpg"
    "$ROOT_DIR/data/imagenet-r/n02113799/sketch_5.jpg"
    "$ROOT_DIR/data/imagenet-r/n02113799/sketch_13.jpg"
    "$ROOT_DIR/data/imagenet-r/n02113799/misc_12.jpg"
  )
fi

for f in "${IMAGES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing image: $f" >&2
    exit 1
  fi
  ARRIVAL_BASENAMES+=("$(basename "$f")")
done

echo "== Sync code/config to Pi =="
rsync -av \
  "$ROOT_DIR/config.toml" \
  "$ROOT_DIR/query.py" \
  "$ROOT_DIR/shutter_deck/encoder.py" \
  "$ROOT_DIR/shutter_deck/preprocess.py" \
  "$ROOT_DIR/shutter_deck/service.py" \
  "$ROOT_DIR/shutter_deck/persistence.py" \
  "$ROOT_DIR/shutter_deck/ingest.py" \
  "$PI_HOST:$REMOTE_ROOT/" >/dev/null
rsync -av \
  "$ROOT_DIR/shutter_deck/encoder.py" \
  "$ROOT_DIR/shutter_deck/preprocess.py" \
  "$ROOT_DIR/shutter_deck/service.py" \
  "$ROOT_DIR/shutter_deck/persistence.py" \
  "$ROOT_DIR/shutter_deck/ingest.py" \
  "$PI_HOST:$REMOTE_ROOT/shutter_deck/" >/dev/null

echo "== Reset Pi state =="
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  mkdir -p \"$REMOTE_ROOT/logs\" /mnt/nvme/incoming /var/lib/shutter-deck
  chown -R \$(id -un):\$(id -gn) /mnt/nvme /var/lib/shutter-deck || true
  if [ -f \"$REMOTE_ROOT/logs/burst.pid\" ]; then kill -TERM \$(cat \"$REMOTE_ROOT/logs/burst.pid\") 2>/dev/null || true; fi
  pgrep -af \"^\\./.venv/bin/python -u -m shutter_deck.service config.toml\$\" | while read -r pid _; do kill -TERM \"\$pid\" 2>/dev/null || true; done || true
  sleep 2
  pgrep -af \"^\\./.venv/bin/python -u -m shutter_deck.service config.toml\$\" | while read -r pid _; do kill -9 \"\$pid\" 2>/dev/null || true; done || true
  rm -f \"$REMOTE_ROOT\"/logs/burst.log \"$REMOTE_ROOT\"/logs/rss.log \"$REMOTE_ROOT\"/logs/burst.pid
  rm -f /var/lib/shutter-deck/cam_state.bin /var/lib/shutter-deck/cam_state.tmp /var/lib/shutter-deck/metadata.db
  rm -f /mnt/nvme/incoming/* || true
'"

echo "== Start service + RSS sampler =="
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  cd \"$REMOTE_ROOT\"
  nohup ./.venv/bin/python -u -m shutter_deck.service config.toml > logs/burst.log 2>&1 < /dev/null &
  svc=\$!
  echo \$svc > logs/burst.pid
  (
    while kill -0 \$svc 2>/dev/null; do
      ts=\$(date +%s.%N)
      rss=\$(ps -o rss= -p \$svc | tr -d \" \")
      echo \"\$ts \${rss:-0}\" >> logs/rss.log
      sleep 0.2
    done
  ) >/dev/null 2>&1 &
'"

echo "== Wait for watcher =="
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  cd \"$REMOTE_ROOT\"
  for _ in \$(seq 1 120); do
    grep -q \"Watching /mnt/nvme/incoming\" logs/burst.log && break
    sleep 1
  done
  tail -n 20 logs/burst.log
'"

echo "== Send burst (${#IMAGES[@]} files, spacing=${SPACING_SEC}s) =="
for f in "${IMAGES[@]}"; do
  scp "$f" "$PI_HOST:/mnt/nvme/incoming/$(basename "$f")" >/dev/null
  date '+sent %H:%M:%S.%3N'
  sleep "$SPACING_SEC"
done

echo "== Wait for queue drain (metadata rows == ${#IMAGES[@]}) =="
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  for _ in \$(seq 1 180); do
    rows=\$(python3 - <<PY
import sqlite3, os
p=\"/var/lib/shutter-deck/metadata.db\"
if not os.path.exists(p):
    print(0)
else:
    conn=sqlite3.connect(p)
    print(conn.execute(\"select count(*) from ingested_images\").fetchone()[0])
PY
)
    if [ \"\$rows\" -ge ${#IMAGES[@]} ]; then break; fi
    sleep 1
  done
  cd \"$REMOTE_ROOT\"
  tail -n 200 logs/burst.log
'"

echo "== Stop service (persist sparse state) =="
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  cd \"$REMOTE_ROOT\"
  kill -TERM \$(cat logs/burst.pid)
  for _ in \$(seq 1 60); do
    kill -0 \$(cat logs/burst.pid) 2>/dev/null || break
    sleep 1
  done
  tail -n 50 logs/burst.log
  ls -lh /var/lib/shutter-deck/cam_state.bin /var/lib/shutter-deck/metadata.db
'"

echo "== Collect metrics =="
FIRST_IMAGE="${ARRIVAL_BASENAMES[0]}"
QUERY_IMAGE="${ARRIVAL_BASENAMES[1]:-$FIRST_IMAGE}"
ssh "$PI_HOST" "bash -lc '
  set -euo pipefail
  cd \"$REMOTE_ROOT\"
  python3 - <<PY
from pathlib import Path
import sqlite3
from shutter_deck.service import load_config, build_cam
from shutter_deck.persistence import load_cam_state

rss_vals = []
for line in Path(\"logs/rss.log\").read_text().splitlines():
    p = line.split()
    if len(p) == 2 and p[1].isdigit():
        rss_vals.append(int(p[1]))
print(\"rss_max_kb\", max(rss_vals) if rss_vals else 0)
print(\"rss_max_mb\", round((max(rss_vals) if rss_vals else 0)/1024, 1))

cfg = load_config(Path(\"config.toml\"))
cam = build_cam(cfg)
ok = load_cam_state(cam, Path(cfg[\"paths\"][\"state_file\"]))
print(\"load_ok\", ok)
print(\"prototype_count\", int(cam.occupied.sum().item()))
conn = sqlite3.connect(cfg[\"paths\"][\"metadata_db\"])
rows = conn.execute(\"select count(*) from ingested_images\").fetchone()[0]
print(\"metadata_rows\", rows)
print(\"state_size_bytes\", Path(cfg[\"paths\"][\"state_file\"]).stat().st_size)
print(\"batch_timing_lines\")
for line in Path(\"logs/burst.log\").read_text().splitlines():
    if \"Batch timings:\" in line:
        print(line)
PY
  /usr/bin/time -v ./.venv/bin/python query.py \"/mnt/nvme/incoming/$QUERY_IMAGE\" --config config.toml --top-k 5
'"

echo "== Done =="

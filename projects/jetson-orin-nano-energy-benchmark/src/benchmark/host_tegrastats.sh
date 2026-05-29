#!/usr/bin/env bash
set -euo pipefail

# Host-seitig tegrastats starten/stoppen und Log ins Projekt verschieben.
# Usage:
#   ./host_tegrastats.sh start OUTPREFIX          # startet Logging -> /tmp/OUTPREFIX_tegrastats.log
#   ./host_tegrastats.sh stop                     # stoppt, verschiebt nach results/, setzt Eigentümer
#   ./host_tegrastats.sh status                   # zeigt laufenden Logger (falls aktiv)

RESULTS_DIR="${RESULTS_DIR:-/home/adem/jetson-containers/workspace/ee-bench/results}"
STATE="/tmp/ee_bench_tegrastats.state"
INTERVAL="${INTERVAL:-1000}"   # ms

require_sudo() {
  sudo -v || { echo "sudo benötigt"; exit 1; }
}

convert_log() {
  local LOG_PATH="$1"
  [[ -f "$LOG_PATH" ]] || { echo "WARN: Logdatei nicht gefunden für Konvertierung: $LOG_PATH"; return; }
  local CSV_PATH="${LOG_PATH%.log}_power.csv"
  LOG="$LOG_PATH" CSV="$CSV_PATH" /usr/bin/python3 - <<'PY'
import csv
import datetime as dt
import os
import re
import sys

log_path = os.environ["LOG"]
csv_path = os.environ["CSV"]

rows = []
rx_ts = re.compile(r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}).*?(?:POM_5V_IN|VDD_IN)\s+(\d+)mW')
rx_no_ts = re.compile(r'(?:POM_5V_IN|VDD_IN)\s+(\d+)\s*/')

with open(log_path, 'r', errors='ignore') as handle:
    for idx, line in enumerate(handle):
        m = rx_ts.search(line)
        if m:
            timestamp = dt.datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
            power_w = float(m.group(2)) / 1000.0
            rows.append((idx, timestamp, power_w))
            continue
        m = rx_no_ts.search(line)
        if m:
            rows.append((idx, None, float(m.group(1)) / 1000.0))

if not rows:
    print(f"Keine Power-Samples gefunden in {log_path}", file=sys.stderr)
    sys.exit(0)

first_ts = next((ts for _, ts, _ in rows if ts is not None), None)

with open(csv_path, 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(["sample_index", "elapsed_s", "power_W"])
    for i, (_, ts, power_w) in enumerate(rows):
        if first_ts and ts:
            elapsed = (ts - first_ts).total_seconds()
        elif first_ts and ts is None:
            # Kein Timestamp für diese Zeile -> approximieren über Index
            elapsed = i
        else:
            elapsed = i
        writer.writerow([i, elapsed, power_w])

print(f"parsed {len(rows)} samples -> {csv_path}")
PY
  sudo chown adem:adem "$CSV_PATH" >/dev/null 2>&1 || true
}

cmd_start() {
  local OUT="${1:-}"
  [[ -n "$OUT" ]] || { echo "Bitte OUTPREFIX angeben"; exit 2; }
  require_sudo
  local TMP="/tmp/${OUT}_tegrastats.log"
  # Falls noch läuft: erst stoppen
  if [[ -f "$STATE" ]]; then
    echo "Hinweis: bestehendes Logging gefunden – stoppe zuerst."; cmd_stop || true
  fi
  # Start
  sudo -n stdbuf -oL -eL tegrastats --interval "$INTERVAL" | tee "$TMP" >/dev/null &
  local PID=$!
  echo "PID=$PID" >"$STATE"
  echo "OUT=$OUT" >>"$STATE"
  echo "TMP=$TMP" >>"$STATE"
  echo "RESULTS_DIR=$RESULTS_DIR" >>"$STATE"
  sleep 2
  echo "gestartet: PID $PID, tmp: $TMP"
  tail -n1 "$TMP" || true
}

cmd_stop() {
  [[ -f "$STATE" ]] || { echo "kein aktives Logging."; return 0; }
  # shellcheck disable=SC1090
  source "$STATE"
  require_sudo
  if ps -p "$PID" >/dev/null 2>&1; then
    sudo kill "$PID" || true
    sleep 1
  fi
  local DEST="${RESULTS_DIR}/${OUT}_tegrastats.log"
  mkdir -p "$RESULTS_DIR"
  # verschieben & Rechte korrigieren
  if [[ -s "$TMP" ]]; then
    sudo mv "$TMP" "$DEST"
    sudo chown adem:adem "$DEST"
    convert_log "$DEST"
    echo "verschoben: $DEST"
  else
    echo "WARN: tmp-Log leer oder fehlt: $TMP"
  fi
  rm -f "$STATE"
}

cmd_status() {
  if [[ -f "$STATE" ]]; then
    echo "Status:"
    cat "$STATE"
    # shellcheck disable=SC1090
    source "$STATE"
    if ps -p "$PID" >/dev/null 2>&1; then
      echo "läuft (PID $PID)"; tail -n1 "$TMP" || true
    else
      echo "nicht aktiv (PID $PID nicht gefunden)"
    fi
  else
    echo "kein aktives Logging."
  fi
}

case "${1:-}" in
  start) shift; cmd_start "$@";;
  stop)  cmd_stop;;
  status) cmd_status;;
  *) echo "Usage: $0 {start OUTPREFIX|stop|status}"; exit 1;;
esac

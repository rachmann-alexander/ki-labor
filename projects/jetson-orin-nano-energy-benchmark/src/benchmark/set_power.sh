#!/usr/bin/env bash
# Steuerung von CPU, GPU und EMC Taktraten auf Jetson Orin
# Nutzung: ./src/set_power.sh {slow|medium|fast}
set -euo pipefail

GPU_MIN="/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq"
GPU_MAX="/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"
EMC_CAP="/sys/kernel/nvpmodel_clk_cap/emc"
CPUS=(0 1 2 3 4 5)

gpu_read_max() {
  local avail="/sys/devices/platform/17000000.gpu/devfreq_dev/available_frequencies"
  if [[ -r "$avail" ]]; then
    awk '{print $NF}' "$avail"
  else
    echo "1300500000"
  fi
}

emc_read_max() {
  local avail="/sys/kernel/nvpmodel_clk_cap/available_frequencies"
  if [[ -r "$avail" ]]; then
    awk '{print $NF}' "$avail"
  else
    echo "3200000000"
  fi
}

cpu_read_max() {
  local freq
  if [[ -r "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies" ]]; then
    freq=$(awk '{print $NF}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
  elif [[ -r "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq" ]]; then
    freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
  else
    freq="2208000"
  fi
  echo "$freq"
}

gpu_set() { echo "$1" | sudo tee "$GPU_MIN" >/dev/null; echo "$2" | sudo tee "$GPU_MAX" >/dev/null; }
emc_set() { echo "$1" | sudo tee "$EMC_CAP" >/dev/null; }
cpu_set() {
  local freq="$1"
  for c in "${CPUS[@]}"; do
    echo "$freq" | sudo tee "/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_max_freq" >/dev/null || true
  done
}

case "${1:-}" in
  slow)
    gpu_set 306000000 408000000
    emc_set 800000000
    cpu_set 729600
    sudo systemctl stop nvpmodel.service >/dev/null 2>&1 || true
    echo "🟡 Set: slow (CPU 729MHz, GPU 408MHz, EMC 800MHz)"
    ;;
  medium)
    gpu_set 306000000 816000000
    emc_set 1600000000
    cpu_set 1200000
    sudo systemctl stop nvpmodel.service >/dev/null 2>&1 || true
    echo "🟢 Set: medium (CPU 1.2GHz, GPU 816MHz, EMC 1600MHz)"
    ;;
  fast)
    if command -v jetson_clocks >/dev/null 2>&1; then
      sudo jetson_clocks --fan
      echo "🔴 Set: fast (jetson_clocks: max clocks)"
    else
      echo "⚠️  jetson_clocks nicht gefunden – setze maximale Frequenzen manuell." >&2
      gpu_max_val=$(gpu_read_max)
      cpu_max_val=$(cpu_read_max)
      emc_max_val=$(emc_read_max)
      gpu_set "$gpu_max_val" "$gpu_max_val"
      emc_set "$emc_max_val"
      cpu_set "$cpu_max_val"
      echo "🔴 Set: fast (manuell GPU ${gpu_max_val}, CPU ${cpu_max_val}, EMC ${emc_max_val})"
    fi
    ;;
  *)
    echo "Usage: $0 {slow|medium|fast}"
    exit 1
    ;;
esac

# Status ausgeben
echo -n "CPU max: "; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo -n "GPU min/max: "; cat "$GPU_MIN" "$GPU_MAX"
echo -n "EMC cap: "; cat "$EMC_CAP"

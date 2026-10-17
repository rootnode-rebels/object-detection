#!/bin/bash
# ==============================================================================
# Raspberry Pi 5 (2GB) System & Assistant Memory Diagnostic Utility
# Run this script anytime to inspect live RAM, swap, temperature, and processes.
# ==============================================================================

echo "================================================================="
echo "  Raspberry Pi 5 (2GB RAM) System & Health Telemetry"
echo "================================================================="
echo ""

echo "--- 1. RAM & Swap Utilization ---"
free -h
echo ""

echo "--- 2. Active Swap Partitions / Files ---"
swapon --show
echo ""

echo "--- 3. Kernel Swappiness Setting ---"
cat /proc/sys/vm/swappiness 2>/dev/null | awk '{print "Current vm.swappiness = " $1 " (15 is optimal for 2GB Pi 5)"}'
echo ""

echo "--- 4. SoC Temperature & CPU Throttling ---"
if command -v vcgencmd &>/dev/null; then
    vcgencmd measure_temp
    vcgencmd get_throttled
else
    TEMP_MC=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0")
    echo "SoC Temperature: $(awk "BEGIN {print $TEMP_MC/1000}")°C"
fi
echo ""

echo "--- 5. Smart Assistant Process Status ---"
PIDS=$(pgrep -f "smart_assistant.py" || true)
if [ -n "$PIDS" ]; then
    echo "Assistant process(es) found:"
    ps -p $PIDS -o pid,user,%cpu,%mem,rss,comm,args
    for p in $PIDS; do
        RSS_KB=$(ps -p "$p" -o rss= 2>/dev/null | tr -d ' ' || echo "0")
        RSS_MB=$(( ${RSS_KB:-0} / 1024 ))
        echo "PID $p Resident Memory (RSS): ${RSS_MB} MB"
    done
else
    echo "Assistant is NOT currently running."
fi
echo ""

echo "--- 6. Systemd Background Service Status ---"
systemctl is-active smart_assistant.service 2>/dev/null | awk '{print "smart_assistant.service state: " $1}'
echo "================================================================="

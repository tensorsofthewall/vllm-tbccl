#!/bin/bash
# One-line AER / link / GPU summary (read-only): correctable Timeout and total, nonfatal, fatal, thunderbolt0 state / MTU / errors+drops, GPU temperature.
t=$(cat /sys/bus/pci/devices/*/aer_dev_correctable 2>/dev/null | awk '/^Timeout/{s+=$2} END{print s+0}')
c=$(cat /sys/bus/pci/devices/*/aer_dev_correctable 2>/dev/null | awk '/^TOTAL_ERR_COR/{s+=$2} END{print s+0}')
n=$(cat /sys/bus/pci/devices/*/aer_dev_nonfatal 2>/dev/null | awk '/^TOTAL_ERR_NONFATAL/{s+=$2} END{print s+0}')
f=$(cat /sys/bus/pci/devices/*/aer_dev_fatal 2>/dev/null | awk '/^TOTAL_ERR_FATAL/{s+=$2} END{print s+0}')
st=$(cat /sys/class/net/thunderbolt0/operstate); mtu=$(cat /sys/class/net/thunderbolt0/mtu)
e=$(ip -s link show thunderbolt0 | awk 'NR==4{rx=$3" "$4} NR==6{tx=$3" "$4} END{print "rx_err/drop="rx" tx_err/drop="tx}')
echo "$(date +%T) AER Timeout=$t cor_total=$c nonfatal=$n fatal=$f if=thunderbolt0 $st mtu=$mtu $e gpu=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C"

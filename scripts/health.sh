#!/bin/bash
# One-line TB4 / GPU health snapshot (Linux): AER counters of the TB4 root port chain, link state, GPU temperature/throttle, RTT to the Mac.
echo "boot=$(cat /proc/sys/kernel/random/boot_id | cut -c1-8) link=$(cat /sys/class/net/thunderbolt0/carrier) mtu=$(cat /sys/class/net/thunderbolt0/mtu)"
for d in /sys/bus/pci/devices/*; do
  [ -f $d/aer_dev_correctable ] || continue
  c=$(awk '/^TOTAL/{print $2}' $d/aer_dev_correctable); n=$(awk '/^TOTAL/{print $2}' $d/aer_dev_nonfatal); f=$(awk '/^TOTAL/{print $2}' $d/aer_dev_fatal)
  t=$(awk '/Timeout/{print $2}' $d/aer_dev_correctable)
  [ "$c$n$f" != "000" ] && echo "AER $(basename $d): correctable=$c (Timeout=$t) nonfatal=$n fatal=$f"
done
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,pstate,power.draw,clocks_throttle_reasons.active --format=csv,noheader | sed 's/^/gpu: /'
echo "rtt: $(ping -c3 -q 192.168.3.1 | tail -1)"
echo "kernel: $(journalctl -k --since '-30min' --no-pager 2>/dev/null | grep -i -E 'aer|thunderbolt|nvrm|xid' | tail -2 | tr '\n' ';')"

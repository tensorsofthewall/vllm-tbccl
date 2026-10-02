#!/bin/bash
# Tear down the deployment started by p48_up.sh / p48_single.sh <tag>: SIGINT the Linux process group, drop the attached ssh child (the
# Mac processes then get SIGHUP), and clear stragglers of THIS deployment on the Mac by its master port. Generic VLLM:: process cleanup on
# the Mac only runs when no other `vllm serve` is left there, so sibling deployments are never touched.
TAG=${1:?tag}
cd /mnt/BigChonk/projects/vllm-tbccl
if [ -f results/${TAG}_linux.pid ]; then
  P=$(cat results/${TAG}_linux.pid); kill -INT -- -"$P" 2>/dev/null || kill -INT "$P" 2>/dev/null
fi
[ -f results/${TAG}_mac.sshpid ] && kill "$(cat results/${TAG}_mac.sshpid)" 2>/dev/null
sleep 3
PORT=$(cat results/${TAG}.port 2>/dev/null)
if [ -n "$PORT" ]; then
  ssh tbccl-mac "pkill -INT -f '[v]llm serve .*--master-port $PORT'; pkill -INT -f '[v]llm serve .*--port $PORT'; sleep 3; pgrep -f '[v]llm serve' >/dev/null || pkill -f '[V]LLM::'" >/dev/null 2>&1
fi
[ -f results/${TAG}_linux.pid ] && { P=$(cat results/${TAG}_linux.pid); kill -9 -- -"$P" 2>/dev/null; }
L=$(pgrep -fl "[V]LLM::" | head -3); [ -n "$L" ] && echo "VLLM:: processes still on Linux (may belong to a sibling deployment): $L"
rm -f results/${TAG}_linux.pid results/${TAG}_mac.sshpid results/${TAG}.port
exit 0

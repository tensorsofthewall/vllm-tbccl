#!/bin/bash
# the CUDA+Metal vLLM 0.30 work physical byte-integrity: CUDA (Linux) <-> Metal (Mac) over Thunderbolt, both rank orders.  p70_phys_integrity.sh <A|B> [iters]
#   A: CUDA is PP rank 0 (the Linux host is the init host)   B: Metal is PP rank 0 (the Mac is the init host)
O=$1; IT=${2:-20}; cd "$(dirname "$0")/.."; PORT=$((29900 + RANDOM % 90)); mkdir -p results docs/data/phase70
if [ $O = A ]; then CR=0; MR=1; INIT=192.168.3.2; else CR=1; MR=0; INIT=192.168.3.1; fi
bash scripts/p69_aer.sh | sed "s/^/before $O: /"
ssh -o BatchMode=yes tbccl-mac "export PATH=/opt/homebrew/bin:\$PATH; cd ~/projects/vllm-tbccl && VLLM_TBCCL_ENABLE=1 VLLM_TBCCL_BACKEND=metal VLLM_TBCCL_ARCHITECTURE=Qwen3ForCausalLM VLLM_USE_V2_MODEL_RUNNER=0 VLLM_METAL_BUILD_FROM_SOURCE=1 VLLM_HOST_IP=192.168.3.1 TBCCL_LOCAL_ENDPOINT=192.168.3.1:0 perl -e 'alarm 600; exec @ARGV' .venv-p70-metal/bin/python examples/p70_metal_integrity_probe.py --role metal --rank $MR --init tcp://$INIT:$PORT --iters $IT --out results/p70_phys_$O" > results/p70_phys_${O}_mac.log 2>&1 &
env VLLM_TBCCL_ENABLE=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_HOST_IP=192.168.3.2 TBCCL_LOCAL_ENDPOINT=192.168.3.2:0 timeout 600 .venv-p70-cuda/bin/python examples/p70_metal_integrity_probe.py --role cuda --rank $CR --init tcp://$INIT:$PORT --iters $IT --out results/p70_phys_$O > results/p70_phys_${O}_linux.log 2>&1
echo "linux rc=$?"; wait
grep -h "RESULT\| ok\|MISMATCH" results/p70_phys_${O}_linux.log results/p70_phys_${O}_mac.log | grep -v "^objc" | cut -c1-330
bash scripts/p69_aer.sh | sed "s/^/after $O: /"

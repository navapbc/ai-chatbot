#!/bin/bash
# Throwaway spike orchestrator. Usage: run-bench.sh
cd "$(dirname "$0")/../.."
export SPIKE_TOKEN=spiketoken
CDP="ws://127.0.0.1:9333/cdp?token=$SPIKE_TOKEN"
LOG=/private/tmp/claude-502/bench
mkdir -p $LOG
stop(){ pkill -f cdp-relay/launch.mjs; pkill -f cdp-relay/relay.mjs; pkill -f cdp-relay/delay.mjs; pkill -f agent-browser; docker rm -f spike-redis >/dev/null 2>&1; sleep 2; }
trap stop EXIT; stop
launch(){ (node spike/cdp-relay/launch.mjs > $LOG/launch.log 2>&1 &); sleep 8; }
direct(){ node spike/cdp-relay/relay.mjs > $LOG/relayA.log 2>&1 & sleep 1; launch; node spike/cdp-relay/bench.mjs "$CDP" "$1"; stop; }
delayed(){ node spike/cdp-relay/relay.mjs > $LOG/relayA.log 2>&1 & sleep 1
  LISTEN=9334 TARGET=9333 DELAY_MS=$2 node spike/cdp-relay/delay.mjs > $LOG/delay.log 2>&1 &
  sleep 1; EXT_URL=ws://127.0.0.1:9334/extension node -e "
    const fs=require('fs');const p='spike/cdp-relay/extension/config.js';
    fs.writeFileSync(p,fs.readFileSync(p,'utf8').replace(/RELAY_URL = '.*'/,\"RELAY_URL = 'ws://127.0.0.1:9334/extension'\"))"
  launch; node spike/cdp-relay/bench.mjs "$CDP" "$1"; stop; }
bus(){ docker run -d --rm --name spike-redis -p 6390:6379 redis:7-alpine >/dev/null; sleep 2
  REDIS_URL=redis://127.0.0.1:6390 INSTANCE=A PORT=9333 node spike/cdp-relay/relay.mjs > $LOG/relayA.log 2>&1 &
  sleep 1; REDIS_URL=redis://127.0.0.1:6390 INSTANCE=B PORT=9335 node spike/cdp-relay/relay.mjs > $LOG/relayB.log 2>&1 &
  sleep 1; launch   # extension -> A (9333); agent-browser -> B (9335)
  node spike/cdp-relay/bench.mjs "ws://127.0.0.1:9335/cdp?token=$SPIKE_TOKEN" "$1"; stop; }
echo "scenario       results (ms; each call includes ~CLI spawn)"
direct direct-local
delayed delay-40ms 40
delayed delay-100ms 100
bus bus-2-instances

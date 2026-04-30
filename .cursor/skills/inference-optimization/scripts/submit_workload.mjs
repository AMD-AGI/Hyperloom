#!/usr/bin/env node
// SaFE PyTorchJob submitter — verified against gptoss120b-opt-v5-9nqc8 (Succeeded).
//
// Usage:
//   node submit_workload.mjs \
//     --api-key "$SAFE_API_KEY" \
//     --workspace core42-hyperloom \
//     --name my-job \
//     --image "vllm/vllm-openai-rocm:v0.17.0" \
//     --script /path/to/entrypoint.sh
//
// The --script file content is read and passed as entryPoints[0] (NOT base64).
// Env fallbacks: SAFE_API_KEY, SANDBOX_WORKSPACE

import { readFileSync } from 'node:fs';
import https from 'node:https';

const args = process.argv.slice(2);
function arg(name, env, fallback) {
  const i = args.indexOf('--' + name);
  if (i >= 0 && args[i + 1]) return args[i + 1];
  if (env && process.env[env]) return process.env[env];
  return fallback || '';
}

const apiKey    = arg('api-key', 'SAFE_API_KEY');
const workspace = arg('workspace', 'SANDBOX_WORKSPACE', 'core42-hyperloom');
const name      = arg('name', null, 'opt-job');
const image     = arg('image');
const scriptPath = arg('script');
const gpu       = arg('gpu', null, '8');
const cpu       = arg('cpu', null, '64');
const memory    = arg('memory', null, '1024Gi');
const ttl       = arg('ttl', null, '7200');
const baseUrl   = arg('base-url', null, 'https://core42.primus-safe.amd.com');

if (!apiKey)     { console.error('Missing --api-key or SAFE_API_KEY'); process.exit(1); }
if (!image)      { console.error('Missing --image'); process.exit(1); }
if (!scriptPath) { console.error('Missing --script'); process.exit(1); }

const scriptContent = readFileSync(scriptPath, 'utf-8');

const body = JSON.stringify({
  displayName: name,
  workspaceId: workspace,
  kind: 'PyTorchJob',
  images: [image],
  resources: [{
    replica: 1,
    cpu,
    gpu,
    gpuName: 'amd.com/gpu',
    memory,
    sharedMemory: '512Gi',
    ephemeralStorage: '100Gi',
  }],
  entryPoints: [scriptContent],
  env: { MODELS_ROOT: '/wekafs/models' },
  isTolerateAll: true,
  ttlSecondsAfterFinished: parseInt(ttl),
});

const url = new URL('/api/v1/workloads', baseUrl);
const req = https.request(url, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  },
  rejectUnauthorized: false,
}, (res) => {
  let data = '';
  res.on('data', c => data += c);
  res.on('end', () => {
    try {
      const j = JSON.parse(data);
      if (j.workloadId) {
        console.log(JSON.stringify({ workloadId: j.workloadId }));
      } else {
        console.error('Error:', data.slice(0, 500));
        process.exit(1);
      }
    } catch { console.error('Response:', data.slice(0, 500)); process.exit(1); }
  });
});
req.on('error', e => { console.error('Fetch error:', e.message); process.exit(1); });
req.write(body);
req.end();

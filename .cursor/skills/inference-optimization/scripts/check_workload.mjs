#!/usr/bin/env node
// Poll SaFE workload status. Usage:
//   node check_workload.mjs --api-key AK --id WORKLOAD_ID [--wait] [--logs]
//
// --wait: poll every 30s until Succeeded/Failed
// --logs: print pod logs when done

import https from 'node:https';

const args = process.argv.slice(2);
function arg(name, env) {
  const i = args.indexOf('--' + name);
  if (i >= 0 && args[i + 1]) return args[i + 1];
  if (env && process.env[env]) return process.env[env];
  return '';
}
const hasFlag = (name) => args.includes('--' + name);

const apiKey  = arg('api-key', 'SAFE_API_KEY');
const wid     = arg('id');
const doWait  = hasFlag('wait');
const doLogs  = hasFlag('logs');
const baseUrl = arg('base-url') || 'https://core42.primus-safe.amd.com';

if (!apiKey || !wid) { console.error('Missing --api-key/--id'); process.exit(1); }

function fetch(path) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, baseUrl);
    https.get(url, {
      headers: { 'Authorization': `Bearer ${apiKey}` },
      rejectUnauthorized: false,
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch { resolve({ raw: data }); }
      });
    }).on('error', reject);
  });
}

async function check() {
  const d = await fetch(`/api/v1/workloads/${wid}`);
  const phase = d.phase || d.status || 'unknown';
  const duration = d.duration || '0s';
  const pods = d.pods || [];
  console.log(`Phase: ${phase}  Duration: ${duration}  Pods: ${pods.length}`);

  if (doLogs && pods.length > 0 && (phase === 'Succeeded' || phase === 'Failed')) {
    const pid = pods[0].podId || pods[0].name;
    try {
      const logs = await fetch(`/api/v1/workloads/${wid}/pods/${pid}/logs?tail=100`);
      for (const l of (logs.logs || [])) console.log(l);
    } catch (e) { console.error('Logs error:', e.message); }
  }

  return phase;
}

if (doWait) {
  let phase = '';
  for (let i = 0; i < 480; i++) { // max 4 hours
    phase = await check();
    if (phase === 'Succeeded' || phase === 'Failed') break;
    await new Promise(r => setTimeout(r, 30000));
  }
  process.exit(phase === 'Succeeded' ? 0 : 1);
} else {
  await check();
}

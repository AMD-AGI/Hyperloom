// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Upsert the one sticky code-metrics report comment on a pull request.
//
// Called from actions/github-script by code-metrics.yml (same-repository PRs) and by
// code-metrics-comment.yml (fork PRs, whose own token cannot comment). The report is
// posted as text and nothing in it is evaluated; for a fork it was produced by code
// the fork controls, so it is checked only for being a report, not trusted further,
// and the caller passes the triggering `run` (github.event.workflow_run): its
// conclusion, head SHA and link go above the report, derived here and not from the
// artifact, so a forged "PASSED" cannot stand in for the job's real result.
'use strict';

const fs = require('fs');

const MARKER = '<!-- code-metrics-report -->';
// GitHub rejects comment bodies over 65536 characters.
const LIMIT = 65000;

// The header a workflow_run poster puts above an untrusted report. Every value comes
// from the event payload GitHub wrote, never from the artifact.
function runHeader(run) {
  const conclusion = /^[a-z_]+$/.test(String(run.conclusion)) ? run.conclusion : 'unknown';
  const sha = /^[0-9a-f]{40}$/.test(String(run.head_sha)) ? run.head_sha : 'unknown';
  const link = /^https:\/\/github\.com\/[^\s()]+$/.test(String(run.html_url)) ? ` ([run](${run.html_url}))` : '';
  return (
    `**Gate job conclusion: ${conclusion}**${link} on \`${sha}\`, as reported by GitHub.\n\n` +
    '_The report below was uploaded by that run; the conclusion above is the result._\n\n---\n'
  );
}

async function postReport({ github, context, core, number, reportPath, run }) {
  let body = fs.readFileSync(reportPath, 'utf8');
  if (!body.startsWith(MARKER)) {
    throw new Error(`${reportPath} is not a code-metrics report`);
  }
  if (run) {
    body = `${MARKER}\n${runHeader(run)}${body.slice(MARKER.length)}`;
  }
  if (body.length > LIMIT) {
    body = `${body.slice(0, LIMIT)}\n\n_Report truncated; the job summary has the full text._\n`;
  }
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: number,
    per_page: 100,
  });
  const mine = comments.find(
    (c) => c.user && c.user.login === 'github-actions[bot]' && (c.body || '').startsWith(MARKER),
  );
  if (mine) {
    await github.rest.issues.updateComment({ ...context.repo, comment_id: mine.id, body });
    core.info(`updated ${mine.html_url}`);
  } else {
    const { data } = await github.rest.issues.createComment({ ...context.repo, issue_number: number, body });
    core.info(`created ${data.html_url}`);
  }
}

module.exports = postReport;
module.exports.runHeader = runHeader;

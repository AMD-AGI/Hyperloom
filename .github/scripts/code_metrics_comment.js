// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Upsert the one sticky code-metrics report comment on a pull request.
//
// Called from actions/github-script by code-metrics.yml (same-repository PRs) and by
// code-metrics-comment.yml (fork PRs, whose own token cannot comment). The report is
// posted as text and nothing in it is evaluated; for a fork it was produced by code
// the fork controls, so it is checked only for being a report, not trusted further.
'use strict';

const fs = require('fs');

const MARKER = '<!-- code-metrics-report -->';
// GitHub rejects comment bodies over 65536 characters.
const LIMIT = 65000;

module.exports = async function postReport({ github, context, core, number, reportPath }) {
  let body = fs.readFileSync(reportPath, 'utf8');
  if (!body.startsWith(MARKER)) {
    core.setFailed(`${reportPath} is not a code-metrics report`);
    return;
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
};

# Requesting Access to the Hyperloom Web UI

The hosted Hyperloom web UI runs on Primus-SaFE. AMD users and approved
partners need Primus-SaFE access before they can open the Hyperloom UI, bind an
LLM Gateway key, or launch hosted optimization jobs.

The same access-request flow applies to the Core42 and TensorWeave (TW)
Primus-SaFE clusters:

| Cluster | URL |
|---|---|
| Core42 | [https://core42.primus-safe.amd.com/](https://core42.primus-safe.amd.com/) |
| TensorWeave (TW) | [https://tw325.primus-safe.amd.com/login](https://tw325.primus-safe.amd.com/login) |

## 1. Submit an access request

Open the [AISE access request form](https://amd-hub.atlassian.net/jira/software/c/projects/AISE/form/35?isEligibleForUserSurvey=true)
and fill in:

- **Issue Sub-Type:** `Access Issue`
- **Components:** `Operations`
- **Description:** include the AMD NTID(s) that need access

<!-- Screenshot placeholder: add the access request form screenshot at slides/primus_safe_access_request.png, then replace this paragraph with:
<p align="center"><img width="600" alt="Primus-SaFE access request form" src="../slides/primus_safe_access_request.png" /></p>
-->

_Screenshot: Primus-SaFE access request form (to be added)._

## 2. Wait for provisioning and account sync

After the request is submitted, an administrator adds the requested account(s)
to the cluster. Once the account is added, allow about 15 minutes for access to
synchronize before trying to sign in.

## 3. Sign in with AMD SSO

Open the cluster URL and choose **Continue with AMD**:

- Core42: [https://core42.primus-safe.amd.com/](https://core42.primus-safe.amd.com/)
- TensorWeave (TW): [https://tw325.primus-safe.amd.com/login](https://tw325.primus-safe.amd.com/login)

<!-- Screenshot placeholder: add the AMD SSO login screenshot at slides/primus_safe_sso_login.png, then replace this paragraph with:
<p align="center"><img width="600" alt="Primus-SaFE AMD SSO login" src="../slides/primus_safe_sso_login.png" /></p>
-->

_Screenshot: Primus-SaFE AMD SSO login (to be added)._

## Next step: bind your LLM Gateway key

After you can sign in to Primus-SaFE, bind your
[LLM Gateway](https://llm.amd.com/) key to
[Hyperloom](https://core42.primus-safe.amd.com/hyperloom/) to obtain your
`AK_YOUR_API_KEY`. That key is required for both the hosted Hyperloom UI and the
local optimization workflow.

See the root [README](../README.md#prerequisites) and the
[Authentication & Environment Guide](ENV_AND_AUTH.md) for the credential setup
that follows access provisioning.

## Reference

This flow mirrors the internal Confluence guidance for
[TensorWeave MI325 cluster with Primus-SaFE](https://amd.atlassian.net/wiki/spaces/~712020ea4fade82ae94a95b7c0ba1cb554d2a8/pages/1178771460/TensorWeave+MI325+cluster+with+Primus-SaFE).

# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | ✅ Yes    |

## Reporting a Vulnerability

If you discover a security vulnerability in Arbor, please **do not** open a public GitHub issue.

Instead, use one of the following channels:

- **GitHub Private Vulnerability Reporting** (preferred): Use the "Report a vulnerability" button on the [Security tab](../../security/advisories/new) of this repository.
- **Email**: Contact the maintainer directly via GitHub (<maintainer>) for the appropriate security contact.

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigations

You can expect an acknowledgment within 5 business days and a resolution plan within 30 days for confirmed vulnerabilities.

## Security Best Practices for Contributors

- Never commit API keys, tokens, passwords, or credentials
- Never commit model weights or proprietary training data
- Always add `.env` and secrets files to `.gitignore`
- Use GitHub Secrets for CI/CD credentials

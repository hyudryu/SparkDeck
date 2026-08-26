# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities through GitHub's private security-advisory feature for this repository. Do not open a public issue with exploit details, credentials, logs containing secrets, or private model prompts.

Include the affected revision, a concise reproduction, the expected impact, and any suggested mitigation. You should receive an acknowledgement through the advisory within seven days. Please allow maintainers a reasonable opportunity to investigate and publish a fix before public disclosure.

## Supported versions

SparkDeck is under active development. Security fixes are applied to the latest commit on the default branch; older snapshots and forks are not maintained by the project.

## Deployment guidance

- Do not expose SparkDeck's management API directly to the public internet. Use a trusted network or an authenticated TLS reverse proxy.
- Keep `data/`, `.env` files, service environment files, Hugging Face tokens, pairing credentials, and runtime API keys out of version control.
- Give the SparkDeck process and Docker socket access only to trusted users. Docker control is effectively host-level administrative access.
- Review benchmark payloads before enabling future community sync. SparkDeck is designed to exclude prompt and response content, but operators remain responsible for their deployment and logs.
- Rotate credentials immediately if they appear in logs, screenshots, issues, or commits.

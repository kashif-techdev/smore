#!/usr/bin/env python3
"""
Update GitHub repository description and topics via API.

Usage (requires a PAT with repo scope):
  set GITHUB_TOKEN=ghp_...
  python scripts/set_github_about.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

OWNER = "kashif-techdev"
REPO = "smore"
DESCRIPTION = "MRI super-resolution & anti-aliasing (DIP project, SMORE-inspired U-Net)"
TOPICS = [
    "mri",
    "super-resolution",
    "pytorch",
    "unet",
    "medical-imaging",
]


def api_request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print(
            "GITHUB_TOKEN not set. Set a personal access token with `repo` scope, then re-run.",
            file=sys.stderr,
        )
        return 1

    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    api_request("PATCH", base, token, {"description": DESCRIPTION})
    api_request("PUT", f"{base}/topics", token, {"names": TOPICS})
    print(f"Updated {OWNER}/{REPO} description and topics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

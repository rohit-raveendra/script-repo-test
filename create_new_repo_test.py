"""create_repo.py — Create the target GitHub repository via the GitHub API.

Uses {{previous.repo_path}} so it chains after any prior step without
depending on that step's name.

Placeholders:
  {{GITHUB_TOKEN}}      — GitHub PAT or app token (rendered as env-var ref)
  {{GITHUB_ORG}}        — Target GitHub organization
  {{TARGET_REPO_NAME}}  — Repository name to create on GitHub
  {{GITHUB_API_URL}}    — GitHub API base URL
                          • github.com       → https://api.github.com
                          • GHES             → https://<hostname>/api/v3
"""
import json
import sys
import requests
from pathlib import Path

GITHUB_TOKEN     = {{GITHUB_TOKEN}}
GITHUB_ORG       = {{GITHUB_ORG}}
TARGET_REPO_NAME = {{TARGET_REPO_NAME}}
GITHUB_API_URL   = {{GITHUB_API_URL}}

# --- Setup ---
base_url = GITHUB_API_URL.rstrip("/")
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# --- Check if repo already exists ---
check_url = f"{base_url}/repos/{GITHUB_ORG}/{TARGET_REPO_NAME}"
check_resp = requests.get(check_url, headers=headers, timeout=30)

if check_resp.status_code == 200:
    print(f"[create_repo] Repository already exists: {GITHUB_ORG}/{TARGET_REPO_NAME}", file=sys.stderr)
    clone_url = check_resp.json().get("clone_url", f"https://github.com/{GITHUB_ORG}/{TARGET_REPO_NAME}.git")
elif check_resp.status_code == 404:
    # Repo doesn't exist — create it
    if GITHUB_ORG:
        create_url = f"{base_url}/orgs/{GITHUB_ORG}/repos"
    else:
        create_url = f"{base_url}/user/repos"

    payload = {
        "name": TARGET_REPO_NAME,
        "private": True,
        "auto_init": False,
    }

    resp = requests.post(create_url, json=payload, headers=headers, timeout=30)

    if resp.status_code == 201:
        clone_url = resp.json().get("clone_url", f"https://github.com/{GITHUB_ORG}/{TARGET_REPO_NAME}.git")
        print(f"[create_repo] Repository created: {GITHUB_ORG}/{TARGET_REPO_NAME}", file=sys.stderr)
    elif resp.status_code == 422:
        # Race condition: repo was created between our check and create call
        clone_url = f"https://github.com/{GITHUB_ORG}/{TARGET_REPO_NAME}.git"
        print(f"[create_repo] Repository created concurrently: {GITHUB_ORG}/{TARGET_REPO_NAME}", file=sys.stderr)
    else:
        raise RuntimeError(
            f"GitHub repo creation failed ({resp.status_code}): {resp.text}"
        )
else:
    raise RuntimeError(
        f"GitHub repo check failed ({check_resp.status_code}): {check_resp.text}"
    )

print(json.dumps({
    "target_repo_url": clone_url,
    "github_org": GITHUB_ORG,
    "target_repo_name": TARGET_REPO_NAME,
}))


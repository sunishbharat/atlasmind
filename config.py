"""
Profile-based configuration for AtlasMind.

Profiles are stored in profiles.json (gitignored).
Copy profiles.json.example to profiles.json and fill in your credentials.

Selection priority (highest first):
  1. --profile CLI flag
  2. ATLASMIND_PROFILE env var
  3. "default" key in profiles.json
  4. First profile in the file
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROFILES_FILE = Path(__file__).parent / "profiles.json"


@dataclass
class Profile:
    name:          str
    jira_url:      str
    email:         str
    token:         str
    client_id:     str = ""
    client_secret: str = ""
    jira_type:     str = "cloud"   # "cloud" | "server"

    @property
    def jira_base_url(self) -> str:
        return self.jira_url.rstrip("/")

    @property
    def is_cloud(self) -> bool:
        return self.jira_type == "cloud"


def load_profiles() -> dict:
    """Load raw profiles dict from profiles.json."""
    if not PROFILES_FILE.exists():
        raise FileNotFoundError(
            f"profiles.json not found. "
            f"Copy profiles.json.example to profiles.json and fill in your credentials."
        )
    with open(PROFILES_FILE) as f:
        return json.load(f)


def list_profiles() -> list[str]:
    """Return names of all configured profiles."""
    data = load_profiles()
    return list(data.get("profiles", {}).keys())


def get_profile(name: str | None = None) -> Profile:
    """
    Return the Profile for the given name.
    If name is None, falls back to ATLASMIND_PROFILE env var,
    then the 'default' key in profiles.json, then the first profile.
    """
    data = load_profiles()
    profiles = data.get("profiles", {})

    if not profiles:
        raise ValueError("No profiles defined in profiles.json.")

    # Resolve name
    resolved = (
        name
        or os.getenv("ATLASMIND_PROFILE")
        or data.get("default")
        or next(iter(profiles))
    )

    if resolved not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(
            f"Profile '{resolved}' not found. Available: {available}"
        )

    raw = profiles[resolved]

    # Token can also come from env var ATLASMIND_<PROFILE_UPPER>_TOKEN
    token = (
        raw.get("token")
        or os.getenv(f"ATLASMIND_{resolved.upper()}_TOKEN")
        or os.getenv("ATLASSIAN_TOKEN")
        or ""
    )

    return Profile(
        name          = resolved,
        jira_url      = raw["jira_url"],
        email         = raw.get("email", ""),
        token         = token,
        client_id     = raw.get("client_id", "") or os.getenv("ATLASSIAN_CLIENT_ID", ""),
        client_secret = raw.get("client_secret", "") or os.getenv("ATLASSIAN_CLIENT_SECRET", ""),
        jira_type     = raw.get("jira_type", "cloud"),
    )


def print_profiles():
    """Pretty-print all configured profiles (masks tokens)."""
    data = load_profiles()
    profiles = data.get("profiles", {})
    default  = data.get("default", next(iter(profiles), ""))

    print(f"{'Profile':<15} {'Jira URL':<45} {'Email':<35} Default")
    print("-" * 105)
    for name, raw in profiles.items():
        is_default = "  ✓" if name == default else ""
        token_set  = "set" if raw.get("token") else "env/unset"
        print(
            f"{name:<15} {raw.get('jira_url',''):<45} "
            f"{raw.get('email',''):<35} {is_default}"
        )


if __name__ == "__main__":
    print_profiles()

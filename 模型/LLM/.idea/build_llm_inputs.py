from typing import Dict, List
import pandas as pd
import numpy as np
import re


IMPORTANT_FEATURES = [
    "repo_age_days",
    "user_age_days",
    "days_since_last_push",
    "time_to_last_push_days",
    "very_short_activity_window_flag",
    "user_repo_creation_gap_days",
    "is_recent_repo_flag",
    "is_stale_repo_flag",
    "watchers_count",
    "stargazers_count",
    "forks_count",
    "open_issues_count",
    "repo_size_kb",
    "has_homepage",
    "has_license_meta",
    "has_license_file",
    "readme_length",
    "desc_length",
    "topics_count",
    "risk_filename_count",
    "external_domains_count",
    "external_domains_average_age",
    "readme_has_sensitive_request_keywords",
    "readme_has_airdrop_claim_keywords",
    "readme_has_approve_keywords",
    "wallet_frontend_risk_combo_flag",
    "binary_download_lure_combo_flag",
    "framework_hint_count",
    "contract_file_ratio",
    "has_contracts_dir",
    "has_test_dir",
    "has_deploy_dir",
    "has_package_json",
    "has_hardhat_config",
    "has_foundry_toml",
    "primary_language_is_solidity",
    "primary_language_is_typescript",
    "primary_language_is_javascript",
    "primary_language_is_python",
]


def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def serialize_structured_features(row: pd.Series, feature_names: List[str], cfg: Dict) -> List[str]:
    mode = cfg["input_build"]["structured_mode"]
    max_items = int(cfg["input_build"]["max_structured_items"])

    names = IMPORTANT_FEATURES.copy()
    if mode == "all_numeric":
        extra = [x for x in feature_names if x not in names]
        names.extend(extra)

    lines = []
    for name in names:
        if name not in row.index:
            continue
        val = row[name]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            val = round(float(val), 6)
        lines.append(f"{name} = {val}")
        if len(lines) >= max_items:
            break
    return lines


def build_prompt(row: pd.Series, feature_names: List[str], cfg: Dict) -> str:
    repo_name = clean_text(row.get("repo_full_name", ""))[:200]
    desc = clean_text(row.get("description_text", ""))[: int(cfg["input_build"]["max_description_chars"])]
    topics = clean_text(row.get("topics_text", ""))[: int(cfg["input_build"]["max_topics_chars"])]
    readme = clean_text(row.get("readme_text", ""))[: int(cfg["input_build"]["max_readme_chars"])]

    struct_lines = serialize_structured_features(row, feature_names, cfg)
    structured_block = "\n".join([f"- {x}" for x in struct_lines])

    prompt = f"""You are a security analyst focusing on blockchain GitHub repositories and accounts.
Determine whether the following sample is suspicious.

[Repository]
{repo_name}

[Description]
{desc}

[Topics]
{topics}

[Structured Signals]
{structured_block}

[README Excerpt]
{readme}

[Label Definition]
0 = benign / normal open-source repository
1 = suspicious repository / account

Return classification based on the whole evidence."""
    return prompt.strip()


def build_prompts(df: pd.DataFrame, feature_names: List[str], cfg: Dict) -> List[str]:
    return [build_prompt(row, feature_names, cfg) for _, row in df.iterrows()]
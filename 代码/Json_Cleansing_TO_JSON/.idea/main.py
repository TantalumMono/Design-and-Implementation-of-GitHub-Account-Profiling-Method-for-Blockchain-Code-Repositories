import os
import json
import math
import base64
from datetime import datetime
from collections import Counter


def calculate_entropy(text):
    """计算字符串信息熵（用于检测随机用户名）"""
    if not text:
        return 0.0
    probs = [n / len(text) for n in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs)


def to_dt_github(ts):
    """兼容 GitHub 常见时间格式，返回 naive datetime"""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)

    ts = str(ts).strip()
    if not ts:
        return None

    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def parse_collected_at(data):
    raw = data.get("collected_at")
    dt = to_dt_github(raw)
    return dt if dt else datetime.now()


def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def maybe_decode_base64(content):
    """尝试解码 base64 README"""
    if not isinstance(content, str):
        return ""
    try:
        decoded = base64.b64decode(content, validate=True)
        text = decoded.decode("utf-8", errors="ignore")
        if text and any(ch.isprintable() for ch in text):
            return text.strip()
    except Exception:
        pass
    return content.strip()


def extract_readme_text(readme_field):
    """
    README 提取：
    1) 如果本身就是字符串，直接返回
    2) 如果是 dict，优先取 text/body/content/readme/markdown
    3) 若标注 encoding=base64，则尝试解码
    """
    if isinstance(readme_field, str):
        return readme_field.strip()

    if isinstance(readme_field, dict):
        encoding = str(readme_field.get("encoding", "")).lower()

        for key in ["text", "body", "readme", "markdown", "content"]:
            val = readme_field.get(key)
            if isinstance(val, str) and val.strip():
                if key == "content" and encoding == "base64":
                    return maybe_decode_base64(val)
                return val.strip()

        return json.dumps(readme_field, ensure_ascii=False)

    return ""


def normalize_tree(tree_field):
    """
    统一 tree/file_tree 结构
    返回标准化后的 tree 列表，仅用于特征计算
    """
    if tree_field is None or not isinstance(tree_field, list):
        return []

    tree_list = []
    for item in tree_field:
        if isinstance(item, dict):
            tree_list.append(item)
        elif isinstance(item, str):
            tree_list.append({"path": item})
    return tree_list


def normalize_external_domains(ext_domains_field):
    """
    兼容 external_domains / readme_domains 的几种格式：
    1) {"abc.com": 123, "x.org": 20}
    2) ["abc.com", "x.org"]
    3) [{"domain": "abc.com", "age": 123}, ...]
    输出统一为 {domain: age_or_0}
    """
    result = {}

    if not ext_domains_field:
        return result

    if isinstance(ext_domains_field, dict):
        for k, v in ext_domains_field.items():
            domain = safe_text(k)
            if not domain:
                continue
            result[domain] = v if isinstance(v, (int, float)) else 0
        return result

    if isinstance(ext_domains_field, list):
        for item in ext_domains_field:
            if isinstance(item, str):
                domain = item.strip()
                if domain:
                    result[domain] = 0
            elif isinstance(item, dict):
                domain = safe_text(item.get("domain") or item.get("host") or item.get("name"))
                age = item.get("age", 0)
                if domain:
                    result[domain] = age if isinstance(age, (int, float)) else 0

    return result


def parse_single_record(data):
    repo = data.get("repo_detail", {}) or {}
    user = data.get("user_detail", {}) or {}

    readme_raw = data.get("readme", "")
    readme_text = extract_readme_text(readme_raw)

    tree_raw = data.get("tree_structure") or data.get("file_tree") or []
    tree = normalize_tree(tree_raw)

    ext_domains_raw = data.get("external_domains", {}) or data.get("readme_domains", {}) or []
    ext_domains = normalize_external_domains(ext_domains_raw)

    collected_at = parse_collected_at(data)
    repo_created_at = to_dt_github(repo.get("created_at"))
    repo_pushed_at = to_dt_github(repo.get("pushed_at"))
    user_created_at = to_dt_github(user.get("created_at"))
    user_updated_at = to_dt_github(user.get("updated_at"))

    BLOCKCHAIN_CORE = {"Solidity", "Go", "Rust"}
    BLOCKCHAIN_SCRIPT = {"JavaScript", "TypeScript", "Python"}
    LANG_MAP = {lang: i + 1 for i, lang in enumerate(
        ["Solidity", "JavaScript", "TypeScript", "Go", "Rust", "Python"]
    )}

    features = {}
    features["repo_full_name"] = data.get("repo_full_name")

    # --- 1. 时间与生命周期特征 ---
    features["repo_age_days"] = (collected_at - repo_created_at).days if (collected_at and repo_created_at) else None
    features["user_age_days"] = (collected_at - user_created_at).days if (collected_at and user_created_at) else None
    features["days_since_last_push"] = (collected_at - repo_pushed_at).days if (collected_at and repo_pushed_at) else None

    if repo_pushed_at and repo_created_at:
        diff_sec = (repo_pushed_at - repo_created_at).total_seconds()
        features["is_one_commit_repo"] = 1 if diff_sec < 3600 else 0
        features["time_to_last_push_days"] = (repo_pushed_at - repo_created_at).days
    else:
        features["is_one_commit_repo"] = 0
        features["time_to_last_push_days"] = None

    features["user_repo_creation_gap_days"] = (
        (repo_created_at - user_created_at).days if (repo_created_at and user_created_at) else None
    )
    features["user_update_frequency"] = (
        (user_updated_at - user_created_at).days if (user_updated_at and user_created_at) else None
    )
    features["Is_user_deleted"] = 1 if (not user or user.get("owner_current_status") == "deleted_or_banned") else 0

    # --- 2. 仓库活跃度与配置特征 ---
    stars = repo.get("stargazers_count", 0) or 0
    forks = repo.get("forks_count", 0) or 0

    features["fork_to_star_ratio"] = forks / (stars + 1)
    features["repo_size_kb"] = repo.get("size", 0) or 0
    features["has_issues_enabled"] = 1 if repo.get("has_issues") else 0
    features["has_wiki_enabled"] = 1 if repo.get("has_wiki") else 0
    features["stargazers_count"] = stars
    features["forks_count"] = forks
    features["open_issues_count"] = repo.get("open_issues_count", 0) or 0
    features["subscribers_count"] = repo.get("subscribers_count", 0) or 0
    features["network_count"] = repo.get("network_count", 0) or 0

    lang = repo.get("language", "Unknown") or "Unknown"
    features["lang_is_blockchain_core"] = 1 if lang in BLOCKCHAIN_CORE else 0
    features["lang_is_scripting"] = 1 if lang in BLOCKCHAIN_SCRIPT else 0
    features["lang_fixed_ordinal"] = LANG_MAP.get(lang, 0)

    features["Has_homepage"] = 1 if repo.get("homepage") else 0
    features["has_license"] = 1 if repo.get("license") else 0
    features["has_discussions_enabled"] = 1 if repo.get("has_discussions") else 0

    # --- 3. 开发者画像特征 ---
    completeness = sum(1 for k in ["company", "blog", "location", "email", "bio"] if user.get(k))
    features["user_profile_completeness"] = completeness

    followers = user.get("followers", 0) or 0
    following = user.get("following", 0) or 0
    features["follower_to_following_ratio"] = followers / (following + 1)
    features["public_repos_count"] = user.get("public_repos", 0) or 0
    features["is_hireable"] = 1 if user.get("hireable") else 0
    features["User_type"] = 1 if user.get("type") == "Organization" else 0
    features["public_gists_count"] = user.get("public_gists", 0) or 0
    features["Followers_count"] = followers
    features["Following_count"] = following
    features["user_login_entropy"] = calculate_entropy(user.get("login", "") or "")
    features["has_twitter"] = 1 if user.get("twitter_username") else 0
    features["has_email"] = 1 if user.get("email") else 0

    u_name = user.get("name")
    u_login = user.get("login")
    features["user_name_matches_login"] = 1 if (u_name == u_login or u_name is None) else 0

    # --- 4. 内容与语义特征 ---
    repo_description = safe_text(repo.get("description", ""))
    repo_topics = repo.get("topics", []) if isinstance(repo.get("topics", []), list) else []

    features["readme_length"] = len(readme_text)
    features["desc_length"] = len(repo_description)
    features["topics_count"] = len(repo_topics)

    # --- 5. 文件结构与外部网络特征 ---
    features["total_files_count"] = len(tree)

    exec_exts = {".exe", ".sh", ".bat", ".dll", ".msi", ".scr"}
    has_exec = any(
        safe_text(file.get("path", "")).lower().endswith(ext)
        for file in tree if isinstance(file, dict)
        for ext in exec_exts
    )
    features["has_executable_files"] = 1 if has_exec else 0

    extensions = {
        safe_text(file.get("path", "")).split(".")[-1].lower()
        for file in tree
        if isinstance(file, dict) and "." in safe_text(file.get("path", ""))
    }
    features["file_extension_diversity"] = len(extensions)

    if ext_domains:
        valid_ages = [v for v in ext_domains.values() if isinstance(v, (int, float)) and v > 0]
        features["external_domains_count"] = len(ext_domains)
        features["External_domains_average_age"] = sum(valid_ages) / len(valid_ages) if valid_ages else 0
    else:
        features["external_domains_count"] = 0
        features["External_domains_average_age"] = 0

    # 输出更精简的 cleaned JSON
    cleaned_record = {
        "readme_text": readme_text,
        "features": features
    }

    return cleaned_record


def parse_github_folder_to_json(input_folder):
    all_records = []

    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, file)
        print(f"Processing: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            if not isinstance(data, list):
                print(f"Skip invalid JSON format: {file}")
                continue

            for item in data:
                if isinstance(item, dict):
                    record = parse_single_record(item)
                    all_records.append(record)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return all_records


def save_as_json(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_as_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(input_folder, output_path, output_format="json"):
    records = parse_github_folder_to_json(input_folder)

    if not records:
        print("No valid JSON data found.")
        return

    if output_format.lower() == "jsonl":
        save_as_jsonl(records, output_path)
    else:
        save_as_json(records, output_path)

    print("=================================")
    print("Data processing completed")
    print("Total samples:", len(records))
    print("Output file:", output_path)
    print("Output format:", output_format)


if __name__ == "__main__":
    input_folder = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\XGBoost_LLM_TF_IDF\.idea\family级分组验证\提升负样本数后\重新筛选正样本\筛选后的正样本"

    output_path = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\XGBoost_LLM_TF_IDF\.idea\family级分组验证\提升负样本数后\重新筛选正样本\raw_positive.json"
    output_format = "json"

    # output_path = "github_dataset_cleaned.jsonl"
    # output_format = "jsonl"

    main(input_folder, output_path, output_format)
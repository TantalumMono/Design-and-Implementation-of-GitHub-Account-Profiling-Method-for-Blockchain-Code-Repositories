import os
import re
import json
import math
import base64
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse


# =========================
# 基础工具函数
# =========================

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


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


def calculate_entropy(text):
    """计算字符串信息熵"""
    text = safe_text(text)
    if not text:
        return 0.0
    probs = [n / len(text) for n in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs)


def maybe_decode_base64(content):
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
    """统一 tree_structure / file_tree 结构"""
    if tree_field is None or not isinstance(tree_field, list):
        return []

    tree_list = []
    for item in tree_field:
        if isinstance(item, dict):
            tree_list.append(item)
        elif isinstance(item, str):
            tree_list.append({"path": item, "type": "file"})
    return tree_list


def normalize_domain(url_or_domain):
    raw = safe_text(url_or_domain).lower()
    if not raw:
        return ""

    if not raw.startswith(("http://", "https://")):
        raw = raw.strip("/")
        raw = raw.replace("www.", "")
        return raw

    try:
        netloc = urlparse(raw).netloc.lower().replace("www.", "")
        return netloc
    except Exception:
        return ""


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
            domain = normalize_domain(k)
            if domain:
                result[domain] = v if isinstance(v, (int, float)) else 0
        return result

    if isinstance(ext_domains_field, list):
        for item in ext_domains_field:
            if isinstance(item, str):
                domain = normalize_domain(item)
                if domain:
                    result[domain] = 0
            elif isinstance(item, dict):
                domain = normalize_domain(
                    item.get("domain") or item.get("host") or item.get("name")
                )
                age = item.get("age", 0)
                if domain:
                    result[domain] = age if isinstance(age, (int, float)) else 0
    return result


def extract_urls(text):
    text = safe_text(text)
    if not text:
        return []
    url_pattern = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
    return url_pattern.findall(text)


def count_keyword_hits(text, keywords):
    text_low = safe_text(text).lower()
    if not text_low:
        return 0
    hits = 0
    for kw in keywords:
        if kw in text_low:
            hits += 1
    return hits


def has_any_keyword(text, keywords):
    return 1 if count_keyword_hits(text, keywords) > 0 else 0


def average(values):
    return sum(values) / len(values) if values else 0


def ratio(num, denom):
    return num / denom if denom else 0


def get_file_paths(tree):
    return [safe_text(item.get("path", "")) for item in tree if isinstance(item, dict)]


def get_tree_items(tree):
    return [item for item in tree if isinstance(item, dict)]


def get_file_extensions(paths):
    exts = []
    for path in paths:
        base = os.path.basename(path)
        if "." in base:
            ext = os.path.splitext(base)[1].lower()
            if ext:
                exts.append(ext)
    return exts


def file_name_exists(paths, names):
    name_set = {n.lower() for n in names}
    return 1 if any(os.path.basename(p).lower() in name_set for p in paths) else 0


def path_contains_prefix(paths, prefixes):
    prefixes = tuple(p.lower().rstrip("/") + "/" for p in prefixes)
    return 1 if any(p.lower().startswith(prefixes) for p in paths) else 0


def count_paths_with_prefix(paths, prefixes):
    prefixes = tuple(p.lower().rstrip("/") + "/" for p in prefixes)
    return sum(1 for p in paths if p.lower().startswith(prefixes))


def count_paths_with_keywords(paths, keywords):
    keywords = [k.lower() for k in keywords]
    return sum(1 for p in paths if any(k in p.lower() for k in keywords))


def count_paths_by_ext(paths, ext_set):
    ext_set = {e.lower() for e in ext_set}
    return sum(1 for p in paths if os.path.splitext(p)[1].lower() in ext_set)


def count_hidden_files(paths):
    return sum(1 for p in paths if os.path.basename(p).startswith("."))


def collect_sizes(tree):
    sizes = []
    for item in tree:
        if item.get("type") == "file":
            size = item.get("size")
            if isinstance(size, (int, float)) and size >= 0:
                sizes.append(float(size))
    return sizes


# =========================
# 关键词表
# =========================
BLOCKCHAIN_KEYWORDS = [
    "blockchain", "crypto", "cryptocurrency", "web3", "dapp", "smart contract", "solidity",
    "evm", "token", "coin", "nft", "dao", "defi", "dex", "amm", "swap", "router", "lp",
    "liquidity", "yield", "farm", "staking", "restaking", "lending", "borrowing", "bridge",
    "cross-chain", "cross chain", "governance", "oracle", "vault", "launchpad", "presale",
    "mint", "airdrop", "mainnet", "testnet", "rollup", "layer2", "ethereum", "eth", "bsc",
    "bnb chain", "arbitrum", "optimism", "base", "polygon", "avalanche", "fantom", "tron",
    "solana", "aptos", "sui", "cosmos", "bitcoin"
]

TOKEN_STD_KEYWORDS = [
    "erc20", "erc721", "erc1155", "erc4626", "erc777", "erc1363", "erc2981",
    "bep20", "trc20", "spl token", "sip-010", "tokenomics"
]

WALLET_KEYWORDS = [
    "wallet", "connect wallet", "walletconnect", "metamask", "phantom", "rabby",
    "trust wallet", "coinbase wallet", "rainbow", "okx wallet", "tronlink"
]

RISKY_README_KEYWORDS = [
    "seed phrase", "mnemonic", "private key", "keystore", "recovery phrase", "recovery key",
    "import wallet", "verify wallet", "synchronize wallet", "sync wallet", "rectify wallet",
    "unlock wallet", "wallet validation", "validate wallet", "connect wallet", "sign message",
    "sign transaction", "approve", "permit", "permit2", "setapprovalforall", "increaseallowance",
    "claim airdrop", "claim reward", "claim now", "airdrop claim", "drainer", "sweep", "sweeper"
]

DOWNLOAD_LURE_KEYWORDS = [
    "download", "download here", "latest release", "installer", "setup.exe", "run this file",
    "double click", "browser extension", "chrome extension", "install extension",
    "install package", "desktop client", "windows client", "macos client", "apk", "zip file"
]

SECURITY_TRUST_KEYWORDS = [
    "audit", "audited", "security", "openzeppelin", "test", "tests", "coverage",
    "ci", "github actions", "hardhat", "foundry", "truffle", "brownie"
]

EXPLORER_DOMAINS = [
    "etherscan.io", "bscscan.com", "arbiscan.io", "basescan.org", "polygonscan.com",
    "snowtrace.io", "ftmscan.com", "tronscan.org", "solscan.io", "blockscout.com"
]

RISK_FILENAME_KEYWORDS = [
    "wallet", "drainer", "airdrop", "claim", "seed", "mnemonic", "privatekey", "keystore",
    "sweep", "approve", "permit", "unlock", "recover"
]

BRANDISH_KEYWORDS = [
    "uniswap", "pancakeswap", "metamask", "trustwallet", "coinbase", "binance",
    "opensea", "blur", "aave", "compound", "curve", "raydium", "jupiter"
]

EXECUTABLE_EXTS = {".exe", ".dll", ".bat", ".cmd", ".ps1", ".msi", ".scr", ".apk", ".jar", ".sh"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz"}
SOLIDITY_EXTS = {".sol"}
JS_TS_EXTS = {".js", ".jsx", ".ts", ".tsx"}
PY_EXTS = {".py"}
GO_EXTS = {".go"}
RUST_EXTS = {".rs"}
MOVE_EXTS = {".move"}
VYPER_EXTS = {".vy"}


# =========================
# 单条样本解析
# =========================

def parse_single_record(data, raw_file_name=None):
    repo = data.get("repo_detail", {}) or {}
    user = data.get("user_detail", {}) or {}

    readme_text = extract_readme_text(data.get("readme", ""))
    repo_description = safe_text(repo.get("description", ""))
    topics = repo.get("topics", []) if isinstance(repo.get("topics", []), list) else []
    topics_text = " ".join(safe_text(x) for x in topics)

    tree = normalize_tree(data.get("tree_structure") or data.get("file_tree") or [])
    tree_items = get_tree_items(tree)
    paths = get_file_paths(tree)
    extensions = get_file_extensions(paths)

    ext_domains = normalize_external_domains(
        data.get("external_domains", {}) or data.get("readme_domains", {}) or []
    )

    readme_urls = extract_urls(readme_text)
    desc_urls = extract_urls(repo_description)
    readme_domains = {normalize_domain(u) for u in readme_urls if normalize_domain(u)}
    desc_domains = {normalize_domain(u) for u in desc_urls if normalize_domain(u)}

    for d in readme_domains | desc_domains:
        ext_domains.setdefault(d, 0)

    collected_at = parse_collected_at(data)
    repo_created_at = to_dt_github(repo.get("created_at"))
    repo_pushed_at = to_dt_github(repo.get("pushed_at"))
    user_created_at = to_dt_github(user.get("created_at"))
    user_updated_at = to_dt_github(user.get("updated_at"))

    repo_name = safe_text(repo.get("name") or data.get("repo_full_name", "")).lower()
    owner_login = safe_text(user.get("login") or repo.get("owner", {}).get("login")).lower()
    homepage = safe_text(repo.get("homepage"))
    homepage_domain = normalize_domain(homepage)
    user_blog = safe_text(user.get("blog"))
    user_blog_domain = normalize_domain(user_blog)

    stars = repo.get("stargazers_count", 0) or 0
    forks = repo.get("forks_count", 0) or 0
    watchers = repo.get("watchers_count", 0) or 0
    followers = user.get("followers", 0) or 0
    following = user.get("following", 0) or 0

    path_depths = [p.count("/") + 1 for p in paths if p]
    top_level_dirs = {p.split("/")[0] for p in paths if "/" in p}
    file_sizes = collect_sizes(tree_items)

    text_all = "\n".join([readme_text, repo_description, topics_text])
    text_low = text_all.lower()

    solidity_file_count = count_paths_by_ext(paths, SOLIDITY_EXTS)
    js_ts_file_count = count_paths_by_ext(paths, JS_TS_EXTS)
    python_file_count = count_paths_by_ext(paths, PY_EXTS)
    go_file_count = count_paths_by_ext(paths, GO_EXTS)
    rust_file_count = count_paths_by_ext(paths, RUST_EXTS)
    move_file_count = count_paths_by_ext(paths, MOVE_EXTS)
    vyper_file_count = count_paths_by_ext(paths, VYPER_EXTS)
    executable_file_count = count_paths_by_ext(paths, EXECUTABLE_EXTS)
    archive_file_count = count_paths_by_ext(paths, ARCHIVE_EXTS)

    test_related_file_count = count_paths_with_keywords(paths, ["test", "tests", "spec", "specs"])
    deploy_related_file_count = count_paths_with_keywords(paths, ["deploy", "deployment", "migrations", "scripts"])
    docs_related_file_count = count_paths_with_keywords(paths, ["docs", "doc", "whitepaper"])
    env_file_count = count_paths_with_keywords(paths, [".env", "env.example", ".env.example", ".env.template"])
    risk_filename_count = count_paths_with_keywords(paths, RISK_FILENAME_KEYWORDS)

    valid_domain_ages = [v for v in ext_domains.values() if isinstance(v, (int, float)) and v > 0]
    explorer_domain_count = sum(1 for d in ext_domains if d in EXPLORER_DOMAINS)

    features = {}
    features["repo_full_name"] = data.get("repo_full_name") or repo.get("full_name")
    features["raw_file_name"] = raw_file_name or ""

    # 1) 时间与生命周期特征
    features["repo_age_days"] = (collected_at - repo_created_at).days if (collected_at and repo_created_at) else None
    features["user_age_days"] = (collected_at - user_created_at).days if (collected_at and user_created_at) else None
    features["days_since_last_push"] = (collected_at - repo_pushed_at).days if (collected_at and repo_pushed_at) else None
    features["time_to_last_push_days"] = (repo_pushed_at - repo_created_at).days if (repo_pushed_at and repo_created_at) else None
    features["very_short_activity_window_flag"] = 1 if (
        repo_pushed_at and repo_created_at and (repo_pushed_at - repo_created_at).total_seconds() <= 24 * 3600
    ) else 0
    features["user_repo_creation_gap_days"] = (repo_created_at - user_created_at).days if (repo_created_at and user_created_at) else None
    features["user_profile_update_span_days"] = (user_updated_at - user_created_at).days if (user_updated_at and user_created_at) else None
    features["is_user_deleted"] = 1 if (not user or user.get("owner_current_status") == "deleted_or_banned") else 0
    features["is_recent_repo_flag"] = 1 if (features["repo_age_days"] is not None and features["repo_age_days"] <= 30) else 0
    features["is_stale_repo_flag"] = 1 if (features["days_since_last_push"] is not None and features["days_since_last_push"] >= 180) else 0

    # 2) 仓库活跃度与配置特征
    features["fork_to_star_ratio"] = ratio(forks, stars + 1)
    features["watchers_count"] = watchers
    features["watch_to_star_ratio"] = ratio(watchers, stars + 1)
    features["issues_to_star_ratio"] = ratio(repo.get("open_issues_count", 0) or 0, stars + 1)
    features["repo_size_kb"] = repo.get("size", 0) or 0
    features["has_issues_enabled"] = 1 if repo.get("has_issues") else 0
    features["has_wiki_enabled"] = 1 if repo.get("has_wiki") else 0
    features["has_discussions_enabled"] = 1 if repo.get("has_discussions") else 0
    features["has_projects_enabled"] = 1 if repo.get("has_projects") else 0
    features["has_downloads_enabled"] = 1 if repo.get("has_downloads") else 0
    features["has_pages_enabled"] = 1 if repo.get("has_pages") else 0
    features["stargazers_count"] = stars
    features["forks_count"] = forks
    features["open_issues_count"] = repo.get("open_issues_count", 0) or 0
    features["subscribers_count"] = repo.get("subscribers_count", 0) or 0
    features["network_count"] = repo.get("network_count", 0) or 0
    features["has_homepage"] = 1 if homepage else 0
    features["has_license_meta"] = 1 if repo.get("license") else 0
    features["has_license_file"] = file_name_exists(paths, ["license", "license.md", "license.txt", "copying"])
    features["repo_is_fork"] = 1 if repo.get("fork") else 0
    features["repo_archived"] = 1 if repo.get("archived") else 0
    features["repo_disabled"] = 1 if repo.get("disabled") else 0
    features["allow_forking"] = 1 if repo.get("allow_forking", True) else 0
    features["is_template_repo"] = 1 if repo.get("is_template") else 0
    features["visibility_is_public"] = 1 if safe_text(repo.get("visibility", "public")).lower() == "public" else 0
    default_branch = safe_text(repo.get("default_branch")).lower()
    features["default_branch_is_main"] = 1 if default_branch == "main" else 0
    features["default_branch_is_master"] = 1 if default_branch == "master" else 0

    # 3) 开发者画像特征
    profile_fields = ["company", "blog", "location", "email", "bio"]
    features["user_profile_completeness"] = sum(1 for k in profile_fields if user.get(k))
    features["follower_to_following_ratio"] = ratio(followers, following + 1)
    features["public_repos_count"] = user.get("public_repos", 0) or 0
    features["is_hireable"] = 1 if user.get("hireable") else 0
    features["user_type_is_org"] = 1 if safe_text(user.get("type")).lower() == "organization" else 0
    features["public_gists_count"] = user.get("public_gists", 0) or 0
    features["followers_count"] = followers
    features["following_count"] = following
    features["user_login_entropy"] = calculate_entropy(owner_login)
    features["user_login_length"] = len(owner_login)
    features["has_twitter"] = 1 if user.get("twitter_username") else 0
    features["has_email"] = 1 if user.get("email") else 0
    u_name = safe_text(user.get("name"))
    features["user_name_matches_login"] = 1 if (not u_name or u_name.lower() == owner_login.lower()) else 0
    features["user_blog_matches_homepage_domain"] = 1 if (homepage_domain and user_blog_domain and homepage_domain == user_blog_domain) else 0
    features["owner_repo_density"] = ratio(features["public_repos_count"], ((features["user_age_days"] or 0) / 365.0) + 1.0)

    # 4) 内容与语义特征（README + description + topics + repo_name）
    features["repo_name_length"] = len(repo_name)
    features["readme_length"] = len(readme_text)
    features["desc_length"] = len(repo_description)
    features["topics_count"] = len(topics)
    features["repo_name_entropy"] = calculate_entropy(repo_name)
    features["readme_url_count"] = len(readme_urls)
    features["desc_url_count"] = len(desc_urls)
    features["readme_unique_domain_count"] = len(readme_domains)
    features["desc_unique_domain_count"] = len(desc_domains)
    features["homepage_domain_in_readme"] = 1 if (homepage_domain and homepage_domain in readme_domains) else 0
    features["homepage_domain_in_description"] = 1 if (homepage_domain and homepage_domain in desc_domains) else 0
    features["repo_name_has_blockchain_keyword"] = has_any_keyword(repo_name, BLOCKCHAIN_KEYWORDS)
    features["repo_name_has_brand_keyword"] = has_any_keyword(repo_name, BRANDISH_KEYWORDS)
    features["description_has_blockchain_keyword"] = has_any_keyword(repo_description, BLOCKCHAIN_KEYWORDS)
    features["description_has_risky_keyword"] = has_any_keyword(repo_description, RISKY_README_KEYWORDS)
    features["description_has_download_lure_keyword"] = has_any_keyword(repo_description, DOWNLOAD_LURE_KEYWORDS)
    features["topic_has_blockchain_keyword"] = has_any_keyword(topics_text, BLOCKCHAIN_KEYWORDS)
    features["blockchain_keyword_count"] = count_keyword_hits(text_all, BLOCKCHAIN_KEYWORDS)
    features["token_standard_keyword_count"] = count_keyword_hits(text_all, TOKEN_STD_KEYWORDS)
    features["wallet_keyword_count"] = count_keyword_hits(text_all, WALLET_KEYWORDS)
    features["risky_keyword_count"] = count_keyword_hits(text_all, RISKY_README_KEYWORDS)
    features["download_lure_keyword_count"] = count_keyword_hits(text_all, DOWNLOAD_LURE_KEYWORDS)
    features["security_trust_keyword_count"] = count_keyword_hits(text_all, SECURITY_TRUST_KEYWORDS)
    features["readme_has_contract_address"] = 1 if re.search(r"0x[a-fA-F0-9]{40}", readme_text) else 0
    features["description_has_contract_address"] = 1 if re.search(r"0x[a-fA-F0-9]{40}", repo_description) else 0
    features["readme_has_block_explorer_link"] = 1 if any(d in readme_text.lower() for d in EXPLORER_DOMAINS) else 0
    features["description_has_block_explorer_link"] = 1 if any(d in repo_description.lower() for d in EXPLORER_DOMAINS) else 0
    features["readme_has_badge"] = 1 if ("shields.io" in readme_text.lower() or "badge" in readme_text.lower()) else 0
    features["readme_has_install_run_commands"] = 1 if any(cmd in text_low for cmd in [
        "npm install", "yarn", "pnpm", "forge test", "forge build", "cargo build",
        "go test", "npm run", "bun install"
    ]) else 0
    features["readme_has_wallet_connect_keywords"] = has_any_keyword(text_all, WALLET_KEYWORDS)
    features["readme_has_sensitive_request_keywords"] = has_any_keyword(text_all, [
        "seed phrase", "mnemonic", "private key", "keystore", "recovery phrase"
    ])
    features["readme_has_airdrop_claim_keywords"] = has_any_keyword(text_all, [
        "airdrop", "claim", "claim airdrop", "claim reward"
    ])
    features["readme_has_approve_keywords"] = has_any_keyword(text_all, [
        "approve", "permit", "permit2", "sign message", "setapprovalforall"
    ])

    # 5) 文件结构与工程骨架特征
    features["total_files_count"] = len(paths)
    features["file_extension_diversity"] = len(set(extensions))
    features["top_level_dir_count"] = len(top_level_dirs)
    features["max_path_depth"] = max(path_depths) if path_depths else 0
    features["avg_path_depth"] = average(path_depths)
    features["hidden_file_count"] = count_hidden_files(paths)
    features["hidden_file_ratio"] = ratio(features["hidden_file_count"], len(paths) + 1)
    features["has_executable_files"] = 1 if executable_file_count > 0 else 0
    features["executable_file_count"] = executable_file_count
    features["has_archive_files"] = 1 if archive_file_count > 0 else 0
    features["archive_file_count"] = archive_file_count
    features["avg_file_size_bytes"] = average(file_sizes)
    features["max_file_size_bytes"] = max(file_sizes) if file_sizes else 0
    features["large_file_count_ge_1mb"] = sum(1 for s in file_sizes if s >= 1024 * 1024)
    features["has_readme_file"] = file_name_exists(paths, ["readme.md", "readme", "readme.txt"])
    features["has_env_file"] = 1 if env_file_count > 0 else 0
    features["env_file_count"] = env_file_count
    features["docs_related_file_count"] = docs_related_file_count
    features["risk_filename_count"] = risk_filename_count

    # 6) 外部链接与域名特征
    features["external_domains_count"] = len(ext_domains)
    features["external_domains_average_age"] = average(valid_domain_ages)
    features["external_domains_with_age_count"] = len(valid_domain_ages)
    features["explorer_domain_count"] = explorer_domain_count
    features["homepage_domain_in_external_domains"] = 1 if (homepage_domain and homepage_domain in ext_domains) else 0
    features["user_blog_domain_in_external_domains"] = 1 if (user_blog_domain and user_blog_domain in ext_domains) else 0

    # 7) 区块链专项特征（仅用当前 JSON 中真正可提取的信息）
    primary_lang = safe_text(repo.get("language")).lower()
    features["primary_language_is_solidity"] = 1 if primary_lang == "solidity" else 0
    features["primary_language_is_go"] = 1 if primary_lang == "go" else 0
    features["primary_language_is_rust"] = 1 if primary_lang == "rust" else 0
    features["primary_language_is_javascript"] = 1 if primary_lang == "javascript" else 0
    features["primary_language_is_typescript"] = 1 if primary_lang == "typescript" else 0
    features["primary_language_is_python"] = 1 if primary_lang == "python" else 0

    features["solidity_file_count"] = solidity_file_count
    features["js_ts_file_count"] = js_ts_file_count
    features["python_file_count"] = python_file_count
    features["go_file_count"] = go_file_count
    features["rust_file_count"] = rust_file_count
    features["move_file_count"] = move_file_count
    features["vyper_file_count"] = vyper_file_count
    features["contract_file_ratio"] = ratio(solidity_file_count, len(paths) + 1)

    features["has_contracts_dir"] = path_contains_prefix(paths, ["contracts", "src/contracts"])
    features["has_test_dir"] = path_contains_prefix(paths, ["test", "tests", "spec", "specs"])
    features["has_deploy_dir"] = path_contains_prefix(paths, ["deploy", "deployments", "scripts"])
    features["has_migrations_dir"] = path_contains_prefix(paths, ["migrations"])
    features["has_ci_workflows"] = path_contains_prefix(paths, [".github/workflows"])
    features["has_frontend_dir"] = path_contains_prefix(paths, ["frontend", "web", "app", "src", "pages", "public"])
    features["has_docs_dir"] = path_contains_prefix(paths, ["docs", "doc"])

    features["has_package_json"] = file_name_exists(paths, ["package.json"])
    features["has_yarn_lock"] = file_name_exists(paths, ["yarn.lock"])
    features["has_package_lock_json"] = file_name_exists(paths, ["package-lock.json"])
    features["has_pnpm_lock"] = file_name_exists(paths, ["pnpm-lock.yaml"])
    features["has_foundry_toml"] = file_name_exists(paths, ["foundry.toml"])
    features["has_hardhat_config"] = file_name_exists(paths, [
        "hardhat.config.js", "hardhat.config.ts", "hardhat.config.cjs", "hardhat.config.mjs"
    ])
    features["has_truffle_config"] = file_name_exists(paths, ["truffle-config.js", "truffle.js"])
    features["has_brownie_config"] = file_name_exists(paths, ["brownie-config.yaml", "brownie-config.yml"])
    features["has_remix_config"] = file_name_exists(paths, ["remix.config.js", "remix.config.ts"])
    features["has_cargo_toml"] = file_name_exists(paths, ["cargo.toml"])
    features["has_go_mod"] = file_name_exists(paths, ["go.mod"])

    features["test_related_file_count"] = test_related_file_count
    features["deploy_related_file_count"] = deploy_related_file_count
    features["contract_test_ratio"] = ratio(test_related_file_count, solidity_file_count + 1)
    features["framework_hint_count"] = sum([
        features["has_foundry_toml"],
        features["has_hardhat_config"],
        features["has_truffle_config"],
        features["has_brownie_config"],
        features["has_remix_config"],
    ])
    features["frontend_contract_combo_flag"] = 1 if (
        features["has_frontend_dir"] and (features["has_contracts_dir"] or solidity_file_count > 0)
    ) else 0
    features["wallet_frontend_risk_combo_flag"] = 1 if (
        features["has_frontend_dir"] and features["readme_has_wallet_connect_keywords"] and features["risky_keyword_count"] > 0
    ) else 0
    features["binary_download_lure_combo_flag"] = 1 if (
        executable_file_count > 0 and features["download_lure_keyword_count"] > 0
    ) else 0

    cleaned = {
        "readme_text": readme_text,
        "description_text": repo_description,
        "topics_text": topics_text,
        "combined_text": "\n".join([repo_name, repo_description, topics_text, readme_text]).strip(),
        "features": features
    }

    for extra_key in ["label", "source", "family", "family_id", "is_malicious"]:
        if extra_key in data:
            cleaned[extra_key] = data[extra_key]

    return cleaned


# =========================
# 文件夹解析与保存
# =========================

def parse_github_folder_to_json(input_folder, skip_filenames=None):
    all_records = []
    skip_filenames = set(skip_filenames or [])

    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue
        if file in skip_filenames:
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
                    record = parse_single_record(item, raw_file_name=file)
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
    skip_filenames = {os.path.basename(output_path), os.path.basename(output_path).replace(".json", ".jsonl")}
    records = parse_github_folder_to_json(input_folder, skip_filenames=skip_filenames)

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
    # 默认处理“脚本所在目录”下的所有原始 JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = r"C:\Users\Dell\Desktop\Grade4\毕业设计\代码\Github_crawler_direct_plus\.idea\data\active_repos"
    output_folder = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\XGBoost_LLM_TF_IDF\.idea\新特征工程表"

    # 你也可以改成：output_format = "jsonl"
    output_format = "json"
    output_name = "positive_github_dataset_cleaned.json" if output_format == "json" else "github_dataset_cleaned.jsonl"
    output_path = os.path.join(output_folder, output_name)

    main(input_folder, output_path, output_format)

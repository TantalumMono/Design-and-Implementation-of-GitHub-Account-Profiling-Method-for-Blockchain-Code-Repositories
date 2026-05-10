import requests
import time
import base64
import re
import whois
import json
import os
import logging
import random
from datetime import datetime,timezone,timedelta
from requests.exceptions import SSLError, ConnectionError, Timeout

TOKEN = "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"

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "collector.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

BLOCKCHAIN_KEYWORDS = [
    "blockchain",
    "smart contract",
    "wallet",
    "crypto",
    "web3",
    "ethereum",
    "bitcoin",
    "solidity",
    "defi",
    "nft"
]

BLOCKCHAIN_LANGUAGES = [
    "Solidity",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Python"
]

def github_get(url, params=None):
    headers = {"Authorization": f"token {TOKEN}"}
    while True:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 200:
            logging.info(f"GitHub API success: {url}")
            return r.json()
        elif r.status_code == 403:
            logging.warning("Rate limit reached, sleeping 60s...")
            time.sleep(60)
        else:
            logging.error(f"GitHub API error {r.status_code}: {r.text}")
            return None



def search_normal_repos(
        min_stars=10,
        pages=100,
        per_page=5,
        # days_active=365,
        sample_size=200
):

    logging.info("Start searching NORMAL blockchain-related repositories")

    all_repos = []

    # 活跃时间阈值
    # since_date = (datetime.now(timezone.utc) - timedelta(days=days_active)).strftime("%Y-%m-%d")

    for page in range(1, pages + 1):
        keywords = random.sample(BLOCKCHAIN_KEYWORDS, 2)
        language = random.choice(BLOCKCHAIN_LANGUAGES)

        logging.info(f"Fetching page {page}, keywords={keywords}, language={language}")

        url = "https://api.github.com/search/repositories"

        # 构造搜索条件
        query_parts = [
            f'"{keywords[0]}"',
            f'"{keywords[1]}"',
            f"language:{language}"
            f"stars:>={min_stars}",
            # f"pushed:>={since_date}"
        ]

        params = {
            "q": " ".join(query_parts),
            "per_page": per_page,
            "page": page
        }

        data = github_get(url, params)
        if not data or "items" not in data:
            continue

        all_repos.extend(data["items"])
        time.sleep(1)

    logging.info(f"Total candidate normal repos: {len(all_repos)}")

    # 随机化，消除 GitHub 默认排序影响
    random.shuffle(all_repos)

    # 若指定采样数量
    if sample_size and len(all_repos) > sample_size:
        all_repos = all_repos[:sample_size]
        logging.info(f"Randomly sampled {sample_size} normal repos")

    return all_repos


def get_repo_detail(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    return github_get(url)


def get_user_detail(username):
    url = f"https://api.github.com/users/{username}"
    return github_get(url)


def get_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    data = github_get(url)
    if data and "content" in data:
        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return ""


def get_tree(owner, repo, path=""):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    items = github_get(url)
    if not items or isinstance(items, dict):
        return []

    structure = []
    for item in items:
        structure.append(item)
        if item["type"] == "dir":
            structure.extend(get_tree(owner, repo, item["path"]))
    return structure


def extract_domains(text):
    urls = re.findall(r'https?://[^\s)]+', text)
    domains = [u.split("/")[2] for u in urls if "/" in u]
    return list(set(domains))


def normalize_datetime(dt):
    if dt is None:
        return None

    # 有些 whois 返回 list
    if isinstance(dt, list):
        dt = dt[0]

    if not isinstance(dt, datetime):
        return None

    # naive → aware
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    # aware → 统一 UTC
    return dt.astimezone(timezone.utc)


def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation = normalize_datetime(w.creation_date)
        if not creation:
            return None

        now = datetime.now(timezone.utc)
        return (now - creation).days

    except Exception as e:
        logging.warning(f"Whois lookup failed for domain {domain}: {e}")
        return None

def collect_repo_all_info(repo):
    owner = repo["owner"]["login"]
    name = repo["name"]

    logging.info(f"Collecting repository: {owner}/{name}")

    detail = get_repo_detail(owner, name)
    user = get_user_detail(owner)
    readme = get_readme(owner, name)
    tree = get_tree(owner, name)

    domains = extract_domains(readme)
    logging.info(f"{owner}/{name} extracted {len(domains)} external domains")

    domain_ages = {}
    for d in domains:
        age = get_domain_age(d)
        domain_ages[d] = age
        logging.info(f"Domain {d}, age: {age}")

    return {
        "repo_full_name": repo["full_name"],
        "repo_detail": detail,
        "user_detail": user,
        "readme": readme,
        "tree_structure": tree,
        "external_domains": domain_ages,
        "collected_at": datetime.now().isoformat()
    }


def save_repo_data(data):
    full_name = data["repo_full_name"]
    filename = full_name.replace("/", "__") + ".json"
    path = os.path.join(DATA_DIR, filename)
    logging.info(f"Saving repository: {data.get('repo_full_name')}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[✓] Saved to {path}")
    logging.info(f"Repository data saved: {path}")


def load_collected_repos(data_dir):
    """
    从 data 目录中加载已采集的仓库 full_name 集合
    """
    collected = set()
    for fname in os.listdir(data_dir):
        if fname.endswith(".json") and "__" in fname:
            full_name = fname.replace("__", "/").replace(".json", "")
            collected.add(full_name)
    return collected


if __name__ == "__main__":
    logging.info("==== Data collection started ====")

    collected_repos = load_collected_repos(DATA_DIR)
    repos = search_normal_repos()

    for repo in repos:
        full_name = repo["full_name"]
        if full_name in collected_repos:
            logging.info(f"[SKIP] {full_name} already collected")
            continue
        try:
            info = collect_repo_all_info(repo)
            save_repo_data(info)
            collected_repos.add(full_name)  # 防止本轮重复
            time.sleep(2)
        except Exception as e:
            logging.error(
                f"Failed on repository {repo.get('full_name', 'unknown')}: {e}"
            )

    logging.info("==== Data collection finished ====")

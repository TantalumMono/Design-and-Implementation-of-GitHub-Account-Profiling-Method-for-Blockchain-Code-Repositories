import requests
import time
import base64
import re
import whois
import json
import os
import logging
from bs4 import BeautifulSoup
from datetime import datetime,timezone,timedelta
from urllib.parse import urlparse

# =========================
# 基础配置
# =========================

TOKEN = "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"
WAYBACK_API = "https://archive.org/wayback/available"

ACTIVE_DIR = "./data/active_repos"
WAYBACK_DIR = "./data/deleted_wayback"

os.makedirs(ACTIVE_DIR, exist_ok=True)
os.makedirs(WAYBACK_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =========================
# GitHub API 请求
# =========================

def github_get(url, params=None, retry=True):
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)

        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", 0))
            sleep_time = max(reset - int(time.time()), 0)
            sleep_time = min(sleep_time, 60)
            logging.warning(f"Rate limit hit. Sleeping {sleep_time}s")
            time.sleep(sleep_time)
            if retry:
                return github_get(url, params, retry=False)

        return {
            "status": r.status_code,
            "data": r.json() if r.status_code == 200 else None
        }

    except Exception as e:
        logging.error(f"GitHub request failed: {url} | {e}")
        return {"status": "error", "data": None}


def get_repo_detail(owner, repo):
    return github_get(f"https://api.github.com/repos/{owner}/{repo}")


def get_user_detail(username):
    return github_get(f"https://api.github.com/users/{username}")


def get_readme(owner, repo):
    resp = github_get(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if resp["status"] == 200 and resp["data"]:
        content = resp["data"].get("content")
        if content:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
    return None


# =========================
# README 域名分析
# =========================

def extract_domains(text):
    urls = re.findall(r'https?://[^\s)]+', text)
    domains = set()

    # 提取域名
    for u in urls:
        try:
            parts = u.split("/")
            if len(parts) > 2:
                domains.add(parts[2])
        except:
            continue

    # 查询年龄
    domains_age = {}
    for domain in domains:
        age = get_domain_age(domain)
        if age is not None:
            domains_age[domain] = age
        else:
            domains_age[domain] = 0

    return domains_age


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

# def extract_domains(text):
#     urls = re.findall(r'https?://[^\s)]+', text)
#     domains = set()
#     for u in urls:
#         try:
#             parsed = urlparse(u)
#             domains.add(parsed.netloc)
#         except:
#             continue
#     return list(domains)
#
#
# def analyze_readme_domains(readme_text):
#     if not readme_text:
#         return []
#
#     domains = extract_domains(readme_text)
#     return [{"domain": d} for d in domains]

def get_tree(owner, repo, path=""):

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = github_get(url)

    if resp["status"] != 200 or not resp["data"]:
        return []

    items = resp["data"]

    if isinstance(items, dict):
        return []

    structure = []

    for item in items:
        structure.append(item)
        if item.get("type") == "dir":
            structure.extend(get_tree(owner, repo, item["path"]))

    return structure

# =========================
# Wayback 查询
# =========================

def check_wayback_snapshot(repo_url):
    try:
        r = requests.get(WAYBACK_API, params={"url": repo_url}, timeout=10)
        data = r.json()
        closest = data.get("archived_snapshots", {}).get("closest")

        if closest and closest.get("available"):
            return closest.get("url")
    except Exception as e:
        logging.error(f"Wayback query failed: {e}")

    return None


# =========================
# Wayback 页面解析
# =========================

def extract_metric(soup, pattern):
    tag = soup.find('a', href=re.compile(pattern))
    if tag:
        txt = tag.get_text(strip=True).lower()
        match = re.search(r'([\d,.]+)', txt)
        if match:
            val = float(match.group(1).replace(',', ''))
            return int(val * 1000) if 'k' in txt else int(val)
    return 0


def parse_wayback_page(snapshot_url):
    try:
        r = requests.get(snapshot_url, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")

        owner_tag = soup.select_one("span.author a, a[rel='author']")
        repo_tag = soup.select_one(
            'strong[itemprop="name"] a, a[data-pjax="#repo-content-pjax-container"]'
        )

        owner = owner_tag.get_text(strip=True) if owner_tag else "Unknown"
        repo = repo_tag.get_text(strip=True) if repo_tag else "Unknown"

        readme_box = soup.find("article", {"class": "markdown-body"})
        readme_text = readme_box.get_text("\n") if readme_box else ""

        snapshot_time_match = re.search(r'/web/(\d+)/', snapshot_url)
        snapshot_time = snapshot_time_match.group(1) if snapshot_time_match else None

        return {
            "repo_full_name": f"{owner}/{repo}",
            "snapshot_url": snapshot_url,
            "snapshot_time": snapshot_time,
            "stars": extract_metric(soup, r'/stargazers'),
            "forks": extract_metric(soup, r'/network/members|/forks'),
            "watches": extract_metric(soup, r'/watchers'),
            "readme": readme_text,
            "readme_domains": extract_domains(readme_text)
        }

    except Exception as e:
        logging.error(f"Wayback parse failed: {e}")
        return None


# =========================
# 主采集逻辑
# =========================

def collect_repo_all_info(owner, repo):

    full_name = f"{owner}/{repo}"
    logging.info(f"Processing {full_name}")

    repo_resp = get_repo_detail(owner, repo)

    # =========================
    # 1️⃣ 仓库存在
    # =========================
    if repo_resp["status"] == 200:

        repo_detail = repo_resp["data"]
        user_resp = get_user_detail(owner)

        readme_text = get_readme(owner, repo)

        tree_structure = get_tree(owner, repo)
        return {
            "repo_full_name": full_name,
            "repo_deleted": False,
            "data_source": "github_api",
            "collected_at": datetime.utcnow().isoformat(),
            "repo_detail": repo_detail,
            "user_detail": user_resp["data"] if user_resp["status"] == 200 else None,
            "readme": readme_text,
            "readme_domains": extract_domains(readme_text),
            "tree_structure": tree_structure
        }

    # =========================
    # 2️⃣ 仓库不可访问 → 查快照
    # =========================
    if repo_resp["status"] in [404, 403]:

        logging.warning(f"{full_name} inaccessible. Checking Wayback...")

        repo_url = f"https://github.com/{owner}/{repo}"
        snapshot_url = check_wayback_snapshot(repo_url)

        if snapshot_url:
            snapshot_data = parse_wayback_page(snapshot_url)

            if not snapshot_data:
                return None

            # ⭐⭐⭐ 关键修改：即使 Wayback 有快照，也调用 GitHub API 查账户
            user_resp = get_user_detail(owner)

            if user_resp["status"] == 200:
                user_detail = user_resp["data"]
                owner_status = "active"
            else:
                user_detail = None
                owner_status = "deleted_or_banned"

            return {
                "repo_full_name": full_name,
                "repo_deleted": True,
                "data_source": "wayback_snapshot",
                "collection_timestamp": datetime.utcnow().isoformat(),
                "repo_detail": snapshot_data,
                "user_detail": user_detail,
                "owner_current_status": owner_status
            }

        else:
            logging.warning("No Wayback snapshot found. Dropped.")
            return None

    return None


# =========================
# 保存函数
# =========================

def save_repo_data(data):

    filename = data["repo_full_name"].replace("/", "__") + ".json"

    if data["data_source"] == "github_api":
        save_path = os.path.join(ACTIVE_DIR, filename)
    else:
        save_path = os.path.join(WAYBACK_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved → {save_path}")


# =========================
# 主入口
# =========================

# =========================
# 主入口
# =========================

if __name__ == "__main__":

    with open("repo_list.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    total_count = len(urls)

    api_success = []
    wayback_success = []
    dropped = []

    for url in urls:

        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
        if not match:
            continue

        owner, repo = match.group(1), match.group(2)

        data = collect_repo_all_info(owner, repo)

        if data:
            save_repo_data(data)

            if data["data_source"] == "github_api":
                api_success.append(data["repo_full_name"])
            elif data["data_source"] == "wayback_snapshot":
                wayback_success.append(data["repo_full_name"])

            time.sleep(2)
        else:
            dropped.append(f"{owner}/{repo}")
            logging.info(f"[DROP] {owner}/{repo} no usable data.")

    # =========================
    # 统计回显
    # =========================

    print("\n======================")
    print("采集完成统计报告")
    print("======================")
    print(f"仓库总数: {total_count}")
    print(f"成功采集总数: {len(api_success) + len(wayback_success)}")
    print(f" - GitHub API 成功: {len(api_success)}")
    print(f" - Wayback 快照成功: {len(wayback_success)}")
    print(f"丢弃数量: {len(dropped)}")

    print("\nGitHub API 成功仓库:")
    for r in api_success:
        print("  ✔", r)

    print("\nWayback 成功仓库:")
    for r in wayback_success:
        print("  ✔", r)

    print("\n被丢弃仓库:")
    for r in dropped:
        print("  ✘", r)

    print("======================\n")
# 负样本扩充报告
## 1. 输入与目标
- 输入：45 条清洗后的原始负样本（正常样本）。
- 目标：扩充至约 300 条，并尽量让各 family 数量接近。
- 约束：遵循上传的特征工程文档中关于时间/活跃度、Owner 画像、文本语义、文件结构、外链域名和区块链专项特征的定义；README、description、topics、combined_text 作为文本字段保留。

## 2. Family 划分
### F1_pumpfun_clone
- 原始数：24
- 代表仓库：2723799947qq2022/solana-pumpfun-bot, 2kwkkk/solana-pumpfun-bot, 790659193qqch/solana-pumpfun-bot, 7arlystar/solana-pumpfun-bot, 918715c83/solana-pumpfun-bot, AmirhBeigi7zch6f/solana-pumpfun-bot
### F2_crypto_trading_bot
- 原始数：4
- 代表仓库：dev-protocol/polymarket-copytrading-bot-sport, FaceOFWood/SniperBot-Solana-PumpSwap, harshith-vaddiparthy/quant-bot, Morning-Star213/Solana-pumpfun-bot
### F3_web3_platform_app
- 原始数：8
- 代表仓库：arliawhite/rentverse, clixlogix/blockchain-dev, knightsdex/knightsbridge-dex, MentarisHub121/TokenPresaleApp, mirzamudassir/blocknovas-nyx-public, SuperDev313/Trading_Platform_Ultrax
### F4_security_osint_cve
- 原始数：5
- 代表仓库：adminlove520/VulnWatchDog, nocomp/poc-CVE-2001-1473, themaxlpalfaboy/CVE-2025-54897-LAB, Zeeeepa/spyder-osint, Zeeeepa/spyder-osint2
### F5_ai_media_demo
- 原始数：4
- 代表仓库：eferos93/test4, Metaldadisbad/HacxGPT, rizvejoarder/SoraMax, shivas1432/sora2-watermark-remover

## 3. 扩充策略
- 采用 family 内 bootstrap near-neighbor 扩充：每个 synthetic 样本都从同 family 的原始样本出发，继承大部分结构化特征骨架，只对少量数值特征做小幅扰动，并重算派生比例特征。
- 文本采用“近邻改写”策略：F1（大模板克隆族）直接保留原始 README/description，以避免引入文本风格偏差；其他 family 仅做非常轻量的措辞改写，尽量保留 section、URL、关键词、项目骨架和主题。
- 为避免训练时把 family/source 当成标签泄漏，family/source 信息仅放入带 meta 的版本中；strict 版本保持原始 schema。

## 4. 数量分布
- F1_pumpfun_clone: 原始 24，生成 36，合计 60
- F2_crypto_trading_bot: 原始 4，生成 56，合计 60
- F3_web3_platform_app: 原始 8，生成 52，合计 60
- F4_security_osint_cve: 原始 5，生成 55，合计 60
- F5_ai_media_demo: 原始 4，生成 56，合计 60

## 5. 生成偏置检查
- 结构化特征区分 original / synthetic 的 5 折 ROC-AUC：0.546
- 文本特征区分 original / synthetic 的 5 折 ROC-AUC：0.578
- 解释：AUC 越接近 0.5，说明 synthetic 与 original 越不容易被机器轻易区分。当前两项均接近 0.5，说明整体生成偏置较弱。

### 5.1 文本族内质心相似度（TF-IDF cosine）
- F1_pumpfun_clone: 1.0000
- F2_crypto_trading_bot: 0.9965
- F3_web3_platform_app: 0.9956
- F4_security_osint_cve: 0.9922
- F5_ai_media_demo: 0.9875

### 5.2 数值特征最大漂移（abs SMD Top 10）
| family                | feature              |   orig_mean |   syn_mean |   abs_smd |
|:----------------------|:---------------------|------------:|-----------:|----------:|
| F5_ai_media_demo      | user_login_length    |      10.75  |    11.4821 | 0.334684  |
| F5_ai_media_demo      | repo_name_length     |      10.5   |    11.9464 | 0.187542  |
| F4_security_osint_cve | user_login_length    |       9.4   |    10.0545 | 0.173279  |
| F4_security_osint_cve | desc_length          |      25.6   |    27.4182 | 0.117742  |
| F3_web3_platform_app  | user_login_length    |      10.125 |    10.4038 | 0.116933  |
| F1_pumpfun_clone      | repo_size_kb         |   12816.2   | 12515.9    | 0.108478  |
| F5_ai_media_demo      | days_since_last_push |     119.75  |   121.25   | 0.0912437 |
| F1_pumpfun_clone      | user_login_length    |      10.25  |    10.5556 | 0.076386  |
| F1_pumpfun_clone      | readme_length        |    6100.04  |  6100.06   | 0.0635147 |
| F3_web3_platform_app  | days_since_last_push |     293.125 |   279.038  | 0.0631961 |

### 5.3 布尔特征最大比例差（Top 10）
| family               | feature                         |   orig_rate |   syn_rate |   abs_diff |
|:---------------------|:--------------------------------|------------:|-----------:|-----------:|
| F3_web3_platform_app | has_license_meta                |   0.375     |  0.346154  | 0.0288462  |
| F3_web3_platform_app | has_license_file                |   0.5       |  0.480769  | 0.0192308  |
| F3_web3_platform_app | has_test_dir                    |   0.25      |  0.230769  | 0.0192308  |
| F3_web3_platform_app | user_type_is_org                |   0.25      |  0.269231  | 0.0192308  |
| F3_web3_platform_app | primary_language_is_javascript  |   0.25      |  0.230769  | 0.0192308  |
| F1_pumpfun_clone     | has_homepage                    |   0.0416667 |  0.0555556 | 0.0138889  |
| F3_web3_platform_app | primary_language_is_typescript  |   0.375     |  0.384615  | 0.00961538 |
| F3_web3_platform_app | has_homepage                    |   0.125     |  0.115385  | 0.00961538 |
| F3_web3_platform_app | repo_is_fork                    |   0.125     |  0.134615  | 0.00961538 |
| F3_web3_platform_app | readme_has_install_run_commands |   0.875     |  0.884615  | 0.00961538 |

## 6. 输出文件说明
- `negative_augmented_300_strict.json`：仅保留原始 schema，可直接接入现有训练脚本。
- `negative_augmented_300_with_meta.json`：额外包含 `meta`，便于 family 级分析和溯源。
- `negative_family_mapping.csv`：每条样本的 family、synthetic 标记与来源映射。
- `negative_augmentation_report.md`：本报告。

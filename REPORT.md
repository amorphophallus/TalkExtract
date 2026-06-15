# Bilibili 视频访谈提取 — 方法报告

## 项目概述

**目标**: 从 Bilibili 视频 [BV1nioTBcEvj](https://www.bilibili.com/video/BV1nioTBcEvj/)（晚点聊 LateTalk — 唐文斌访谈）提取完整文本转录，带说话人标签。

**日期**: 2026-06-14 ~ 2026-06-15  
**硬件**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM), CUDA 12.0  
**视频时长**: 126 分钟 | 输出: ~46,000 中文字符 | 5,650 个段落

---

## 技术方案

### 整体架构

```
Bilibili 视频 URL
    │
    ├─ bilibili-api-python → 提取 DASH 音频流 (m4a, 146kbps AAC)
    │
    ├─ faster-whisper large-v3 (GPU float16) → 语音转文字
    │   · 5,650 segments, 语言检测: zh (概率 1.00)
    │   · 耗时 ~5 分钟 (RTX 4070)
    │
    └─ pyannote.audio 4.0 (speaker-diarization-3.1) → 说话人分离
        · 1,427 个说话人片段, 识别出 2 位说话人
        · 耗时 ~10 分钟
        │
        └─ 合并 → SRT + 对话文本
```

### 工具选型对比

| 候选方案 | 优点 | 缺点 | 结论 |
|----------|------|------|------|
| Bilibili AI 字幕 | 零成本, 最快 | 此视频无 AI 字幕 | ❌ 不可用 |
| BibiGPT / 在线工具 | 无需本地搭建 | 免费额度有限, 无说话人分离 | ❌ 不满足需求 |
| yt-dlp + Whisper | 成熟生态 | Bilibili 反爬 412 错误, 无法下载 | ❌ 受阻 |
| bilibili-api + faster-whisper + pyannote | 全本地, 可定制, GPU 加速 | 配置复杂 | ✅ 最终采用 |

### 关键问题与解决

1. **Bilibili 下载受阻**: yt-dlp 遭遇 HTTP 412，API 需要 wbi 签名 → 改用 `bilibili-api-python` 直接调用 API 提取 DASH 音频 URL
2. **CUDA 版本不匹配**: 系统 CUDA 12.0 / ctranslate2 需要 CUDA 12，但 PyTorch 默认装了 CUDA 13 → 补充安装 `nvidia-cublas-cu12`，设置 `LD_LIBRARY_PATH`
3. **HuggingFace 国内不可达**: 模型下载超时 → 配置 `HF_ENDPOINT=https://hf-mirror.com`
4. **pyannote 门控仓库**: pyannote 的三个子模型 (speaker-diarization-3.1, segmentation-3.0, speaker-diarization-community-1) 需要分别接受使用条款
5. **HF_TOKEN 在非交互 Shell 不可用**: bashrc 有 `[ -z "$PS1" ] && return` 保护 → 写 shell wrapper 用 grep 直接从 bashrc 提取 token
6. **pyannote 4.x API 变更**: `use_auth_token` → `token`; 返回值 `Annotation` → `DiarizeOutput` → 适配新 API

### 本地模型

| 模型 | 大小 | 用途 |
|------|------|------|
| Systran/faster-whisper-large-v3 | ~2.9 GB | 中文语音转文字 |
| pyannote/speaker-diarization-3.1 | ~1.5 GB | 说话人分离 |

---

## 输出文件

`/home/huyue/projects/talk-extract/`

| 文件 | 说明 |
|------|------|
| `晚点聊LateTalk_labeled.txt` | 带 [主持人]/[唐文斌] 标签的对话文本（推荐阅读） |
| `晚点聊LateTalk.srt` | 带时间戳和说话人标签的字幕文件 |
| `晚点聊LateTalk_raw.txt` | 无标签纯文本，适合全文搜索 |

---

## 说话人统计

| 说话人 | 发言段数 | 总时长 | 推定身份 |
|--------|---------|--------|---------|
| SPEAKER_01 | 560 | 19.9 min | 主持人 (曼琪) |
| SPEAKER_00 | 867 | 99.4 min | 嘉宾 (唐文斌) |

---

## 可复现性

所有核心脚本：

- `transcribe.py` — 完整流程 (whisper + pyannote)，自动处理 CUDA 库路径
- `diarize.py` — 仅 diarization (基于已保存的 whisper 结果)
- `download_audio.py` — Bilibili 音频下载
- `run_diarize.sh` — 非交互 Shell 启动器 (处理 bashrc token 提取)

Python 环境: conda `talk-extract`, Python 3.11。依赖清单见 `requirements.txt`。

要重新运行, 确保 `~/.bashrc` 中设置了正确的 `HF_TOKEN`, 然后执行 `bash run_diarize.sh`。

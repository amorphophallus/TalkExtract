# TalkExtract

> 让每一段对话，都能被 AI 读懂。

TalkExtract 是一个视频访谈/播客文本提取工具。支持 Bilibili、YouTube，将没有字幕的视频自动转写成带说话人标签的结构化文本，使 NotebookLM、ChatGPT 等 AI 工具可以直接引用和学习。

## 为什么需要这个工具？

**NotebookLM** 是目前最强的长文 AI 学习工具之一，但它只接受有字幕的视频作为 source，没有字幕的优质访谈、播客、讲座无法被 AI 索引和总结。

TalkExtract 解决的就是这个问题：**自动下载音频 → GPU 语音识别 → 说话人分离 → 输出结构化对话文本**，让每一段有价值的对话都能成为 AI 知识库的原料。

---

## Demo: 唐文斌访谈

**[晚点聊 LateTalk] 从旷视到原力灵机，唐文斌的第二次创业**

| | |
|---|---|
| 来源 | [Bilibili BV1nioTBcEvj](https://www.bilibili.com/video/BV1nioTBcEvj/) |
| 时长 | 126 分钟 |
| 语言 | 中文 |
| 说话人 | 2 人（主持人曼琪 + 唐文斌） |
| 处理耗时 | ~15 分钟 (RTX 4070 SUPER) |

**输出效果**：

```
[主持人] 你们最多的时候是不是有十几个IOI的金牌

[唐文斌] 我们NOI的金牌更多

[主持人] 当时范浩强作为高中生是怎么跑到你们这上班的

[唐文斌] 反正这也是个有趣故事
      ...
[唐文斌] 大家都想去解决那个本质的问题
[唐文斌] 但实际上当你真正变成一个商业化的产品的时候
[唐文斌] 那你所有对客户有影响的东西都是本质的对吧
```

**完整输出文件**：[`晚点聊LateTalk_labeled.txt`](./从旷视到原力灵机_唐文斌的第二次创业_晚点聊LateTalk_labeled.txt)（~46,000 字）

---

## 特性

- 🎬 **多平台支持**: Bilibili、YouTube
- 🎙️ **说话人分离**: 自动识别并标注每位发言人
- 🚀 **GPU 加速**: 支持 NVIDIA GPU (RTX 系列)，2 小时音频约 15 分钟处理
- 📝 **多种输出**: 带标签对话文本、SRT 字幕（带时间戳）、纯文本
- 🧠 **AI 友好**: 输出可直接导入 NotebookLM、ChatGPT、Claude 作为 source
- 💻 **全本地运行**: 数据不出本地，隐私安全

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/talkextract.git
cd talkextract

# 2. 创建 conda 环境
conda create -n talkextract python=3.11 -y
conda activate talkextract

# 3. 安装依赖（使用国内镜像加速）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install faster-whisper pyannote.audio yt-dlp bilibili-api-python nvidia-cublas-cu12 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装 ffmpeg
conda install -c conda-forge ffmpeg -y

# 5. 设置 HuggingFace Token（pyannote 说话人分离需要）
#    注册 https://huggingface.co 并创建 token，然后：
export HF_TOKEN="hf_..."
#    并逐一接受以下模型的用户条款：
#    - https://huggingface.co/pyannote/speaker-diarization-3.1
#    - https://huggingface.co/pyannote/segmentation-3.0
#    - https://huggingface.co/pyannote/speaker-diarization-community-1

# 6. (国内用户) 设置 HuggingFace 镜像
export HF_ENDPOINT="https://hf-mirror.com"
```

## 使用

```bash
# 完整流程：下载 + 转写 + 说话人分离
python transcribe.py "https://www.bilibili.com/video/BV1nioTBcEvj/"

# 或者分步执行
python download_audio.py "https://www.youtube.com/watch?v=VIDEO_ID"  # 下载音频
python transcribe.py audio.m4a                                        # 转写 + 分离

# 如果已有 whisper 结果，可单独运行 diarization
python diarize.py audio.m4a whisper_segments.json
```

## 工作原理

```
视频 URL
  │
  ├─ [下载] yt-dlp / bilibili-api → 提取音频流 (m4a)
  │
  ├─ [转写] faster-whisper large-v3 (GPU float16) 
  │    PyTorch CTranslate2 后端, 比原版 Whisper 快 4 倍
  │    支持 VAD (语音活动检测), 自动跳过静音
  │
  ├─ [分离] pyannote.audio speaker-diarization-3.1
  │    深度学习声纹模型, 识别每位说话人
  │
  └─ [合并] 对齐时间戳 → 输出带标签文本 + SRT 字幕
```

## 硬件要求

| GPU | 2 小时音频耗时 | 模型 |
|-----|-------------|------|
| RTX 4070 SUPER (12GB) | ~15 分钟 | large-v3 + pyannote |
| RTX 3060 (12GB) | ~20 分钟 | large-v3 + pyannote |
| RTX 2060 (6GB) | ~25 分钟 | medium + pyannote |
| CPU only | ~1-2 小时 | small (int8) + pyannote |

纯 CPU 也可运行，但需将 `transcribe.py` 中的 `DEVICE` 改为 `"cpu"`，`COMPUTE_TYPE` 改为 `"int8"`。

## 项目结构

```
talkextract/
├── transcribe.py          # 核心转写脚本 (whisper + pyannote)
├── diarize.py             # 独立说话人分离脚本
├── download_audio.py      # 视频音频下载脚本
├── run.sh                 # 非交互 Shell 启动器
├── run_diarize.sh         # Diarization 启动器
├── requirements.txt       # Python 依赖
├── REPORT.md              # 技术报告与踩坑记录
└── demo/                  # Demo 输出文件
    ├── 晚点聊LateTalk_labeled.txt
    ├── 晚点聊LateTalk.srt
    └── 晚点聊LateTalk_raw.txt
```

## 常见问题

**Q: 为什么不用 yt-dlp 直接下载？**
Bilibili 加强了反爬 (wbi 签名)，yt-dlp 常返回 HTTP 412。本工具对 Bilibili 使用 `bilibili-api-python` 作为备选。

**Q: HuggingFace 模型下载太慢？**
设置国内镜像：`export HF_ENDPOINT="https://hf-mirror.com"`

**Q: 说话人标签是随机的 (SPEAKER_00, SPEAKER_01)？**
是的，pyannote 无法识别身份。需要根据对话内容手动推断（如 Demo 中根据发言时长推断 SPEAKER_00 为唐文斌）。

**Q: pip 安装太慢？**
使用清华镜像：`pip install ... -i https://pypi.tuna.tsinghua.edu.cn/simple`

## License

MIT

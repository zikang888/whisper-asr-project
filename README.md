# Whisper ASR Training Project

一个完整的 Whisper 语音识别模型训练项目，基于 LibriSpeech 数据集。

## 📋 项目简介

这个项目提供了一个完整的 Whisper 模型训练流程，包括：
- 快速验证（小规模训练）
- 完整训练（大规模数据集）
- 模型评估（WER/CER 指标）
- 训练报告生成

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/zikang888/whisper-asr-training-report.git
cd whisper-asr-training-report
```

### 2. 安装依赖

```bash
# 国内用户推荐先配置 pip 清华镜像，加速下载
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 然后安装依赖
pip install -r requirements.txt
```

### 3. 配置环境

```bash
# 复制示例配置文件
cp .env.example .env

# 根据需要编辑 .env 文件（可选）
```

### 4. 下载模型和测试数据

```bash
chmod +x scripts/download.sh
./scripts/download.sh
```

### 5. 开始快速训练（验证）

```bash
chmod +x scripts/*.sh
./scripts/quick_train.sh
```

### 6. 完整训练

```bash
./scripts/full_train.sh
```

## 📁 项目结构

```
.
├── .env.example              # 环境变量示例配置
├── requirements.txt          # Python 依赖包
├── README.md                 # 项目说明文档
├── configs/                  # 训练配置文件
│   ├── gpu_config.yaml
│   ├── optimized_config.yaml
│   ├── quick_test_config.yaml
│   ├── full_train_config.yaml
│   ├── full_train_config_v2.yaml
│   └── final_train_config.yaml
├── scripts/                  # 所有脚本
│   ├── download.sh          # 模型和数据下载脚本
│   ├── hfd.sh               # Hugging Face 下载工具
│   ├── prepare_dataset.sh   # 数据集准备脚本
│   ├── quick_start.sh       # 快速启动脚本
│   ├── start_training.sh    # 开始训练脚本
│   ├── quick_train.sh       # 快速训练脚本
│   ├── full_train.sh        # 完整训练脚本
│   ├── run_optimized_training.sh  # 优化训练脚本
│   ├── monitor_training.sh  # 训练监控脚本
│   ├── train.py             # 主要训练代码（环境变量版）
│   ├── train_yaml.py        # 训练代码（YAML 配置版）
│   ├── train_complete.py    # 完整训练流水线
│   ├── train_optimized.py   # 优化训练代码
│   ├── evaluate_accuracy.py # 准确率评估
│   ├── test_trained_model.py # 模型测试
│   ├── generate_report.py   # 报告生成（中文）
│   └── generate_report_en.py # 报告生成（英文）
├── output/                   # 训练输出（自动生成）
│   └── final/               # 最终模型
├── models/                   # 预训练模型（自动下载）
│   └── whisper-small/
└── logs/                     # 日志文件
```

## ⚙️ 环境配置说明

在 `.env` 文件中可以配置以下参数：

### Hugging Face 设置
```bash
HF_HOME="./hf_cache"          # 缓存目录
HF_ENDPOINT="https://hf-mirror.com"  # 镜像源（国内加速）
```

### pip 清华镜像（国内用户推荐）
```bash
# 永久配置 pip 镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或临时使用：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 快速测试模式
```bash
QUICK_TEST_MODE=true          # 启用快速测试
QUICK_TEST_SAMPLES=1000       # 训练样本数
QUICK_TEST_VALIDATION=100     # 验证样本数
```

### 训练参数
```bash
MAX_TRAIN_STEPS=500           # 最大训练步数
EVAL_STEPS=250                # 验证频率
SAVE_STEPS=500                # 检查点保存频率
```

### 数据加载优化
```bash
NUM_WORKERS=4                 # 数据加载工作进程数（关键性能优化）
```

### 混合精度训练
```bash
FP16_ENABLED=true             # 启用 FP16（加速训练）
```

## 📊 训练模式

### 快速训练 (Quick Training)
- 使用少量数据（默认 1000 样本）
- 验证整个流程是否正常工作
- 适合调试和快速迭代

### 完整训练 (Full Training)
- 使用完整 LibriSpeech 数据集（13万+样本）
- 训练更多步数
- 最终性能更好

## 🏗️ 关键优化点

### 1. 数据加载优化
- 使用多进程数据加载（`NUM_WORKERS=4`）
- GPU 利用率从 30% 提升到 80-90%
- 训练速度提升 4-8 倍

### 2. 混合精度训练 (FP16)
- 显存占用减半
- 训练速度提升约 50%

### 3. 梯度累积
- 小显存也能使用大的有效 batch size
- 配置：`batch_size=8, accumulation=4 → effective_batch=32`

## 📈 评估指标

- **WER (Word Error Rate)**: 词错误率
- **CER (Character Error Rate)**: 字符错误率
- **Word Accuracy**: 词准确率 = 100 - WER
- **Character Accuracy**: 字符准确率 = 100 - CER

## 🔧 遇到的问题与解决方案

### 问题 1: 数据加载很慢
**症状**: GPU 利用率很低（< 30%），经常空闲等待  
**解决**: 配置 `NUM_WORKERS=4`（或更大）

### 问题 2: 显存不足
**症状**: CUDA out of memory 错误  
**解决**: 
1. 减小 batch size
2. 启用 FP16 (已默认)
3. 减小 `MAX_LENGTH`

### 问题 3: 模型输出错误文本
**症状**: 输出全是 "the the the..."  
**解决**: 
1. 确保下载的模型正确
2. 明确设置 `language="english"` 和 `task="transcribe"`

### 问题 4: 中文字体显示问题
**症状**: 训练图表的中文标签不显示  
**解决**: 使用英文标签（已在代码中实现）

## 📂 输出文件说明

训练完成后，你会在 `output/` 目录看到：
- `final/`: 最终训练好的模型
- `training_report.md`: 训练报告
- 检查点文件：训练过程中的模型快照

## 💻 使用训练好的模型

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 加载模型
model_path = "./output/final"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

if torch.cuda.is_available():
    model = model.cuda()

# 语音识别示例
# audio_array = ... (你的音频数据，16kHz 采样率)
# input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
# 
# with torch.no_grad():
#     predicted_ids = model.generate(
#         input_features,
#         language="english",
#         task="transcribe"
#     )
# 
# transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
# print(transcription)
```

## 📊 训练结果示例

在快速测试模式下，我们获得：
- **WER**: 7.45%
- **Word Accuracy**: 92.55%
- **训练时间**: 约 23 分钟（RTX 4090）

## 🔗 参考资料

- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LibriSpeech Dataset](https://www.openslr.org/12/)

## 📄 许可证

MIT License

## 👤 作者

GitHub: [zikang888](https://github.com/zikang888)

---

**开始使用：** 只需运行 `./scripts/download.sh` 然后 `./scripts/quick_train.sh` 就可以开始训练了！

# Whisper ASR 相关论文合集

本文档整理了 Whisper 语音识别项目相关的核心论文、数据集论文和改进方法论文。

---

## 一、核心论文

### 1. Whisper: Robust Speech Recognition via Large-Scale Weak Supervision

- **作者**: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- **发表时间**: 2022年12月
- **期刊/会议**: ICML 2023
- **arXiv**: [2212.04356](https://arxiv.org/abs/2212.04356)
- **PDF**: [https://arxiv.org/pdf/2212.04356](https://arxiv.org/pdf/2212.04356)
- **代码**: [https://github.com/openai/whisper](https://github.com/openai/whisper)
- **许可证**: MIT

**简介**: Whisper 是 OpenAI 发布的通用语音识别模型，在 68 万小时的多语言、多任务监督数据上进行训练。模型采用编码器-解码器 Transformer 架构，支持多语言语音识别、语音翻译、语言识别和时间戳预测。该模型在零样本迁移设置下，无需任何微调即可在标准基准测试上达到与先前全监督结果相当的性能。Whisper 提供了五种模型尺寸：tiny (39M)、base (74M)、small (244M)、medium (769M)、large (1550M)。

**关键贡献**:
- 提出弱监督大规模训练范式，替代自监督预训练+微调的传统方案
- 在 680,000 小时多语言音频上训练，涵盖 96 种语言
- 零样本泛化能力强，无需针对特定数据集微调
- 开源模型权重和推理代码

---

### 2. Whisper large-v3

- **发布方**: OpenAI
- **发布时间**: 2023年11月 (OpenAI DevDay)
- **模型**: [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)

**简介**: Whisper large-v3 是 Whisper 系列的第三代大模型，在约 500 万小时音频数据上训练。相比 large-v2，主要改进包括：使用 128 个 Mel 频率 bin（之前为 80 个），支持 99 种语言（新增粤语等），在多种语言上 WER 降低 10%-20%。支持句级和词级时间戳预测。

---

### 3. Whisper large-v3-turbo

- **发布方**: OpenAI
- **发布时间**: 2024年10月
- **模型**: [https://huggingface.co/openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

**简介**: Whisper large-v3-turbo 是 large-v3 的精简加速版，参数从 1550M 减少到 809M，仅保留 4 层解码器。转录速度比 large-v3 快约 8 倍，同时保持接近的准确率。

---

## 二、数据集论文

### 4. LibriSpeech: An ASR Corpus Based on Public Domain Audio Books

- **作者**: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur (Johns Hopkins University)
- **发表时间**: 2015年 (ICASSP 2015)
- **PDF**: [https://www.danielpovey.com/files/2015_icassp_librispeech.pdf](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
- **数据集**: [https://www.openslr.org/12/](https://www.openslr.org/12/)
- **许可证**: CC BY 4.0

**简介**: LibriSpeech 是基于 LibriVox 项目公有领域有声读物构建的英语朗读语音语料库，包含约 1000 小时 16kHz 采样的语音数据。数据集分为 train-clean-100、train-clean-360、train-other-500 等子集，以及 dev-clean、dev-other、test-clean、test-other 等评估集。该语料库已成为 ASR 领域最广泛使用的基准数据集之一。论文还提供了 Kaldi 工具包脚本和预构建的语言模型。

**关键特性**:
- 总时长约 1000 小时（960h 训练 + 约 5h dev + 约 5h test）
- 16kHz 采样率
- 使用 Smith-Waterman 算法进行长音频对齐和分割
- 提供语言模型训练数据和预训练语言模型

---

## 三、Whisper 改进与蒸馏论文

### 5. Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling

- **作者**: Sanchit Gandhi, Patrick von Platen, Alexander M. Rush (Hugging Face)
- **发表时间**: 2023年11月
- **arXiv**: [2311.00430](https://arxiv.org/abs/2311.00430)
- **代码**: [https://github.com/huggingface/distil-whisper](https://github.com/huggingface/distil-whisper)
- **许可证**: MIT

**简介**: Distil-Whisper 是 Whisper 的知识蒸馏版本，专为英语语音识别优化。通过保留完整编码器（冻结）并将解码器蒸馏至仅 2 层，实现了 6 倍加速、49% 体积缩减，同时在分布外评估集上 WER 仅相差 1% 以内。训练使用 20,000 小时开放音频数据，结合 KL 散度损失和伪标签训练。在长音频转录中，由于减少了幻觉，甚至超过原始 Whisper 的表现。

**关键结果**:
- 6 倍推理速度提升
- 模型体积缩小 49%
- 短格式 WER 与 large-v2 相差 <1%
- 长格式 WER 优于 large-v2（减少幻觉）

---

### 6. OWSM: Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data

- **作者**: Yifan Peng, Jinchuan Tian, Brian Yan, Shinji Watanabe 等 (CMU)
- **发表时间**: 2023年10月
- **arXiv**: [2309.13876](https://arxiv.org/abs/2309.13876)
- **PDF**: [https://arxiv.org/pdf/2309.13876](https://arxiv.org/pdf/2309.13876)

**简介**: OWSM (Open Whisper-Style Speech Model) 使用开源工具包和公开可用数据复现了 Whisper 风格的训练流程。该工作开源了数据准备、训练、推理和评分的完整流程，支持任意语言到任意语言的语音翻译。虽然数据集规模仅为 Whisper 的约四分之一，但在部分基准上达到甚至超过 Whisper 的表现。

---

### 7. Using Fine-Tuning and Min Lookahead Beam Search to Improve Whisper

- **作者**: 多作者
- **发表时间**: 2023年9月
- **arXiv**: [2309.10299](https://arxiv.org/abs/2309.10299)

**简介**: 该论文系统研究了 Whisper 的微调策略和解码算法改进。通过实验对比全参数微调和 LoRA 微调（r=192 高秩设置），发现高秩 LoRA 可以媲美全参数微调的性能。同时提出 Filter-Ends 和 Min Lookahead 两种改进的束搜索解码算法，在多种语言上平均降低 2.26 WER。论文还从理论上证明了 Min Lookahead 优于标准束搜索。

---

### 8. Fine-tuning Whisper on Low-Resource Languages for Real-World Applications

- **作者**: Vincenzo Timmel, Claudio Paonessa, Reza Kakooee 等
- **发表时间**: 2024年12月
- **arXiv**: [2412.15726](https://arxiv.org/abs/2412.15726)

**简介**: 提出一种新颖的数据生成方法，将句子级数据转换为长格式语料库，用于对低资源语言微调 Whisper。以瑞士德语为案例，生成的模型成为瑞士德语 STT 领域的新 SOTA。该方法可将句子级数据转换为保留分段能力的长格式训练数据，适用于其他低资源语言。

---

### 9. On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR

- **作者**: Ganesh Pavan Kartikeya Bharadwaj Kolluri, Michael Kampouridis, Ravi Shekhar
- **发表时间**: 2026年3月
- **arXiv**: [2603.27981](https://arxiv.org/abs/2603.27981)

**简介**: 分析了 Whisper 编码器层剪枝在 SLAM-ASR 框架中的影响。实验覆盖三种 Whisper 变体 (Small, Medium, Large-v2) 和三种语言 (丹麦语、荷兰语、英语)，超过 200 个训练轮次。结果表明剪枝 2 层编码器仅导致 2-4% WER 退化，结合 LoRA 微调后甚至超过未剪枝基线，同时参数总量减少 7-14%。

---

## 四、语音识别基础论文

### 10. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

- **作者**: Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli (Facebook AI)
- **发表时间**: 2020年6月
- **arXiv**: [2006.11477](https://arxiv.org/abs/2006.11477)
- **代码**: [https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq)

**简介**: wav2vec 2.0 是自监督语音表示学习的里程碑工作，也是 Whisper 论文中重点对比的基线方法。模型在 53,000 小时无标签语音数据上预训练，仅需 10 分钟标注数据微调即可在 LibriSpeech 上达到 5.7 WER（test-clean）。该工作证明了自监督预训练在语音识别中的巨大潜力。

**关键贡献**:
- 提出量化模块将连续语音表示离散化
- 对比学习目标：从干扰项中识别真实的量化表示
- 仅需极少标注数据即可达到优秀性能

---

### 11. Attention Is All You Need

- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar 等 (Google)
- **发表时间**: 2017年6月
- **arXiv**: [1706.03762](https://arxiv.org/abs/1706.03762)
- **会议**: NeurIPS 2017

**简介**: Transformer 架构的开创性论文，提出了完全基于注意力机制的序列到序列模型。Whisper 的编码器-解码器架构直接继承自 Transformer。论文引入多头自注意力、位置编码、残差连接和层归一化等关键组件，已成为现代深度学习的基石。

---

## 五、评估指标

### 12. WER/CER 评估指标

Whisper ASR 项目中使用的主要评估指标：

- **WER (Word Error Rate)** = (S + D + I) / N，其中 S 为替换错误数、D 为删除错误数、I 为插入错误数、N 为参考文本词数
- **CER (Character Error Rate)** = 字符级别的错误率
- **Word Accuracy** = 100% - WER
- 评估工具 [evaluate](https://github.com/huggingface/evaluate) (Hugging Face)

---

## 论文引用格式

如需在学术工作中引用，推荐使用以下 BibTeX 格式：

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}

@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}

@article{gandhi2023distil,
  title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling},
  author={Gandhi, Sanchit and von Platen, Patrick and Rush, Alexander M},
  journal={arXiv preprint arXiv:2311.00430},
  year={2023}
}

@article{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}

@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

---

> 最后更新: 2026年5月31日
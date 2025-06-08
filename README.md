# 🐦 BirdCLEF 2025 – Bronze Medal Solutions (Private LB 0.893)

This repository hosts two high-scoring inference notebooks submitted to the [BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025) competition, both earning Bronze Medals with Private Leaderboard scores of 0.893. Additionally, it includes a community-released single-model inference notebook that provided significant inspiration for our work.

## 📂 Project Structure

- bird25-inference-seresnext-nfnet-bronze.ipynb  
  Weighted ensemble using ECA-NFNet-L0 and SeresNext26t_32x4d backbones. Achieved Private LB 0.893 (Bronze Medal).

- bird25-openvino-ensemble-infer-baseline-bronze.ipynb  
  Multi-stage OpenVINO-optimized ensemble. Also scored 0.893 (Bronze Medal).

- birdclef2025-single-sed-model-inference.ipynb  
  High-quality public baseline by i2nfinit3y. This single-model SED notebook provided the design inspiration for our solutions.

## 📌 Project Overview

BirdCLEF 2025 is a bio-acoustics challenge to detect bird species in noisy rainforest soundscapes. Our solutions combine CNN backbones, test-time augmentation, post-processing, and strategic ensembling to balance accuracy and runtime.

## 🥇 Solution A — Weighted Ensemble (PB 0.893)

- Backbones: ECA-NFNet-L0, SeresNext26t_32x4d  
- Blend Weights: 0.6 (NFNet) + 0.4 (SeresNext)  
- Post-processing: Power-adjusted thresholding  
- Pooling: Attention-based  
- Runtime: ~70 min  
- Key References:  
  https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank  
  https://www.kaggle.com/datasets/i2nfinit3y/bird2025-sed-ckpt

## 🥇 Solution B — Multi-Stage OpenVINO Ensemble (PB 0.893)

- Pipeline: Single SED → OpenVINO-optimized model → 3-Fold SED ensemble  
- Runtime: ~80 min  
- Optimizations: OpenVINO multithreaded CPU inference, Test-Time Augmentation (TTA)  
- Bottleneck: 3-Fold SED mel-spectrogram generation (consider `librosa` instead of `torchaudio`)  
- Key References:  
  https://www.kaggle.com/code/kurisew/lb0-855-openvino-multithread-tta-ensemble-infer  
  https://www.kaggle.com/code/hideyukizushi/bird25-weightedblend-nfnet-convnextv2-lb-860  
  https://www.kaggle.com/code/i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857

## 🏆 Community Baseline

- birdclef2025-single-sed-model-inference.ipynb   
https://www.kaggle.com/code/i2nfinit3y/bird2025-single-sed-model-inference-lb-0-857 
  Author: i2nfinit3y  
  This single-model SED notebook served as the high-quality baseline and design reference for our winning solutions.

## 🔧 Setup

### Requirements

pip install -r requirements.txt

Core packages: torch>=2.0, torchaudio, timm, numpy, pandas, librosa, scipy, opencv-python, openvino-runtime (optional), tqdm, joblib, matplotlib

### Directory Layout

Inference/
    bird25-inference-seresnext-nfnet-bronze.ipynb
    bird25-openvino-ensemble-infer-baseline-bronze.ipynb
    birdclef2025-single-sed-model-inference.ipynb
LICENSE
README.md

### Running

Open any notebook in Inference/ and follow the instructions cell-by-cell.  
Prepare the data directory with:  
- test_soundscapes/*.ogg  
- taxonomy.csv  
- sample_submission.csv  
(Download from Kaggle BirdCLEF 2025)

## 📌 Best Practices

- Always map class indices with taxonomy.csv
- Tune blend weights privately to avoid leaderboard overfitting
- Blend logits before applying thresholds
- Replace torchaudio with librosa for faster mel generation if runtime is critical

## 🧑‍💻 Team

Ryan Ren
Zhenghao Ni

## 🏁 Results

| Solution | Private LB | Medal |
|----------|-----------:|-------|
| A        | 0.893      | 🥉    |
| B        | 0.893      | 🥉    |

Final rank: top ~5% among ~2,000 teams

## 📫 Contact

GitHub Issues  
Github Profile: https://github.com/Ryan-Ren0330
Kaggle: https://www.kaggle.com/ryanren
Email: ryan.ren@mail.utoronto.ca


&lt;div align="center"&gt;

&lt;!-- ANIMATED HEADER --&gt;
&lt;img src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:4F46E5,100:06B6D4&height=220&section=header&text=TactIntentNet&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Decode%20Latent%20Tactical%20Intent%20from%20Broadcast%20Positions&descAlignY=55&descSize=18&descColor=ffffff" width="100%" /&gt;

&lt;!-- LANGUAGE TOGGLE --&gt;
[ 🇬🇧 English ](README.md) | [ 🇯🇵 日本語 ](README_JP.md)

&lt;br /&gt;

&lt;!-- TECH BADGES --&gt;
[![AMD](https://img.shields.io/badge/AMD-MI300X-ED1C24?style=flat-square&logo=amd&logoColor=white)](https://www.amd.com)
[![ROCm](https://img.shields.io/badge/ROCm-6.2-ED1C24?style=flat-square)](https://rocm.docs.amd.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.7-3C2179?style=flat-square)](https://pytorch-geometric.readthedocs.io)
[![Gradio](https://img.shields.io/badge/Gradio-5.25-FF6B6B?style=flat-square)](https://gradio.app)
[![StatsBomb](https://img.shields.io/badge/StatsBomb-Open%20Data-2E7D32?style=flat-square)](https://github.com/statsbomb/open-data)

&lt;br /&gt;

&lt;p&gt;
  &lt;b&gt;TactIntentNet&lt;/b&gt; is the first open-source system to decode football tactical intent from broadcast video using &lt;b&gt;causal graph neural networks&lt;/b&gt;. It detects real-time opponent deviations via Gaussian Mixture Model fingerprinting and generates coaching alerts through a &lt;b&gt;Qwen 2.5 1.5B LLM&lt;/b&gt; — all running on &lt;b&gt;AMD Instinct MI300X&lt;/b&gt;.
&lt;/p&gt;

&lt;br /&gt;

&lt;!-- ACTION BUTTONS --&gt;
[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-HuggingFace-8B5CF6?style=for-the-badge&labelColor=1E293B)](https://huggingface.co/spaces/shafayatsaad/tactintentnet)
[![GitHub](https://img.shields.io/badge/⭐_Star_This_Repo-181717?style=for-the-badge&labelColor=1E293B&logo=github)](https://github.com/shafayatsaad/tactintentnet)
[![AMD Hackathon](https://img.shields.io/badge/🔥_AMD_Hackathon-lablab.ai-ED1C24?style=for-the-badge&labelColor=1E293B)](https://lablab.ai/ai-hackathons/amd-developer)

&lt;/div&gt;

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚨 The Tactical Intelligence Gap](#-the-tactical-intelligence-gap)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [⚡ Performance](#-performance)
- [🚀 Quick Start](#-quick-start)
- [📊 Benchmarks](#-benchmarks)
- [🛠️ Tech Stack](#️-tech-stack)
- [👥 Team](#-team)

---

## 🎯 Overview

**TactIntentNet** bridges the gap between broadcast video and elite tactical analysis. Built for the **AMD Developer Hackathon 2026**, it transforms freely available StatsBomb 360 freeze-frame data into actionable tactical intelligence using a **3-layer GATv2 Graph Neural Network** — no proprietary tracking hardware required.

### Why TactIntentNet?

- 🎥 **Broadcast Only**: Works with free video — no $500K tracking contracts
- 🧠 **Causal Reasoning**: Learns player-to-player influence weights, not just correlations
- ⚡ **Real-Time**: 12ms inference latency at 80 fps on AMD MI300X
- 🔮 **Counterfactuals**: Drag any player to a new position, see intent shift instantly
- 💬 **LLM Coaching**: Qwen 2.5 generates plain-English tactical alerts

---

## 🚨 The Tactical Intelligence Gap

Elite tactical analysis is locked behind proprietary tracking systems. TactIntentNet democratizes it:

| Problem | Impact | TactIntentNet Solution |
|---------|--------|------------------------|
| ❌ **Proprietary Tracking** | $500K+/year for hardware | **Free broadcast video only** |
| ❌ **Post-Hoc Analysis** | No real-time deviation detection | **Streaming OTDS timeline** |
| ❌ **Correlation Metrics** | xG/PPDA describe what, not why | **Causal GNN attention graph** |
| ❌ **No Counterfactuals** | Can't test "what-if" scenarios | **Interactive player explorer** |
| ❌ **No LLM Intelligence** | Static dashboards | **Qwen 2.5 coaching alerts** |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔴 **Live Intent Feed** | Real-time tactical intent prediction across 12 classes with confidence scoring |
| 📈 **Opponent Deviation** | OTDS timeline with GMM fingerprinting + side-by-side tactical comparison |
| 🧪 **Counterfactual Explorer** | Drag any player to a new position; observe intent probability shift in 80ms |
| 🧠 **Causal Graph** | Visualize learned GATv2 attention weights between all 22 players |
| 🤖 **Tactical Assistant** | Qwen 2.5 1.5B LLM answers coaching questions in natural language |
| 📋 **Match Report** | One-click exportable scouting report with phase breakdown & action items |

---

## 🏗️ Architecture

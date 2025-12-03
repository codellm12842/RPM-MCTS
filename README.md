# RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation <!-- omit in toc -->

<p align="center">
  <a href="https://arxiv.org/abs/2511.19895"><img src="https://img.shields.io/badge/arXiv-2511.19895-AE4F55.svg?logo=arxiv" alt="Paper"></a>
  <a href="https://github.com/codellm12842/RPM-MCTS"><img src="https://img.shields.io/badge/Github-RPM--MCTS-8D4859.svg?logo=github" alt="Code"></a>
  <a href="https://openreview.net/forum?id=o3FLpLuj0Y"><img src="https://img.shields.io/badge/OpenReview-RPM--MCTS-6D415E.svg" alt="Paper"></a>
  <a href="https://aaai.org/conference/aaai/aaai-26/"><img src="https://img.shields.io/badge/Accepted-AAAI'26-4E3B64.svg" alt="License"></a>
</p>

This repository provides the implementation for **RPM-MCTS**.

<p align="center"><img src="figures/overview.png" width=100%></p>
<p align="center"><em>Figure.</em> Overview of RPM-MCTS.</p>

## Table of Contents <!-- omit in toc -->

- [Getting Started](#getting-started)
- [Reference Code](#reference-code)
- [Citation](#citation)

## Getting Started

1. Install dependencies
```
pip install -e .
```

2. Download embedding model
```
python huggingface/download.py
```

3. Build knowledge base
```
python rpm_mcts_tools/knowledge_base/vector_db_build_kb2.py
```

4. Run
```
python baselines/run_all.py
```

## Reference Code

- SRA-MCTS: [https://github.com/DIRECT-BIT/SRA-MCTS](https://github.com/DIRECT-BIT/SRA-MCTS)
- ReST-MCTS*: [https://github.com/THUDM/ReST-MCTS](https://github.com/THUDM/ReST-MCTS)

## Citation

```

```
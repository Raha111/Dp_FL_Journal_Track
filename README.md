# Differentially Private Federated Learning Journal

This repository documents an ongoing study of **differential privacy in federated learning (DP-FL)**. It brings together paper reviews, concept notes, implementation practice, and experiments so that the progression from theory to reproducible results can be followed in one place.

## Research Scope

The work focuses on the privacy guarantees and practical trade-offs involved in federated learning, including:

- record-level and client-level differential privacy;
- central, local, and distributed DP trust models;
- clipping, noise calibration, and privacy accounting;
- client sampling and participation;
- learning under non-IID data;
- the relationship between privacy budgets and model utility.

## Repository Map

| Path | Contents |
| --- | --- |
| [`lit review/`](lit%20review/) | Source PDFs: ten numbered papers and nine additional papers |
| [`Notes/Papers/`](Notes/Papers/) | Structured reviews, questions, and takeaways from research papers |
| [`Code/`](Code/) | MNIST baseline, shared neural network, and manual FedAvg implementation |
| [`src/main.py`](src/main.py) | Initial script trial |
| `Experiments/` | Experiment configurations, results, and interpretations (planned) |

## Reading Index

Titles and author names below are transcribed from the supplied PDFs. Numbering follows the numbered PDF filenames.

| No. | Paper and authors | Status | References |
| ---: | --- | --- | --- |
| 01 | *Differentially Private Federated Learning: A Systematic Review* - Jie Fu, Yuan Hong, Xinpeng Ling, Leixia Wang, Xun Ran, Zhiyu Sun, Wendy Hui Wang, Zhili Chen, Yang Cao | Completed | [PDF](lit%20review/paper%201.pdf) / [Notes](Notes/Papers/01_fu_dpfl_systematic_review.md) |
| 02 | *Communication-Efficient Learning of Deep Networks from Decentralized Data* - H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas | Notes template created; review pending | [PDF](lit%20review/paper%202.pdf) / [Notes](Notes/Papers/02_Comm_Eff_deep_net.md) |
| 03 | *Learning Differentially Private Recurrent Language Models* - H. Brendan McMahan, Daniel Ramage, Kunal Talwar, Li Zhang | Review pending | [PDF](lit%20review/paper%203.pdf) |
| 04 | *Rényi Differential Privacy* - Ilya Mironov | Review pending | [PDF](lit%20review/paper%204.pdf) |
| 05 | *Subsampled Rényi Differential Privacy and Analytical Moments Accountant* - Yu-Xiang Wang, Borja Balle, Shiva Prasad Kasiviswanathan | Review pending | [PDF](lit%20review/paper%205.pdf) |
| 06 | *Understanding Clipping for Federated Learning: Convergence and Client-Level Differential Privacy* - Xinwei Zhang, Xiangyi Chen, Mingyi Hong, Zhiwei Steven Wu, Jinfeng Yi | Review pending | [PDF](lit%20review/paper%206.pdf) |
| 07 | *Differentially Private Learning with Adaptive Clipping* - Galen Andrew, Om Thakkar, H. Brendan McMahan, Swaroop Ramaswamy | Review pending | [PDF](lit%20review/paper%207.pdf) |
| 08 | *Privacy Odometers and Filters: Pay-as-you-Go Composition* - Ryan Rogers, Aaron Roth, Jonathan Ullman, Salil Vadhan | Review pending | [PDF](lit%20review/paper%208.pdf) |
| 09 | *Fully-Adaptive Composition in Differential Privacy* - Justin Whitehouse, Aaditya Ramdas, Ryan Rogers, Zhiwei Steven Wu | Review pending | [PDF](lit%20review/paper%209.pdf) |
| 10 | *Privacy Amplification via Random Check-Ins* - Borja Balle, Peter Kairouz, H. Brendan McMahan, Om Thakkar, Abhradeep Thakurta | Review pending | [PDF](lit%20review/paper%2010.pdf) |

### Additional Literature

These PDFs are available for future review; reading completion has not yet been recorded.

| Paper and authors | Reference |
| --- | --- |
| *User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization* - Kang Wei, Jun Li, Ming Ding, Chuan Ma, Hang Su, Bo Zhang, H. Vincent Poor | [PDF](lit%20review/lit/2003.00229v2.pdf) |
| *Differentially Private Federated Learning with Time-Adaptive Privacy Spending* - Shahrzad Kiani, Nupur Kulkarni, Adam Dziedzic, Stark Draper, Franziska Boenisch | [PDF](lit%20review/lit/2502.18706v1.pdf) |
| *An Adaptive Differentially Private Federated Learning Framework* - Jin Wang, Hui Ma, Yajun Zhang, Xinjun Pei, Ming Yan, Fei Xing, Yikun Chen | [PDF](lit%20review/lit/2602.06838v3.pdf) |
| *AdaDpFed: A Differentially Private Federated Learning Algorithm With Adaptive Noise on Non-IID Data* - Zirun Zhao, Yi Sun, Ali Kashif Bashir, Zhaowen Lin | [PDF](lit%20review/lit/AdaDpFed_A_Differentially_Private_Federated_Learning_Algorithm_With_Adaptive_Noise_on_Non-IID_Data.pdf) |
| *Adap DP-FL: Differentially Private Federated Learning with Adaptive Noise* - Jie Fu, Zhili Chen, Xiao Han | [PDF](lit%20review/lit/Adap%20DP-FL%20Differentially%20Private%20Federated%20Learning%20with%20Adaptive%20Noise-with-annotations.pdf) |
| *Differentially Private Federated Learning With an Adaptive Noise Mechanism* - Rui Xue, Kaiping Xue, Bin Zhu, Xinyi Luo, Tianwei Zhang, Qibin Sun, Jun Lu | [PDF](lit%20review/lit/Differentially_Private_Federated_Learning_With_an_Adaptive_Noise_Mechanism.pdf) |
| *FedAdaClip: Adaptive Clipping for Private Federated Learning* - Yifan Sun, Rui Zhang, Yang Tao | [PDF](lit%20review/lit/FedAdaClip_Adaptive_Clipping_for_Private_Federated_Learning.pdf) |
| *Differentially Private Federated Learning with Adaptive Clipping Thresholds* - Jianhua Liu, Yanglin Zeng, Zhongmei Wang, Weiqing Zhang, Yao Tong | [PDF](lit%20review/lit/futureinternet-18-00148-v2.pdf) |
| *Dynamic privacy budget allocation for enhanced differential privacy in federated learning* - Libo Zhu, Hongbin Song, Xiang Chen | [PDF](lit%20review/lit/s10586-025-05699-x.pdf) |

## How to Navigate

Start with the **Reading Index** above for the theoretical background. Each paper entry records the citation, motivation, important concepts, research questions, and a concise summary. As implementations and experiments are added, their corresponding sections will link the relevant code and results back to the papers or concepts they examine.

## Results and Highlights

The repository now includes literature notes and initial implementation practice. No saved experimental results are available yet. The current training scripts implement non-private baselines; clipping, noise addition, and privacy accounting have not been implemented.

## Code Practice

Requires Python with `torch` and `torchvision` installed. Run these commands from the repository root:

```powershell
python Code/mnist_baseline.py
python Code/fedavg_manual.py
```

- [`models.py`](Code/models.py): shared network with a 784-input layer, 128 hidden units, ReLU, and 10 output classes.
- [`mnist_baseline.py`](Code/mnist_baseline.py): centralized MNIST training for five epochs using SGD, a learning rate of 0.01, and batches of 32; prints training metrics and test accuracy.
- [`fedavg_manual.py`](Code/fedavg_manual.py): five equal random client partitions, five communication rounds, two local epochs per round, sample-weighted model averaging, and test evaluation after each round.

Both training scripts download MNIST to `data/`, which is excluded from Git. Training outputs and accuracy measurements have not yet been recorded in this repository.

## Latest Update — 2026-10-16

Added the MNIST baseline, shared model, and manual FedAvg scripts under `Code/`. Indexed all 19 supplied PDFs with their titles, authors, and source links; Paper 1 is completed. Downloaded datasets are excluded from Git.

## Current Progress

- **Literature review:** Paper 1 completed; Paper 2 review pending
- **Code practice:** MNIST and manual FedAvg implementations added; full training validation pending
- **Experiments:** not started

## Maintenance Workflow

For each completed repository update, update this README to reflect the changes, then commit and push the changes together to the configured remote.


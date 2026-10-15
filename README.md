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
| [`Notes/Papers/`](Notes/Papers/) | Structured reviews, questions, and takeaways from research papers |
| [`Code/`](Code/) | MNIST baseline, shared neural network, and manual FedAvg implementation |
| [`src/main.py`](src/main.py) | Initial script trial |
| `Experiments/` | Experiment configurations, results, and interpretations (planned) |

## Reading Index

| No. | Paper | Status | Entry |
| ---: | --- | --- | --- |
| 01 | *Differentially Private Federated Learning: A Systematic Review* | Reading complete; notes recorded | [Open notes](Notes/Papers/01_fu_dpfl_systematic_review.md) |
| 02 | Communication-efficient deep network training | Notes template created; review pending | [Open notes](Notes/Papers/02_Comm_Eff_deep_net.md) |

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

Added the MNIST baseline, shared model, and manual FedAvg scripts under `Code/`. Updated the reading index and excluded downloaded datasets from Git.

## Current Progress

- **Literature review:** active
- **Code practice:** MNIST and manual FedAvg implementations added; full training validation pending
- **Experiments:** not started

## Maintenance Workflow

For each completed repository update, update this README to reflect the changes, then commit and push the changes together to the configured remote.


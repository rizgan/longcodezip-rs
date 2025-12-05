# Experiments

This folder contains the old experiments for the three code-related tasks. Some codes may be outdated after refactoring.

### Quick Start

Each task directory contains a `run.sh` script for easy experimentation. Simply navigate to the desired task directory and run:

```bash
cd <task_directory>
bash run.sh
```

### Code Retrieval (RepoQA)

Navigate to the `repo-qa` directory and run experiments with different compression ratios:

```bash
cd repo-qa
bash run.sh
```

The script will evaluate LongCodeZip on the RepoQA dataset with compression ratios, running experiments in parallel on multiple GPUs.

**Key Parameters:**
- `--compression-ratio`: Controls the compression level
- `--model`: Specifies the base LLM model
- `--backend`: Backend for model inference (vllm)

### Code Completion

Navigate to the `long-code-completion` directory:

```bash
cd long-code-completion
bash run.sh
```

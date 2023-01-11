# Viral ProtBert
ProtBert for modeling viral escape.

## Creating Environment

```bash
conda env create -n <env_name> -f environment.yaml
```

:warning: We have downgraded to ``tokenizers==0.10.3`` and ``transformers==4.10.0`` for GLIBC compatibility with TRUBA servers (CentOS 7.3 and GLIBC 2.17).

## Running Instructions

```bash
python cov.py --test --checkpoint='models/checkpoint_4.pt' --masking_prob=0.01 --minibatch_size=240
```

:warning: Use ``--minibatch_size=240`` (which requires ~110 Gb of GPU RAM) on HPC Vega server (4 x 40 Gb GPU RAM).

:warning: Use ``--minibatch_size=24`` (which requires ~11 Gb of GPU RAM) on DARG server (12 Gb GPU RAM).

```bash
python cov.py --train --epochs=4 --masking_prob=0.01 --minibatch_size=12
```
:warning: Use ``--minibatch_size=12`` (which requires ~120 Gb of GPU RAM) on HPC Vega server (4 x 40 Gb GPU RAM).

```bash
srun --jobid=<job_id> nvidia-smi
```

```bash
python cov.py --embed --use_cache
```

## SARS-CoV-2 Experiment Notes

Train (3336) + Test (836) = 4172 protein sequences (with a maximum length of 1582)

**In silico, all single-residue mutations (``--masking_prob=1``):**

* **4,363,061** training sequences
* **1,092,599** testing sequences

**Training ProtBERT model 4 epochs (``--minibatch_size=12``):**

* IYTE-DARG (12Gb GPU, 64Gb RAM) :x: OOM error 
* TRUBA-AKYA (64Gb GPU, 380Gb RAM) :x: OOM error
* HPC-VEGA (160Gb GPU, 500Gb RAM) :warning: 120 hours/epoch

**Training results with ``--masking_prob=0.01`` (Use only 44,635 of 4,363,061 train sequences):**

* HPC-VEGA (160Gb GPU, 500Gb RAM) :white_check_mark: 80 minutes/epoch
* Accuracy for each epoch = [0.88, 0.88, 0.91, 0.93]
* Perplexity for each epoch = [1.61, 1.58, 1.46, 1.31]
* Loss for each epoch = [0.48, 0.46, 0.38, 0.27]

**Test results with original ProtBERT model:**

* Accuracy = 0.57, Perplexity = 4.32, Loss = 1.46, Time = 12 hours

**Test results with fine-tuned ProtBERT model (``checkpoint_4.pt``):**

* Accuracy = 0.16, Perplexity = 33.2, Loss = 3.50, Time = 12 hours

**Test results with ``--masking_prob=0.01`` (Use only 11,176 of 1,092,599 test sequences):**

* Accuracy = 0.88, Perplexity = 1.46, Loss = 0.38, Time = 6 minutes

## References

[https://huggingface.co/Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)
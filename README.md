# Viral ProtBert
ProtBert for modeling viral escape.

## Creating Environment

```bash
conda env create -n <env_name> -f environment.yaml
```

:warning: We have downgraded to ``tokenizers==0.10.3`` and ``transformers==4.10.0`` for GLIBC compatibility with TRUBA servers (CentOS 7.3 and GLIBC 2.17).

## Running Instructions

```bash
python cov.py --test --batch_size=38 --minibatch_size=128
```

:warning: Use ``--minibatch_size=24`` (which requires ~11 Gb of GPU RAM) on DARG server.

:information_source: There are **1,092,599** single residue mutants (in silico) for **836** test sequences.

```bash
python cov.py --train --batch_size=834 --minibatch_size=128 --epochs=4
```

:information_source: There are **4,363,061** single residue mutants (in silico) for **3336** train sequences.

```bash
srun --jobid=<job_id> nvidia-smi
```

```bash
python cov.py --embed --use_cache
```

## References

[https://huggingface.co/Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)
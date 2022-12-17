# Viral ProtBert
ProtBert for modeling viral escape.

## Creating Environment

```bash
conda env create -n <env_name> -f environment.yaml
```

**Note:** We have downgraded to ``tokenizers==0.10.3`` and ``transformers==4.10.0`` for GLIBC compatibility with TRUBA servers (CentOS 7.3 and GLIBC 2.17).

## Running Instructions

```bash
python cov.py --test --inference_batch_size=38 --minibatch_size=128
```

**Note:** Use ``--minibatch_size=24`` (which requires ~11 Gb of GPU RAM) on DARG server.

```bash
srun --jobid=<job_id> nvidia-smi
```

```bash
python cov.py --embed --use_cache
```

## References

[https://huggingface.co/Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)
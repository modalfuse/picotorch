# Data access

PicoTorch does not redistribute raw WISDM text or CIFAR-10 images. On-device inference reads the shipped weight headers.

## Datasets

| Dataset | Role in this archive | Source |
| --- | --- | --- |
| WISDM Actitracker v1.1 | Optional refresh of `examples/wisdm/wisdm_weights.hpp`; 6-class IMU Encoder | [Fordham WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php) |
| CIFAR-10 official split | Optional refresh of `examples/cifar10/cifar_weights.hpp`; Conv + Encoder | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |

This is the Kwapisz 2011 phone-accelerometer set, not UCI 507.

## WISDM

```bash
mkdir -p examples/wisdm/data
curl -L -o examples/wisdm/data/WISDM_ar_latest.tar.gz \
  https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
tar -xzf examples/wisdm/data/WISDM_ar_latest.tar.gz -C examples/wisdm/data
ln -sfn WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt examples/wisdm/data/WISDM_ar_v1.1_raw.txt
python examples/wisdm/train_export.py --raw examples/wisdm/data/WISDM_ar_v1.1_raw.txt
```

## CIFAR-10

```bash
mkdir -p examples/cifar10/data
curl -L -o examples/cifar10/data/cifar-10-python.tar.gz \
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
python examples/cifar10/train_export.py --epochs 8
```

If the tarball is missing or incomplete, `train_export.py` loads the same official split from Hugging Face into `examples/cifar10/data/hf/`.

## Glucose counterpart

The PicoTorch glucose example uses Pico-Former weights. Point the export scripts at `ceqt_weights.h`:

```bash
export PICO_FORMER_WEIGHTS=/path/to/rice-s3/firmware/include/ceqt_weights.h
```

Pico-Former does not redistribute CGM recordings. AZT1D remains available from Mendeley Data, https://doi.org/10.17632/gk9m674wcx.1. The host protocol is archived as RICE-Former, https://doi.org/10.5281/zenodo.22171786.

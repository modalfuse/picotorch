# WISDM Actitracker data (local)

Do not commit the archive or the extracted text. Download the classic 6-class set from the [Fordham WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php) (activity-prediction / Actitracker v1.1).

```bash
mkdir -p examples/wisdm/data
curl -L -o examples/wisdm/data/WISDM_ar_latest.tar.gz \
  https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
tar -xzf examples/wisdm/data/WISDM_ar_latest.tar.gz -C examples/wisdm/data
ln -sfn WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt examples/wisdm/data/WISDM_ar_v1.1_raw.txt
```

On-device inference uses `wisdm_weights.hpp` and does not read this directory. Re-run `train_export.py` only when you need to refresh that header.

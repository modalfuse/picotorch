# CIFAR-10 data (local)

Do not commit the official tarball, the extracted batches, or the Hugging Face cache. Obtain the official train/test split from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

```bash
mkdir -p examples/cifar10/data
curl -L -o examples/cifar10/data/cifar-10-python.tar.gz \
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 170498071 bytes; train_export.py extracts cifar-10-batches-py when the tarball is complete
```

If the tarball is missing or incomplete, `train_export.py` loads the same official split from Hugging Face (`cifar10`, `plain_text/*.parquet`) into `data/hf/`.

On-device inference uses `cifar_weights.hpp` and does not read this directory. Re-run `train_export.py` only when you need to refresh that header.

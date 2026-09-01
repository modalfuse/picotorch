# Public-release checklist

- [x] Public tree is an independent git repository, not a commit of the lab `PicoTorch/` workspace.
- [x] Firmware sources copied without `.pio/`, `build/`, or `managed_components/`.
- [x] Exported weight headers and contrast `.tflite` / `.espdl` included.
- [x] Raw WISDM and CIFAR-10 archives excluded (`examples/*/data/` keeps only `README.md`).
- [x] Local absolute paths replaced by flags or environment variables (`PICO_FORMER_WEIGHTS`, `PICOTORCH_SERIAL`, `IDF_PATH`).
- [x] PlatformIO `upload_port` / `monitor_port` left for auto-detect.
- [x] Code license recorded (`LICENSE`, MIT).
- [x] Citation and Zenodo metadata templates included.
- [x] GitHub repository name is `picotorch`.
- [x] Enable `picotorch` in the Zenodo GitHub integration.
- [x] Create GitHub release `v1.0.0`, then `v1.0.1` after the hook was enabled.
- [x] Concept DOI `10.5281/zenodo.22231092` added to `CITATION.cff` and `README.md`.
- [x] Zenodo description aligned with the Pico-Former record (https://doi.org/10.5281/zenodo.22218446).

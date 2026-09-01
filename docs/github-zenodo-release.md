# GitHub and Zenodo release procedure

GitHub repository name is `picotorch` (lowercase). The display title in README and Zenodo is **PicoTorch: an open-source machine learning framework for constrained embedded platforms**.

## Before publishing

1. Confirm `CITATION.cff` and `.zenodo.json` use `https://github.com/modalfuse/picotorch`.
2. Run the checks in `docs/release-checklist.md`.
3. Push to GitHub `picotorch` and tag the release version.

## Connect GitHub to Zenodo

1. Sign in to Zenodo with the GitHub account that owns the repository.
2. Open the Zenodo GitHub integration and enable archiving for `picotorch`.
3. Create a GitHub release from a tag **after** the hook is enabled. A release published before the hook is not archived.
4. Zenodo archives the release and issues a **version DOI** plus a **concept DOI**.

Use the **concept DOI** in the manuscript Data and code section (it always resolves to the latest version). Pair it with the Pico-Former concept DOI `10.5281/zenodo.22218446` and the RICE-Former concept DOI `10.5281/zenodo.22171786` when those archives are cited.

Recorded identifiers for this tree:

| Identifier | Value |
| --- | --- |
| Concept DOI | [10.5281/zenodo.22231092](https://doi.org/10.5281/zenodo.22231092) |
| Version DOI (`v1.0.1`) | [10.5281/zenodo.22231093](https://doi.org/10.5281/zenodo.22231093) |
| GitHub | https://github.com/modalfuse/picotorch |

`v1.0.0` was published before the hook. The first archived snapshot is `v1.0.1`.

## After Zenodo issues the DOI

1. Add the concept DOI to `CITATION.cff` as `doi`.
2. Add the DOI badge and the Zenodo line to `README.md`.
3. Align `.zenodo.json` `description` with this README (title, four design points, included materials, layout, install, data boundary, citation).
4. Commit the metadata update. Do not cut a new GitHub release solely for that commit.

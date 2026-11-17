# Public data provenance

The fixture uses `sklearn.datasets.load_digits(n_class=10)` from scikit-learn **1.8.0**, not a new scrape or private input. This is the bundled 1,797-row subset of the UCI Optical Recognition of Handwritten Digits dataset. It is not the complete 5,620-row UCI train/test collection, and our fixed stratified 75/25 split is not UCI's original writer-disjoint split.

- Dataset authors: E. Alpaydin and C. Kaynak (1998).
- Dataset/attribution: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- DOI: https://doi.org/10.24432/C50P49
- UCI's stated license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Loader documentation: https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html
- Package source: https://pypi.org/project/scikit-learn/1.8.0/

The upstream 8×8 pixels are integers 0..16. The derived float inputs divide them by16; Q10 inputs multiply by64. The derived outputs include the fixed stratified split, trained CNN parameters, integer intermediate values and predictions. These are generated locally by the scripts, not vendored in the proposed public reproduction package. Upstream package licenses remain their own; the dataset attribution above applies to derived data.

The new 778-parameter model is trained once under `training_plan.json`. Its random initialization and optimizer/split seeds are public reproducibility seeds, not cryptographic private coins. No selected proving secret seed or Gaussian mask is present in this fixture.

# Dataset provenance and reuse

This packet uses `sklearn.datasets.load_digits(n_class=10)` from the installed
scikit-learn 1.8.0, with no dataset download or package installation.
The official [scikit-learn load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
identifies its 1,797 examples as a copy of the UCI handwritten-digits **test**
set. Each example has 64 integer features from 0 to 16.

The source is Alpaydin, E. and Kaynak, C. (1998), *Optical Recognition of
Handwritten Digits*, UCI Machine Learning Repository,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
The official [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
explicitly lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The UCI source describes 43 contributors and 8×8 counts derived from 32×32
bitmaps. Its full 5,620-example dataset is not the dataset trained here.
These official sources were checked on 2026-10-30.

Changes in this packet: preserve bundled row order; make a new stratified
train/holdout split; normalize by 16; train a new MLP; quantize parameters
and inputs to Q10. Original author attribution and license apply to the
redistributed example pixels and labels. This new split is **not** UCI's
original train/test evaluation and is not a writer-disjoint evaluation.
It is a small useful classification demonstration, not a state-of-the-art
benchmark, one of the paper's eleven legacy models, or evidence of
100-bit cryptographic security.

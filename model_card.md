# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier built with scikit-learn's
`RandomForestClassifier` (100 trees, `random_state=42`). It predicts whether an
individual's annual income exceeds $50,000 based on demographic and
employment attributes drawn from U.S. Census data. Categorical features are
encoded with a one-hot encoder and the target label is binarized with a label
binarizer; both transformers are fit on the training split and saved alongside
the model as pickle artifacts (`model/model.pkl`, `model/encoder.pkl`). The
model was developed for the WGU D501 / Udacity Machine Learning DevOps project
and is intended as a teaching example of a reproducible, deployable ML pipeline
rather than a production system.

## Intended Use

The model is intended to demonstrate an end-to-end machine learning workflow:
data processing, training, slice-based evaluation, unit testing, and
deployment behind a FastAPI REST endpoint. It is appropriate for educational
and demonstration purposes. It is not intended to inform real-world decisions
about individuals (for example, lending, hiring, or eligibility decisions),
because the underlying data is dated, exhibits demographic imbalance, and the
target is a coarse income threshold.

## Training Data

The training data is the publicly available Census Income ("Adult") dataset
containing 32,561 rows. Each record has 14 features (a mix of continuous
attributes such as `age`, `hours-per-week`, and `capital-gain`, and eight
categorical attributes such as `workclass`, `education`, `occupation`, and
`native-country`) plus the binary `salary` label. The label is imbalanced:
roughly 76% of records are `<=50K` and 24% are `>50K`. The raw data was
cleaned by removing extraneous spaces. The model was trained on an 80% split
of the data, stratified on the label to preserve the class ratio.

## Evaluation Data

The model was evaluated on the held-out 20% test split (also stratified on the
label). The same one-hot encoder and label binarizer fit on the training split
were applied to the test split, so no information from the test data leaked
into preprocessing.

## Metrics

The model is evaluated using precision, recall, and F1 (F-beta with beta = 1).
On the held-out test set the overall performance is:

- **Precision: 0.7353**
- **Recall: 0.6378**
- **F1: 0.6831**

Performance was also computed on slices of each categorical feature; the full
per-slice results are recorded in `slice_output.txt`. Performance varies
substantially across slices. Among slices with at least 100 test records,
the strongest groups include `education: Prof-school` (F1 = 0.8764),
`education: Masters` (F1 = 0.8343), and `occupation: Prof-specialty`
(F1 = 0.7989), while the weakest include `education: 7th-8th` (F1 = 0.0000),
`education: 10th` (F1 = 0.2353), and `education: 9th` (F1 = 0.2500). The low
scores generally occur on slices where high earners are rare, so the model
predicts the majority `<=50K` class and recall on the positive class collapses.

## Ethical Considerations

The dataset encodes protected attributes including `race`, `sex`, and
`native-country`, and the underlying population is imbalanced across these
groups. A model trained on this data can reproduce or amplify historical
disparities, and the slice metrics confirm that accuracy is not uniform across
groups. Because income is correlated with these sensitive attributes, using
the model for any consequential decision about a person could produce unfair
outcomes. Any deployment beyond demonstration should include a fairness audit,
disparate-impact analysis, and human oversight.

## Caveats and Recommendations

The Census Income dataset reflects a specific historical population and a fixed
$50,000 threshold, so the model does not generalize to current incomes or other
populations. The hyperparameters were not tuned and the continuous features
were not scaled, so there is headroom to improve performance through
cross-validation, hyperparameter search, and class-imbalance handling (for
example, class weighting or resampling). The per-slice results should guide
those improvements, with priority on the low-F1 education slices. The model
should be retrained on current, representative data and re-evaluated on slices
before any use beyond this educational project.

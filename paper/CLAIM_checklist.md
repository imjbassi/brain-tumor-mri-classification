# CLAIM 2024 Checklist

Checklist for Artificial Intelligence in Medical Imaging, completed for:

**Complete Patient-Level Leakage in a Widely Used Brain Tumor MRI Benchmark**
Bassi, J. (2026)

Section numbers refer to the manuscript. Items that do not apply to a
retrospective secondary analysis of public data are marked N/A with a reason,
rather than left blank.

| # | Item | Where addressed |
|---|---|---|
| **Title and abstract** |
| 1 | Identification as a study of AI methodology | Title; Abstract |
| 2 | Structured summary of design, methods, results, conclusions | Abstract |
| **Introduction** |
| 3 | Scientific and clinical background | §1 |
| 4 | Study objectives and hypotheses | §1 (contributions list) |
| **Methods** |
| 5 | Prospective or retrospective study | Retrospective secondary analysis of public data (§3, §"Ethics Statement") |
| 6 | Study goal | §1: audit an existing benchmark and re-evaluate a standard baseline on a valid split |
| 7 | Data sources | §3; Mendeley/Kaggle release, traced to the figshare collection of Cheng et al. (2015) |
| 8 | Eligibility criteria | §3.1–3.4: all 7,023 released images; exclusions documented (duplicates, untraced images) |
| 9 | Data pre-processing steps | §4.1 |
| 10 | Selection of data subsets | §3.2, §3.4, §6 (patient-disjoint fold construction) |
| 11 | Definitions of data elements | §3, Table 1 |
| 12 | De-identification methods | N/A — source collections distributed de-identified; recovered identifiers are arbitrary within-dataset labels (§"Ethics Statement") |
| 13 | How missing data were handled | §6: 437 untraced tumor images assigned to training only, never test |
| 14 | Definition of ground truth reference standard | §3; labels as distributed, with 3 disagreements against the source documented in §3.1 |
| 15 | Rationale for choosing the reference standard | §3 |
| 16 | Source of ground-truth annotations | Source collections (Cheng et al. 2015 and the aggregation) |
| 17 | Annotation tools | N/A — labels used as distributed |
| 18 | Measurement of inter- and intra-rater variability | N/A — no new annotation performed |
| **Model** |
| 19 | Detailed description of model | §4.2 (ResNet-18, ImageNet-pretrained, four-way head) |
| 20 | Software libraries, frameworks, packages | §5; `requirements.txt` with pinned versions |
| 21 | Initialization of model parameters | §4.2, §4.3 |
| **Training** |
| 22 | Details of training approach | §4.3, §5, Table 7 |
| 23 | Method of selecting the final model | §4.3; **and see §6**, which documents and corrects a contaminated validation split affecting exactly this item |
| 24 | Ensembling techniques | N/A — single model per run |
| **Evaluation** |
| 25 | Metrics of model performance | §6, §8; accuracy, per-class F1, macro AUC, ECE, NLL |
| 26 | Statistical measures of significance and uncertainty | §6 (patient bootstrap, variance decomposition), §7 (CI on the difference); §11 documents two withdrawn statistical claims |
| 27 | Robustness or sensitivity analysis | §3.1 (provenance sensitivity control), §6 (5-fold CV, variance components), §8.3 (ablation) |
| 28 | Methods for explainability | §10 (Grad-CAM), reported for both leaky and patient-disjoint models |
| 29 | Validation or testing on external data | **Not performed.** Stated as the principal limitation in §12 |
| **Results** |
| 30 | Flow of participants or cases | Table 1, Table 8; §6 fold construction |
| 31 | Demographic and clinical characteristics | N/A — not distributed with the source data; noted as a limitation in §12 |
| 32 | Performance metrics for optimal model | §6 (patient-disjoint, the figure we consider valid); §8 (released split, for comparability) |
| 33 | Failure analysis of incorrectly classified cases | §7 (error rates by duplication status), Appendix A |
| **Discussion** |
| 34 | Study limitations, including potential bias | §12; also §3.3, §6, §11 |
| 35 | Implications for practice | §13, including a recommended evaluation protocol for this dataset |
| **Other** |
| 36 | Registration number and registry name | N/A — not a clinical trial |
| 37 | Where the full study protocol can be accessed | Repository and archived release (see Code and Data Availability) |
| 38 | Sources of funding and other support; role of funders | §"Funding" — none |

## Notes on partial and negative items

- **Item 29 (external validation)** is the most significant unmet item. The
  paper's claim is deliberately scoped to what a corrected split measures on
  this collection, and it states explicitly that this is not a claim about
  clinical performance.
- **Items 31, 12, 17, 18** are unmet because the source collections do not
  distribute demographic data or annotation provenance. This is itself part
  of the paper's argument about the dataset.
- **Item 23** is the item this paper found itself failing, and §6 and §11
  document the failure and the correction rather than reporting only the
  corrected state.

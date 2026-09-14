# CLAIM 2024 Checklist

Completed for **Complete Patient-Level Leakage Among Traceable Images in a Widely Used Brain Tumor MRI Benchmark** by Jaiveer Bassi (2026).

Items that do not apply to this retrospective secondary analysis of public data are marked N/A with a reason.

| # | Item | Where addressed |
|---|---|---|
| **Title and abstract** |||
| 1 | Identification as a study of AI methodology | Title and Abstract |
| 2 | Summary of design, methods, results, and conclusions | Unstructured Abstract, as required by the target journal |
| **Introduction** |||
| 3 | Scientific and clinical background | Introduction |
| 4 | Study objectives and hypotheses | Final paragraph of Introduction |
| **Methods** |||
| 5 | Prospective or retrospective study | Methods; Ethics Statement (retrospective secondary analysis) |
| 6 | Study goal | Introduction (benchmark audit and corrected evaluation) |
| 7 | Data sources | Methods: Dataset and provenance recovery |
| 8 | Eligibility criteria | Methods: Dataset and provenance recovery; Evaluation designs |
| 9 | Data preprocessing steps | Methods: Model and training |
| 10 | Selection of data subsets | Methods: Evaluation designs |
| 11 | Definitions of data elements | Results, Table 1; Supplementary Table S1 |
| 12 | De-identification methods | N/A - source data were distributed de-identified; recovered identifiers are arbitrary within-dataset labels (Ethics Statement) |
| 13 | Handling of missing data | Methods: Evaluation designs (437 unmatched tumor images were training-only) |
| 14 | Ground-truth reference standard | Published folder labels; three source-label disagreements documented in Supplementary Methods |
| 15 | Rationale for reference standard | Methods: Dataset and provenance recovery |
| 16 | Source of ground-truth annotations | Source collections cited in Methods and Data Availability |
| 17 | Annotation tools | N/A - no new annotation was performed |
| 18 | Inter- and intra-rater variability | N/A - not reported by the aggregated source and no new annotation was performed |
| **Model and training** |||
| 19 | Detailed model description | Methods: Model and training |
| 20 | Software libraries, frameworks, packages | `requirements.txt` and Code Availability |
| 21 | Parameter initialization | Methods: Model and training (ImageNet-pretrained backbone; new four-class head) |
| 22 | Training approach | Methods: Model and training; Supplementary Table S4 |
| 23 | Final model selection | Methods: Model and training (validation-loss checkpointing and group-disjoint validation) |
| 24 | Ensembling | N/A - no ensemble was used |
| **Evaluation** |||
| 25 | Performance metrics | Results; Methods: Statistical analysis |
| 26 | Statistical significance and uncertainty | Methods: Statistical analysis; Results, Tables 2-4 |
| 27 | Robustness or sensitivity analysis | Matched random-fold control; provenance sensitivity control; seed and fold variance |
| 28 | Explainability | Results: Calibration and saliency; Figure 4 |
| 29 | External validation | Not performed; identified as the principal limitation in Discussion |
| **Results** |||
| 30 | Flow of participants or cases | Results, Table 1; Methods: Evaluation designs |
| 31 | Demographic and clinical characteristics | N/A - not distributed with the source collection; stated in Discussion as a limitation |
| 32 | Performance of the primary model | Results, Table 2; patient-disjoint result emphasized |
| 33 | Failure analysis | Results: Removing exact duplicates; Calibration and saliency; Supplementary Figure S3 |
| **Discussion** |||
| 34 | Limitations and potential bias | Discussion |
| 35 | Implications for practice | Discussion; explicit statement that the model is not for clinical use |
| **Other** |||
| 36 | Registration | N/A - not a clinical trial |
| 37 | Protocol access | GitHub release v1.2.2 in Data and Code Availability |
| 38 | Funding and role of funders | Funding - none |

## Material limitations

- External, institution-disjoint validation was not performed. The manuscript limits its claim to evaluation within this collection.
- The source collection does not provide demographic metadata or annotation-rater information.
- Patient identifiers were recovered for 91.3% of tumor images; unmatched tumor images were excluded from testing and retained only in training.
- The no-tumor class lacks subject identifiers and is grouped by near-duplicate cluster, so tumor-only performance is reported separately.

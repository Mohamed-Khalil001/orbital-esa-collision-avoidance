# Phase scripts (provenance only)

These scripts come from the six-phase research process. **They cannot be run from this repository.**
They read intermediate files (for example `01_train_event_cohort_audit.csv`, `04_cutoff_event_summary.csv` and pickled out-of-fold predictions) from the analysis sandbox where they were first executed (paths such as `/working_dir/...`), and those intermediate files are not part of the repository.

They are kept so that each phase of the report can be traced to the code that produced it. For code that runs end to end from the raw ESA files, use the notebooks in `research/notebooks/`.

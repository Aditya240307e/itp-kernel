# data/results -- result files of runs

`v1/` holds the result files of the runs made on **dataset v1** (82 problems, 9 IMO; `data/dataset_v1` has been replaced by `data/dataset_v2`): the blind search on the three splits, one behaviour-cloning run, a smoke
test of the policy-guided search and of the RL loop.  They are a historical record of the pipeline's first measurements (`docs/TRAINING.md`, "what the first learner did"), they name problems of dataset v1 (some
of them have since become library lemmas or were replaced), and they are NOT baselines of dataset v2.  Baselines and learned results on dataset v2 are produced by the scripts in `scripts/` (see
`docs/EXPERIMENTS.md`) and are written under `results/` of the directory they are run from.

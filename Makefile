# Makefile -- the common tasks.  `make help` lists them.   PY=python3.13 make gates   J=1 make gates   (J: how many gates run at once)
# The experiments (docs/EXPERIMENTS.md) are run by scripts/*.sh, which read their settings from the environment and from scripts/local.env; the targets below only call them:
#   PROFILE=small SEEDS=0-2 make experiments        DEVICE=cuda PROFILE=full SEEDS=0-6 make experiments
PY ?= python3
J  ?= 2

.PHONY: help kernel gates gates-ml gates-repo gates-synth gates-all dataset-verify examples mutants-check sanitize dist clean check-env smoke experiments status report paper

help:
	@echo "make kernel          build the hybrid kernel and the frozen reference kernel into build/"
	@echo "make gates           the 47 kernel / tactic / environment / bank gates (about 13 minutes on two cores)"
	@echo "make gates-ml        the learner's gates S44 (machinery) and S51 (training suite, experiment runner); they need torch and gymnasium"
	@echo "make gates-repo      the repository gates: S45 layout, S46 examples"
	@echo "make gates-synth     the synthetic theorem generator's gate S48"
	@echo "make gates-all       every gate"
	@echo "make dataset-verify  replay every recorded proof of data/dataset_v2 in a fresh engine (about 1 minute)"
	@echo "make examples        run the examples"
	@echo "make mutants-check   check that every mutant of the catalogue still applies to the code (seconds); the mutation run itself: python3 mutants/mutate.py PREFIX"
	@echo "make sanitize        build the kernel with ASan + UBSan and run the gates under it (long)"
	@echo "make dist            pack the repository into dist/*.tar.gz and dist/*.zip   (ARTIFACTS=dir adds trained models under artifacts/)"
	@echo "make check-env       is this machine ready for the experiments? (python, torch, CUDA, memory, disk)"
	@echo "make smoke           the whole experiment pipeline on a tiny world (1-2 hours on two cores): do this before a real run"
	@echo "make experiments     the experiments of the paper, resumable   (PROFILE=small|full  SEEDS=0-4  DEVICE=cuda  JOBS=n ...: scripts/env.sh)"
	@echo "make status          what is done, running, failed"
	@echo "make report          tables, figures and numbers from the runs so far (+ paper/generated/ for the full profile)"
	@echo "make paper           build paper/main.pdf (needs latexmk)"
	@echo "make clean           remove build/, dist/ and the Python caches"

kernel:
	$(PY) src/python/build_kernel.py all

gates: kernel
	$(PY) tests/run_gates.py -j $(J)

gates-ml: kernel
	$(PY) tests/run_gates.py -j 1 s44_ml_tests.py s51_suite_tests.py

gates-repo: kernel
	$(PY) tests/run_gates.py -j 1 s45_layout_tests.py s46_examples_tests.py

gates-synth: kernel
	$(PY) tests/run_gates.py s48_synth_tests.py

gates-all: kernel
	$(PY) tests/run_gates.py -j $(J) --all

dataset-verify: kernel
	$(PY) src/python/problems/dataset.py --verify data/dataset_v2

examples: kernel
	@for f in examples/[0-9][0-9]_*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

mutants-check:
	$(PY) mutants/mutate.py --check

sanitize: kernel
	$(PY) tools/sanitize.py

dist:
	$(PY) tools/make_dist.py $(if $(ARTIFACTS),--artifacts $(ARTIFACTS),)

# the scripts take PROFILE and PY from the call, then from scripts/local.env, then from their defaults: pass on only what the caller set (this Makefile's own default PY is not a setting)
SETTINGS = $(if $(filter-out file,$(origin PY)),PY=$(PY) )$(if $(PROFILE),PROFILE=$(PROFILE) )

check-env: kernel
	$(PY) tools/check_env.py

smoke: kernel
	$(SETTINGS)./scripts/smoke_test.sh

experiments: kernel
	$(SETTINGS)./scripts/run_all.sh

status:
	$(SETTINGS)./scripts/status.sh

report:
	$(SETTINGS)./scripts/07_report.sh

paper:
	$(MAKE) -C paper

clean:
	rm -rf build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

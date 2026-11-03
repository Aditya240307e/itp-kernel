# Makefile -- the common tasks.  `make help` lists them.   PY=python3.13 make gates   J=1 make gates   (J: how many gates run at once)
# The experiments (docs/EXPERIMENTS.md) are run by scripts/*.sh, which read their settings from the environment and from scripts/local.env; the targets below only call them:
#   PROFILE=small SEEDS=0-2 make experiments        DEVICE=cuda PROFILE=full SEEDS=0-6 make experiments
PY ?= python3
J  ?= 2

.PHONY: help kernel gates gates-ml gates-repo gates-synth gates-all dataset-verify examples mutants-check tool-mutants soundness audit sanitize dist clean check-env smoke experiments status report paper

help:
	@echo "make kernel          build the hybrid kernel and the frozen reference kernel into build/"
	@echo "make gates           the 49 default gates: kernel, tactic, environment, bank and contest proofs (about 45 minutes on two cores)"
	@echo "make gates-ml        the learner's gates S44 (machinery) and S51 (training suite, experiment runner); they need torch and gymnasium"
	@echo "make gates-repo      the repository gates: S45 layout, S46 examples"
	@echo "make gates-synth     the synthetic theorem generator's gate S48"
	@echo "make gates-all       every gate (54; about an hour on two cores)"
	@echo "make dataset-verify  replay every recorded proof and sub-theorem of data/dataset_v2 in a fresh engine (about 25 minutes, 3 GB)"
	@echo "make examples        run the examples"
	@echo "make mutants-check   check that every mutant of the catalogues (the kernel's, and the faults of the tools below) still applies to the code (seconds); the mutation run itself: python3 mutants/mutate.py PREFIX"
	@echo "make tool-mutants    inject faults into the tools that make the evidence (soundness folder, statement audit, number checker, identity of the code) and check that their tests fail for each one; 252 faults, 47 minutes on two cores (J=8: a few)"
	@echo "make soundness      the soundness evidence in one stamped folder (results/soundness): every gate, every mutant, the fuzz campaigns, kernel against the independent checker, the findings; hours (J=8 for the big machine)"
	@echo "make audit          the statement audit in one stamped folder (results/audit): every statement of the bank against its intended meaning, the cautions, the ledger of second readers, a review sheet; minutes"
	@echo "make sanitize        build the kernel with ASan + UBSan and run the kernel and bank gates that can run under them (long; it lists the ones it leaves out, and why)"
	@echo "make dist            pack the repository into dist/*.tar.gz and dist/*.zip   (GIT=1 adds the history as a .git directory; ADD='paper/main.pdf' adds files git does not hold; ARTIFACTS=dir adds trained models under artifacts/)"
	@echo "make check-env       is this machine ready for the experiments? (python, torch, CUDA, memory, disk)"
	@echo "make smoke           the whole experiment pipeline on a tiny world (about 70 minutes of run time on two cores): do this before a real run"
	@echo "make experiments     the experiments of the paper, resumable   (PROFILE=small|full  SEEDS=0-4  DEVICE=cuda  JOBS=n ...: scripts/env.sh)"
	@echo "make status          what is done, running, failed"
	@echo "make report          the results folder (results/$$PROFILE) from the runs so far: tables, figures, per-problem outcomes, statistics, provenance, manifest (PUBLISH=1: also paper/generated/)"
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
	$(PY) mutants/check_tools.py --check

tool-mutants: kernel
	$(PY) mutants/check_tools.py --jobs $(J)

sanitize: kernel
	$(PY) tools/sanitize.py

dist:
	$(PY) tools/make_dist.py $(if $(GIT),--git,) $(if $(ARTIFACTS),--artifacts $(ARTIFACTS),) $(foreach f,$(ADD),--add $(f))

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

soundness: kernel
	$(SETTINGS)JOBS=$(J) ./scripts/08_soundness.sh

audit: kernel
	$(SETTINGS)./scripts/09_audit.sh

paper:
	$(MAKE) -C paper

clean:
	rm -rf build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

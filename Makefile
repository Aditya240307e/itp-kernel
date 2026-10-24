# Makefile -- the common tasks.  `make help` lists them.   PY=python3.13 make gates   J=1 make gates   (J: how many gates run at once)
PY ?= python3
J  ?= 2

.PHONY: help kernel gates gates-ml gates-repo gates-all dataset-verify examples mutants-check sanitize dist clean

help:
	@echo "make kernel          build the hybrid kernel and the frozen reference kernel into build/"
	@echo "make gates           the 47 kernel / tactic / environment / bank gates (about 10 minutes on two cores)"
	@echo "make gates-ml        the learner's gate S44 (needs torch and gymnasium)"
	@echo "make gates-repo      the repository gates: S45 layout, S46 examples"
	@echo "make gates-all       every gate"
	@echo "make dataset-verify  replay every recorded proof of data/dataset_v1 in a fresh engine (about 1 minute)"
	@echo "make examples        run the examples"
	@echo "make mutants-check   check that every mutant of the catalogue still applies to the code (seconds); the mutation run itself: python3 mutants/mutate.py PREFIX"
	@echo "make sanitize        build the kernel with ASan + UBSan and run the gates under it (long)"
	@echo "make dist            pack the repository into dist/*.tar.gz and dist/*.zip   (ARTIFACTS=dir adds trained models under artifacts/)"
	@echo "make clean           remove build/, dist/ and the Python caches"

kernel:
	$(PY) src/python/build_kernel.py all

gates: kernel
	$(PY) tests/run_gates.py -j $(J)

gates-ml: kernel
	$(PY) tests/run_gates.py s44_ml_tests.py

gates-repo: kernel
	$(PY) tests/run_gates.py -j 1 s45_layout_tests.py s46_examples_tests.py

gates-all: kernel
	$(PY) tests/run_gates.py -j $(J) --all

dataset-verify: kernel
	$(PY) src/python/problems/dataset.py --verify data/dataset_v1

examples: kernel
	@for f in examples/[0-9][0-9]_*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

mutants-check:
	$(PY) mutants/mutate.py --check

sanitize: kernel
	$(PY) tools/sanitize.py

dist:
	$(PY) tools/make_dist.py $(if $(ARTIFACTS),--artifacts $(ARTIFACTS),)

clean:
	rm -rf build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

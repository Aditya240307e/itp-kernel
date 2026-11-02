# PRIOR_WORK.md -- what the literature says (checked 2026-11-02), what it does not, and what is left to read

A working note for the person who writes the related-work section. It is **not** text for the paper, and it
is not a literature review: it records what was looked up, so that "first", "novel" and "unlike" are
written only after the systems below have been read by you.

How to read the tags. **[read]**: extracted on 2026-11-02 from the paper or its page by a web-fetch tool that summarises
a page with a small model; check every number and quote in the PDF before it enters a paper. **[not found]**: the text I
was given did not say it; that is not proof that the paper does not. **[memory]**: from my memory of the paper, not
read in this session; verify before use. Nothing here was measured on this system.

## 1. The systems closest to this one

| system | what it is | facts [read] | not found in the text read | where it touches this work |
|:---|:---|:---|:---|:---|
| **Peano** (Poesia, Goodman; Phil. Trans. R. Soc. A 2023, arXiv 2211.15864) | a dependently typed theorem-proving environment with a *finite* action space | case study on five Khan Academy algebra sections over the reals; forward steps only; arithmetic evaluation is an atomic axiom outside the formal system; ConPoLe (contrastive policy, character-level GRU), tactic induction by anti-unification, an induced curriculum; stated limits: no inline lambdas/lemmas, no backward steps, no recursion in tactics | kernel size or trusted base; throughput or latency; the implementation language (the code is referenced on GitHub, which was not reachable from here) | the closest system. Different: a larger bank with an official-contest tier, a kernel with a deterministic work meter, an independent checker and a mutation suite, a typed term builder instead of enumerating all actions (**to verify** against Peano's code, not assumed) |
| **Minimo** (Poesia, Broman, Haber, Goodman; NeurIPS 2024 [memory: the PDF read carries a NeurIPS checklist and names no venue], arXiv 2407.00695) | an agent that poses conjectures and proves them in Peano, starting from axioms only | propositional logic, natural-number arithmetic and group theory; MCTS with the language model as policy and value; **hindsight relabeling on proof search trees**; about 8.45 M parameters, 288 GPU hours for the main results; extrinsic test on 35 theorems from Kleene and 10 Natural Number Game levels, "admittedly small scale"; stated limits: "its library remains fixed to the definitions and axioms that it starts with", and "a large library can cause the current action enumeration algorithm in Peano to become prohibitively slow" | any claim about speed of the environment in numbers | **hindsight is not new.** What this system adds is a library that grows (`ei_her_boot`: a hindsight theorem is replayed by the kernel and admitted as a global) and an olympiad-oriented bank. Both are *candidate* differences: read the follow-up work before saying so |
| **INT** (Wu, Jiang, Ba, Grosse; ICLR 2021, arXiv 2007.02924) | a synthetic benchmark of inequality theorems with a generator and a lightweight proof assistant | theorems generated from the ordered-field axioms by applying axioms to a core statement; six generalization axes (iid, initial conditions, axiom orders, axiom combinations, number of axioms, proof length); "1.28 ms per step versus 7.96 ms in HOL Light" (6.2x); behaviour cloning with a transformer or a GNN, MCTS on top | a statement about what the assistant trusts | **the nearest prior art for "a fast lightweight assistant plus a synthetic theorem generator".** The paper must cite it, compare the design of the generator, and not claim a first. Different: algebra and number theory, kernel-checked proofs, and a real-problem transfer test (`test`, `test_comp`) rather than generalization axes |
| **LeanDojo** (Yang et al., NeurIPS 2023, arXiv 2306.15626) | a Lean toolkit: data extraction, a gym-like `Dojo` environment, a retrieval-augmented prover | 98 734 theorems and 217 776 tactics from Lean 3 mathlib in the benchmark; `initialize` and `run_tac`, failed tactics return an error state; "proof-checking misjudgments" of 21.1 % (lean-gym) down to 1.4 %; a 10-minute wall-time limit per theorem; docs: tracing mathlib takes about one hour on 32 CPUs and needs at least 32 GB of memory, default tactic timeout 5000 ms (Lean 3), a Lean 4 version exists | per-step latency or throughput measurements | the usual answer to "why not Lean". Different layer: it makes the *interface* to Lean reliable; this system makes the *kernel and the cost model* small and auditable |
| **Pantograph** (Aniva et al., arXiv 2410.16429) | a machine-to-machine interface to Lean 4 | supports proof search such as MCTS, handles Lean 4 inference steps more robustly, a use case with a model and proof sketches | benchmarks, timings and memory (the abstract has none) | same role as LeanDojo for Lean 4 |
| **Kimina Lean Server** (Dos Santos et al., arXiv 2504.21230) | a high-performance Lean 4 verifier for RL pipelines | 9 419 NuminaMath-LEAN proofs verified in 42:40 on 8 cores down to 7:56 on 64 (about 0.27 s down to 0.05 s per proof; about 20 proofs/s at 64 cores, my arithmetic), against 87:35 down to 12:56 for the next tool; 72 vCPU, 540 GB machine, Lean v4.15.0; cached imports | memory measurements | **the throughput bar of the Lean side**, and for *whole proofs*, not one action. A probe here is one action, so the units differ; say so |
| **PVSGym** (Acharya, Nukala, Shankar; NeurIPS 2025 MATH-AI workshop) | an RL gym for the PVS prover with a dataset from the NASA library | abstract only | theorem count, API, trust, speed | shows that "a gym for a proof assistant" is not new; cite it |

Numbers for other systems come from their papers, on their hardware and tasks. They are not a comparison with this system.

## 2. Still to read before the related-work section is written ([memory] only)

CoqGym (Yang, Deng 2019), GamePad (Huang et al. 2019), HOList and DeepHOL (Bansal et al. 2019), Holophrasm and GPT-f on Metamath, miniF2F
(Zheng et al.), ProofNet, PutnamBench, Lean-gym and LLMSTEP, DeepSeek-Prover (synthetic data by autoformalization: a different meaning of
"synthetic"; arXiv 2405.14333, found in a search), Goedel-Prover-V2, Seed-Prover, AlphaProof (Hubert et al. 2025). For each, answer in one
line: *what does it trust? how fast is a step? how is cost counted? does it have hindsight, a growing library, a generator?*

Also read about the Lean kernel: it is small and has independent re-implementations (lean4checker, Nanoda, Lean4Lean -- **[memory]**).
A reviewer will say "Lean's kernel is also small and has second checkers". The answer is in the scope: here the *whole* stack up to the
action interface is small and mutation-tested, and costs are deterministic; but write it only after checking what those projects provide.

## 3. What this changes in our claims (see also `docs/PAPER_PLAN.md`)

1. **Hindsight is not a contribution**; Minimo already uses it on proof trees (and the idea is hindsight experience replay). Say "we use", never "we introduce". The title may keep the word if the abstract says what is used.
2. **A growing library** is a candidate difference from Minimo's stated limit; claim it only as an ablation result (`ei_her_boot` vs `ei_her`), with its interval, and only if the result is positive.
3. **Synthetic theorems with proofs** exist (INT). Do not write "first"; write what the generator is: one family, algebra and number theory, kernel-checked, with yield and saturation statistics.
4. **Speed**: INT reported 1.28 ms per step for its own assistant; Kimina reports 0.05 to 0.27 s per whole Lean proof (a different unit). Report probes per second, latency percentiles and memory from `scripts/06_benchmarks.sh` on an idle machine and say exactly what a probe is. The smoke numbers of the development machine (blind search about 890 probes/s on one shared core, one probe including the interface) show only that the engine is in the millisecond range per probe; they are not for the paper.
5. **Trust**: the claim that survives is "a small trusted base, tested four ways, and every number in the paper is bound to the code that produced it" -- no claim that it is more trustworthy than Lean's kernel.
6. **Benchmarks**: 268 problems is small beside miniF2F's 488 or PutnamBench; the selling point is the level structure and the family-disjoint split, not the size.

## 4. Feature matrix (fill with [read] entries only)

`+` the text read says it has it; `-` the text read says it does not, or describes a different thing (named in the cell); `?` not found in the text read, to be checked in the paper or the code.

| feature | Peano | Minimo | INT | LeanDojo | this system |
|:---|:---:|:---:|:---:|:---:|:---:|
| a gym-like single-step interface | + | + | + | + | + |
| size or trust of the kernel discussed | ? | ? | ? | ? | + |
| a deterministic work meter in the kernel | ? | ? | ? | ? (wall-clock timeouts are what the docs mention) | + |
| an independent second checker | ? | ? | ? | ? | + |
| step latency reported | ? | ? | + (1.28 ms) | ? | + (to run) |
| a generator of theorems with proofs | - (problem templates) | - (conjectures; proofs by search) | + | - (extracted from mathlib) | + |
| hindsight relabeling | ? | + | ? | ? | + |
| new theorems added to the library during training | - (tactics only) | - (library fixed) | ? | ? | + (`ei_her_boot`) |
| contest-level statements | - (Khan Academy algebra) | - (Kleene, Natural Number Game) | - (generated inequalities) | via miniF2F, evaluated | statements, few proofs |

The LeanDojo cell for the meter is `?` on purpose: the text read mentions a 5000 ms default tactic timeout and a 10-minute wall limit, which are wall-clock; it does not say that Lean lacks a
deterministic option (Lean has a `maxHeartbeats` counter, **[memory]**; check it before writing about it).

## 5. A Lean comparison (your decision; nothing was done)

* **A. None.** State that the published numbers are on other tasks and machines. Cheapest; weakest; most honest default.
* **B. Micro-comparison.** Formalize 30 to 50 level-1 and level-2 bank lemmas in Lean 4 without Mathlib, time the check of each proof file and, through the Lean REPL, each tactic step, on the same machine and report both
  with the caveat that Lean elaborates more. A day or two of your time plus the Lean toolchain (not installed in this sandbox).
* **C. Run the bank in a Lean-based system.** Out of scope for this paper.

Whatever you choose, say in the text which option it was; a reviewer will ask.

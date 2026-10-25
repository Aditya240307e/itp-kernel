"""bank_all.py -- the whole problem bank, loaded as one world, and the splits that tell a learner what it may train on.

The modules register their problems in `bank.BANK` when they are imported, in the order below; the world below the problems of a module is the one the modules before it built (so the digests pinned in the gates for the
21 problems of bank_lib, for the 51 of bank_lib2, ... do not move when a module is added at the end).  268 problems: 187 PROVED (a kernel-checked reference proof) and 81 STATEMENT-ONLY (the formal statement, its
checked meaning and brute-force evidence, no proof):

    bank_lib       21  the first graded problems (gate S41); the library below them ends at the theorem imo1959_p1
    bank_lib2      51  more graded problems, level 1-4, proved (gate S43)
    bank_amc       AMC-style and AIME-style problems with reference proofs
    bank_amc2      72 problems written in the manner of AMC (origin 'style': no claim to be AMC problems), with reference proofs
    bank_aime2     AIME-style problems with larger answers
    bank_imo       IMO problems, statement-only (the formal statement, no proof; faithfulness-checked and truth-screened)
    bank_imo2      ISL problems, statement-only
    bank_imo3      olympiad lemmas and a few contest problems WITH reference proofs (IMO 1977 P6, IMO 1983 P6, ISL 2005 N3, Vasc, ..)
    bank_imo4      more shortlist problems and classical exercises, statement-only
    bank_classic   classical results and exercises of olympiad number theory, statement-only

The contest modules (the AMC/AIME tier and the IMO tier) are checked by gate S49 (provenance, composition, statements, the statement-only machinery) and S50 (premises mask and proofs).

    import bank_all
    bk = bank_all.load()                    # the world is built once (about a minute)
    bank_all.names()                        # the names of the problems, in the order of the world
    bank_all.NEW                            # the names bank_lib2 adds (the world below them is the one of the first 21)
    bank_all.MODULES                        # {module name: [problem names]}
    bank_all.proved_names() / bank_all.statement_only_names()   # the two tiers
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
import bank_lib, bank_lib2                  # importing registers the problems in bank.BANK, in this order
import bank_amc, bank_amc2, bank_aime2, bank_imo, bank_imo2, bank_imo3, bank_imo4, bank_classic
from bank import BANK

NEW = bank_lib2.NAMES
MODULES = {m.__name__: list(getattr(m, 'NAMES', [])) for m in (bank_lib2, bank_amc, bank_amc2, bank_aime2, bank_imo, bank_imo2, bank_imo3, bank_imo4, bank_classic)}


def load(only=None):
    return bank_lib.load(only)


def names(): return list(BANK.problems)


def proved_names(): return [n for n, p in BANK.problems.items() if p.proved]


def statement_only_names(): return [n for n, p in BANK.problems.items() if not p.proved]

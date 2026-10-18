"""bank_all.py -- the whole problem bank: the 28 problems of bank_lib.py (gate S41) and the problems of bank_lib2.py (gate S43), loaded as one world, and the splits that tell a learner what it may train on.

    import bank_all
    bk = bank_all.load()                    # 82 problems; the world is built once (a few seconds)
    bank_all.NAMES                          # their names, in the order of the world
    bank_all.NEW                            # the names bank_lib2 adds (the world below them is the one of the first 28)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
import bank_lib, bank_lib2                  # importing registers the problems in bank.BANK, in this order
from bank import BANK

NEW = bank_lib2.NAMES


def load(only=None):
    return bank_lib.load(only)


def names(): return list(BANK.problems)

"""Inspect and load downloaded pickles without running their code (E2 and E5 downloads; gate G4).

`globals_in(path)` lists every (module, name) a pickle would import, read from its opcodes with `pickletools`
and nothing executed. `load(path, allowed)` unpickles with an Unpickler whose find_class refuses everything outside
`allowed`; plain containers, strings and numbers need no global at all.
"""
from __future__ import annotations

import io
import pickle
import pickletools

NUMPY_ARRAYS = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


def globals_in(path) -> set:
    """Every (module, name) the pickle would import; STACK_GLOBAL operands are resolved from the preceding strings."""
    with open(path, "rb") as handle:
        data = handle.read()
    found, strings = set(), []
    for opcode, argument, _ in pickletools.genops(io.BytesIO(data)):
        if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING", "STRING"):
            strings.append(argument)
        elif opcode.name == "GLOBAL":
            module, name = argument.split(" ", 1)
            found.add((module, name))
        elif opcode.name == "STACK_GLOBAL":
            if len(strings) < 2:
                found.add(("<unresolved>", "<unresolved>"))
            else:
                found.add((strings[-2], strings[-1]))
        elif opcode.name in ("INST", "OBJ", "REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4"):
            found.add(("<opcode>", opcode.name))
    return found


class _Restricted(pickle.Unpickler):
    def __init__(self, handle, allowed: set):
        super().__init__(handle)
        self.allowed = allowed

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError("refused global %s.%s" % (module, name))
        return super().find_class(module, name)


def load(path, allowed: set = frozenset()):
    with open(path, "rb") as handle:
        return _Restricted(handle, set(allowed)).load()

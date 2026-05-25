"""CDFG JSON schema constants."""

NODE_INSTRUCTION = 0
NODE_VARIABLE = 1
NODE_CONSTANT = 2

FLOW_CONTROL = 0
FLOW_DATA = 1
FLOW_CALL = 2

NODE_TYPES = ["instruction", "variable", "constant"]

EDGE_TYPES = [
    ("instruction", "control", "instruction"),
    ("instruction", "data", "variable"),
    ("variable", "data", "instruction"),
    ("constant", "data", "instruction"),
    ("instruction", "call", "instruction"),
]

EDGE_TYPES_WITH_ATTR = {
    ("instruction", "control", "instruction"),
    ("variable", "data", "instruction"),
    ("constant", "data", "instruction"),
}

LABEL_KEYS = [
    "lut",
    "ff",
    "dsp",
    "bram",
    "uram",
    "cycles_max",
    "cycles_min",
    "estimated_clock",
    "interval_max",
    "interval_min",
    "target_clock",
]


def safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return -1

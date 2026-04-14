"""
Sentinel values used across the schema system for kwargs metadata.

These interned strings enable fast identity comparison (via `is`) instead of
equality comparison when checking whether a node argument is required or
accepts any type.
"""

import sys

# Marks a kwarg/forward_kwarg as mandatory -- the caller must supply a value.
__REQUIRED__ = sys.intern("__required__")

# Marks a kwarg type as unconstrained -- any value/type is accepted.
__ANY__ = sys.intern("__any__")

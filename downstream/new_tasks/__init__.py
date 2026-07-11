"""Strict, evidence-aware downstream tasks for the maintained ChatPathway2 pipeline.

The package is intentionally separate from :mod:`downstream.tasks`.  It
defines the revised Task 0--6 contracts without changing the historical task
entry points used by existing result artifacts.
"""

from .schemas import SchemaError

__all__ = ["SchemaError"]

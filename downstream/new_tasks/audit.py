#!/usr/bin/env python3
"""Audit the revised task matrix and import every declared evaluator."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from downstream.new_tasks.schemas import SchemaError


def audit() -> dict[str, object]:
    matrix_path = Path(__file__).with_name("matrix.json")
    with matrix_path.open(encoding="utf-8") as handle:
        matrix = json.load(handle)
    if matrix.get("schema_version") != 1 or not isinstance(matrix.get("tasks"), list):
        raise SchemaError("new_tasks/matrix.json has an unsupported schema.")
    expected_ids = [f"task{index}" for index in range(7)]
    task_ids = [task.get("id") for task in matrix["tasks"]]
    if task_ids != expected_ids:
        raise SchemaError(f"task IDs must be exactly {expected_ids}; got {task_ids}.")
    modules = []
    for index, task in enumerate(matrix["tasks"]):
        if not isinstance(task, dict) or not task.get("module"):
            raise SchemaError(f"matrix task[{index}] needs a module.")
        module = importlib.import_module(str(task["module"]))
        if not callable(getattr(module, "main", None)):
            raise SchemaError(f"{task['module']} does not expose main().")
        modules.append(str(task["module"]))
    return {"matrix": str(matrix_path), "num_tasks": len(modules), "modules": modules}


def main() -> None:
    result = audit()
    print(f"Revised downstream audit passed for {result['num_tasks']} tasks.")
    for module in result["modules"]:
        print(f"  {module}")


if __name__ == "__main__":
    main()

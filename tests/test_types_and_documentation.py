"""Check project-wide documentation coverage and resolvable type annotations.

The source scan requires a module description and docstrings for every class
and function, plus explicit parameter and return annotations except self/cls.
Runtime checks resolve annotations on supported import-safe modules to catch
missing type names that compilation alone would not detect.

The scan excludes virtual environments and image datasets by enumerating only
project source directories. Historical Caffe and plotting scripts are parsed
for documentation coverage but not imported, because their original dependencies
and import-time side effects remain unsupported.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def python_sources() -> list[Path]:
    """List project Python sources without traversing environments or image datasets."""
    sources = list(REPOSITORY_ROOT.glob("*.py"))
    for directory in (
        "models",
        "ssd_detector",
        "testing_helpers",
        "tests",
    ):
        sources.extend((REPOSITORY_ROOT / directory).rglob("*.py"))
    return sorted(sources)


def test_all_source_interfaces_have_types_and_docstrings() -> None:
    """Require module/class/function docs and all non-self parameter/return annotations."""
    for source_path in python_sources():
        syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
        module_description = ast.get_docstring(syntax_tree)
        assert module_description, f"Missing module docstring: {source_path}"
        assert len([line for line in module_description.splitlines() if line.strip()]) >= 5, (
            f"Module description must explain responsibilities and limitations: {source_path}"
        )
        for syntax_node in ast.walk(syntax_tree):
            if isinstance(syntax_node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                location = f"{source_path.relative_to(REPOSITORY_ROOT)}:{syntax_node.lineno} {syntax_node.name}"
                assert ast.get_docstring(syntax_node), f"Missing docstring: {location}"
                if isinstance(syntax_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert syntax_node.returns is not None, f"Missing return annotation: {location}"
                    parameters = [
                        *syntax_node.args.posonlyargs,
                        *syntax_node.args.args,
                        *syntax_node.args.kwonlyargs,
                    ]
                    parameters.extend(
                        parameter
                        for parameter in (syntax_node.args.vararg, syntax_node.args.kwarg)
                        if parameter
                    )
                    for parameter in parameters:
                        if parameter.arg not in {"self", "cls"}:
                            assert (
                                parameter.annotation is not None
                            ), f"Missing parameter annotation: {location}.{parameter.arg}"


def check_module_hints(module: ModuleType) -> None:
    """Resolve annotations on module-owned functions, classes, and methods."""
    get_type_hints(module)
    for module_member in vars(module).values():
        if getattr(module_member, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(module_member):
            get_type_hints(module_member)
        elif inspect.isclass(module_member):
            get_type_hints(module_member)
            for class_member in vars(module_member).values():
                if inspect.isfunction(class_member):
                    get_type_hints(class_member)


def test_runtime_annotations_resolve() -> None:
    """Catch unresolved annotation names without executing archival scripts or demos."""
    module_names = [
        "type_definitions",
        "device_and_initialization",
        "command_line_options",
        "train_gaze_model",
        "check_pytorch_runtime",
        "gaze_metrics",
        "training_support",
        "gazefollow_dataset",
        "gaze_augmentations",
        "training_engine",
        "models",
        "models.training_setup",
        "models.gaze_following_network",
        "visualize_gaze_predictions",
        "predict_gaze_from_faces",
        "ssd_detector.s3fd_network",
        "ssd_detector.bounding_boxes",
        "ssd_detector.face_detection",
        "ssd_detector.evaluate_wider_faces",
    ]
    for module_name in module_names:
        check_module_hints(importlib.import_module(module_name))

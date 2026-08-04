from __future__ import annotations

import ast
import re
import sys
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.domains.python import _parse_annotation
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypeAliasInfo:
    name: str
    alias_of: str
    description_lines: list[str]
    alias_detail_lines: list[str]
    top_level: str
    canonical_name: str
    defining_name: str
    path: Path
    lineno: int
    target_id: str
    import_aliases: dict[str, str] = field(default_factory=dict)
    produced_by: list[str] = field(default_factory=list)
    consumed_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskInfo:
    qualname: str
    inputs: list[str]
    output: str | None


def _module_name(package_root: Path, package_name: str, path: Path) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    return ".".join((package_name, *relative.parts))


def _top_level_name(package_root: Path, package_name: str, path: Path) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    if len(relative.parts) <= 1:
        return package_name
    return relative.parts[0]


def _canonical_name(package_name: str, top_level: str, alias_name: str) -> str:
    if top_level == package_name:
        return f"{package_name}.{alias_name}"
    return f"{package_name}.{top_level}.{alias_name}"


def _target_id(canonical_name: str) -> str:
    return "mimodium-type-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", canonical_name)


def _reference_names(alias: TypeAliasInfo) -> set[str]:
    return {alias.name, alias.canonical_name, alias.defining_name}


def _doc_comment_lines(source_lines: list[str], lineno: int) -> list[str]:
    previous_index = lineno - 2
    if previous_index < 0 or not source_lines[previous_index].startswith("#:"):
        return []

    doc_lines: list[str] = []
    index = previous_index
    while index >= 0 and source_lines[index].startswith("#:"):
        line = source_lines[index][2:]
        if line.startswith(" "):
            line = line[1:]
        doc_lines.append(line)
        index -= 1
    doc_lines.reverse()
    return doc_lines


def _is_alias_detail_line(line: str) -> bool:
    return re.match(r"\s*:(shape|dtype):", line) is not None


def _split_doc_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    description_lines: list[str] = []
    alias_detail_lines: list[str] = []
    for line in lines:
        if _is_alias_detail_line(line):
            alias_detail_lines.append(line)
            continue
        description_lines.append(line)

    while description_lines and not description_lines[-1].strip():
        description_lines.pop()

    return description_lines, alias_detail_lines


def _is_task_decorator(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Name) and decorator.id == "task"


def _simple_annotation_name(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    return None


def _type_alias_value(source: str, node: ast.TypeAlias) -> str:
    segment = ast.get_source_segment(source, node.value)
    if segment is None:
        segment = ast.unparse(node.value)
    return textwrap.dedent(segment).strip()


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    imported_name = alias.name.split(".", 1)[0]
                    aliases[imported_name] = imported_name
                    continue
                aliases[alias.asname] = alias.name
            continue

        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in {"typing", "typing_extensions"}:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported_name = f"{node.module}.{alias.name}"
                aliases[alias.asname or alias.name] = imported_name

    return aliases


class _ImportAliasResolver(ast.NodeTransformer):
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.expr:
        replacement = self.aliases.get(node.id)
        if replacement is None or replacement == node.id:
            return node
        replacement_node = ast.parse(replacement, mode="eval").body
        return ast.copy_location(replacement_node, node)


def _referencable_annotation(annotation: str, import_aliases: dict[str, str]) -> str:
    try:
        tree = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return annotation

    resolved = _ImportAliasResolver(import_aliases).visit(tree)
    ast.fix_missing_locations(resolved)
    try:
        return ast.unparse(resolved)
    except Exception:
        return annotation


def _iter_python_files(package_root: Path) -> list[Path]:
    return sorted(
        path
        for path in package_root.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def _note_python_file_dependencies(app: Sphinx, package_root: Path) -> None:
    for path in _iter_python_files(package_root):
        app.env.note_dependency(str(path))


def _parse_module(
    path: Path, package_root: Path, package_name: str
) -> tuple[list[TypeAliasInfo], list[TaskInfo]]:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    module = _module_name(package_root, package_name, path)
    top_level = _top_level_name(package_root, package_name, path)
    import_aliases = _import_aliases(tree)

    aliases: list[TypeAliasInfo] = []
    tasks: list[TaskInfo] = []

    for node in tree.body:
        if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            name = node.name.id
            canonical = _canonical_name(package_name, top_level, name)
            doc_lines = _doc_comment_lines(source_lines, node.lineno)
            description_lines, alias_detail_lines = _split_doc_lines(doc_lines)
            aliases.append(
                TypeAliasInfo(
                    name=name,
                    alias_of=_type_alias_value(source, node),
                    description_lines=description_lines,
                    alias_detail_lines=alias_detail_lines,
                    top_level=top_level,
                    canonical_name=canonical,
                    defining_name=f"{module}.{name}",
                    path=path,
                    lineno=node.lineno,
                    target_id=_target_id(canonical),
                    import_aliases=import_aliases,
                )
            )
            continue

        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_task_decorator(decorator) for decorator in node.decorator_list):
            continue

        inputs: list[str] = []
        output: str | None = None
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__call__":
                continue
            inputs = [
                name
                for arg in item.args.args[1:]
                if (name := _simple_annotation_name(arg.annotation)) is not None
            ]
            output = _simple_annotation_name(item.returns)
            break

        tasks.append(
            TaskInfo(
                qualname=f"{module}.{node.name}",
                inputs=inputs,
                output=output,
            )
        )

    return aliases, tasks


def _build_aliases(app: Sphinx, warn_missing_docs: bool = False) -> list[TypeAliasInfo]:
    package_root = Path(app.config.mimodium_type_package_root).resolve()
    package_name = app.config.mimodium_type_package_name

    aliases: list[TypeAliasInfo] = []
    tasks: list[TaskInfo] = []
    for path in _iter_python_files(package_root):
        module_aliases, module_tasks = _parse_module(path, package_root, package_name)
        aliases.extend(module_aliases)
        tasks.extend(module_tasks)

    seen_aliases: set[str] = set()
    duplicate_aliases: set[str] = set()
    for alias in aliases:
        if alias.name in seen_aliases:
            duplicate_aliases.add(alias.name)
            continue
        seen_aliases.add(alias.name)

    if duplicate_aliases:
        LOGGER.warning(
            "Duplicate Mimodium semantic type names: %s",
            ", ".join(sorted(duplicate_aliases)),
        )

    produced_by: dict[str, list[str]] = {alias.name: [] for alias in aliases}
    consumed_by: dict[str, list[str]] = {alias.name: [] for alias in aliases}
    for task in tasks:
        if task.output in produced_by:
            produced_by[task.output].append(task.qualname)
        for input_name in task.inputs:
            if input_name in consumed_by:
                consumed_by[input_name].append(task.qualname)

    aliases_with_usage = [
        replace(
            alias,
            produced_by=sorted(
                produced_by[alias.name], key=lambda value: value.rsplit(".", 1)[-1]
            ),
            consumed_by=sorted(
                consumed_by[alias.name], key=lambda value: value.rsplit(".", 1)[-1]
            ),
        )
        for alias in aliases
    ]

    for alias in aliases_with_usage:
        if (
            warn_missing_docs
            and not alias.description_lines
            and not alias.alias_detail_lines
        ):
            LOGGER.warning(
                "Missing #: documentation for semantic type %s (%s:%s)",
                alias.defining_name,
                alias.path,
                alias.lineno,
            )

    app.env.mimodium_semantic_type_aliases = aliases_with_usage
    return aliases_with_usage


def _get_aliases(app: Sphinx) -> list[TypeAliasInfo]:
    aliases = getattr(app.env, "mimodium_semantic_type_aliases", None)
    if aliases is None:
        aliases = _build_aliases(app)
    return aliases


def _plural_classes(count: int) -> str:
    noun = "class" if count == 1 else "classes"
    return f"{count} {noun}"


def _rst_entry(directive: SphinxDirective, lines: list[str]) -> nodes.entry:
    entry = nodes.entry()
    if not lines:
        return entry
    content = StringList()
    for line in lines:
        content.append(line, source=directive.env.doc2path(directive.env.docname))
    directive.state.nested_parse(content, 0, entry)
    return entry


def _plain_entry(text: str) -> nodes.entry:
    entry = nodes.entry()
    if text:
        entry += nodes.paragraph(text=text)
    return entry


def _type_entry(directive: SphinxDirective, alias: TypeAliasInfo) -> nodes.entry:
    entry = nodes.entry()
    target = nodes.target("", "", ids=[alias.target_id])
    entry += target
    directive.state.document.note_explicit_target(target)
    entry += nodes.paragraph("", "", nodes.literal(text=alias.name))
    entry.extend(_usage_entry(directive, "Produced by", alias.produced_by).children)
    entry.extend(_usage_entry(directive, "Consumed by", alias.consumed_by).children)
    return entry


def _alias_entry(
    directive: SphinxDirective, alias: TypeAliasInfo, detail_lines: list[str]
) -> nodes.entry:
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    annotation = _referencable_annotation(alias.alias_of, alias.import_aliases)
    try:
        paragraph.extend(_parse_annotation(annotation, directive.env))
        entry += paragraph
    except SyntaxError:
        if "\n" not in alias.alias_of:
            entry = _rst_entry(directive, [f"``{alias.alias_of}``"])
        else:
            lines = [".. code-block:: python", ""]
            lines.extend(f"   {line}" for line in alias.alias_of.splitlines())
            entry = _rst_entry(directive, lines)

    entry.extend(_rst_entry(directive, detail_lines).children)
    return entry


def _usage_entry(
    directive: SphinxDirective, label: str, task_qualnames: list[str]
) -> nodes.entry:
    if not task_qualnames:
        return nodes.entry()

    lines = [f".. dropdown:: {label}: {_plural_classes(len(task_qualnames))}", ""]
    lines.extend(f"   * :py:class:`~{qualname}`" for qualname in task_qualnames)
    return _rst_entry(directive, lines)


def _row(cells: list[nodes.entry]) -> nodes.row:
    row = nodes.row()
    for cell in cells:
        row += cell
    return row


def _header_row(headers: list[str]) -> nodes.row:
    return _row([_plain_entry(header) for header in headers])


def _table_for_aliases(
    directive: SphinxDirective, aliases: list[TypeAliasInfo]
) -> nodes.table:
    table = nodes.table(classes=["mimodium-semantic-type-table"])
    tgroup = nodes.tgroup(cols=3)
    table += tgroup
    for _ in range(3):
        tgroup += nodes.colspec()

    thead = nodes.thead()
    thead += _header_row(["Type", "Alias of", "Description"])
    tgroup += thead

    tbody = nodes.tbody()
    for alias in aliases:
        tbody += _row(
            [
                _type_entry(directive, alias),
                _alias_entry(directive, alias, alias.alias_detail_lines),
                _rst_entry(directive, alias.description_lines),
            ]
        )
    tgroup += tbody
    return table


def _package_title(package_name: str) -> str:
    if package_name == "mimodium":
        return "Mimodium"
    return package_name.replace("_", " ").title()


def _package_sort_key(package_name: str, preferred_order: list[str]) -> tuple[int, str]:
    if package_name in preferred_order:
        return (preferred_order.index(package_name), package_name)
    return (len(preferred_order), package_name)


class MimodiumTypeCatalogDirective(SphinxDirective):
    has_content = False

    def run(self) -> list[nodes.Node]:
        package_root = Path(self.env.app.config.mimodium_type_package_root).resolve()
        _note_python_file_dependencies(self.env.app, package_root)
        aliases = _build_aliases(self.env.app, warn_missing_docs=True)
        self._register_python_type_targets(aliases)
        self._register_reference_targets(aliases)

        preferred_order = list(self.env.app.config.mimodium_type_package_order)
        grouped: dict[str, list[TypeAliasInfo]] = {}
        for alias in aliases:
            grouped.setdefault(alias.top_level, []).append(alias)

        output: list[nodes.Node] = []
        for package_name in sorted(
            grouped, key=lambda name: _package_sort_key(name, preferred_order)
        ):
            aliases = sorted(grouped[package_name], key=lambda alias: alias.name)
            section = nodes.section(ids=[nodes.make_id(package_name)])
            section += nodes.title(text=_package_title(package_name))
            section += _table_for_aliases(self, aliases)
            output.append(section)

        return output

    def _register_python_type_targets(self, aliases: list[TypeAliasInfo]) -> None:
        domain = self.env.get_domain("py")
        for alias in aliases:
            for ref_name in _reference_names(alias):
                domain.note_object(
                    ref_name,
                    "type",
                    alias.target_id,
                    aliased=ref_name != alias.canonical_name,
                    location=(str(alias.path), alias.lineno),
                )

    def _register_reference_targets(self, aliases: list[TypeAliasInfo]) -> None:
        reference_targets = getattr(self.env, "mimodium_semantic_type_refs", {}).copy()
        for alias in aliases:
            for ref_name in _reference_names(alias):
                reference_targets[ref_name] = (
                    self.env.docname,
                    alias.target_id,
                    alias.name,
                )
        self.env.mimodium_semantic_type_refs = reference_targets


def _task_owner(obj: Any) -> Any | None:
    owner_name, _, _ = getattr(obj, "__qualname__", "").rpartition(".")
    module = sys.modules.get(getattr(obj, "__module__", ""))
    owner = getattr(module, owner_name, None)
    if owner is not None and hasattr(owner, "__task_spec__"):
        return owner
    return None


def skip_members(
    app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool | None:
    if type(obj).__name__ == "TypeAliasType":
        return True
    if name == "__call__":
        return False
    if _task_owner(obj) is not None and callable(obj):
        return True
    return None


def process_task_call_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    if what != "method" or not name.endswith(".__call__") or _task_owner(obj) is None:
        return
    lines[:] = [
        "Automatically called as part of a "
        "`Dagreon Workflow "
        "<https://dagreon.readthedocs.io/en/latest/api.html#dagreon.Workflow>`_."
    ]


def process_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None] | None:
    if signature is None and return_annotation is None:
        return None

    replacements: dict[str, str] = {}
    for alias in _get_aliases(app):
        replacements[alias.canonical_name] = alias.name
        replacements[alias.defining_name] = alias.name

    def replace_aliases(value: str | None) -> str | None:
        if value is None:
            return None
        value = re.sub(r"^\(self,\s*", "(", value)
        value = value.replace("(self)", "()")
        for full_name, short_name in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = value.replace(full_name, short_name)
        return value

    return replace_aliases(signature), replace_aliases(return_annotation)


def resolve_semantic_type_reference(
    app: Sphinx, env: Any, node: nodes.Element, contnode: nodes.Node
) -> nodes.Node | None:
    if node.get("refdomain") != "py":
        return None
    if node.get("reftype") not in {"type", "class", "obj", "data"}:
        return None

    target = node.get("reftarget", "")
    ref_targets = getattr(env, "mimodium_semantic_type_refs", {})
    if target not in ref_targets:
        return None
    docname, target_id, title = ref_targets[target]

    return make_refnode(
        app.builder,
        node.get("refdoc", env.docname),
        docname,
        target_id,
        contnode,
        title,
    )


def prepare_semantic_types(app: Sphinx, env: Any, docnames: list[str]) -> None:
    _build_aliases(app)


def purge_semantic_type_references(app: Sphinx, env: Any, docname: str) -> None:
    reference_targets = getattr(env, "mimodium_semantic_type_refs", {})
    env.mimodium_semantic_type_refs = {
        name: target
        for name, target in reference_targets.items()
        if target[0] != docname
    }


def merge_semantic_type_references(
    app: Sphinx, env: Any, docnames: set[str], other: Any
) -> None:
    reference_targets = getattr(env, "mimodium_semantic_type_refs", {}).copy()
    for name, target in getattr(other, "mimodium_semantic_type_refs", {}).items():
        if target[0] in docnames:
            reference_targets[name] = target
    env.mimodium_semantic_type_refs = reference_targets


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("mimodium_type_package_root", "../mimodium", "env")
    app.add_config_value("mimodium_type_package_name", "mimodium", "env")
    app.add_config_value(
        "mimodium_type_package_order",
        ["scenario", "propagation", "algorithms", "evaluation", "visualization"],
        "env",
    )
    app.add_directive("mimodium-type-catalog", MimodiumTypeCatalogDirective)
    app.connect("autodoc-skip-member", skip_members)
    app.connect("autodoc-process-docstring", process_task_call_docstring)
    app.connect("autodoc-process-signature", process_signature)
    app.connect("missing-reference", resolve_semantic_type_reference)
    app.connect("env-before-read-docs", prepare_semantic_types)
    app.connect("env-purge-doc", purge_semantic_type_references)
    app.connect("env-merge-info", merge_semantic_type_references)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}

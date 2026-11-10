"""`decision/policy.py` must stay pure. Enforced by inspecting its imports, so it cannot rot.

No `torch`, no `httpx`/`requests`, no `pathlib` I/O, no
`os.environ`, no `time` inside the policy. The point is not tidiness -- it is that the policy stays
testable on a laptop with no GPU, which is the entire value of the refactor.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src" / "cybergate"
POLICY = SRC / "decision" / "policy.py"

#: Modules the policy may never import, directly or via `from X import Y`.
FORBIDDEN_MODULES = {
    "torch",
    "httpx",
    "requests",
    "urllib",
    "urllib.request",
    "json",
    "os",
    "time",
    "datetime",
    "pathlib",
    "hashlib",
    "random",
    "socket",
    "subprocess",
    "threading",
}

#: The policy's whole permitted import surface.
ALLOWED_IMPORTS = {
    "__future__",
    ".inputs",
    ".outcome",
    ".rules",
}


def _imports_of(path: Path) -> set[str]:
    """Return every module name `path` imports, with relative imports as `.module`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            found.add(f"{prefix}{node.module or ''}")
    return found


def test_policy_imports_nothing_impure() -> None:
    imported = _imports_of(POLICY)
    forbidden = {name for name in imported if name.split(".")[0] in FORBIDDEN_MODULES}
    assert not forbidden, f"policy.py imports impure modules: {sorted(forbidden)}"


def test_policy_import_surface_is_exactly_the_three_siblings() -> None:
    imported = _imports_of(POLICY)
    unexpected = imported - ALLOWED_IMPORTS
    assert not unexpected, (
        f"policy.py grew new imports: {sorted(unexpected)}. "
        "Anything the policy needs belongs on GateInputs instead."
    )


def test_policy_has_no_module_level_mutable_state() -> None:
    """A module-level dict or list in the policy would be shared state across decisions."""
    tree = ast.parse(POLICY.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id != "__all__":
                    offenders.append(target.id)
    assert not offenders, f"policy.py has module-level mutable state: {offenders}"


@pytest.mark.parametrize(
    "module",
    [
        "cybergate/config.py",
        "cybergate/boilerplate.py",
        "cybergate/messages.py",
        "cybergate/families/base.py",
        "cybergate/families/gemma4/family.py",
        "cybergate/decision/inputs.py",
        "cybergate/decision/outcome.py",
        "cybergate/decision/rules.py",
        "cybergate/serving/postprocess.py",
        "cybergate/serving/proxy.py",
        "cybergate/state/keys.py",
    ],
)
def test_shared_modules_do_not_import_torch(module: str) -> None:
    """Modules importable in the API-server process must not pull in torch.

    The entry point runs in both processes; the API server does not need torch and should not pay to
    import it.
    """
    imported = _imports_of(SRC.parent / "cybergate" / Path(module).relative_to("cybergate"))
    assert "torch" not in {name.split(".")[0] for name in imported}


def test_importing_the_package_does_not_import_torch() -> None:
    """Definition of done #6: the API-server process starts without importing torch."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src');"
            " import cybergate, cybergate.decision, cybergate.serving, cybergate_plugin;"
            " print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=SRC.parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing the serving path pulled in torch"

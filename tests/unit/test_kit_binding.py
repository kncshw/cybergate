"""The kit binding check: an operator must agree with the family and checkpoint it is loaded under.

Why this exists at all: nothing bound an operator to a model. A direction fitted on a different
checkpoint of the same architecture loaded without complaint and stopped firing, which is how the
shipped Qwen operator came to clear its own threshold on 18 of its own 100 fitted training positives.

`tests/conftest.py` sets `CYBERGATE_SKIP_KIT_CHECK=1` for the whole suite, because the fixture
operators are 4-dimensional stand-ins that contradict their families on purpose. Every test here that
exercises the enforcing path clears that flag explicitly, so the skip is never what makes these pass.
"""
from __future__ import annotations

import pytest

from cybergate.families import family_by_name
from cybergate.ops import CyberGateOperator
from cybergate.ops.kit import (
    SKIP_ENV,
    KitMismatch,
    check_model,
    check_operator,
    enforce,
)


def _op(
    dim: int,
    gate_layer: int,
    pool: str,
    model_key: str = "gemma",
    category: str = "cyber_offense",
) -> CyberGateOperator:
    return CyberGateOperator(
        c=[0.0] * dim,
        thresh=0.0,
        gate_layer=gate_layer,
        pool=pool,
        category=category,
        model_key=model_key,
        meta={},
    )


@pytest.fixture
def enforcing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the suite-wide skip flag so `enforce` actually raises."""
    monkeypatch.delenv(SKIP_ENV, raising=False)


# --- the declarations themselves ----------------------------------------------------------------


def test_the_two_shipped_families_declare_a_kit() -> None:
    """Gemma-4 and Qwen-3.6 ship artifacts, so they must name a kit and a checkpoint shape."""
    for name, kit, dim, layers, gate in (
        ("gemma4", "gemma4", 2816, 30, (16, "mean_user")),
        ("qwen36", "qwen36", 2048, 40, (21, "mean_user")),
    ):
        family = family_by_name(name)
        assert family.kit_name == kit
        assert family.hidden_size == dim
        assert family.num_layers == layers
        assert family.expected_gate == gate


def test_expected_gate_matches_the_head_of_the_operator_ladder() -> None:
    """The declared gate and the first default operator must describe the same file.

    These drifted apart before: both families' ladders led with an operator at a different layer
    from the one their docstrings and `expected_gate` name, so a deployment without an explicit
    `CYBERGATE_OP_PATH` silently served the wrong one.
    """
    for name, expected_head in (
        ("gemma4", "gemma4_L16_M1.json"),
        ("qwen36", "qwen36_L21_M1_refit.json"),
    ):
        family = family_by_name(name)
        head = family.default_op_names[0]
        assert head == expected_head
        assert f"L{family.expected_gate[0]}" in head


def test_a_subclass_that_forgets_to_override_the_kit_binding_is_caught() -> None:
    """Subclassing a kit-bearing family inherits its kit binding, which is a trap worth pinning.

    The removed `Llama31Family` subclassed `Qwen36Family` and had to null all four kit fields
    explicitly; without that it claimed Qwen's kit, Qwen's 2048 dimensions and Qwen's L21 gate, and
    rejected its own 4096-dim L24 operator. No such subclass ships today, so this uses a synthetic
    one -- the hazard is in the inheritance, not in any particular family.
    """
    from cybergate.families.qwen36.family import Qwen36Family

    class _Forgetful(Qwen36Family):
        name = "forgetful"
        default_op_names = ("forgetful_L24_M1.json",)

    # Inherited Qwen binding, so a 4096-dim L24 operator of its own is rejected on three counts.
    forgetful = _Forgetful()
    assert forgetful.kit_name == "qwen36"
    problems = check_operator(_op(4096, 24, "mean_user", "forgetful"), forgetful)
    assert len(problems) == 2, problems
    assert any("4096" in p and "2048" in p for p in problems)
    assert any("L24" in p and "L21" in p for p in problems)

    class _Careful(_Forgetful):
        name = "careful"
        kit_name = None
        hidden_size = 4096
        num_layers = 0
        expected_gate = None

    assert check_operator(_op(4096, 24, "mean_user", "careful"), _Careful()) == []


# --- check_operator -----------------------------------------------------------------------------


def test_the_shipped_operator_shape_passes_its_own_family() -> None:
    gemma, qwen = family_by_name("gemma4"), family_by_name("qwen36")
    assert check_operator(_op(2816, 16, "mean_user", "gemma"), gemma) == []
    assert check_operator(_op(2048, 21, "mean_user", "qwen36"), qwen) == []


def test_wrong_architecture_is_caught() -> None:
    """A Gemma-shaped direction under the Qwen family: the cross-architecture case."""
    problems = check_operator(_op(2816, 21, "mean_user"), family_by_name("qwen36"))
    assert any("2816" in p and "2048" in p for p in problems)


def test_the_stale_ladder_operator_is_caught() -> None:
    """L10 under Gemma-4, which is exactly what the stale `default_op_names` ladder resolved to."""
    problems = check_operator(_op(2816, 10, "mean_user"), family_by_name("gemma4"))
    assert len(problems) == 1
    assert "L10" in problems[0] and "L16" in problems[0]


def test_the_wrong_pooling_is_caught() -> None:
    """Right layer, wrong pooling. `last` and `mean_user` are different directions at one layer."""
    problems = check_operator(_op(2816, 16, "last"), family_by_name("gemma4"))
    assert len(problems) == 1
    assert "'last'" in problems[0] and "'mean_user'" in problems[0]


def test_a_gate_layer_off_the_end_of_the_checkpoint_is_caught() -> None:
    """The hook clamps with `min(gate_layer, len(layers) - 1)`, so this would otherwise be silent."""
    problems = check_operator(_op(2816, 99, "mean_user"), family_by_name("gemma4"))
    assert any("99" in p and "clamp" in p for p in problems)


def test_an_undeclared_family_reports_nothing() -> None:
    """A family that measured nothing must not guess. Neutral values stand the check down."""

    class _Unmeasured:
        name = "unmeasured"
        hidden_size = 0
        num_layers = 0
        expected_gate = None

    assert check_operator(_op(1234, 7, "last"), _Unmeasured()) == []


# --- check_model --------------------------------------------------------------------------------


def test_the_served_checkpoint_layer_count_is_checked() -> None:
    """Gemma's artifacts were fitted on a 30-layer checkpoint; a 40-layer model is a different one."""
    op = _op(2816, 16, "mean_user")
    assert check_model(op, family_by_name("gemma4"), 30) == []
    problems = check_model(op, family_by_name("gemma4"), 40)
    assert any("40" in p and "30" in p for p in problems)


def test_check_model_catches_the_clamp_even_when_the_family_is_undeclared() -> None:
    """`op.gate_layer >= total_layers` is checked unconditionally: the clamp is always wrong."""

    class _Unmeasured:
        name = "unmeasured"
        num_layers = 0

    problems = check_model(_op(2816, 44, "mean_user"), _Unmeasured(), 30)
    assert any("44" in p and "clamp" in p for p in problems)


# --- enforce ------------------------------------------------------------------------------------


def test_enforce_raises_and_names_the_escape_hatch(enforcing: None) -> None:
    with pytest.raises(KitMismatch) as excinfo:
        enforce(["a problem"], "operator fake.json")
    message = str(excinfo.value)
    assert "operator fake.json" in message
    assert "a problem" in message
    assert SKIP_ENV in message


def test_enforce_passes_a_clean_result_through(enforcing: None) -> None:
    assert enforce([], "operator fake.json") == []


def test_the_escape_hatch_downgrades_to_a_returned_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag set the caller gets the problems back to log, rather than an exception."""
    monkeypatch.setenv(SKIP_ENV, "1")
    assert enforce(["a problem"], "operator fake.json") == ["a problem"]


def test_only_the_exact_value_one_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truthy-looking value is not an opt-out; this flag disables a safety check."""
    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv(SKIP_ENV, value)
        with pytest.raises(KitMismatch):
            enforce(["a problem"], "operator fake.json")


# --- end to end through the real loader ---------------------------------------------------------


def test_loading_every_shipped_artifact_passes_the_check(enforcing: None) -> None:
    """The check must not reject what we actually ship -- and we ship four files, not two.

    This table held only the two category operators until 2026-11-01. Both refusal directions were
    shipped, both were loaded through the same `load_cybergate_op`, and neither was ever loaded
    here with enforcement live, so `check_operator` rejecting them went unnoticed through 1039
    passing tests. It took a live serve to find. See `test_a_refusal_direction_is_not_held_to_the_gate`.
    """
    from pathlib import Path

    from cybergate.ops import load_cybergate_op

    root = Path(__file__).resolve().parents[2]
    for kit, name in (
        ("gemma4", "gemma4_L16_M1.json"),
        ("gemma4", "gemma4_L21_refusal_dir.json"),
        ("qwen36", "qwen36_L21_M1_refit.json"),
        ("qwen36", "qwen36_L28_refusal_dir.json"),
    ):
        path = root / "kits" / kit / name
        if not path.exists():
            pytest.skip(f"{path} not present")
        op = load_cybergate_op(path)
        assert check_operator(op, family_by_name(kit)) == [], f"{kit}/{name} was rejected"


def test_a_refusal_direction_is_not_held_to_the_gate() -> None:
    """A refusal direction is `last`-pooled at its own layer, so the gate check cannot apply.

    The consequence of getting this wrong is not a noisy failure. `refusal_monitor.py` loads the
    direction through `load_cybergate_op`, so a raise there takes the whole monitor down: no
    `s_ref_z`, no two-factor veto, no mid/late rescue, and nothing in the decision record to say the
    second factor is missing. Measured live on gemma4 while it was broken: 482 rows where the kit
    expects 558, 0 of 1149 rows carrying `s_ref_z`.
    """
    gemma4 = family_by_name("gemma4")
    assert gemma4.expected_gate == (16, "mean_user")

    ref = _op(2816, 21, "last", category="refusal_monitor")
    assert check_operator(ref, gemma4) == []

    # Exempt from the GATE check only. A refusal direction of the wrong width is still a real
    # mismatch -- it would be a direction from another architecture.
    wrong_width = _op(2048, 21, "last", category="refusal_monitor")
    problems = check_operator(wrong_width, gemma4)
    assert len(problems) == 1 and "2048-dimensional" in problems[0]

    # And a CATEGORY operator at the refusal direction's layer is still caught.
    assert check_operator(_op(2816, 21, "last"), gemma4) != []


def test_loading_a_superseded_operator_raises_through_the_loader(
    enforcing: None, tmp_path: "pathlib.Path"
) -> None:
    """An operator of the right WIDTH at the wrong gate layer must fail to load, not serve.

    This read `ops/gemma4_L10_M1.json` -- the superseded direction that was a live default until
    2026-10-31 -- and skipped when it was absent, so trimming `ops/` silently removed the
    coverage. The case is built here instead: same loader path, same assertion, no 157 KB
    artifact shipped to support it, and nothing to skip on.
    """
    import json

    from cybergate.ops import load_cybergate_op

    path = tmp_path / "gemma4_L10_M1.json"
    path.write_text(json.dumps({
        "c": [1.0] + [0.0] * 2815,  # correct width for gemma4, so only the gate layer trips
        "thresh": 0.10807704,
        "meta": {"gate_layer": 10, "pool": "mean_user", "model_key": "gemma"},
    }))
    with pytest.raises(KitMismatch) as excinfo:
        load_cybergate_op(path)
    assert "L10" in str(excinfo.value)


# --- the kits on disk ---------------------------------------------------------------------------


def _kit_root() -> "pathlib.Path":
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2] / "kits"


def _manifest(kit: str) -> dict:
    import json

    path = _kit_root() / kit / "MANIFEST.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_the_manifest_gate_agrees_with_the_family_declaration(kit: str) -> None:
    """The manifest restates the gate so a kit is readable on its own. That restatement must agree.

    This is the assertion the old arrangement lacked: the same claim lived in a module docstring,
    nothing checked it, and it drifted to name a layer the family does not ship.
    """
    manifest = _manifest(kit)
    family = family_by_name(manifest["family"])
    assert family.kit_name == kit
    assert (manifest["gate"]["layer"], manifest["gate"]["pool"]) == family.expected_gate
    assert manifest["checkpoint"]["hidden_size"] == family.hidden_size
    assert manifest["checkpoint"]["num_layers"] == family.num_layers


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_the_manifest_default_operator_is_the_head_of_the_ladder(kit: str) -> None:
    """A kit's declared default operator and the family's first default name are one file."""
    manifest = _manifest(kit)
    family = family_by_name(manifest["family"])
    assert manifest["files"]["operator"] == family.default_op_names[0]
    assert manifest["files"]["refusal_direction"] == family.default_ref_dir_name
    assert manifest["files"]["candidate_bundle"] == family.default_bundle_name


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_every_file_the_manifest_names_is_present(kit: str) -> None:
    """A kit that names a file it does not carry is worse than one that names nothing."""
    manifest = _manifest(kit)
    root = _kit_root() / kit
    named = [v for k, v in manifest["files"].items() if k != "alternates"]
    named += list(manifest["files"]["alternates"])
    for name in named:
        assert (root / name).is_file(), f"{kit}: manifest names {name}, which is not in the kit"


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_the_manifest_threshold_is_the_operator_threshold(kit: str) -> None:
    """The headline number in the manifest must be the one the gate actually compares against."""
    import json

    manifest = _manifest(kit)
    op_file = _kit_root() / kit / manifest["files"]["operator"]
    assert json.loads(op_file.read_text())["thresh"] == manifest["gate"]["threshold"]


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_an_in_sample_threshold_is_declared_as_such(kit: str) -> None:
    """Whether the threshold rule saw an evaluation corpus is the single biggest reviewer risk here.

    gemma4's R3 is set on the fit reference only, so its CatQA leak is a real out-of-sample number.
    qwen36's rule is set against CatQA itself, so its leak is the constraint and not a measurement.
    Both must say which they are, because the two are not comparable and look identical in a table.
    """
    manifest = _manifest(kit)
    expected = {"gemma4": False, "qwen36": True}[kit]
    assert manifest["gate"]["in_sample"] is expected


@pytest.mark.parametrize("kit", ["gemma4", "qwen36"])
def test_canaries_are_present_and_internally_consistent(kit: str) -> None:
    """Canary verdicts must agree with their own recorded margins, where a margin was measured."""
    import json

    path = _kit_root() / kit / "canaries.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    doc = json.loads(path.read_text())
    canaries = doc["canaries"]
    assert canaries, f"{kit}: canaries.json carries no canaries"
    assert any(c["expect_fired"] for c in canaries), f"{kit}: no positive canary"
    for c in canaries:
        assert c["prompt"].strip(), f"{kit}/{c['key']}: empty prompt"
        import hashlib

        assert hashlib.sha256(c["prompt"].encode()).hexdigest() == c["prompt_sha256"]
        if c["margin"] is not None:
            # A measured margin must point the same way as the verdict it is shipped with.
            assert (c["margin"] > 0) is c["expect_fired"], f"{kit}/{c['key']}: margin contradicts verdict"


def test_the_gemma_kit_ships_both_halves_of_the_canary_set() -> None:
    """gemma4 has per-prompt measurements for both fit corpora, so it has no excuse for one-sided canaries."""
    import json

    path = _kit_root() / "gemma4" / "canaries.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    roles = [c["role"] for c in json.loads(path.read_text())["canaries"]]
    assert roles.count("positive") >= 3
    assert roles.count("negative") >= 3


def test_the_qwen_kit_states_its_missing_negatives() -> None:
    """qwen36 cannot assert a quiet prompt: the aggregate says 17 of 140 fired, not which 17.

    Shipping no negative canary is the correct call. Shipping one anyway, or shipping none silently,
    would both be wrong -- so the gap has to be written down in the kit.
    """
    import json

    path = _kit_root() / "qwen36" / "canaries.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    doc = json.loads(path.read_text())
    assert not any(c["role"] == "negative" for c in doc["canaries"])
    assert doc.get("known_gaps"), "qwen36 canaries omit negatives without saying so"


# --- artifact resolution ------------------------------------------------------------------------
#
# `tests/conftest.py` sets CYBERGATE_OPS_DIR, so `OPS_DIR_IS_EXPLICIT` is True for the whole suite
# and the kit path is never taken by accident. These tests patch the module constants directly,
# because both are read at import time.


def test_an_explicit_ops_dir_suppresses_the_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for a directory and silently getting a different one is worse than either outcome."""
    from cybergate.ops import loader

    monkeypatch.setattr(loader, "OPS_DIR_IS_EXPLICIT", True)
    assert loader.artifact_dirs(family_by_name("gemma4")) == (loader.OPS_DIR,)


def test_the_kit_is_searched_first_when_nothing_was_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment that sets no environment must still resolve the operator we ship.

    This is the regression that matters: `ops/` still holds superseded per-layer operators, and the
    ladder picks whichever exists. That is how Gemma's default resolved to L10 and Qwen's to a
    pre-refit L15, both of which load cleanly and gate the wrong thing.
    """
    import pathlib

    from cybergate.ops import loader

    monkeypatch.setattr(loader, "OPS_DIR_IS_EXPLICIT", False)
    monkeypatch.setattr(loader, "KITS_DIR", pathlib.Path("/kits"))
    dirs = loader.artifact_dirs(family_by_name("gemma4"))
    assert dirs == (pathlib.Path("/kits/gemma4"), loader.OPS_DIR)


def test_a_family_with_no_kit_searches_only_the_ops_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A family declaring no kit must not be sent looking for one.

    Every shipped family now has a kit, so this uses a stub. The branch still has to work: it is
    what a family measured offline but never served would take.
    """
    from cybergate.ops import loader

    class _NoKit:
        name = "nokit"
        kit_name = None

    monkeypatch.setattr(loader, "OPS_DIR_IS_EXPLICIT", False)
    assert loader.artifact_dirs(_NoKit()) == (loader.OPS_DIR,)


@pytest.mark.parametrize(
    ("kit", "operator"),
    [("gemma4", "gemma4_L16_M1.json"), ("qwen36", "qwen36_L21_M1_refit.json")],
)
def test_resolution_with_nothing_configured_lands_in_the_kit(
    kit: str, operator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end against the real tree: no environment set, the shipped operator is what resolves."""
    import pathlib

    from cybergate.ops import loader

    kits = pathlib.Path(__file__).resolve().parents[2] / "kits"
    if not (kits / kit / operator).exists():
        pytest.skip(f"{kits / kit / operator} not present")
    monkeypatch.setattr(loader, "OPS_DIR_IS_EXPLICIT", False)
    monkeypatch.setattr(loader, "KITS_DIR", kits)
    family = family_by_name(kit)
    assert loader.default_op_path(family) == kits / kit / operator
    assert loader.resolve_ref_dir_path(family) == kits / kit / family.default_ref_dir_name

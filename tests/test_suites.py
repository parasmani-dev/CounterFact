import json

from counterfact.reference import answer
from counterfact.schemas import PairSpec
from counterfact.suites import HardSuite, answer_change_stats, build_suite, generate_suite


def test_seeded_hard_suite_is_valid_balanced_and_deterministic():
    first = build_suite(729)
    second = build_suite(729)
    assert first.model_dump() == second.model_dump()
    suite = HardSuite.model_validate(first.model_dump())
    assert len(suite.families) == 20
    counts = {
        kind: sum(f.spec.question.type == kind for f in suite.families)
        for kind in ("largest_category", "value_lookup", "above_threshold")
    }
    assert counts == {"largest_category": 7, "value_lookup": 7, "above_threshold": 6}
    assert all(f.synthetic is True for f in suite.families)
    assert all(12 <= len(f.spec.chart.categories) <= 20 for f in suite.families)
    assert all(
        f.spec.question.type != "value_lookup" or f.spec.show_value_labels for f in suite.families
    )
    assert all(
        f.spec.question.type == "value_lookup" or not f.spec.show_value_labels
        for f in suite.families
    )
    assert all(PairSpec.model_validate(f.spec.model_dump()) == f.spec for f in suite.families)
    assert answer_change_stats(suite) == {"changed": 13, "unchanged": 7}
    for family in suite.families:
        original = answer(family.spec.chart, family.spec.question)
        transformed = answer(family.spec.transformed(), family.spec.question)
        assert (original != transformed) == (family.spec.question.type != "value_lookup")


def test_generate_suite_renders_all_pairs_and_writes_json(tmp_path):
    output = tmp_path / "suites" / "hard_v1.json"
    suite = generate_suite(42, tmp_path / "data", output)
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["synthetic"] is True
    assert len(serialized["families"]) == 20
    for family in suite.families:
        folder = tmp_path / "data" / "pairs" / family.pair_id
        assert (folder / "original.png").is_file()
        assert (folder / "transformed.png").is_file()

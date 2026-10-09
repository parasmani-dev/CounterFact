"""Seeded, synthetic hard-suite generation for repeatable demos and CI."""

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from counterfact.reference import answer
from counterfact.schemas import (
    Category,
    ChangeValue,
    Chart,
    PairSpec,
    Question,
    Reorder,
    StrictModel,
)
from counterfact.storage import PairStore


class SuiteFamily(StrictModel):
    family_id: str
    synthetic: Literal[True] = True
    spec: PairSpec
    pair_id: str | None = None


class HardSuite(StrictModel):
    schema_version: Literal[1] = 1
    suite_id: Literal["hard_v1"] = "hard_v1"
    synthetic: Literal[True] = True
    seed: int = Field(ge=0, le=2**32 - 1)
    families: list[SuiteFamily] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_family_mix(self) -> "HardSuite":
        expected = {"largest_category": 7, "value_lookup": 7, "above_threshold": 6}
        counts = {kind: 0 for kind in expected}
        for family in self.families:
            counts[family.spec.question.type] += 1
            if not family.family_id.startswith(f"hard-{family.spec.question.type}-"):
                raise ValueError("Family ID must identify its question type")
        if counts != expected:
            raise ValueError(f"Expected hard-suite question mix {expected}")
        if len({f.family_id for f in self.families}) != 20:
            raise ValueError("Family IDs must be unique")
        return self


def build_suite(seed: int = 42) -> HardSuite:
    rng = random.Random(seed)
    families: list[SuiteFamily] = []
    cases = [("largest_category", 7), ("value_lookup", 7), ("above_threshold", 6)]
    for question_type, count in cases:
        for index in range(1, count + 1):
            family_id = f"hard-{question_type}-{index:02d}"
            category_count = rng.randint(12, 20)
            ids = [f"group_{n:02d}" for n in range(1, category_count + 1)]
            labels = [f"Group {chr(ord('A') + n)}" for n in range(category_count)]
            values = [
                round(1000 + n * (40 / (category_count - 1)), 3) for n in range(category_count)
            ]
            rng.shuffle(values)
            categories = [
                Category(id=cid, label=label, value=value)
                for cid, label, value in zip(ids, labels, values, strict=True)
            ]
            maximum_index = max(range(category_count), key=lambda n: categories[n].value)

            if question_type == "largest_category":
                runner_up = max(
                    (n for n in range(category_count) if n != maximum_index),
                    key=lambda n: categories[n].value,
                )
                new_value = round(categories[maximum_index].value * 1.02, 3)
                question = Question(type="largest_category")
                mutation = ChangeValue(
                    type="change_value", category_id=ids[runner_up], value=new_value
                )
                show_labels = False
            elif question_type == "value_lookup":
                target_index = rng.randrange(category_count)
                question = Question(type="value_lookup", target_id=ids[target_index])
                order = ids[1:] + ids[:1]
                mutation = Reorder(type="reorder", order=order)
                show_labels = True
            else:
                target_index = maximum_index
                target_value = categories[target_index].value
                threshold = round(target_value * 1.01, 3)
                updated = round(target_value * 1.03, 3)
                # Keep the threshold within 2% of the target and make the answer flip.
                question = Question(
                    type="above_threshold", target_id=ids[target_index], threshold=threshold
                )
                mutation = ChangeValue(
                    type="change_value", category_id=ids[target_index], value=updated
                )
                show_labels = False

            spec = PairSpec(
                id=family_id.replace("-", "_"),
                seed=seed,
                chart=Chart(categories=categories),
                question=question,
                show_value_labels=show_labels,
                mutations=[mutation],
            )
            families.append(SuiteFamily(family_id=family_id, spec=spec))
    return HardSuite(seed=seed, families=families)


def answer_change_stats(suite: HardSuite) -> dict[str, int]:
    changed = sum(
        answer(f.spec.transformed(), f.spec.question) != answer(f.spec.chart, f.spec.question)
        for f in suite.families
    )
    return {"changed": changed, "unchanged": len(suite.families) - changed}


def generate_suite(
    seed: int = 42,
    data_dir: Path = Path("data"),
    output_path: Path = Path("data/suites/hard_v1.json"),
) -> HardSuite:
    suite = build_suite(seed)
    store = PairStore(data_dir)
    store.initialize()
    rendered = []
    for family in suite.families:
        manifest = store.create(family.spec)
        rendered.append(family.model_copy(update={"pair_id": manifest["id"]}))
    suite = suite.model_copy(update={"families": rendered})
    suite = HardSuite.model_validate(suite.model_dump())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(suite.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return suite


def balanced_order(suite: HardSuite) -> HardSuite:
    selected = []
    for kind in ("largest_category", "value_lookup", "above_threshold"):
        selected.extend([f for f in suite.families if f.spec.question.type == kind][:2])
    chosen = {f.family_id for f in selected}
    remaining = [f for f in suite.families if f.family_id not in chosen]
    return HardSuite.model_validate(
        suite.model_copy(update={"families": selected + remaining}).model_dump()
    )

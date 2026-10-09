"""Strict, bounded chart specifications. No model-generated Python is executed."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,40}$")]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Value = Annotated[float, Field(ge=0, le=1_000_000, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Category(StrictModel):
    id: Identifier
    label: Label
    value: Value

    @model_validator(mode="after")
    def validate_label(self) -> "Category":
        if any(not c.isprintable() for c in self.label):
            raise ValueError("Category labels must be printable, single-line text")
        if not self.label.isascii():
            raise ValueError("Phase 1 supports ASCII labels with the bundled DejaVu Sans font")
        return self


class Chart(StrictModel):
    type: Literal["bar"] = "bar"
    categories: Annotated[list[Category], Field(min_length=2, max_length=20)]

    @model_validator(mode="after")
    def unique_categories(self) -> "Chart":
        ids = [c.id for c in self.categories]
        labels = [c.label.casefold() for c in self.categories]
        if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
            raise ValueError("Category IDs and labels must be unique")
        return self


class Question(StrictModel):
    type: Literal["largest_category", "value_lookup", "above_threshold"]
    target_id: Identifier | None = None
    threshold: Value | None = None

    @model_validator(mode="after")
    def matching_arguments(self) -> "Question":
        if self.type == "largest_category":
            if self.target_id is not None or self.threshold is not None:
                raise ValueError("largest_category takes no target or threshold")
        elif self.target_id is None:
            raise ValueError("This question requires target_id")
        if self.type == "above_threshold" and self.threshold is None:
            raise ValueError("above_threshold requires threshold")
        if self.type != "above_threshold" and self.threshold is not None:
            raise ValueError("Only above_threshold takes a threshold")
        return self


class ChangeValue(StrictModel):
    type: Literal["change_value"]
    category_id: Identifier
    value: Value


class Reorder(StrictModel):
    type: Literal["reorder"]
    order: Annotated[list[Identifier], Field(min_length=2, max_length=20)]


Mutation = Annotated[ChangeValue | Reorder, Field(discriminator="type")]


class PairSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: Identifier
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    chart: Chart
    question: Question
    show_value_labels: bool = True
    mutations: Annotated[list[Mutation], Field(min_length=1, max_length=8)]

    @model_validator(mode="before")
    @classmethod
    def default_label_visibility(cls, value: Any) -> Any:
        """Lookup questions need labels; other types default to uncluttered bars."""
        if isinstance(value, dict) and "show_value_labels" not in value:
            value = dict(value)
            question = value.get("question")
            value["show_value_labels"] = (
                isinstance(question, dict) and question.get("type") == "value_lookup"
            )
        return value

    @model_validator(mode="after")
    def valid_pair(self) -> "PairSpec":
        from counterfact.reference import answer

        if self.question.type == "value_lookup" and not self.show_value_labels:
            raise ValueError("value_lookup requires show_value_labels=true")
        answer(self.chart, self.question)
        transformed = self.transformed()
        answer(transformed, self.question)
        if transformed == self.chart:
            raise ValueError("The transformation must change the chart")
        return self

    def transformed(self) -> Chart:
        categories = list(self.chart.categories)
        for mutation in self.mutations:
            by_id = {c.id: c for c in categories}
            if isinstance(mutation, ChangeValue):
                if mutation.category_id not in by_id:
                    raise ValueError("Mutation targets a missing category")
                if by_id[mutation.category_id].value == mutation.value:
                    raise ValueError("Value mutation must change a value")
                categories = [
                    Category(id=c.id, label=c.label, value=mutation.value)
                    if c.id == mutation.category_id
                    else c
                    for c in categories
                ]
            else:
                if len(mutation.order) != len(by_id) or set(mutation.order) != set(by_id):
                    raise ValueError("Reorder must be a permutation of every category ID")
                if mutation.order == [c.id for c in categories]:
                    raise ValueError("Reorder must change category order")
                categories = [by_id[key] for key in mutation.order]
        return Chart(categories=categories)

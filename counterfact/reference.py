"""Executable answers from source data, never from an LLM."""

from counterfact.schemas import Chart, Question

Answer = str | float | bool


def format_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def answer(chart: Chart, question: Question) -> Answer:
    if question.type == "largest_category":
        highest = max(c.value for c in chart.categories)
        winners = [c for c in chart.categories if c.value == highest]
        if len(winners) != 1:
            raise ValueError("largest_category requires a unique maximum on both charts")
        return winners[0].label
    target = next((c for c in chart.categories if c.id == question.target_id), None)
    if target is None:
        raise ValueError("Question targets a missing category")
    if question.type == "value_lookup":
        return target.value
    assert question.threshold is not None  # Enforced by the Question schema.
    return target.value > question.threshold


def question_text(chart: Chart, question: Question) -> str:
    if question.type == "largest_category":
        return "Which category has the largest value? Return its label."
    label = next(c.label for c in chart.categories if c.id == question.target_id)
    if question.type == "value_lookup":
        return f'What is the value of category "{label}"?'
    assert question.threshold is not None
    threshold = format_value(question.threshold)
    return f'Is the value of category "{label}" strictly greater than {threshold}?'

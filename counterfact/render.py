"""Deterministic, headless rendering within a pinned runtime and font environment."""

import hashlib
import io
import threading
from typing import Protocol

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from counterfact.reference import format_value  # noqa: E402
from counterfact.schemas import Chart  # noqa: E402

RENDERER_VERSION = f"bar-v2-matplotlib-{matplotlib.__version__}"
_SIMILAR_PALETTE = ("#5572a3", "#5975a5", "#526f9f", "#5b77a7")
_render_lock = threading.Lock()  # Matplotlib rc/font state is process-global.


class ChartRenderer(Protocol):
    def render(self, chart: Chart) -> bytes: ...


class BarRenderer:
    def render(self, chart: Chart, *, show_value_labels: bool = False) -> bytes:
        # Revalidate because lists nested inside frozen Pydantic models remain mutable.
        chart = Chart.model_validate(chart.model_dump())
        with (
            _render_lock,
            matplotlib.rc_context(
                {
                    "font.family": "DejaVu Sans",
                    "font.size": 9,
                    "text.usetex": False,
                    "text.parse_math": False,
                    "axes.formatter.useoffset": False,
                    "path.simplify": False,
                }
            ),
        ):
            figure = Figure(figsize=(10, 6), dpi=100, facecolor="white")
            canvas = FigureCanvasAgg(figure)
            axes = figure.add_subplot(111)
            values = [c.value for c in chart.categories]
            # Derive each shade from the stable category ID so reordering does not recolor bars.
            colors = [
                _SIMILAR_PALETTE[
                    int.from_bytes(hashlib.sha256(c.id.encode()).digest()[:2], "big")
                    % len(_SIMILAR_PALETTE)
                ]
                for c in chart.categories
            ]
            bars = axes.bar(range(len(values)), values, color=colors, width=0.65)
            axes.set_xticks(range(len(values)), [c.label for c in chart.categories])
            axes.tick_params(axis="x", labelrotation=35, labelsize=9)
            axes.set_ylabel("Value")
            axes.set_ylim(0, max(max(values) * 1.2, 1))
            if show_value_labels:
                axes.bar_label(
                    bars, labels=[format_value(v) for v in values], padding=4, fontsize=8
                )
            axes.spines[["top", "right"]].set_visible(False)
            axes.set_axisbelow(True)
            axes.grid(axis="y", alpha=0.2)
            figure.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.36)
            buffer = io.BytesIO()
            canvas.print_png(buffer, metadata={"Software": f"CounterFact {RENDERER_VERSION}"})
            figure.clear()
            return buffer.getvalue()


def image_hash(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()

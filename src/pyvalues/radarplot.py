from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def plot_radar(
    dim_names: list[str],
    valuess: list[list[float]],
    labels: list[str | None] = [None],
    linecolors: list[str] = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999"
    ],
    fillcolors: list[str | None] = [None],
    fillalphas: list[float] = [0.25]
):
    theta = radar_factory(len(dim_names))
    fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.25, 0.50, 0.75], fontsize="small")  # type: ignore
    ax.set_varlabels(dim_names)  # type: ignore
    has_label = False
    for i in range(len(valuess)):
        label = labels[i % len(labels)]
        ax.plot(theta, list(valuess[i]),
                color=linecolors[i % len(linecolors)],
                label=label)
        if fillcolors[i % len(fillcolors)] is not None:
            ax.fill(theta, list(valuess[i]),
                    facecolor=fillcolors[i % len(fillcolors)],
                    alpha=fillalphas[i % len(fillalphas)],
                    label='_nolegend_')
        if label is not None:
            has_label = True
    if has_label:
        fig.legend(loc="upper right", labelspacing=0.1, fontsize='small')
    return plt


def radar_factory(num_vars):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Based on
    https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(2 * np.pi, 0, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs) -> list[Line2D]:
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), fmt="")
            for i in range(len(labels)):
                horizontal_alignment = None
                if i == 0 or i == len(labels) / 2:
                    horizontal_alignment = "center"
                elif i < len(labels) / 2:
                    horizontal_alignment = "left"
                else:
                    horizontal_alignment = "right"

                vertical_alignment = None
                if i == len(labels) / 4 or i == (len(labels) * 3 / 4):
                    vertical_alignment = "center"
                elif i < len(labels) / 4 or i > (len(labels) * 3 / 4):
                    vertical_alignment = "bottom"
                else:
                    vertical_alignment = "top"

                self.text(theta[i], 1.025, labels[i], fontsize="small",
                          horizontalalignment=horizontal_alignment,
                          verticalalignment=vertical_alignment)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            return RegularPolygon((0.5, 0.5), num_vars,
                                  radius=.5, edgecolor="k")

        def _gen_axes_spines(self):
            # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
            spine = Spine(axes=self,
                          spine_type='circle',
                          path=Path.unit_regular_polygon(num_vars))
            # unit_regular_polygon gives a polygon of radius 1 centered at
            # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
            # 0.5) in axes coordinates.
            spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

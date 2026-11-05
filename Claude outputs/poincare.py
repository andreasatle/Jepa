"""
Interactive Poincare disk hyperbolic distance visualizer.

Click anywhere inside the unit circle to set the reference point.
A grid of points inside the disk is then colored by their hyperbolic
distance to that reference point.

Run with: python poincare_distance.py
"""

import numpy as np
import matplotlib.pyplot as plt


def poincare_distance(u, v):
    """Hyperbolic distance between two points in the Poincare disk.

    u, v: arrays of shape (..., 2), each row a point with norm < 1.
    """
    diff_sq = np.sum((u - v) ** 2, axis=-1)
    u_norm_sq = np.sum(u ** 2, axis=-1)
    v_norm_sq = np.sum(v ** 2, axis=-1)
    denom = (1 - u_norm_sq) * (1 - v_norm_sq)
    denom = np.clip(denom, 1e-12, None)  # avoid divide-by-zero at the boundary
    arg = 1 + 2 * diff_sq / denom
    return np.arccosh(np.clip(arg, 1.0, None))


def make_grid(n=200):
    """A dense regular grid over the unit square, masked to the disk with NaN."""
    xs = np.linspace(-0.999, 0.999, n)
    ys = np.linspace(-0.999, 0.999, n)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def hyperbolic_ring_lattice(radii, base_count=6):
    """Points placed at equal hyperbolic-distance rings from the origin.

    Each ring is spaced evenly in *hyperbolic* radius (not Euclidean), and
    the number of points on a ring grows like sinh(r) -- the same
    exponential growth in circumference that makes hyperbolic space fit
    a tree. Returns Euclidean (x, y) coordinates for plotting.
    """
    xs, ys = [], []
    base_sinh = np.sinh(radii[0])
    for r in radii:
        n = max(base_count, int(round(base_count * np.sinh(r) / base_sinh)))
        euclid_r = np.tanh(r / 2)  # hyperbolic radius -> Euclidean radius, origin-centered
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs.append(euclid_r * np.cos(thetas))
        ys.append(euclid_r * np.sin(thetas))
    return np.concatenate(xs), np.concatenate(ys)


class PoincareApp:
    LEVELS = 12  # number of isolines

    def __init__(self):
        self.xx, self.yy = make_grid(n=250)
        self.ref = np.array([0.0, 0.0])  # start with the root at the center
        self.contour_set = None
        self.contour_labels = []

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        boundary = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5)
        self.ax.add_patch(boundary)

        # static overlay: points placed at equal hyperbolic spacing from the
        # origin, showing directly that equal hyperbolic steps hold
        # exponentially more points -- this doesn't move when you click,
        # it's a property of the space, not of the reference point
        lattice_x, lattice_y = hyperbolic_ring_lattice(
            radii=np.arange(0.5, 3.5, 0.5), base_count=6
        )
        self.ax.scatter(
            lattice_x, lattice_y, s=10, color="gray", alpha=0.55, zorder=2,
            label="equally spaced in hyperbolic distance",
        )
        self.ax.legend(loc="lower left", fontsize=8)

        self.ref_marker, = self.ax.plot(
            [self.ref[0]], [self.ref[1]], "o", color="red", markersize=10,
            markeredgecolor="white", zorder=5,
        )

        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect("equal")
        self.ax.set_title("Click inside the disk to move the reference point")

        self.redraw_isolines()

    def _distance_field(self):
        pts = np.stack([self.xx, self.yy], axis=-1)
        dists = poincare_distance(pts, self.ref)
        outside = (self.xx ** 2 + self.yy ** 2) >= 0.999 ** 2
        dists[outside] = np.nan
        return dists

    def redraw_isolines(self):
        if self.contour_set is not None:
            try:
                self.contour_set.remove()  # matplotlib >= 3.8
            except AttributeError:
                for coll in self.contour_set.collections:
                    coll.remove()
        for label in self.contour_labels:
            try:
                label.remove()
            except (ValueError, NotImplementedError):
                pass
        self.contour_labels = []

        dists = self._distance_field()
        # even spacing in the interior; the last levels bunch up near the
        # boundary because that's exactly where hyperbolic distance blows up
        levels = np.linspace(np.nanmin(dists), np.nanmax(dists), self.LEVELS)
        self.contour_set = self.ax.contour(
            self.xx, self.yy, dists, levels=levels, cmap="viridis", linewidths=1.2
        )
        self.contour_labels = self.ax.clabel(
            self.contour_set, inline=True, fontsize=8, fmt="%.1f"
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if x ** 2 + y ** 2 >= 0.98 ** 2:
            return  # ignore clicks outside the disk (near/on the boundary)

        self.ref = np.array([x, y])
        self.ref_marker.set_data([x], [y])
        self.redraw_isolines()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = PoincareApp()
    app.show()
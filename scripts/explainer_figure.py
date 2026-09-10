"""What the problem actually is — the README's opening figure, DTFM-059.

    python scripts/explainer_figure.py

The headline figure shows the *result*. This shows the *problem*, for a reader
who has never met ellipsometry: light goes in, bounces off both surfaces of a
film, the two reflections interfere, and the colour pattern that comes back
depends on how thick the film is.

The third panel is the one that matters. It shows why this is not a lookup: two
films of different thickness and different refractive index produce spectra that
overlap almost exactly, so the map from spectrum to thickness is not one-to-one.
That degeneracy is the whole reason the project is about uncertainty rather than
about accuracy.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import baseline as bl  # noqa: E402
from src import dispersion as dp  # noqa: E402
from src import generate as gen  # noqa: E402
from src import losses as ls  # noqa: E402

INK, MUTED = "#161923", "#5b6273"
FILM, LIGHT = "#9694ee", "#b0721a"


def observe(thickness_nm, cauchy_a, wavelengths, substrate):
    theta = np.array([thickness_nm, cauchy_a, 0.004])
    with torch.no_grad():
        return bl.forward_observable(theta, wavelengths, gen.Measurement(), substrate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures/what_this_is.png")
    args = parser.parse_args()

    prior = gen.Prior()
    wavelengths = np.linspace(400.0, 800.0, 200)
    n_si, k_si = dp.load_nk(prior.substrate, wavelengths)
    substrate = torch.tensor(n_si + 1j * k_si)

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # --- 1. the measurement, drawn ------------------------------------------
    ax = axes[0]
    ax.set(xlim=(0, 10), ylim=(0, 6.4), xticks=[], yticks=[],
           title="1 · light in, light out")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(plt.Rectangle((1, 0.6), 8, 1.1, color="#8f95a5", zorder=1))
    ax.add_patch(plt.Rectangle((1, 1.7), 8, 0.75, color=FILM, zorder=1))
    ax.text(9.15, 1.1, "silicon", fontsize=8, va="center", color=MUTED)
    ax.text(9.15, 2.08, "the film", fontsize=8, va="center", color=FILM, fontweight="bold")
    ax.annotate("", xy=(4.2, 2.45), xytext=(1.6, 5.6),
                arrowprops={"arrowstyle": "-|>", "color": LIGHT, "lw": 2})
    ax.text(1.35, 5.75, "light in, at 70°", fontsize=8.5, color=LIGHT)
    ax.annotate("", xy=(6.6, 5.4), xytext=(4.2, 2.45),
                arrowprops={"arrowstyle": "-|>", "color": LIGHT, "lw": 1.6, "alpha": 0.85})
    ax.annotate("", xy=(5.4, 1.7), xytext=(4.2, 2.45),
                arrowprops={"arrowstyle": "-", "color": LIGHT, "lw": 1.2, "alpha": 0.55})
    ax.annotate("", xy=(7.8, 4.6), xytext=(5.4, 1.7),
                arrowprops={"arrowstyle": "-|>", "color": LIGHT, "lw": 1.2, "alpha": 0.55})
    ax.text(6.75, 5.55, "reflects off the top", fontsize=7.5, color=MUTED)
    ax.text(7.5, 4.25, "…and off the bottom", fontsize=7.5, color=MUTED)
    ax.text(0.55, 0.05,
            "The two reflections interfere. How they add up depends on the film's\n"
            "thickness — so the reflected spectrum carries the thickness in it.",
            fontsize=8.2, color=INK)

    # --- 2. thickness changes the pattern -----------------------------------
    ax = axes[1]
    for thickness, colour in ((120.0, "#3b3aa6"), (300.0, "#0f7050"), (900.0, "#a33232")):
        _, delta = ls.split_psi_delta(observe(thickness, 1.46, wavelengths, substrate)[None, :])
        ax.plot(wavelengths, np.degrees(delta[0]), lw=1.5, color=colour,
                label=f"{thickness:.0f} nm thick")
    ax.set(xlabel="wavelength (nm)", ylabel="Δ, one of the two measured angles (°)",
           title="2 · a thicker film, a busier pattern")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # --- 3. why it is hard ---------------------------------------------------
    # The pair below was FOUND, not assumed. Matching optical thickness n*d and
    # calling the result degenerate does not survive a look: at 400 nm the best
    # alternative index still leaves a residual 209x the instrument noise, because
    # dispersion across a 400 nm band breaks the tie easily on a thick film.
    #
    # The degeneracy is real at the THIN end, which is where DTFM-053 located it:
    # |rho(thickness, index)| exceeds 0.99 below about 21 nm of thickness. At 25 nm
    # a 0.28 nm difference in thickness, paired with an index shift of 0.01, is
    # indistinguishable — the residual is half the instrument noise.
    ax = axes[2]
    reference_nm, reference_n = 25.0, 1.46
    twin_nm, twin_n = 24.72, 1.47
    for thickness, index, colour, style, width in (
        (reference_nm, reference_n, "#3b3aa6", "-", 2.6),
        (twin_nm, twin_n, "#b0721a", "--", 1.6),
    ):
        psi, _ = ls.split_psi_delta(observe(thickness, index, wavelengths, substrate)[None, :])
        ax.plot(wavelengths, np.degrees(psi[0]), style, lw=width, color=colour,
                label=f"{thickness:.2f} nm at n={index:.2f}")
    residual = bl.wrapped_residual(
        observe(twin_nm, twin_n, wavelengths, substrate).numpy(),
        observe(reference_nm, reference_n, wavelengths, substrate).numpy(),
        "ellipsometry",
    )
    rms = float(np.sqrt(np.mean(residual**2)))
    ax.set(xlabel="wavelength (nm)", ylabel="Ψ, the other measured angle (°)",
           title="3 · and why it needs an error bar")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    # Top-right: the curve falls from top-left to bottom-right and the legend
    # holds the bottom-right corner, so this is the only clear space.
    ax.text(0.97, 0.96,
            "Two different films, one curve.\n"
            f"They differ by {rms / 1e-3:.2f} mrad — half the\n"
            "instrument's noise, so nothing can tell\n"
            "them apart. The answer is not a number;\n"
            "it is a number and a spread.",
            transform=ax.transAxes, fontsize=8.2, color=INK, va="top", ha="right")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"  figure -> {out}")
    print(f"  panel 3: {reference_nm} nm at n={reference_n} vs {twin_nm} nm at "
          f"n={twin_n} — {rms / 1e-3:.2f} mrad apart, noise is 1.00 mrad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

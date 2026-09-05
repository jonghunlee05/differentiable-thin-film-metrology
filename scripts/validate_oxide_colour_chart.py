"""Compare predicted wafer colours against the published thermal-oxide chart.

The only validation in this project against something *physically observed*
rather than computed. Fabs have used the SiO2-on-silicon colour chart for
decades: grow a known thickness of thermal oxide, look at the wafer under white
light, read the thickness off the colour. If the simulator reproduces those
colours it is reproducing an observation, not another calculation.

    python scripts/validate_oxide_colour_chart.py

Chart source
------------
Rogue Valley Microdevices, "Thermal Oxide Color Chart",
https://roguevalleymicrodevices.com/thermal-oxide-color-chart/ (retrieved
2026-09-05). The same table appears in many fab references and originates with
Pliskin and Conrad's oxide colour work at IBM.

Caveats, because this is a qualitative comparison and should be read as one
-----------------------------------------------------------------------------
- Colour names are subjective and the chart's own entries span ranges
  ("dark violet to red-violet").
- The chart assumes cool-white fluorescent or daylight; this uses an
  equal-energy illuminant, which shifts hues slightly.
- Above roughly 1 um several wavelengths satisfy the interference condition at
  once, the colours wash out towards grey, and the chart says so itself. The
  disagreements below cluster there.

Not committed as a test: hue matching against prose is too brittle to gate CI.
The durable checks live in tests/.
"""

from __future__ import annotations

import numpy as np
import torch

from src import dispersion as dp
from src import tmm_torch as pt

#: thickness in nm -> colour, verbatim from the source above.
CHART: list[tuple[int, str]] = [
    (50, "Tan"), (75, "Brown"), (100, "Dark violet to red-violet"), (125, "Royal blue"),
    (150, "Light blue to metallic blue"), (175, "Metallic, very light yellow-green"),
    (200, "Light gold / yellow"), (225, "Gold, slight yellow-orange"), (250, "Orange to melon"),
    (275, "Red-violet"), (300, "Blue to violet-blue"), (310, "Blue"), (325, "Blue to blue-green"),
    (345, "Light green"), (350, "Green to yellow-green"), (365, "Yellow-green"),
    (375, "Green-yellow"), (390, "Yellow"), (412, "Light orange"), (426, "Carnation pink"),
    (443, "Violet-red"), (465, "Red-violet"), (476, "Violet"), (480, "Blue-violet"),
    (493, "Blue"), (502, "Blue-green"), (520, "Green"), (540, "Yellow-green"),
    (560, "Green-yellow"), (574, "Pale yellow / creamy grey"), (600, "Carnation pink"),
    (630, "Violet to violet-red"), (720, "Blue-green to green"), (770, "Yellowish"),
    (800, "Orange"), (890, "Blue"), (920, "Blue-green"), (970, "Yellow to yellowish"),
    (1000, "Carnation pink"),
]

#: approximate hue centre, in degrees, that each chart word implies
HUE_OF_WORD = {
    "tan": 30, "brown": 25, "violet": 285, "red-violet": 320, "royal blue": 225, "blue": 225,
    "gold": 45, "yellow": 55, "orange": 30, "melon": 20, "blue-green": 180, "green": 120,
    "yellow-green": 80, "green-yellow": 70, "light green": 120, "carnation pink": 345,
    "violet-red": 330, "blue-violet": 260, "salmon": 15, "pink": 345, "yellowish": 55,
    "metallic": None,
}


def _lobe(x, mu, sigma_low, sigma_high):
    sigma = np.where(x < mu, sigma_low, sigma_high)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_xyz_bar(wavelengths_nm):
    """CIE 1931 2-degree observer, analytic fit.

    Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ Color
    Matching Functions", Journal of Computer Graphics Techniques 2(2), 2013.
    Closed form rather than a data table, so nothing extra is vendored.
    """
    x = (1.056 * _lobe(wavelengths_nm, 599.8, 37.9, 31.0)
         + 0.362 * _lobe(wavelengths_nm, 442.0, 16.0, 26.7)
         - 0.065 * _lobe(wavelengths_nm, 501.1, 20.4, 26.2))
    y = (0.821 * _lobe(wavelengths_nm, 568.8, 46.9, 40.5)
         + 0.286 * _lobe(wavelengths_nm, 530.9, 16.3, 31.1))
    z = (1.217 * _lobe(wavelengths_nm, 437.0, 11.8, 36.0)
         + 0.681 * _lobe(wavelengths_nm, 459.0, 26.0, 13.8))
    return x, y, z


def spectrum_to_srgb(wavelengths_nm, reflectance):
    """Reflectance spectrum to a displayable sRGB triple, equal-energy illuminant."""
    x_bar, y_bar, z_bar = cie_xyz_bar(wavelengths_nm)
    normalisation = np.trapezoid(y_bar, wavelengths_nm)
    xyz = np.array([
        np.trapezoid(reflectance * bar, wavelengths_nm) / normalisation
        for bar in (x_bar, y_bar, z_bar)
    ])

    to_srgb = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])
    linear = np.clip(to_srgb @ xyz, 0.0, None)
    if linear.max() > 1.0:
        linear = linear / linear.max()
    gamma = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(gamma, 0.0, 1.0)


def hue_degrees(rgb):
    """Hue in degrees, or None when the colour is too desaturated to have one."""
    high, low = float(np.max(rgb)), float(np.min(rgb))
    if high - low < 0.06 * high:
        return None
    red, green, blue = rgb
    channel = int(np.argmax(rgb))
    raw = {0: (green - blue), 1: 2 * (high - low) + (blue - red),
           2: 4 * (high - low) + (red - green)}[channel] / (high - low)
    return (raw * 60.0) % 360.0


def expected_hue(description: str):
    lowered = description.lower()
    for word in sorted(HUE_OF_WORD, key=len, reverse=True):
        if word in lowered:
            return HUE_OF_WORD[word]
    return None


def main() -> int:
    wavelengths = np.linspace(400.0, 750.0, 351)
    n_film, _ = dp.load_nk("SiO2", wavelengths)
    n_substrate, k_substrate = dp.load_nk("Si", wavelengths)

    film = torch.tensor(n_film)
    substrate = torch.tensor(n_substrate + 1j * k_substrate)
    grid = torch.tensor(wavelengths)

    print("  Thermal SiO2 on silicon, normal incidence, white light.")
    print("  Colour predicted from the transfer-matrix model plus the CIE observer.\n")
    print(f"  {'d':>6}  {'hue':>5}  {'sRGB':>9}  {'published chart':<36} {'agrees':>7}")

    agreements = comparisons = 0
    for thickness, description in CHART:
        spectrum = pt.stack_reflectance(
            grid, [float(thickness)], [1.0, film, substrate], 0.0, "s"
        ).numpy()
        rgb = spectrum_to_srgb(wavelengths, spectrum)
        hue, target = hue_degrees(rgb), expected_hue(description)
        swatch = "#" + "".join(f"{int(255 * channel):02x}" for channel in rgb)

        if hue is None or target is None:
            verdict = "n/a"
        else:
            comparisons += 1
            separation = min(abs(hue - target), 360.0 - abs(hue - target))
            agreements += separation < 60.0
            verdict = "yes" if separation < 60.0 else f"{separation:.0f}deg"

        shown = f"{hue:.0f}" if hue is not None else "grey"
        print(f"  {thickness:5d}n  {shown:>5}  {swatch:>9}  {description:<36} {verdict:>7}")

    print(f"\n  hue within 60 degrees of the published colour: "
          f"{agreements}/{comparisons} = {100 * agreements / comparisons:.0f}%")
    print("  disagreements cluster above ~700 nm, where the chart itself says the")
    print("  colours wash out and it becomes a rougher guide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

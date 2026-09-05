# Vendored optical constants

Real refractive index data for the materials in `PROJECT_SPEC.md` §4.4. Files are
**unmodified** copies from the refractiveindex.info database, kept in their
original YAML so the citation and licence headers travel with the numbers.

## Source

- **Database:** [refractiveindex.info](https://refractiveindex.info/), maintained by Mikhail Polyanskiy
- **Repository:** <https://github.com/polyanskiy/refractiveindex.info-database>
- **Path within repository:** `database/data/main/<material>/nk/<reference>.yml`
- **Retrieved:** 2026-09-05, from `master`
- **Licence:** **CC0 1.0 Universal** — the database is dedicated to the public
  domain, copyright and related rights waived. No attribution is legally
  required; the citations below are given because the underlying measurements
  are other people's work and §3a's honesty statements apply to data as much as
  to results.

## Datasets

| Material | File | Reference | Range | Form |
|---|---|---|---|---|
| SiO₂ | `SiO2/Malitson.yml` | I. H. Malitson, *J. Opt. Soc. Am.* **55**, 1205 (1965) — [doi](https://doi.org/10.1364/JOSA.55.001205) | 0.21–6.7 µm | Sellmeier |
| Si₃N₄ | `Si3N4/Luke.yml` | K. Luke *et al.*, *Opt. Lett.* **40**, 4823 (2015) — [doi](https://doi.org/10.1364/OL.40.004823) | 0.310–5.504 µm | Sellmeier |
| TiO₂ | `TiO2/Siefke.yml` | T. Siefke *et al.*, *Adv. Opt. Mater.* **4**, 1780 (2016) — [doi](https://doi.org/10.1002/adom.201600250) | 0.12–124 µm | tabulated n,k |
| Si | `Si/Aspnes.yml` | D. E. Aspnes and A. A. Studna, *Phys. Rev. B* **27**, 985 (1983) — [doi](https://doi.org/10.1103/PhysRevB.27.985) | 0.2066–0.826 µm | tabulated n,k |

## Why these four

Each is the standard reference for its material, and each matches what this
project actually models — §1.1 restricts scope to blanket dielectric films
after deposition, on a silicon substrate.

- **SiO₂ (Malitson)** — fused silica at 20 °C. The canonical dispersion for
  thermal oxide. Transparent across the visible, so `k = 0`.
- **Si₃N₄ (Luke)** — measured on *340 nm of Si₃N₄ on thermal SiO₂ on silicon*,
  which is the exact stack geometry this project inverts, rather than bulk
  crystal.
- **TiO₂ (Siefke)** — an **ALD** film, 350 nm. §1.1 singles ALD out as the
  deposition method the industry leans on hardest and the regime where the
  degeneracy of §5.2(c) bites, so film data beats bulk-crystal data here.
- **Si (Aspnes & Studna)** — the standard substrate optical constants, with a
  genuine non-zero `k` through the visible. Silicon absorbs, which is what
  makes DTFM-010's branch selection load-bearing rather than decorative.

## Two things to be careful about

**Sign convention.** These files use `n + ik` with `k ≥ 0`. `src/tmm_torch.py`
rejects `Im(ñ) < 0` for exactly this reason — some sources tabulate `n − ik`,
and ingesting under the wrong convention would yield a plausible spectrum from
an impossible material with nothing raising.

**Validity range.** Every dataset is valid only over the range in the table.
`src/dispersion.py` refuses to extrapolate rather than silently returning a
number, because an extrapolated index is an invented one and §4.4's instruction
is to fit real data rather than invent coefficients.

## Refreshing

`scripts/fetch_optical_constants.py` re-downloads these files from the source
repository. It is not run automatically: the pinned copies are what every result
in this project was computed against, and a silent upstream revision would
change published numbers without any commit recording it.

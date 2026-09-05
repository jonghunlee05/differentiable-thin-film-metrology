# PROJECT_SPEC.md

**Project:** `differentiable-thin-film-metrology`
**Working title:** Thickness from Light
**One line:** Physics-informed neural inversion of thin-film reflectance spectra, with calibrated uncertainty.

---

## What you are building

**A program that looks at reflected light and says how thick a film is — and works out when that program is lying.**

Four pieces of code:

| # | Does | Section |
|---|---|---|
| 1 | `film → spectrum` — the physics simulator | §4 |
| 2 | `spectrum → film`, slowly — guess and check | §6 |
| 3 | `spectrum → film`, instantly — a neural network | §7 |
| 4 | **when is #3 wrong, and does it know?** ← the part that matters | §8 |

Everything else in this document is either *why anyone cares* or *how to do those four well*. **If the rest of it ever stops making sense, come back to this table.** These four have not changed and will not change.

---

## 0. Purpose of this document

This is the source of truth for what the project is, what the physics is, what gets built, and in what order. It is written to be read by (a) me, months from now, (b) anyone reviewing the repository, and (c) an AI coding assistant that needs to be oriented at the start of a session.

If something in the code contradicts this document, one of them is wrong and it should be resolved rather than left.

---

## 0.1 Orientation for a new session

**Picking this up cold — whether that is me after a break, a reviewer, or an AI coding assistant? Read this section, then §1, then §5. That is enough to start being useful.**

### The project in three lines

1. A **differentiable transfer-matrix simulator** (PyTorch) predicts a reflectance spectrum from a film's thickness and refractive index.
2. A **learned inverse solver** runs that backwards in one forward pass instead of ~50 fitting iterations.
3. The real work is establishing **when the fast solver should not be believed** — calibrated uncertainty, and the regimes where the physics guarantees ambiguity.

### Standing decisions — settled, do not relitigate

These were argued out already. Reopening them costs time and changes nothing.

| Decision | Reason |
|---|---|
| **Post-deposition, not post-CMP** | Blanket film, so TMM is unambiguously correct. Post-CMP is flat but compositionally patterned, its interesting quantities are topographic, and pattern-density effects break the blanket-pad shortcut. §1.1 |
| **Dielectric films only** | Metals are opaque — no transmitted light, no second reflection, no fringes. Different measurement problem entirely. |
| **TMM, not RCWA** | Flat unpatterned films need nothing more. RCWA excluded for implementation risk, not relevance. §3 |
| **Write the forward model; import `tmm` only to validate** | The forward model is what proves the physics is mine, and it is the easy part. |
| **Uncertainty is the centre of gravity, not speed** | The speed result is well-trodden and commercially done. The trust question is not. §3a |
| **Synthetic data is the baseline; real data is a live goal** | No lab access at time of writing. Two cheap routes remain open. §3 |
| **No novelty claimed on speed** | Would be immediately corrected by anyone in the field. §3a |
| **MLP and 1D CNN only** | Architecture novelty is not what is being assessed. A transformer here signals inexperience, not ambition. §7.2 |
| **Baseline before network, always** | An accuracy number with nothing to compare against is meaningless, and benchmarking against baselines is explicitly requested by the target roles. |

### Open decisions

- **Ellipsometry (Ψ, Δ) — decide by week 3.** Cheap to add once both polarisations are tracked, and it substantially breaks the thickness–index degeneracy. Recommended. §3
- **The week 7½ fork** — finish the safe version, or push at an open question. Decide with real information about pace, not in advance. §12

### Status

Update this as phases complete. It is the fastest way for anyone — including me in three weeks — to know where things stand.

```
[ ] Week 0    ramp: repo, env, Fresnel, Brewster check, autograd check
[ ] Weeks 1-2 simulator: full TMM, three validations
[ ] Week 3    materials + dataset generator     ← ellipsometry decision here
[ ] Week 4    classical baseline
[ ] Weeks 5-6 network + reconstruction loss
[ ] Week 7    uncertainty + calibration          ← the headline result
[ ] Week 7.5  fork decision
[ ] Week 8    failure atlas
[ ] Week 9    hybrid
[ ] Week 10   packaging + report
```

### Commands

```bash
pytest tests/ -v                 # physics claims must stay green
python -m src.generate --config configs/default.yaml
python -m src.evaluate  --config configs/default.yaml
python scripts/make_figures.py   # regenerates every figure in figures/
```

### How to work in this repository

- **Update the status block above** when a phase completes. Nothing else tracks progress.
- **If code and this spec disagree**, resolve it — do not leave the contradiction.
- **Never report a fit without an error bar.** In this field that is a disqualifying instinct, not an oversight.
- **Tests encode the physics claims.** Brewster, quarter-wave null, agreement with `tmm`, autograd vs finite differences. Keep them passing; add to them when a new physical claim is made.
- **Logic lives in `src/`, narrative in `notebooks/`.** Not the other way round.
- **Commit small and often.** The history is itself evidence of sustained work.

---

## 1. What this is

A semiconductor fab deposits thin films onto wafers and must verify their thickness before building the next layer on top. Thickness cannot be measured directly, so it is inferred from a reflectance spectrum: light reflecting off the top and bottom surfaces of the film interferes, producing wavelength-dependent fringes whose structure encodes the thickness.

Recovering thickness from that spectrum is an **inverse problem**. It is currently solved by iterative fitting, which is accurate and well-characterised but slow — roughly a second per measurement site. That cost limits how densely a wafer can be sampled, typically to 10–20 predetermined sites per wafer, which means most of the wafer surface is never actually measured.

This project builds a learned inverse solver that performs the same inversion in a single forward pass, and — more importantly — establishes **when that fast solver should not be trusted**.

### 1.1 Where this sits in the process flow

A chip is built by repeating one cycle fifty to a hundred times, once per layer:

```
DEPOSIT → COAT resist → EXPOSE → DEVELOP → ETCH → STRIP → CLEAN → CMP → MEASURE → (repeat)
```

**This project targets the measurement immediately after deposition, before lithography.** That position is chosen deliberately:

- **The film is blanket.** No pattern has been printed yet, so the layer is genuinely flat and uniform and the transfer matrix method is unambiguously the correct tool. No diffraction, no topography.
- **It is the earliest possible check.** A bad film is caught before any lithography time is spent on it.
- **Rework is still possible.** Strip and redeposit, or hold the lot. Ten layers later neither option exists.

**Restricted to dielectric films** — CVD or ALD oxide and nitride, or thermal oxide from a furnace step. Optical thickness metrology requires light to pass through the film, reflect off the buried interface, and interfere with the top reflection. Metals deposited by PVD are opaque, produce no second reflection and no fringes, and are measured by other means entirely (four-point probe, X-ray fluorescence, picosecond ultrasonics). Those are out of scope.

**Why not post-CMP.** Considered and rejected. After CMP the surface is flat geometrically but patterned compositionally, the quantities of interest (dishing, erosion) are topographic and need RCWA, many CMP steps are on opaque metal, and — decisively — CMP removal rate depends on local pattern density, so a blanket monitor pad does not represent what happened on the array. Post-deposition avoids all of this and makes TMM the obviously correct choice rather than a defended one.

**Relevance note worth keeping.** ALD is used where films must be very thin with tight control — single-digit to tens of nanometres. That is exactly the regime of degeneracy (c) in §5.2, where the spectrum stops responding to refractive index and the fit returns a confident but meaningless number. The failure regime this project characterises is therefore the regime that matters most for the deposition method the industry is leaning on hardest.

## 2. Problem statement

**Industrial framing.** Measurement throughput limits process control. Denser sampling would give a true thickness map rather than a scatter of points, revealing spatial non-uniformity that sparse sampling misses. The obstacle is inversion cost.

**Technical framing.** Given a measured reflectance spectrum `R_obs(λ)`, recover the film parameters `θ = (d, n(λ))` that produced it. The forward map `θ → R` is cheap and exact. The inverse map is expensive (iterative) and, in identifiable regimes, non-unique.

**The gap this project targets.** Classical inversion is slow but its uncertainty is rigorously characterised — the fit hands you a covariance matrix as a by-product, and there is a mature metrology discipline built on that. Learned inversion is fast but its uncertainty estimates are far weaker and far less studied. **The project's centre of gravity is closing that second gap, not the first.**

## 3. Scope

**In scope**
- Blanket dielectric films immediately after deposition, before lithography (§1.1)
- Flat, unpatterned multilayer stacks at normal or near-normal incidence
- Reflectance (intensity) measurement
- Synthetic data generated from the project's own forward model
- Thickness and dispersion-parameter recovery, with calibrated uncertainty
- Comparison against a classical least-squares baseline

**Recommended extension — decide by week 3**

*Ellipsometry (Ψ, Δ).* Cheaper than it sounds: once the transfer matrix tracks both polarisations, which it does anyway, the ellipsometric quantities follow from `tan(Ψ)·e^(iΔ) = r_p / r_s` — a handful of lines. The gain is disproportionate. Ellipsometry provides two measured quantities per wavelength instead of one, is a ratio and therefore insensitive to source intensity drift, and **substantially breaks the thickness–index degeneracy that is the central difficulty of this project**.

It also unlocks a result not otherwise available: quantify with Fisher information *how much* the degeneracy shrinks moving from reflectance-only to full ellipsometry. That is an information-content question with a real number attached, and a better use of effort than adding another network architecture.

**Out of scope, with reasons**

| Excluded | Reason |
|---|---|
| Patterned / periodic structures | Requires RCWA, not TMM. Excluded for **implementation risk**, not relevance — patterned metrology (OCD) is arguably the more industrially current problem. Writing RCWA from scratch is a multi-month effort with unstable gradients and convergence pitfalls. **Middle path if ever wanted:** differentiable RCWA libraries already exist (`grcwa`, `TORCWA`, `rcwa_tf`) and could replace the hand-written simulator — at the cost of losing the "I wrote the forward model" credential. |
| Opaque metal films (PVD) | No transmitted light, no second reflection, no fringes. A different measurement problem requiring different instruments. |
| Post-CMP measurement | See §1.1. |
| Full 2D wafer-map reconstruction | A different problem (spatial interpolation between sites) from this one (per-site inversion). Worth knowing the distinction; not in scope. |

**Real measured data — a live goal, not an exclusion**

No lab access at time of writing, so the project is built to work entirely on synthetic data. But real data is the single largest available credibility upgrade and two routes cost almost nothing:

1. **Literature digitisation.** Extract measured reflectance or ellipsometry spectra from published papers where the authors report their fitted thickness, and test the pipeline against their answer. That is validation against ground truth someone else established.
2. **Ask.** Email materials groups for ten minutes on an ellipsometer with any sample to hand. Cost: one email.

If either lands, add a sim-to-real section and quantify the gap. Treat this as an open action item throughout, not a closed door.

---

## 3a. Honesty statements

Not scope decisions — claims deliberately *not* made. These cost nothing and prevent avoidable damage in an interview.

**All data is synthetic and no lab access was available.** State this in the README. Any reviewer works it out in seconds; pre-empting it reads as judgement rather than as a gap.

**No novelty is claimed on the speed result.** Optical thickness metrology is decades old and mature. ML-accelerated inversion is done commercially. Claiming otherwise to a metrology engineer would be immediately corrected and would cost credibility.

**Measurement of thickness by spectroscopy is a solved problem, and so is the uncertainty of a classical fit.** The open question is narrower: *when the slow, well-characterised method is replaced by a fast learned one, can the trustworthiness be brought back up to the standard the old method already had?* That is the gap this project addresses. Phrasing it any wider is inaccurate.

**The project's value is demonstrated competence on a recognisable industrial problem, plus a genuinely under-explored angle on uncertainty.** That is sufficient for its purpose. Legibility to a reviewer beats novelty they would have to be talked into.

---

## 4. Physics

### 4.1 The measurement

A film of thickness `d` and complex refractive index `ñ = n + ik` sits on a substrate. Incident light partially reflects at the top interface and partially at the bottom; the two paths differ in optical path length and interfere. The result is a reflectance `R(λ)` with fringes whose spacing depends on `n·d`.

This is the same physics as the colours in a soap bubble or an oil film.

### 4.2 Fresnel coefficients

At an interface between media `i` and `j`, for the two polarisations:

```
r_s = (ñ_i cosθ_i − ñ_j cosθ_j) / (ñ_i cosθ_i + ñ_j cosθ_j)
r_p = (ñ_j cosθ_i − ñ_i cosθ_j) / (ñ_j cosθ_i + ñ_i cosθ_j)
```

with angles related by Snell's law `ñ_i sinθ_i = ñ_j sinθ_j`, so

```
cosθ_j = sqrt(1 − (ñ_i/ñ_j)² sin²θ_i)
```

**Branch cut warning.** For absorbing media `ñ` is complex and this square root has two roots. The physical one satisfies **`Im(ñ_j cosθ_j) ≥ 0`**. Choosing the wrong branch produces gain instead of absorption — this is the single most common bug in a from-scratch implementation and it can cost an evening. Assert the sign explicitly in code.

The condition is on `ñ_j cosθ_j`, not on `cosθ_j` alone, because the transmitted wave carries `exp(i·(2π/λ)·ñ_j·d·cosθ_j)` and it is that product which sets whether the amplitude decays or grows. The two conditions coincide whenever the incident medium is transparent, which is why the simpler form is often quoted — but they diverge once the incident medium absorbs, and every interior interface of a stack containing an absorbing film is exactly that case. Measured over an index-and-angle sweep spanning weak absorbers to metals: the two rules disagree on 3% of cases, and `Im(cosθ_j) ≥ 0` admits gain in all of them. The naive principal root is unphysical in roughly 14%.

### 4.3 Transfer matrix method

Propagation through layer `j` accumulates phase

```
δ_j = (2π / λ) · ñ_j · d_j · cosθ_j
```

Build 2×2 matrices for each interface and each layer:

```
I_ij = (1/t_ij) · [[1,    r_ij],
                   [r_ij,    1]]

L_j  =            [[exp(−i·δ_j),           0],
                   [0,           exp(+i·δ_j)]]
```

The full stack is the ordered product

```
M = I_01 · L_1 · I_12 · L_2 · … · I_{N−1,N}
```

from which

```
r = M[1,0] / M[0,0]
R = |r|²
```

Compute independently for s and p; at normal incidence they coincide.

### 4.4 Dispersion models

Refractive index varies with wavelength. Implement:

| Model | Form | Use for |
|---|---|---|
| Cauchy | `n(λ) = A + B/λ² + C/λ⁴` | Transparent dielectrics below the absorption edge |
| Sellmeier | `n² = 1 + Σ Bᵢλ²/(λ² − Cᵢ)` | Transparent materials, wider range, physically better behaved |
| Lorentz oscillator | `ε(E) = ε_∞ + Σ Aₖ/(Eₖ² − E² − iΓₖE)` | Absorbing materials |

Real `n, k` data for SiO₂, Si₃N₄, TiO₂ and Si comes from **refractiveindex.info**. Fit the models to that data rather than inventing coefficients.

**Physics you should be able to defend in an interview:** why a Cauchy model is appropriate below the absorption edge and wrong above it; why Kramers–Kronig relations constrain `n` and `k` to be consistent with each other.

### 4.5 Non-idealities

Real spectra are messier than ideal ones. Each of these is a real effect and each degrades recoverability in a different way. Implement at least three.

| Effect | Model | Consequence |
|---|---|---|
| Surface roughness | Bruggeman effective-medium layer of mixed film/void | Damps fringe amplitude |
| Interfacial layer | Extra thin layer between film and substrate | Biases thickness if unmodelled |
| Thickness non-uniformity in spot | Average `R` over `d ~ N(d₀, s²)` | Damps high-order fringes |
| Finite spectrometer bandwidth | Convolve `R(λ)` with Gaussian of FWHM `Δλ` | Same signature as above — and that ambiguity is itself interesting |
| Wavelength calibration error | `λ → λ(1+α) + β` | Systematic thickness bias |
| Baseline drift | `R → aR + b` | Correlates with dispersion parameters |
| Backside reflection | Incoherent addition from the wafer's rear face | Adds an offset |

---

## 5. The inverse problem

### 5.1 Formal statement

Let `θ` be the parameter vector (thickness plus dispersion coefficients) and `f` the forward model. The observation is

```
R_obs = f(θ) + ε ,   ε ~ N(0, σ²)
```

and the classical estimator is

```
θ̂ = argmin_θ ‖ R_obs − f(θ) ‖²
```

### 5.2 Why it is ill-posed — three specific degeneracies

These are not vague difficulties. They are precise, computable, and they are the intellectual core of the project.

**(a) Thickness–index correlation.** At a single wavelength the measurement constrains only the optical thickness `n·d`. Thickness and index are perfectly degenerate. Spectroscopic measurement partially breaks this via dispersion — but only partially, and how much can be quantified exactly.

**(b) Fringe-order ambiguity.** `R` is quasi-periodic in `n·d/λ`. A film of thickness `d` and one of `d + λ/(2n cosθ)` produce near-identical spectra over a limited spectral band. The likelihood is genuinely **multimodal**, which matters enormously for how uncertainty should be represented.

**(c) Thin-film insensitivity.** For `d ≪ λ` the derivative `∂R/∂n → 0`. The fit still returns a number. The number is meaningless. Locating that threshold quantitatively is one of the project's best figures.

### 5.3 Identifiability analysis

Build the Jacobian of the forward model at the true parameters:

```
J[i,k] = ∂f(λ_i) / ∂θ_k
```

Then

```
Fisher information   F = JᵀJ / σ²
Parameter covariance C = F⁻¹              (Cramér–Rao lower bound)
Correlation          ρ_jk = C_jk / sqrt(C_jj · C_kk)
```

`ρ` between thickness and index will frequently exceed 0.99. **Reporting that alongside the fitted value is what separates a metrologist from someone who called `curve_fit`.**

The CRB is the theoretical floor on the variance of any unbiased estimator for this measurement design. If your estimator sits near the bound, the algorithm is fine and the *measurement* is the limit — a conclusion worth stating out loud.

With a differentiable forward model, `J` comes from autograd rather than finite differences.

---

## 6. Classical inversion — the baseline

**This must exist before any network is trained.** Without it, a learned model's accuracy number is meaningless, and benchmarking against a baseline is explicitly what the target job descriptions ask for.

**Method.** Nonlinear least squares via `scipy.optimize.least_squares` (Levenberg–Marquardt, or trust-region reflective for bounded problems). Additionally: gradient-descent inversion straight through the autograd simulator, as a second baseline.

**Handle multimodality** with multi-start from many initial guesses; record where each converges. That landscape *is* the fringe-order ambiguity made visible.

**Record for every case:** recovered parameters, error vs truth, wall-clock time per fit, iteration count, convergence success/failure, and the covariance from the converged Jacobian.

**Model selection.** When deciding whether the data justifies an extra parameter (a roughness layer, say), use AIC or BIC, or a nested F-test. The question *"is my model too complex for the information my measurement contains"* is one almost nobody asks and asking it is a strong signal.

---

## 7. Learned inversion

### 7.1 Data generation

The simulator is the labeller, so training data is unlimited and free. Generate on the fly rather than storing a dataset.

```
for each batch:
    sample θ ~ prior over (thickness, dispersion coefficients)
    R_clean = f(θ)
    R_obs   = corrupt(R_clean)     # §4.5 non-idealities + detector noise
    yield (R_obs, θ)
```

**Sampling design is a decision, not a detail.** Uniform in thickness produces a different model than log-uniform. State the choice and the reason. The prior you sample from *is* the model's implicit prior — everything outside it is out-of-distribution at test time.

**Noise model.** Shot noise `σ ∝ sqrt(R)`, plus additive detector noise, plus the systematic effects in §4.5.

### 7.2 Architecture

- **MLP** — the first baseline. Spectrum vector in (~200 values), parameters out.
- **1D CNN** — better suited; fringes are local repeating structure.
- Two output heads: parameter estimate `θ̂` and log-variance `log σ̂²`.

**Do not reach for a transformer.** Architecture novelty is not what is being assessed here, and over-reaching signals inexperience rather than ambition.

### 7.3 Loss

```
L = L_nll  +  λ_recon · L_recon
```

where

```
L_nll   = ½ Σ_k [ (θ̂_k − θ_k)² / σ̂_k²  +  log σ̂_k² ]        # Gaussian NLL
L_recon = ‖ f(θ̂) − R_obs ‖²                                   # physics term
```

`L_recon` is only possible because `f` is written in PyTorch and is therefore differentiable. **This term is what makes the project scientific machine learning rather than generic regression** — it keeps the physics inside the training signal and rejects parameter estimates that fit the numbers but could not have produced the observed curve.

Run an ablation on `λ_recon` and show cases the physics term rescues.

---

## 8. Uncertainty quantification — the heart of the project

### 8.1 Two kinds of uncertainty

- **Aleatoric** — irreducible, from measurement noise and from genuine physical degeneracy. Captured by the `σ̂` head.
- **Epistemic** — the model's own ignorance, reducible with more data. Captured by an ensemble.

**Deep ensemble** of `M ≈ 5` models with different seeds:

```
mean = (1/M) Σ_m θ̂_m
var  = (1/M) Σ_m (σ̂_m² + θ̂_m²) − mean²
```

### 8.2 Calibration — is the model telling the truth?

A predicted `±2 nm` is a claim. Test it.

- **Coverage.** For nominal levels (68%, 95%), measure the empirical fraction of test cases where the truth falls inside the predicted interval. Plot nominal vs empirical — the reliability diagram.
- **Expected calibration error.** Aggregate the deviation from the diagonal into one number.
- **Comparison to the Cramér–Rao bound.** The physics gives a floor on achievable variance. If the network claims uncertainty *below* the CRB, it is claiming something impossible and you have caught it doing so. This check is only available because you have the physics, and it is the strongest single result the project can produce.

### 8.3 Why a Gaussian head is not sufficient

Fringe-order ambiguity makes the posterior **bimodal**: two well-separated thicknesses can both be plausible. A mean-and-variance output cannot represent that — it will average two correct answers into one wrong one, and report a confident, incorrect number.

**Extension (optional, if time permits):** neural posterior estimation with a normalizing flow, which can represent a multimodal posterior. Validate with simulation-based calibration — the rank statistics should be uniform. The argument for using a flow here is a *physics* argument, which is the right register. Only do this if you can defend it; a technique you cannot justify is worse than one you did not use.

---

## 9. The hybrid

The engineering conclusion, and what a real tool would ship:

```
θ_init = network(R_obs)                      # microseconds
θ̂      = least_squares(R_obs, x0=θ_init)     # a few iterations, not fifty
```

Expected result: the network's speed with the classical method's reliability, and resolution of the fringe-order ambiguity that defeats a cold-started fit.

This is a more mature position than "my network beat the baseline." Frame it as **ML removing the expensive part of a physics method**, not as ML replacing physics.

---

## 10. Evaluation protocol

Every claim gets a number, computed the same way for every method.

| Metric | Applies to |
|---|---|
| Thickness RMSE and median absolute error | all methods |
| Error stratified by regime (thin / mid / thick, low / high SNR) | all methods |
| Wall-clock time per inversion | all methods |
| Convergence failure rate | classical, hybrid |
| Empirical coverage at 68% / 95% | learned, hybrid |
| Expected calibration error | learned |
| Ratio of claimed σ̂ to Cramér–Rao σ | learned |
| Out-of-distribution behaviour | learned |

**The failure atlas.** A deliberate table of regimes where things break: near-degenerate `n`–`d`, fringe-order ambiguity, `d ≪ λ`, unmodelled roughness, out-of-prior stacks. For each: what the classical method does, what the network does, and **whether the uncertainty head noticed.**

The expected and most valuable finding is that the classical method fails *loudly* and the network fails *silently*. In a fab a confidently wrong measurement is more dangerous than a slow one, because it gets acted upon.

---

## 11. Workflow

```
  θ (sampled truth)
        │
        │  differentiable TMM  ── forward, exact, fast
        ▼
  R(λ) + realistic noise
        │
        ├─────────────► classical inversion (LM)  ──► θ̂, covariance, timing
        │
        └─────────────► network  ──► θ̂, σ̂
                            │
                            │  θ̂ pushed back through TMM
                            ▼
                     reconstruction loss  ── physics inside the training signal

  then:  network θ̂  ──►  LM refinement  ──►  hybrid result
```

Two loops to hold in mind:

- **Training loop** — expensive, offline, runs once.
- **Inference loop** — cheap, runs on every measurement forever after.

Moving cost from the second into the first is called **amortisation**, and it is the entire industrial argument for the project.

---

## 12. Roadmap

Each phase ends with something committed, so stopping early still leaves a coherent project.

### Week 0 — ramp
Repository, environment. Fresnel at a single interface, validated against Brewster's angle (`tan θ_B = n₂/n₁`, where `r_p = 0` exactly). Rewrite in torch, verify identical numbers, take an autograd derivative and check against finite differences. Read Byrnes, *Multilayer optical calculations*.
**Ships:** validated single-interface model, ~5 commits.

### Weeks 1–2 — simulator
Full transfer matrix method in PyTorch. Complex indices, both polarisations, arbitrary layer count. Validate three ways: analytic single-interface Fresnel; a quarter-wave AR coating that nulls at its design wavelength; agreement with the `tmm` package. **Write your own; import theirs only to check.**
**Ships:** differentiable simulator, three validation plots, gradient check.

### Week 3 — materials and data
Cauchy, Sellmeier, Lorentz fitted to real `n, k` from refractiveindex.info. Dataset generator with the §4.5 noise model. Document and justify the sampling prior.
**Ships:** materials module, reproducible generator.

### Week 4 — baseline
Levenberg–Marquardt inversion and autograd gradient-descent inversion. Full performance record per §10. Multi-start to expose local minima.
**Ships:** baseline table, parity plot with timings.

### Weeks 5–6 — the network
MLP then 1D CNN. Benchmark against baseline. Add the reconstruction loss; ablate it.
**Ships:** two trained models, benchmark table, physics-loss ablation.

### Week 7 — uncertainty
Heteroscedastic head, Gaussian NLL, deep ensemble. Reliability diagram, ECE, comparison to the Cramér–Rao bound.
**Ships:** the calibration figure — the strongest result in the project.

### Week 7½ — the fork
Everything to here is identical whichever way you go, so the decision is made *here*, with real information about your own pace.

**Stop and finish** (weeks 8–10 below) for a complete, defensible project. Safe, and perfectly good for an application.

**Or point the remaining weeks at an open question.** Two candidates, both studiable entirely in simulation:

1. *Does the network's uncertainty match the ambiguity the physics predicts?* The degeneracy structure is analytically computable, so you hold a ground truth for what the uncertainty **ought** to look like — something almost nobody in ML has. Has not been systematically checked for a problem where the true structure is known.
2. *What happens when the simulator is wrong?* Train on simplified physics, test on richer physics. Can the model detect that its own generator was mis-specified, rather than answering confidently anyway? The more industrially serious of the two, and an open problem in the wider simulation-based-inference literature.

**Cost, stated honestly:** novel work can end in a thin result, and for an application a finished ordinary project beats an unfinished ambitious one. Take the fork only if week 7 arrives early.

### Week 8 — failure atlas
Test deliberately in the regimes §5.2 predicts are hard, plus out-of-distribution stacks. Record whether the uncertainty head noticed.
**Ships:** the failure atlas.

### Week 9 — hybrid
Warm-started refinement. Three-way comparison: classical, learned, hybrid.
**Ships:** the comparison, and the engineering conclusion.

### Week 10 — packaging
README opening with the calibration figure and three sentences. Package structure, pinned dependencies, fixed seeds, one command that regenerates every figure. A 4–6 page report with the physics stated properly and the uncertainty analysis as the centrepiece, not an appendix.
**Ships:** the repository as a portfolio object.

---

## 13. Repository structure

```
differentiable-thin-film-metrology/
├── README.md                  calibration figure, three sentences, how to run
├── PROJECT_SPEC.md            this file
├── pyproject.toml             pinned dependencies
├── src/
│   ├── fresnel.py             single-interface reference, numpy (§4.2)
│   ├── tmm_torch.py           differentiable forward model
│   ├── dispersion.py          Cauchy, Sellmeier, Lorentz; real n,k loaders
│   ├── noise.py               §4.5 non-idealities
│   ├── generate.py            dataset sampling
│   ├── baseline.py            Levenberg–Marquardt + autograd inversion
│   ├── models.py              MLP, 1D CNN, heteroscedastic head, ensemble
│   ├── losses.py              parameter, reconstruction, Gaussian NLL
│   ├── uncertainty.py         Jacobian covariance, Cramér–Rao, calibration
│   └── evaluate.py            metrics, failure atlas, OOD probes
├── tests/
│   ├── test_fresnel.py        Brewster angle null
│   ├── test_tmm.py            quarter-wave null; agreement with `tmm` package
│   ├── test_autograd.py       autograd vs finite differences
│   └── test_reproducible.py   fixed seed → fixed output
├── notebooks/
│   ├── 01_simulator_validation.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_training.ipynb
│   └── 04_calibration_and_failure.ipynb
├── configs/
├── data/
│   └── refractiveindex/       vendored n,k with provenance (§4.4); CC0
├── scripts/
├── figures/
└── report.md
```

---

## 14. Prerequisites

**Must be solid already:** complex arithmetic (fluent, not nodding acquaintance); waves and interference; 2×2 matrix multiplication (that is the entire linear algebra requirement); partial derivatives; mean/variance/Gaussian and what a confidence interval claims; Python with numpy.

**Learn during, in the week it is needed:** PyTorch tensors and autograd (week 0); Fresnel and TMM (weeks 1–2); `scipy.optimize` (week 4); dispersion models (week 3); fit covariance and Cramér–Rao (weeks 6–7); git (day 1).

**Not needed:** quantum mechanics, solid-state physics, PDEs, ML theory beyond knowing what a loss is, a GPU, cloud compute, a lab. If a resource opens with band structure or measure theory, it is aimed at a different problem.

---

## 15. Engineering conventions

- **Reproducibility is a requirement, not a nicety.** Fixed seeds, config files, and a single command that regenerates every figure in the repository.
- **Tests encode correctness.** The four in `tests/` are non-negotiable; they are what make the physics claims checkable.
- **Package, not notebooks.** Notebooks are for narrative; logic lives in `src/`. A repository of numbered notebooks with no package structure fails the "clean, reusable, well-documented Python workflows" bar explicitly named in the ASML posting.
- **Commit from day one**, small and often. The commit history is itself evidence of sustained work.
- **Never report a fit without an error bar.** In this field that is a disqualifying instinct rather than an oversight.
- **Do not commit generated data.** The point is that it regenerates.

---

## 16. Definition of done

The project is finished when a reader who knows the field can, in four minutes:

1. See from the README what was built and what the headline result is.
2. Find one figure showing the network is faster than the baseline and by how much.
3. Find one figure showing whether its uncertainty is honest.
4. Find a clear statement of the regimes where it fails and why the physics made that inevitable.
5. Clone it, run one command, and reproduce every figure.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **TMM** | Transfer matrix method — the forward optical model for layered films |
| **Inverse problem** | Recovering causes (film parameters) from effects (a spectrum) |
| **Ill-posed** | Solution may be non-unique or unstable under small perturbations |
| **Degeneracy** | Different parameters producing indistinguishable observations |
| **Fringe order** | Which interference cycle a feature belongs to — ambiguous over a limited band |
| **Jacobian** | Matrix of derivatives of every output with respect to every parameter |
| **Fisher information** | How much a measurement tells you about a parameter |
| **Cramér–Rao bound** | Theoretical floor on the variance of any unbiased estimator |
| **Aleatoric / epistemic** | Irreducible noise-driven uncertainty / reducible model ignorance |
| **Heteroscedastic** | Uncertainty that varies from input to input, rather than being constant |
| **Calibration** | Whether stated confidence matches observed accuracy |
| **Amortisation** | Paying a cost once at training so every later inference is cheap |
| **OOD** | Out of distribution — inputs unlike anything in training |
| **ME** | Metrology, in fab terminology — the measurement step this project sits in |
| **OCD** | Optical critical dimension — the patterned-structure cousin of this problem |

---

## 18. References

- S. J. Byrnes, *Multilayer optical calculations* — arXiv. The clearest statement of the TMM; documentation for the reference `tmm` package. **Read first.**
- H. Fujiwara, *Spectroscopic Ellipsometry: Principles and Applications*. Standard textbook; the early chapters suffice.
- *Numerical Recipes*, chapter on modelling of data. Still the clearest account of where a fit's error bars come from.
- C. Guo et al., *On Calibration of Modern Neural Networks*. Short; the origin of the observation that networks are systematically overconfident.
- B. Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*.
- K. Cranmer, J. Brehmer, G. Louppe, *The frontier of simulation-based inference*, PNAS 2020. Only if pursuing the extended version.
- refractiveindex.info — optical constants database.
- Semiconductor Engineering, *Full Wafer OCD Metrology* — industry statement of the sparse-sampling problem this project is motivated by.

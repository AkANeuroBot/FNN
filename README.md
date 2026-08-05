# FNN — Fourier Neural Network from scratch

A single-hidden-layer feed-forward network written entirely in NumPy, with no machine-learning framework of any kind, used to compare two activation functions on a function-approximation task: the **cosine squasher** of Gallant and White and the standard **logistic sigmoid**.

The purpose of the project is not to obtain a competitive regressor, but to reproduce, with a hand-written implementation, the claim that a network whose hidden units use a cosine squasher can approximate an arbitrary band-limited signal and encode its spectral content in its own weights, in the manner of a truncated Fourier series.

This code was produced for my Bachelor's thesis in Physics at the University of Trieste.

---

## Background

Gallant and White (1988) showed that a single-hidden-layer network with the *cosine squasher* activation

$$
\sigma(z) =
\begin{cases}
0, & z < -\pi/2 \\
\tfrac{1}{2}\left[\cos\left(z + \tfrac{3}{2}\pi\right) + 1\right], & -\pi/2 \le z \le \pi/2 \\
1, & z > \pi/2
\end{cases}
$$

can embed a Fourier series exactly: an appropriate pair of hidden units reproduces a sinusoidal term, so the hidden layer acts as a learnable basis of harmonics rather than as a generic set of soft thresholds. The logistic sigmoid is also a universal approximator (Hornik, Stinchcombe and White, 1989), but it does not carry this spectral interpretation.

The experiments in this repository ask whether that structural difference is visible in practice — in the convergence rate, in the number of epochs needed to reach a target fit, and in the sensitivity to the number of samples, the width of the hidden layer and the learning rate.

---

## Model

Both networks share the same architecture and the same training loop; only the hidden activation differs.

**Forward pass** for a single sample $x_i \in \mathbb{R}$:

$$
h_i = \sigma\!\left(x_i W^{(1)} + b^{(1)}\right), \qquad
\hat{y}_i = h_i W^{(2)} + b^{(2)}
$$

with $W^{(1)} \in \mathbb{R}^{1 \times H}$, $W^{(2)} \in \mathbb{R}^{H \times 1}$ and a linear output unit.

**Backward pass** — the gradients of the squared error are derived and coded by hand:

$$
\delta_i = \hat{y}_i - y_i, \qquad
\nabla_{W^{(2)}} = h_i^{\top}\delta_i, \qquad
\nabla_{W^{(1)}} = x_i^{\top}\left[\left(\delta_i W^{(2)\top}\right) \odot \sigma'\!\left(x_i W^{(1)} + b^{(1)}\right)\right]
$$

**Update** — plain stochastic gradient descent, applied after every single sample (batch size 1, no momentum, no adaptive step):

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta
$$

Weights are initialised from a standard normal distribution, biases at zero.

**Metrics logged per epoch**

| Quantity | Definition |
|---|---|
| Loss | $\tfrac{1}{2}\sum_i (\hat{y}_i - y_i)^2$ |
| "Efficiency" | $100 \cdot \max\left(0,\; 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i y_i^2}\right)$ |
| Wall-clock time | cumulative seconds since the start of training |

Training stops early when the efficiency exceeds 99.9%.

Note that the efficiency is an *uncentred* coefficient of determination: the denominator is $\sum_i y_i^2$ rather than $\sum_i (y_i - \bar{y})^2$. It is therefore a normalised residual rather than a textbook $R^2$, and its values are not directly comparable with those reported by standard regression software.

---

## Target signal

All experiments approximate the same function:

$$
f(x) = e^{-x^2/2}\cos(4x)\sin(4x) + 0.3\sin(2x) + 0.1\cos(4x)
$$

It combines a Gaussian-damped high-frequency beat with two undamped harmonics, so it is neither purely periodic nor purely localised — a deliberately awkward case for a fixed harmonic basis.

<!-- VERIFICA: prog_th1.py and animation_th.py sample the domain through Punti.generatore with epsilon = -pi and dom = 2*pi, which yields points in [-pi, 2*pi); prog_th2-4.py use np.linspace(1e-6, 2*pi - 1e-6, N), i.e. [0, 2*pi]. Decide which domain is canonical and align the scripts, otherwise the single-run figures and the parameter sweeps are not describing the same problem. -->

---

## Repository structure

| File | Role |
|---|---|
| `net.py` | Core library. `Punti` generates the sampling grid; `Network` holds both activations, their derivatives, the two forward/backward routines (`NN_cos`, `NN_log`) and the two training loops (`Cosine_Network`, `logistic_Network`). |
| `prog_th1.py` | Single training run of both networks on 1000 points; produces the loss, efficiency and fit-comparison figures. |
| `prog_th2.py` | Sweep over the number of sampled points (10 → 1000), 10 repetitions each. |
| `prog_th3.py` | Sweep over the width of the hidden layer (3 → 600 units), 10 repetitions each. |
| `prog_th4.py` | Sweep over the learning rate ($10^{-5}$ → $10^{-1}$), 10 repetitions each. |
| `load_data.py` | Reads the CSV files produced by the sweeps, averages over repetitions and plots the three comparison figures. |
| `animation_th.py` | Renders the evolution of the cosine network's output across epochs as an animated GIF. |
| `main.py` | Convenience runner that executes the sweeps and then the plotting stage. |
| `LICENSE` | MIT. |

### Fixed settings in the sweeps

| | `prog_th2` | `prog_th3` | `prog_th4` |
|---|---|---|---|
| Swept parameter | number of points | hidden units | learning rate |
| Points | swept | 600 | 600 |
| Hidden units | 128 | swept | 128 |
| Learning rate | $10^{-2}$ | $10^{-2}$ | swept |
| Epochs (cap) | 50 | 50 | 50 |
| Repetitions | 10 | 10 | 10 |

---

## Requirements

```
numpy
matplotlib
```

```bash
pip install numpy matplotlib
```

`animation_th.py` additionally requires **ImageMagick** on the system path in order to write the GIF.

---

## Usage

### Single run and diagnostic figures

```bash
python prog_th1.py
```

Writes `Loss.png`, `Loss_time.png`, `Confronto.png`, `Adapt.png` and `Efficienza.png` to the working directory.

### Parameter sweeps

```bash
python prog_th2.py   # number of points
python prog_th3.py   # hidden-layer width
python prog_th4.py   # learning rate
```

Each script writes six CSV files. The naming convention is `<metric><activation><script-number>.csv`, where the metric is one of `eff`, `loss` or `epo`, and the activation is `cos` or `log`:

| Script | Files |
|---|---|
| `prog_th2.py` | `effcos2.csv`, `efflog2.csv`, `losscos2.csv`, `losslog2.csv`, `epocos2.csv`, `epolog2.csv` |
| `prog_th3.py` | the same, with suffix `3` |
| `prog_th4.py` | the same, with suffix `4` |

Every file is a matrix of shape `(repetitions, number of swept values)`, so a column is a fixed parameter value and a row is one independent run.

### Aggregation and comparison plots

```bash
python load_data.py
```

Averages each CSV over the repetition axis and produces, for each swept parameter, three figures: efficiency (with the relative difference between the two activations in the lower panel), loss and epochs to convergence.

An `IMG/` directory must exist beforehand.

### Animation

```bash
python animation_th.py
```

---

## Known limitations

The code is a thesis artefact rather than a maintained package. The points below are known and documented rather than fixed, so that the results can be read for what they are.

- **No random seed is fixed.** Weight initialisation is drawn from `np.random.randn` without seeding, so individual runs are not reproducible. The sweeps average over 10 repetitions, but the average itself is not reproducible either.
- **Element-wise activations.** `cosine_squasher`, `logistic_function` and their derivatives loop over the array in pure Python. This dominates the runtime and, since wall-clock time is one of the reported quantities, it affects the timing comparison between the two activations.
- **The loss is measured during the epoch, not after it.** Predictions are collected while the weights are still being updated within the same pass, so the reported per-epoch loss is a running quantity rather than an evaluation of the final state of the epoch.
- **No held-out set.** Every metric is computed on the same points used for training; the experiment measures approximation capacity, not generalisation.
- **Hard-coded paths.** `main.py` calls `os.chdir("home/THESIs/")` and `animation_th.py` calls `os.chdir("home")`; `load_data.py` writes with a Windows-style separator (`'IMG\\'`) and appends a second `.png` to filenames that already carry the extension. All three need adjusting before the scripts will run outside their original environment.
- **`main.py` does not run `prog_th1.py`**, so the single-run figures must be produced separately.
- **The animation fits the FFT coefficients.** `animation_th.py` trains against `np.fft.fft(yy)`, which is complex-valued, with a real-valued network; the frame count passed to `FuncAnimation` also exceeds the number of stored epochs.

---

## Possible extensions

Directions that would strengthen the original comparison without changing its nature:

1. **Spectral read-out of the weights.** The central claim is that the cosine network encodes the spectrum of the signal in its parameters. Extracting the learned frequencies from $W^{(1)}$ and comparing them directly against the FFT of the target would test that claim quantitatively, instead of leaving it to visual inspection of the fit.
2. **A vectorised reference implementation.** Replacing the Python loops with `np.where` keeps the "no ML libraries" constraint intact while making the timing comparison meaningful and the sweeps cheap enough to extend.
3. **Mini-batch and full-batch variants**, to separate the effect of the activation from the noise of per-sample updates.
4. **A second, purely periodic target** (a truncated Fourier series with known coefficients), where the cosine squasher should have a clear structural advantage and the recovered coefficients can be checked against ground truth.

---

## References

1. Gallant, A. R. and White, H. (1988). There exists a neural network that does not make avoidable mistakes. *Proceedings of the IEEE International Conference on Neural Networks*.
2. Hornik, K., Stinchcombe, M. and White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.
3. Silvescu, A. (1999). Fourier neural networks. *Proceedings of the International Joint Conference on Neural Networks*.

<!-- VERIFICA: add the bibliographic reference of the thesis itself (title, supervisor, year, and the OpenStarTs handle if it has been deposited), so that the repository can be cited. -->

---

## Licence

Released under the MIT Licence. See [`LICENSE`](LICENSE).

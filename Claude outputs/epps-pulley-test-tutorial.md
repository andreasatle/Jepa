# The Epps–Pulley test, from scratch

The test behind LeJEPA's SIGReg. Original: T. W. Epps and L. B. Pulley, "A test for normality based on the empirical characteristic function", *Biometrika* 70(3), 723–726 (1983). This tutorial explains the idea of that paper in the form LeJEPA uses, and derives every formula so each step can be checked; it is not a summary of the paper's text. Every number below was computed and checked (the closed form against numerical integration, to four decimals).

## 1. The question

You have $n$ plain numbers $x_1,\ldots,x_n$. Could they be a sample from the standard bell curve $\mathcal{N}(0,1)$?

A single number can't answer that; only the collection can. A test turns the collection into one score: small if the numbers look like $\mathcal{N}(0,1)$, large if they don't. In LeJEPA, the numbers are a batch of embeddings projected onto one random direction, and the score is added to the loss.

## 2. The characteristic function: a fingerprint of a distribution

For a random number $X$, the characteristic function is one complex number for every real $t$:

$$
\varphi(t) = \mathbb{E}\!\left[e^{itX}\right] = \mathbb{E}[\cos(tX)] + i\,\mathbb{E}[\sin(tX)].
$$

**A picture that makes it concrete.** $e^{itx}$ is a point on the unit circle, at angle $tx$. So for a given $t$, wrap every sample point around the circle (point $x$ goes to angle $tx$), and $\varphi(t)$ is the centre of mass of all those points on the circle.

- At $t=0$ every point sits at angle 0, so the centre of mass is exactly 1. Always: $\varphi(0)=1$.
- For small $t$ the points spread only a little around the circle, so the centre stays near 1. The wider the distribution, the faster the points spread as $t$ grows, and the faster $\varphi(t)$ shrinks toward 0.
- The real part (cos) responds to how spread out the points are; the imaginary part (sin) responds to lopsidedness (skew): for a distribution symmetric around 0, the sin terms cancel and the imaginary part is 0.

**Three examples:**

| Distribution | $\varphi(t)$ | What the circle picture says |
| --- | --- | --- |
| All points at 0 (collapse) | $1$ for every $t$ | every point stays at angle 0 forever |
| Coin flip, ±1 | $\cos(t)$ | two points rotating in opposite directions; the centre swings between +1 and −1 |
| $\mathcal{N}(0,1)$ | $e^{-t^2/2}$ | spreads smoothly; the centre fades to 0, never comes back |

**It is the Fourier transform.** For a distribution with density $p(x)$,

$$
\varphi(t) = \int_{-\infty}^{\infty} e^{itx}\,p(x)\,\mathrm{d}x,
$$

the same transform used to solve the wave or heat equation, with $x$ as position and $t$ as frequency. Probabilists drop the $2\pi$ and use $e^{+itx}$; otherwise nothing differs. The three examples above are the standard Fourier pairs:

| Distribution | Density $p(x)$ | Fourier transform $\varphi(t)$ |
| --- | --- | --- |
| Collapse at 0 | $\delta(x)$ | $1$ |
| Coin, ±1 | $\tfrac12\delta(x-1)+\tfrac12\delta(x+1)$ | $\cos t$ |
| $\mathcal{N}(0,1)$ | $e^{-x^2/2}/\sqrt{2\pi}$ | $e^{-t^2/2}$ |

The properties that make Fourier useful for PDEs carry over:

| PDE world | Probability world |
| --- | --- |
| Convolution becomes multiplication | Adding independent random numbers convolves their densities, so their characteristic functions multiply: $\varphi_{X+Y}=\varphi_X\,\varphi_Y$ |
| Derivatives become powers of $it$ | Moments are derivatives at 0: $\mathbb{E}[X]=-i\varphi'(0)$, $\mathbb{E}[X^2]=-\varphi''(0)$ |
| The Gaussian is the heat kernel | Adding Gaussian noise of variance $s^2$ runs the heat equation on the density; $\varphi$ is multiplied by $e^{-s^2t^2/2}$ |
| The transform can be inverted | Equal $\varphi$ means equal distribution (below) |

**The scale.** Width $\sigma$ in $x$ becomes width $1/\sigma$ in $t$:

$$
\mathcal{N}(0,\sigma^2) \;\longmapsto\; \varphi(t)=e^{-\sigma^2t^2/2}.
$$

A wide distribution has a narrow $\varphi$ (its points spread around the circle quickly), and collapse ($\sigma\to 0$) is the flat limit $\varphi=1$. No constant in front is needed: the density integrates to 1, so $\varphi(0)=1$ always. (Other Fourier conventions put a $\sqrt{2\pi}$ in front of the transform of $e^{-x^2/2}$; for a probability density it cancels.)

**Why it is a perfect fingerprint.** The Fourier transform can be inverted: two distributions with the same $\varphi$ are the same distribution. This holds for every distribution, not only Gaussians; it is a different fact from "the transform of a Gaussian is a Gaussian", which only tells us what the target curve $e^{-t^2/2}$ is. The test uses both: the second gives the curve to compare against, the first guarantees that a sample whose $\varphi$ matches it for every $t$ is $\mathcal{N}(0,1)$ and nothing else. Moments (mean, variance, skewness, ...) don't have this property: different distributions can share all of their first few moments, as the coin shows next.

### The coin and the Gaussian, derived

Expand the exponential inside the expectation:

$$
\varphi(t) = \mathbb{E}\!\left[e^{itX}\right] = \sum_{m=0}^{\infty}\frac{(it)^m\,\mathbb{E}[X^m]}{m!}.
$$

For a symmetric distribution the odd moments are 0 (the sin terms cancelling), and the even terms alternate in sign:

$$
\varphi(t) = \sum_{k=0}^{\infty}(-1)^k\,\frac{t^{2k}\,\mathbb{E}[X^{2k}]}{(2k)!}.
$$

**The coin.** $X^2=1$ always, so every even moment is 1:

$$
\varphi(t) = 1-\frac{t^2}{2!}+\frac{t^4}{4!}-\frac{t^6}{6!}+\cdots = \cos t.
$$

On the circle: the two points start at angle 0 and rotate in opposite directions, at angles $+t$ and $-t$. Their sin parts cancel and the centre is $\cos t$.

| $t$ | Point at $+t$ | Point at $-t$ | Centre $\varphi(t)$ |
| --- | --- | --- | --- |
| 0 | at 1 | at 1 | 1 |
| $\pi/2$ | top | bottom | 0 |
| $\pi$ | at −1 | at −1 | −1 |
| $3\pi/2$ | bottom | top | 0 |
| $2\pi$ | at 1 | at 1 | 1 |

The two points keep meeting again, so $\varphi$ never fades. Any distribution on a regular grid of points has a periodic $\varphi$ like this; in Fourier terms, two spikes give a cosine, as in two-slit interference.

**The Gaussian.** The even moments of $\mathcal{N}(0,1)$ are $1, 3, 15, 105, \ldots$, in general $\mathbb{E}[X^{2k}]=(2k-1)!!=1\cdot3\cdot5\cdots(2k-1)$ (one integration by parts gives $\mathbb{E}[X^{2k}]=(2k-1)\,\mathbb{E}[X^{2k-2}]$). Splitting the even factors of $(2k)!$ from the odd ones, $(2k)!=2^k\,k!\,(2k-1)!!$, so

$$
\frac{\mathbb{E}[X^{2k}]}{(2k)!}=\frac{1}{2^k\,k!}
\qquad\Longrightarrow\qquad
\varphi(t)=\sum_{k=0}^{\infty}\frac{(-t^2/2)^k}{k!}=e^{-t^2/2}.
$$

A second route: differentiate $\varphi(t)=\mathbb{E}[\cos tX]$ and integrate by parts with $x\,p(x)=-p'(x)$; this gives $\varphi'(t)=-t\,\varphi(t)$, $\varphi(0)=1$, whose solution is $e^{-t^2/2}$.

**Side by side.** The coin has mean 0 and variance 1, exactly like $\mathcal{N}(0,1)$, so the series agree up to $t^2$; they first differ in the fourth moment (1 for the coin, 3 for the Gaussian):

| Term | Coin (moments 1, 1, 1, ...) | Gaussian (moments 1, 3, 15, ...) |
| --- | --- | --- |
| $t^2$ | $-1/2$ | $-1/2$ |
| $t^4$ | $+1/24$ | $+3/24=1/8$ |
| $t^6$ | $-1/720$ | $-15/720=-1/48$ |

The Gaussian's growing moments turn a series that oscillates forever ($\cos$) into one that dies out ($\exp$). A test comparing only mean and variance would pass the coin as Gaussian; the curves separate beyond $t\approx1$, where the weight of section 4 still looks (the coin scores 3.1 in section 7).

## 3. The empirical characteristic function

From the sample, replace the expectation by an average:

$$
\widehat{\varphi}(t) = \frac{1}{n}\sum_{j=1}^{n} e^{itx_j} = \frac{1}{n}\sum_{j=1}^{n}\cos(tx_j) + \frac{i}{n}\sum_{j=1}^{n}\sin(tx_j).
$$

That is the centre of mass of the $n$ actual points on the circle.

**A small example**, $x=(-1,0,0.5,2)$:

| $t$ | Mean $\cos(tx)$ | Mean $\sin(tx)$ | Gaussian $e^{-t^2/2}$ |
| --- | --- | --- | --- |
| 0 | 1.000 | 0.000 | 1.000 |
| 0.5 | 0.847 | 0.152 | 0.882 |
| 1 | 0.500 | 0.137 | 0.607 |
| 2 | 0.118 | −0.206 | 0.135 |

The real part falls a bit faster than the Gaussian's (this sample is a bit wide), and the imaginary part isn't 0 (the sample leans right: 2 sits further out than −1).

**How noisy is the estimate?** Even for a truly Gaussian sample, $\widehat{\varphi}$ wobbles around $\varphi$. The average squared wobble is

$$
\mathbb{E}\!\left[\left|\widehat{\varphi}(t)-\varphi(t)\right|^2\right] = \frac{1-\varphi(t)^2}{n}.
$$

Here, each $e^{itx_j}$ has squared size 1, and $\varphi(t)^2$ of it is the part every point shares. Two things follow: the noise shrinks like $1/n$, and it is largest where $\varphi(t)$ is near 0, i.e. at large $t$. At large $t$, $\widehat{\varphi}$ is mostly noise.

## 4. The test statistic: a weighted distance between two curves

Compare the two curves over all $t$, squaring the gap and weighting it:

$$
\mathrm{EP} = n\int_{-\infty}^{\infty} \left|\widehat{\varphi}(t)-e^{-t^2/2}\right|^2 w(t)\,\mathrm{d}t, \qquad w(t)=e^{-t^2}.
$$

- **The squared gap** $\lvert\widehat{\varphi}-\varphi\rvert^2$ is (real gap)² + (imaginary gap)².
- **The weight** $w(t)=e^{-t^2}$ concentrates the comparison on small $t$, where $\widehat{\varphi}$ is reliable and describes the coarse shape (spread, skew), and fades out large $t$, where $\widehat{\varphi}$ is mostly noise (section 3). Beyond $t\approx 3$ the weight is below 0.0001. Section 6 explains what the weight really is: a smoothing of the sample.
- **The factor $n$** cancels the $1/n$ of the noise, so that for a true Gaussian sample the score sits at the same level whatever the sample size (section 7).

Mathematically, $\mathrm{EP}/n$ is a squared weighted $L^2$ norm, $\lVert\widehat{\varphi}-\varphi\rVert_w^2$: a distance between two functions. That's why it can be read both as "a norm" and as "a test".

## 5. No integral needed: the closed form

Expanding the square and using one standard integral, $\int_{-\infty}^{\infty}\cos(at)e^{-ct^2}\,\mathrm{d}t = \sqrt{\pi/c}\,e^{-a^2/(4c)}$, the integral disappears:

$$
\begin{aligned}
\mathrm{EP} &= n\bigl(A-B+C\bigr), \\[4pt]
A &= \frac{\sqrt{\pi}}{n^2}\sum_{j=1}^{n}\sum_{k=1}^{n}
\exp\!\left(-\frac{(x_j-x_k)^2}{4}\right), \\[4pt]
B &= \frac{2}{n}\sqrt{\frac{2\pi}{3}}\sum_{j=1}^{n}
\exp\!\left(-\frac{x_j^2}{6}\right), \\[4pt]
C &= \sqrt{\frac{\pi}{2}}.
\end{aligned}
$$

How it comes out: $\lvert\widehat{\varphi}-\varphi\rvert^2 = \lvert\widehat{\varphi}\rvert^2 - 2\varphi\,\operatorname{Re}\widehat{\varphi}+\varphi^2$ ($\varphi$ is real here). $\lvert\widehat{\varphi}\rvert^2$ is a double sum of $\cos\!\bigl(t(x_j-x_k)\bigr)$, giving (A); $\varphi\,\operatorname{Re}\widehat{\varphi}$ is a sum of $\cos(tx_j)e^{-t^2/2}$, giving (B); $\varphi^2$ gives (C).

Each part has a meaning:

- **(A) rewards spreading out.** Each pair of points contributes $e^{-\mathrm{distance}^2/4}$: close pairs add a lot, far pairs nearly nothing. Clumps and collapse make (A) large.
- **(B) rewards being near 0, at a bell-curve scale.** Points within the bell's range subtract a lot; points far out subtract little.
- **(C) is a constant.**

So the score balances "don't clump" against "don't drift too far": the balance is exactly a bell curve. (This is the kernel form of the test; the same expression is a maximum mean discrepancy, MMD, with a Gaussian kernel, the bridge to the machine-learning literature.)

**Check, the collapsed sample** (all $x_j=0$): every pair gives $e^0=1$ and every point gives $e^0=1$, so $\mathrm{EP}=n\bigl(\sqrt{\pi}-2\sqrt{2\pi/3}+\sqrt{\pi/2}\bigr)\approx 0.131n$. For 64 points that is 8.41, for 128 points 16.8: collapse is punished, and more the larger the batch.

## 6. The weight is a smoothing kernel

**The weight smooths the sample.** The weight is the squared Fourier transform of a standard Gaussian bump $g$:

$$
w(t)=e^{-t^2}=\left(e^{-t^2/2}\right)^2=\lvert\widehat{g}(t)\rvert^2,\qquad g=\text{the }\mathcal{N}(0,1)\text{ density}.
$$

Multiplying by $\widehat{g}$ in $t$ is convolving with $g$ in $x$, and Parseval's theorem (the Fourier transform preserves $L^2$ distances, up to $2\pi$) turns the statistic into a plain $L^2$ distance between two smoothed curves:

$$
\frac{\mathrm{EP}}{n} = 2\pi\int_{-\infty}^{\infty}\Bigl(\underbrace{\tfrac{1}{n}\textstyle\sum_j g(x-x_j)}_{\text{sample, smoothed}}-\underbrace{\mathcal{N}(x;\,0,2)}_{\text{bell, smoothed}}\Bigr)^2\mathrm{d}x.
$$

- **The sample, smoothed:** an $\mathcal{N}(0,1)$ bump on every sample point, averaged (a kernel density estimate).
- **The bell, smoothed:** $\mathcal{N}(0,1)$ convolved with $\mathcal{N}(0,1)$ is $\mathcal{N}(0,2)$.
- **The bump has width 1.** The width $\sqrt2$ in the pair term (A) of section 5, $e^{-(x_j-x_k)^2/4}$, is $g$ convolved with itself: how much two points' bumps overlap. Overlapping bumps (a clump) push the score up, the "don't clump" force.

(Checked numerically: for 50 points, the closed form and this integral agree to ten digits.)

**Why there must be a weight: $w=1$ fails.** A sample is $n$ spikes, and spikes never fade in frequency: at large $t$, $\widehat{\varphi}$ is $n$ points spinning around the circle at unrelated speeds, with squared size about $1/n$ (section 3's noise with $\varphi\approx0$). Integrating that over all $t$ gives infinity, for every sample, even a perfectly Gaussian one. In the kernel view, $w=1$ means a delta-spike kernel, no smoothing: a set of spikes can never match a smooth bell. Cutting the integral off at $t=5$, as the code does, keeps it finite but still bad: a true Gaussian would score about 8.2 on average instead of 0.52, almost all of it high-$t$ noise, and the kernel becomes $\sin(5d)/d$, which changes sign, so points at some distances would attract instead of repel.

**Changing the weight changes the ruler, not the zero.** The bump's transform $e^{-t^2/2}$ is never zero, so smoothing is invertible: two distributions that differ still differ after smoothing. The distance is 0 for exactly $\mathcal{N}(0,1)$ and nothing else, whatever the weight. With infinite data any positive weight detects every departure. With finite data the weight decides which departures count most, i.e. how many samples it takes to see each one. A weight $e^{-at^2}$ (ours is $a=1$) has pair kernel $e^{-(x_j-x_k)^2/(4a)}$:

| Weight | Frequencies compared | Sees quickly | Sees slowly | As a loss |
| --- | --- | --- | --- | --- |
| Narrow in $t$ (large $a$, wide bumps) | low only | mean, spread | shape, tails, clusters | smooth, low noise; points repel over long distances |
| $e^{-t^2}$ ($a=1$) | up to $t\approx2$ | spread, skew, fourth moment (the coin) | heavy tails | |
| Wide in $t$ (small $a$, narrow bumps) | high too | fine shape, tails, clumps | | noisy (high $t$ is mostly noise); points repel only close neighbours |

No weight is fastest for every departure, so $e^{-t^2}$ is a compromise, not an optimum. Its **shape** is a convention (a Gaussian gives the closed form); its **scale** is not arbitrary: it matches the target, whose features live at scale 1 ($t\approx1$, bump width 1). A weight a hundred times narrower or wider would still work in the limit, but would be blind or noisy in practice. Because the target $\mathcal{N}(0,1)$ is fixed, the choice is made once, which is why the weight is a constant in LeJEPA's code and not a tuned hyperparameter. For LeJEPA it matters even less than for a test: what the loss most needs to catch, collapse and a wrong spread, sits at low $t$, which every reasonable weight sees.

## 7. Reading the score

**If the sample really is $\mathcal{N}(0,1)$**, the average score is the same for every $n$:

$$
\mathbb{E}[\mathrm{EP}] = \int_{-\infty}^{\infty}\left(1-e^{-t^2}\right)e^{-t^2}\,\mathrm{d}t = \sqrt{\pi}-\sqrt{\frac{\pi}{2}} \approx 0.519.
$$

This is the noise formula of section 3, multiplied by $n$ and integrated against the weight. Simulated, 300 samples each:

| $n$ | Mean score | 95% of scores are below |
| --- | --- | --- |
| 16 | 0.53 | 1.54 |
| 128 | 0.52 | 1.59 |
| 1024 | 0.51 | 1.66 |

**If it isn't**, the gap $\widehat{\varphi}-\varphi$ doesn't shrink as $n$ grows, so the score grows proportionally to $n$. Examples with $n=128$:

| Sample | Score |
| --- | --- |
| Gaussian $\mathcal{N}(0,1)$ | ~0.5 (one draw: 0.02; it varies) |
| Coin flip, ±1 | 3.1 |
| Too wide, $\mathcal{N}(0,4)$ | 16.3 |
| Collapsed to 0 | 16.8 |

**As a test** (the 1983 use): reject "the sample is $\mathcal{N}(0,1)$" if the score exceeds the 95% line, about 1.6. With enough data it detects *any* departure from $\mathcal{N}(0,1)$ (statisticians call that consistent), because the characteristic function is a complete fingerprint and the weight is positive for every $t$.

## 8. From the 1983 test to LeJEPA's SIGReg

The idea is the same; the use differs in four ways.

1. **No standardisation.** The classic test first rescales the sample to mean 0 and standard deviation 1, because it asks "is this *some* normal distribution?", whatever the mean and spread. SIGReg tests against exactly $\mathcal{N}(0,1)$, unstandardised. That is what makes it fight collapse: a collapsed sample has spread 0, and only an unstandardised test sees that (section 5's 16.8). Standardising first would hide it.
2. **The weight's width.** The weight's width is a free choice (it decides how much the test cares about coarse versus fine shape, section 6); LeJEPA uses $e^{-t^2}$.
3. **One direction at a time.** SIGReg applies the test to the batch projected onto many random directions and averages the scores (sliced). By the Cramér–Wold theorem, a distribution in many dimensions is Gaussian if every one-dimensional projection is.
4. **A loss, not a verdict.** Nobody rejects anything: the score itself is minimised by gradient descent. The test's properties become a loss's properties: the score depends smoothly on every $x_j$ (through $\cos$, $\sin$ and $e^{(\cdots)}$), and the gradient stays bounded however far out a point is, so a single outlier can't blow up training.

In LeJEPA's code the integral is computed with the trapezoid rule at 17 points of $t$ instead of the closed form; for this smooth integrand both agree (9 points already do).

## 9. Try it yourself

```python
import math, torch

def epps_pulley(x):                       # closed form, section 5
    n = len(x)
    d = x[:, None] - x[None, :]
    a = math.sqrt(math.pi) / n**2 * torch.exp(-d**2 / 4).sum()
    b = 2 * math.sqrt(2 * math.pi / 3) / n * torch.exp(-x**2 / 6).sum()
    return n * (a - b + math.sqrt(math.pi / 2))

print(epps_pulley(torch.randn(128)))          # ~0.5 on average
print(epps_pulley(torch.zeros(128)))          # 16.8: collapse
print(epps_pulley(torch.randn(128) * 2))      # too wide
print(epps_pulley(torch.randn(128) + 1))      # shifted
```

Things worth trying: a shifted sample (`+ 1`) and one that is slightly too narrow (`* 0.5`); how the score of a non-Gaussian sample grows with $n$ while a Gaussian's doesn't; heavy tails (`torch.distributions.StudentT(3).sample((128,))`), which this weight detects only weakly, because tails live at large $t$, where the weight is small.

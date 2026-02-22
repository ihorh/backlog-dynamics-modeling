# From Random Walk to Inverse Gaussian: A Diffusion Approximation

> 💡 **Hypothesis:** Under a few simplifying assumptions, project duration can be reasonably approximated
> by an inverse Gaussian distribution 📈. The following derivation shows why this makes
> sense and motivates its use.


## 1. Discrete Backlog Dynamics

We begin with the sprint-level model:

$$
B_{i+1} = B_i - X_i
$$

where:

- $B_i$ is the backlog after sprint $i$,
- $X_i = v_i - d_i$ is the net backlog reduction in sprint $i$,
- $v_i$ is completed work,
- $d_i$ is added scope or re-estimation.

Assume that:

1. $X_i$ are independent and identically distributed (i.i.d.),
2. $\mathbb{E}[X_i] = \mu > 0$,
3. $\mathrm{Var}(X_i) = \sigma^2 < \infty$.

Under these assumptions, $B_i$ is a **discrete-time Markov chain** and, more specifically,
a **random walk with drift**.

Project completion occurs at the stopping time

$$
T = \inf \{ i \ge 0 : B_i \le 0 \}.
$$

This is a **first-passage time problem** for a random walk.

---

## 2. Continuous Approximation

The discrete first-passage time distribution of a random walk does not have a simple closed
form. To obtain analytical structure, we introduce a **diffusion approximation**.

This requires two modeling steps.


### 2.1. Assumption 1: Time Rescaling

Let each sprint correspond to a time increment $\Delta t$.

We embed the discrete process into continuous time by writing increments as:

$$
B(t + \Delta t) - B(t) = -X_i.
$$

This step is a mathematical idealization. Sprints are discrete and finite, and we
do not claim the process is truly continuous. We introduce a continuous limit as
an analytical tool and later check whether it approximates the discrete model
sufficiently well.


### 2.2. Assumption 2: Diffusion Scaling of Variance

To make the math easier, we treat each sprint as contributing relatively small,
continuous change to the backlog:

$$
X_i = \mu \Delta t + \sigma \sqrt{\Delta t}\,\varepsilon_i,
\quad \varepsilon_i \sim \mathcal{N}(0,1),
$$

where $\mu$ and $\sigma^2$ come from the mean and variance of the actual discrete net sprint progress
(available empirical data).  

Here we assume that the net sprint progress (net backlog delta) is normally distributed. This is purely
a modeling decision, which lets us treat the discrete backlog changes as a continuous process and study
completion times using standard tools from stochastic calculus.  

The $\sqrt{\Delta t}$ factor makes sure that the variability grows with time in a reasonable way.
Over a project of length $t$, the total variance ends up as

$$
\sigma^2 t,
$$

so fluctuations stay proportional as we shrink $\Delta t$ and move toward the continuous limit.


### 2.3. The Diffusion Limit

Under these assumptions, the rescaled random walk converges (in distribution) to

$$
dB_t = -\mu\, dt + \sigma\, dW_t.
$$

Here:

- $W_t$ is standard Brownian motion (Wiener process, approximated in discrete time as $\sqrt{\Delta t}\,\varepsilon$),
- $-\mu dt$ represents deterministic average burn,
- $\sigma dW_t$ represents continuous Gaussian fluctuations.

This is the continuous-time approximation of the discrete backlog process.

---

## 3. Completion as First-Passage Time

Project completion occurs when backlog first reaches zero:

$$
T = \inf \{ t > 0 : B_t \le 0 \}.
$$

For Brownian motion with constant drift starting at $B_0 > 0$, the distribution of this first-passage time is known in closed form.

> :small[ $\inf\{...\}$ - standard notation for first hitting time (infimum). ]

It follows an **inverse Gaussian distribution**:

$$
T \sim \mathrm{IG}\!\left(
\text{mean} = \frac{B_0}{\mu},
\quad
\text{shape} = \frac{B_0^2}{\sigma^2}
\right).
$$

---

## 4. Interpretation

- The expected duration is $\mathbb{E}[T] = B_0 / \mu$.
- Variability increases with sprint volatility $\sigma^2$.
- The distribution is right-skewed: unlucky sequences of weak sprints push completion further out.

---

## 5. What We Have So Far

With a few simplifying assumptions and modeling choices, we derived a closed-form solution. 😎 

The inverse Gaussian distribution is a candidate for describing project duration, capturing expected
completion time and right-skewed uncertainty, though we’ll test its accuracy in the next article.
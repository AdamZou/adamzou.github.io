# Bayesian meta-learning

This is my first post


In the following discussion, we try to:

1.  use probability density functions for all Lebesgue integrals,
    without loss of generality.

2.  use capital letters for random variables.

3.  use short notations like $p(x)$ and $p(y \vert x)$ for $p(X = x)$
    and $p(Y = y \vert X = x)$, respectively.

# Preliminary {#sect: preliminary}

In this section, we provide a general framework for meta-learning. We
first introduce multi-task setting as the typical data structure for
meta-learning problems, then formulate components and objective of
meta-learning. Lastly, we introduce a more general concept:
hyper-learning (a word that I made up), to provide an interesting
perspective on meta-learning.

## Multi-Task Setting

In multi-task setting, data generator consists of two hierarchical
parts: meta-generator and task-generator:

-   Meta-generator samples task $T \in \mathcal{T}$ from distribution
    $p_\ast(t)$.

-   Given task $T = t$, task-generator samples task-specific data
    $D_t \in \mathcal{D}$ from distribution $p_\ast(d \vert t)$.

Here we use subscript $\ast$ for data generator (ground truth).

::: {.remark}
**Remark 1** (Marginal Distribution). *The distribution of data from a
randomly sampled task is given by
$$p_\ast(d) = \int_\mathcal{T} p_\ast(d \vert t)\, p_\ast(t)\, dt$$*
:::

In multi-task problems, the discussion is always around task-learners.
We denote:

-   $J_t \in \mathcal{J}$ as task-specific decision rule (for task $t$).

-   $L_t: \mathcal{J} \times \mathcal{D} \mapsto \mathbb{R}$ as
    task-specific loss function, to evaluate decision rule when applied
    on data.

-   $\operatorname{TL}: \mathcal{D} \mapsto \mathcal{J}$ as
    task-learner. It trains decision rule from task-specific training
    data: $J_t = \operatorname{TL}(D_t^{\textrm{train}})$.

::: {.example}
**Example 1** (Supervised-Learning Tasks). *It can be formulated as
follows:*

-   *$D = (X, Y)$, where $X \in \mathcal{X}$ and $Y \in \mathcal{Y}$.*

-   *$J_t: \mathcal{X} \mapsto \mathcal{Y}$ is a mapping to be learned
    for task $t$.*

-   *$L_t(J_t, D) \equiv L(J_t, D) = \lVert Y - J_t(X) \Vert_2^2$, if we
    focus on mean squred error.*

-   *If we use linear regression as task-learner, then
    $\operatorname{TL}(D_t^{\textrm{train}}) = f_t$, where
    $$f_t(X) = X\beta_t = X \left[(X^T X)^{-1} X^T Y \Big\vert_{D = D_t^{\textrm{train}}} \right]$$*
:::

## Meta-Learning

Meta-learning is a type of problem that builds on multi-task setting,
the **objective of meta-learning** is "learning to learn", which can be
formulated as risk (or generalization error) minimization:
$$\label{eq: risk minimization}
    \min_{\operatorname{TL}}\, \mathbb{E}_\ast L\left(\operatorname{TL}(D_T^{\textrm{train}}), D_T^{\textrm{test}}\right)$$
The interpretation is, we seek a task-learner $\operatorname{TL}$, such
that when applied on a newly sampled task $T$ (trained on
$D_T^{\textrm{train}}$ then evaluated on $D_T^{\textrm{train}}$, where
$(D_T^{\textrm{train}}, D_T^{\textrm{test}}) \sim_{iid} p_\ast(d \vert T)$),
the expected loss is minimal.

Similar to training and testing in supervised learning, there are two
phases in meta-learning:

#### Meta-Training

This is the phase where $\operatorname{TL}$ trained, using what we
called meta-training data.

#### Meta-Testing

This is when the risk in
Eq [\[eq: risk minimization\]](#eq: risk minimization){reference-type="eqref"
reference="eq: risk minimization"} is estimated, using sample mean on
meta-testing data set.

::: {.remark}
**Remark 2**. *Meta-training and meta-testing can happen in alternating
form. For example, in incremental algorithms where tasks of data are fed
to meta-learner in mini-batches, data can be meta-tested before
meta-trained within each iteration, so that the loss curve is monitored
while training.*
:::

::: {.remark}
**Remark 3**. *Task learner $\operatorname{TL}$ can have many forms. For
example, if $J$ is a neural network parameterized by $w$, then
$\operatorname{TL}$ can contain information such as "$w$ is distributed
on certain manifold" or "$\alpha=0.1$ is the best learning rate",
providing hyper-parameter (in statistical and/or machine learning sense)
for training $J$.*
:::

## Hyper-Learning: A More General Framework

From
Eq [\[eq: risk minimization\]](#eq: risk minimization){reference-type="eqref"
reference="eq: risk minimization"} we can see that, meta-learning has
similar formulation as supervised learning. In fact, meta-learning is
supervised learning on a higher level:

::: {#tab: supervised learning vs meta-learning}
                     Supervised Learning                                              Meta-Learning
  --------- -------------------------------------- -----------------------------------------------------------------------------------
  Mapping    $f: \mathcal{X} \mapsto \mathcal{Y}$                 $\operatorname{TL}: \mathcal{D} \mapsto \mathcal{J}$
  Loss         $L(\widehat{Y}, Y) = L(f(X), Y)$     $L(\widehat{J}, D) = L(\operatorname{TL}(D^{\textrm{train}}), D^{\textrm{test}})$

  : Comparison between supervised learning and meta-learning.
:::

::: {.remark}
**Remark 4**. *One may find the bottom right cell inconsistent since
it's $L(\widehat{J}, D)$, not $L(\widehat{J}, J)$. However, for any data
$D$ there is a corresponding "empirical" decision rule $J$. For example,
in supervised learning $D_n = \left\{(X_i, Y_i)\right\}_{i=1}^n$, the
corresponding decision rule is
$J_n(x) = \sum_{i=1}^n Y_i\, \delta(x - X_i)$, or more rigorously
allowing non-deterministic decision rule
$P(J_n(x) \le y) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}_{Y_i \le y}\, \delta(x - X_i)$.*
:::

However, this doesn't mean meta-learning is simple aggregation of two
levels of supervised learning. Complication comes from the entangled
multi-level structure, just as training a mixed model is not simply
summarizing over stratified models.

From this modular point of view, we can conjecture the existence of a
more general type of machine learning problems. If we consider
meta-learning as supervised learning on top of machine learning
problems, then we may think about other machine learning problems as the
higher-level component. For instance:

#### Unsupervised Learning as Second Level

For example, we may do clustering on different tasks, and see which
tasks are similar. A potential application is to guide us design better
learners: say we found that task A and task B are very close to each
other, then we may expect good performance if we use SOTA architect of
task A on task B.

#### Reinforcement Learning as Second Level

My imagination is limited. I'm not sure what it can do.

We give the name **hyper-learning** to this family of multi-level
machine learning framework ("learning from learning"). Question: can we
have more than two levels, and what does it mean? \[Doge Head\]

[THIS IS CACHE AREA FOR MR. ZOU'S PAPER]{style="color: blue"}

# Derive $\nabla_{\Theta} L(\Theta; D)$ for linear model, assuming Gaussian

Let $\theta \in \mathbb{R}^m$ be task-specific parameters, and
$\Theta = (\mu \in \mathbb{R}^m, \Sigma \in \mathbb{R}^{m \times m})$ be
meta-parameters/hyper-parameters. Noise level is controlled by
$\sigma \in \mathbb{R}_+$. The hierarchical model is formulated as
$$\begin{aligned}
    Y_i \vert X_i, \theta, \sigma &\sim \mathcal{N}(X_i^T \theta, \sigma^2), \quad i= 1, \ldots, n\\
    \theta \vert \Theta &\sim \mathcal{N}(\mu, \Sigma)\end{aligned}$$ ,
that is, $$\begin{aligned}
    \log p(Y_i \vert X_i, \theta, \sigma) &= - \frac{1}{2} \left[\log (2\pi) + \log\lvert\sigma^2\rvert + \frac{1}{\sigma^2}(Y_i - X_i^T \theta)^2\right]\\
    \log p(\theta \vert \Theta) &= - \frac{1}{2}\left[m \log (2\pi) + \log\lvert\Sigma\rvert + (\theta - \mu)^T \Sigma^{-1} (\theta - \mu) \right]\label{eq: logp hyper}\end{aligned}$$
We assume $p(X)$ does not depend on $\theta$ or $\Theta$. In the
following calculation, we focus on data from single task. Because tasks
are independent from each other, we can sum over log-likelihoods of
multiple tasks later.

## Direct Calculation

$$\begin{aligned}
    & L(\Theta, \sigma^2; D)\\
    =& \log p(D \vert \Theta, \sigma^2)\\
    =& \log \int p(D \vert \theta, \sigma^2) p(\theta \vert \Theta) d\theta\\
    =& \log \int \left(\prod_{i=1}^n p(Y_i \vert X_i, \theta, \sigma^2) p(X_i)\right) p(\theta \vert \Theta) d\theta\\
    =& C_1(X) - n \log \sigma - \frac{1}{2} \log\lvert\Sigma\rvert + \log \int \exp\left[- \frac{1}{2 \sigma^2} \lVert Y - X \theta\rVert_2^2 - \frac{1}{2} (\theta - \mu)^T \Sigma^{-1} (\theta - \mu)\right] d\theta\\
    =& C_2(X, Y, \sigma) - \frac{1}{2} \log\lvert\Sigma\rvert + \log \int \exp\left[- \frac{1}{2} \left( \theta^T A^{-1} \theta - 2 b^T A^{-1} \theta + \mu^T \Sigma^{-1} \mu \right) \right] d\theta\\
    =& C_3(X, Y, \sigma) - \frac{1}{2} \log\lvert\Sigma\rvert + \frac{1}{2} \log\lvert A \rvert - \frac{1}{2}\left(\mu^T \Sigma^{-1} \mu - b^T A^{-1} b\right)\\
    =& C_3(X, Y, \sigma) - \frac{1}{2} \log\left\lvert I_m + \frac{1}{\sigma^2} \Sigma X^T X\right\rvert \\
    &+ \frac{1}{2\sigma^2}\left[ 2 Y^T X \mu + \frac{1}{\sigma^2} Y^T X \Sigma X^T Y  - \left(\mu + \frac{1}{\sigma^2} \Sigma X^T Y\right)^T X^T \left(I_n + \frac{1}{\sigma^2} X \Sigma X^T\right)^{-1} X \left(\mu + \frac{1}{\sigma^2} \Sigma X^T Y\right)\right]\end{aligned}$$
where $$\begin{aligned}
    A^{-1} &= \Sigma^{-1} + \frac{1}{\sigma^2} X^T X\\
    A^{-1} b &= \Sigma^{-1} \mu + \frac{1}{\sigma^2} X^T Y\\
    A &= \Sigma - \Sigma X^T \left(\sigma^2 I_n + X \Sigma X^T\right)^{-1} X \Sigma\end{aligned}$$
and $X \in \mathbb{R}^{n \times m}$ $Y \in \mathbb{R}^{n \times 1}$ are
sample data stacked together.

## Using Corollary 1

$$\begin{aligned}
    \nabla_{\Theta} L(\Theta, \sigma; D) &= \nabla_{\Theta} \log p(D \vert \Theta, \sigma)\\
    &= \frac{1}{p(D \vert \Theta, \sigma)} \nabla_{\Theta} \int p(D \vert \theta, \sigma) p(\theta \vert \Theta) d\theta\\
    &= \frac{1}{p(D \vert \Theta, \sigma)} \int p(D \vert \theta, \sigma) \nabla_{\Theta} p(\theta \vert \Theta) d\theta\\
    &= \frac{1}{p(D \vert \Theta, \sigma)} \int p(D \vert \theta, \sigma) p(\theta \vert \Theta) \nabla_{\Theta} \log p(\theta \vert \Theta) d\theta\\
    &= \int p(\theta \vert D, \Theta, \sigma) \nabla_{\Theta} \log p(\theta \vert \Theta) d\theta\end{aligned}$$
Now, from
Eq [\[eq: logp hyper\]](#eq: logp hyper){reference-type="eqref"
reference="eq: logp hyper"} we have $$\begin{aligned}
    d\log p(\theta \vert \Theta) &= \frac{1}{2} \Big\langle \Sigma - (\mu - \theta)(\mu - \theta)^T,\, d\Sigma^{-1}\Big\rangle - (\mu - \theta)^T \Sigma^{-1} d\mu\\
    \nabla_{\mu}\log p(\theta \vert \Theta) &= - \Sigma^{-1} (\mu - \theta)\\
    \nabla_{\Sigma^{-1}}\log p(\theta \vert \Theta) &= \frac{1}{2} \left(\Sigma - (\mu - \theta)(\mu - \theta)^T\right)\end{aligned}$$
On the other hand,
$$\theta \vert D, \Theta, \sigma^2 \sim \mathcal{N}(b, A)$$ Therefore
$$\begin{aligned}
    \nabla_{\mu}L(\Theta, \sigma; D) &= - \Sigma^{-1} (\mu - b)\\
    \nabla_{\Sigma^{-1}}L(\Theta, \sigma; D) &= \frac{1}{2} \left(\Sigma - A - (\mu - b)(\mu - b)^T\right) \label{28}\\
    \nabla_{\Sigma}L(\Theta, \sigma; D) &= - \Sigma^{-1} \left[\nabla_{\Sigma^{-1}}L(\Theta, \sigma; D)\right] \Sigma^{-1}\\
    &= \frac{1}{2}\left( - \Sigma^{-1} + \Sigma^{-1} A \Sigma^{-1} + \left(\nabla_{\mu}L(\Theta, \sigma; D)\right) \left(\nabla_{\mu}L(\Theta, \sigma; D)\right)^T \right)\end{aligned}$$
For data from multiple tasks and setting gradients to zero, we have MLE
of $\Theta$ satisfying $$\label{mle}
    \begin{cases}
        \mu = \overline{b}\\
        \Sigma = \overline{A} + \overline{(b - \mu)(b - \mu)^T}
    \end{cases}$$ where $\overline{\cdot}$ is (weighted) average over
all tasks.

## asymptotic performance of EM

According to equation ([\[mle\]](#mle){reference-type="ref"
reference="mle"}), in iteration $k$ $$\label{em}
    \begin{cases}
        \mu^{[k+1]} = \overline{b_{\tau}^{[k]}}\\
        \Sigma^{[k+1]} = \overline{A_{\tau}^{[k]}} + \overline{(b_{\tau}^{[k]} - \mu^{[k]})(b_{\tau}^{[k]} - \mu^{[k]})^T}
    \end{cases}$$ where $\tau \in \mathcal{T}$, $A$ and $b$ are given in
(15), (16). Consider the diagonal case where $\Sigma$ and
$\frac{EX^T X}{\sigma^2}  = C$ are both diagonal,
$EX^T Y = EX^T (X \theta_\tau + \hat{z}) = C \theta_\tau$ we have
$$\begin{cases}
        b_{\tau}^{[k]} = \frac{{\Sigma^{[k]}}^{-1} }{{\Sigma^{[k]}}^{-1} + C} \mu^{[k]} + 
         \frac{C }{{\Sigma^{[k]}}^{-1} + C} \theta_\tau \\
        
        A_{\tau}^{[k]} =\frac{1}{{\Sigma^{[k]}}^{-1} +  C}
    \end{cases}$$ Substitute them into
[\[em\]](#em){reference-type="ref" reference="em"} we have $$\label{em1}
    \begin{cases}
        \mu^{[k+1]} - \mu^{[k]}= b_{\tau}^{[k]} - \mu^{[k]}  =  \frac{C }{{\Sigma^{[k]}}^{-1} + C}  (\overline{\theta_\tau} - \mu^{[k]})\\
        \Sigma^{[k+1]} = \frac{1}{{\Sigma^{[k]}}^{-1} +  C} + (\frac{C }{{\Sigma^{[k]}}^{-1} + C})^2 [\overline{(\theta_{\tau} - \mu^{[k]})(\theta_{\tau} - \mu^{[k]})^T}  ]
    \end{cases}$$ From [\[em1\]](#em1){reference-type="ref"
reference="em1"} we can conclude that as $k \rightarrow \infty$,
$\mu^{[k]} \rightarrow \overline{\theta_\tau}$ and thus
$\overline{(\theta_{\tau} - \mu^{[k]})(\theta_{\tau} - \mu^{[k]})^T} \rightarrow S^2_T(\theta_\tau)$,
$\Sigma^{[k]} \rightarrow S^2_T(\theta_\tau) - \frac{1}{C}$. Notice that
$C=EX^T X \propto n$ where $n$ is the number of data per task.\
\
If we consider EM-MLE in which we set $A$ to 0, then we have
$$\label{em-mle}
    \begin{cases}
        \mu^{[k+1]} - \mu^{[k]} = \frac{C }{{\Sigma^{[k]}}^{-1} + C}  (\overline{\theta_\tau} - \mu^{[k]})\\
        \Sigma^{[k+1]} = (\frac{C }{{\Sigma^{[k]}}^{-1} + C})^2 [\overline{(\theta_{\tau} - \mu^{[k]})(\theta_{\tau} - \mu^{[k]})^T}  ]
    \end{cases}$$ $\mu^{[k]} \rightarrow \overline{\theta_\tau}$ still
holds. When $C$ is large we approximately have
$\Sigma^{[k]} \rightarrow S^2_T(\theta_\tau) - \frac{2}{C}$.

The reason of underestimation of $\Sigma$ is that when $n$ is finite,
the posterior mean $b_\tau$ tend to be closer to $\mu$ than true task
mean $\theta_\tau$ so the second term of equation
([\[em\]](#em){reference-type="ref" reference="em"}) tend to be
underestimated. But this underestimation is somehow mitigated by the
first term posterior variance $A_\tau$. EM-MLE sets $A_\tau=0$ and thus
leads to more underestimation of $\Sigma$.\
\
However, we would prefer underestimation rather than overestimation in
our algorithm. First, according to equation
([\[28\]](#28){reference-type="ref" reference="28"}) $A_\tau < \Sigma$
is necessary for GEM to converge. This makes sense because the posterior
should always have less entropy than the prior. But this only holds when
there's only one local minimum in loss function in which case the
posterior converges to Dirac distribution given infinite data. In the NN
case where multiply local minimums exist, $A_\tau > \Sigma$ can easily
happen, making the training unstable. In fact, assume there are a set of
true parameters in each task $\{ \theta^{i}_\tau\}, i \in \tau$, and we
can choose one of them for each task to form a task parameter
distribution $P(\mathcal{T})$ then we have a set of this kind of
distributions $P(\mathcal{T}) \in \mathcal{P}$. We actually want to find
the one with the smallest possible variance among them. One way to
assure this is to restrict $\Sigma$ to a small value in the first phase
of training and let $\mu$ converge first, then train on $\Sigma$ in the
second phase. In anyway, underestimation of $\Sigma$ comes with a lot of
benefits and doesn't hurt much if the posterior mean works almost as
good as the true mean.\
\
Next step:\
\* Asymptotic behavior of GEM-MLE using (27) - (30) and set $A_\tau=0$.\
\* Variance behavior(Asymptotic behavior of number of tasks)

## Variational Inference

Instead of computing task-parameter
$\theta \vert \Theta, \sigma^2, D \sim \mathbal{N}(b, A)$, some
algorithms use the variational inference approach, approximating by some
$\theta \sim \mathcal{N}(u, S)$ (denoted by $q(\theta; u, S)$) with
diagonal matrix $S$. Define

$$\begin{aligned}
    \operatorname{ELBO}(q, \Theta, \sigma^2) &= L(\Theta, \sigma^2; D) - D_{\textrm{KL}}\left( q(\theta; u, S)\, \Vert\, p(\theta \vert \Theta, \sigma^2, D) \right)\label{eq: F1}\\
    &= \mathbb{E}_{\theta \sim q(\theta; u, S)} \left[ \log p(D, \theta \vert \Theta, \sigma^2) - \log q(\theta; u, S) \right]\label{eq: F2}\\
    &= \mathbb{E}_{\theta \sim q(\theta; u, S)} \left[ \log p(D \vert \theta, \sigma^2) \right] - D_{\textrm{KL}}\left( q(\theta; u, S)\, \Vert\, p(\theta \vert \Theta) \right)\label{eq: F3}\end{aligned}$$

::: {.remark}
**Remark 5**. *EM algorithm is equivalent to iterative update of
$q(\theta; u, S)$ and $(\Theta, \sigma^2)$ to optimize
$\operatorname{ELBO}$:*

-   *E-step: update $q(\theta; u, S)$ to decrease KL-divergence in
    Eq [\[eq: F1\]](#eq: F1){reference-type="eqref"
    reference="eq: F1"}.*

-   *M-step: update $(\Theta, \sigma^2)$ to increase expected complete
    log-likelihood in Eq [\[eq: F2\]](#eq: F2){reference-type="eqref"
    reference="eq: F2"}.*
:::

::: {.remark}
**Remark 6**.
*$\operatorname{ELBO}(q, \Theta, \sigma^2) \le L(\Theta, \sigma^2; D)$
provides a lower bound for the log-likelihood function.*
:::

Since $p(\theta \vert \Theta, \sigma^2, D)$ generally does not have
closed form, alternatively we can optimize over
Eq [\[eq: F2\]](#eq: F2){reference-type="eqref" reference="eq: F2"} or
Eq [\[eq: F3\]](#eq: F3){reference-type="eqref" reference="eq: F3"}. In
the linear modeling case, $$\begin{aligned}
    &\operatorname{ELBO}(q, \Theta, \sigma^2)\\
    =& C_1(\sigma^2) + \frac{1}{2} \mathbb{E}_{\theta \sim q}  \left( - \frac{1}{\sigma^2} \lVert Y - X \theta \rVert_2^2 - \lVert \theta - \mu \Vert_{\Sigma^{-1}}^2 + \lVert \theta - u \Vert_{S^{-1}}^2 \right) + \frac{1}{2}\left( \log\lvert S \rvert -  \log\lvert \Sigma \rvert\right)\\
    =& C_2(\sigma^2) - \frac{1}{2} \left[ \frac{1}{\sigma^2} \left(\lVert Xu - Y \Vert_2^2 + \lVert u - \mu \Vert_{\Sigma^{-1}}^2\right) + \left\langle S, \, \frac{1}{\sigma^2} X^T X + \Sigma^{-1} \right\rangle - \log\lvert S \rvert + \log\lvert \Sigma \rvert\right]\end{aligned}$$
Therefore $$\begin{aligned}
    \nabla_{u} \operatorname{ELBO}(q, \Theta, \sigma^2) &= - \left[\frac{1}{\sigma^2}X^T (Xu - Y) + \Sigma^{-1} (u - \mu)\right] = - A^{-1} (u - b)\\
    \nabla_{S} \operatorname{ELBO}(q, \Theta, \sigma^2) &= - \frac{1}{2}\left(\frac{1}{\sigma^2} X^T X + \Sigma^{-1} - S^{-1}\right) = \frac{1}{2}\left( S^{-1} - A^{-1} \right)\\
    \nabla_{\mu} \operatorname{ELBO}(q, \Theta, \sigma^2) &= \Sigma^{-1} (u - \mu)\\
    \nabla_{\Sigma^{-1}} \operatorname{ELBO}(q, \Theta, \sigma^2) &= -\frac{1}{2} \left[ (u - \mu) (u - \mu)^T  + S - \Sigma\right]\end{aligned}$$

## Gradient Comparison

# General Properties

## Consistency of L2

Let $D_T = (D_T^{\textrm{tr}}, D_T^{\textrm{val}})$ with
$D_T^{\textrm{tr}} \in \mathcal{D}^k$,
$D_T^{\textrm{val}} \in \mathcal{D}^{m - k}$. By definition,
$$L_2(\Theta; D_T) = \log p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)$$
For a sample containing $n$ tasks:
$$L_2^{(n)}(\Theta) = \frac{1}{n} \sum_{i = 1}^n L_2(\Theta; D_{T_i})$$
Our estimator is provided by (assuming existence and uniqueness)
$$\widehat{\Theta}^{(n)} = \underset{\Theta}{\arg\max}\, L_2^{(n)}(\Theta)$$
Define $$\begin{aligned}
    J(\Theta) &= \mathbb{E}_\ast L_2(\Theta; D_T)\\
    &= \int_{\mathcal{D}^m} p(D_T \vert \Theta_\ast) \log p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta) dD_T\\\end{aligned}$$
where subscript $\ast$ denotes the underlying truth, then
$$\begin{aligned}
    J(\Theta) - J(\Theta_\ast) &= \int_{\mathcal{D}^m} p(D_T \vert \Theta_\ast) \log \frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} dD_T\\
    &= \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \left[\int_{\mathcal{D}^{m - k}} p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast) \log \frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} dD_T^{\textrm{val}} \right] dD_T^{\textrm{tr}}\\
    &\le \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \left[\int_{\mathcal{D}^{m - k}} p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast) \left(\frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} - 1\right) dD_T^{\textrm{val}} \right] dD_T^{\textrm{tr}}\\
    &= \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \cdot 0 \cdot dD_T^{\textrm{tr}} = 0\end{aligned}$$
The inequality is due to $\log x \le x - 1$. So we have established that
$\Theta_\ast$ maximizes $J$. To finish the proof, we provide the sketch
as follows:

1.  By law of large numbers (LLN),
    $L_2^{(n)}(\Theta) \rightarrow_p J(\Theta)$ as
    $n \rightarrow \infty$, for all $\Theta$. (Yayi and Xiaoqi have an
    unresolved misunderstanding about $E_\ast$, I will explain and
    justify the use of LLN later.)

2.  $\widehat{\Theta}^{(n)}$ maximizes $L_2^{(n)}(\Theta)$, while
    $\Theta_\ast$ maximizes $J$ (proved above).

3.  By argmax continuous mapping theorem (or the M-estimator theorem, I
    need to check conditions but I am very sure it holds),
    $\widehat{\Theta}^{(n)} \rightarrow_p \Theta_\ast$ as
    $n \rightarrow \infty$.

Done.

# YAYI's notes

Another way to derive GEM gradient(Corolloary 1):

We denote $\hat{q}=\delta_{\hat{\theta}}(\theta)$ as the task ground
truth or ultimate posterior with $D \rightarrow \Tilde{D}$. According to
equation ([\[eq: F3\]](#eq: F3){reference-type="ref"
reference="eq: F3"}),
$L(\Theta;\Tilde{D}) = ELBO(\hat{q},\Theta, \sigma^2) = \log p(\Tilde{D}|\hat{\theta},\sigma^2) + \log p(\hat{\theta}|\Theta)$.
So we have
$\nabla_{\Theta} L(\Theta;\Tilde{D}) = \nabla_{\Theta} \log p(\hat{\theta}|\Theta)$.
When observed finite data $D$, according to the Bayesian rule, the
probability of the task ground truth $\hat{\theta}$ is the posterior
$p(\theta|D,\Theta,\sigma^2)$. Consequently, we get the probability
distribution of $\nabla_{\Theta} L(\Theta;\Tilde{D})$ through the
posterior distribution of $\hat{\theta}$ and its expectation is
therefore derived as
$E_{p(\theta|D,\Theta,\sigma^2)} \nabla_{\Theta} \log p(\theta|\Theta)$.

GEM-gradient is using expectation estimator while GEM-MLE is using MAP
estimator of $\nabla_{\Theta} L(\Theta;\Tilde{D})$.

## Optimality of Empirical Bayes

$\Theta^{[n]}=\arg \max_{\Theta} E_{\tau \in \mathcal{T}} L(\Theta ; D_\tau)$
where $|D_\tau|=n$. $\Theta^{[n]}$ is **not** the optimal prior/ true
task distribution.

$\hat{\Theta}^{[n]}=\arg \max_{\Theta} E_{\tau \in \mathcal{T}, D_\tau}  L(\Theta ; D_\tau)$

$$E_{\tau \in \mathcal{T}} L(\Theta ; D_\tau) = E_{\tau \in \mathcal{T}} \log E_{\theta_\tau \sim \Theta}  p(D_\tau|\theta_\tau)  = E_{\tau \in \mathcal{T}} \log \int  p(D_\tau|\theta_\tau) p(\theta_\tau|\Theta) d\theta = \int L(\Theta ; D_t) p(t) dt$$

$$E_{\tau \in \mathcal{T} , D_\tau} L(\Theta ; D_\tau) = \iint L(\Theta ; D) p(D \vert t) p(t) dD dt$$

$\hat{D}_\tau$ with batch size $n$ ($\in \mathcal{D}^n$) (is r.v., can
take expectation).\
1. $E_{\tau \in \mathcal{T}} f(\hat{D}_\tau)$\
2. $E_{\tau \in \mathcal{T} , \hat{D}_\tau} f(\hat{D}_\tau)$\
This is the ground truth: task $\tau$ can be parameterized by $w_\tau$,
and

$w_\tau \sim p(w_\tau|\mathcal{T}),  \hat{D}_\tau(w_\tau) \vert w_\tau \sim f(w_\tau)$\
1.
$E_{\tau \in \mathcal{T}} f(\hat{D}_\tau)  = E_{ w_\tau \sim p(w_\tau|\mathcal{T})} f(\hat{D}_\tau(w_\tau))$

In (16):

First term: $E_{d \vert t} L_t(w, d)$

Second term: $L_t(w, d)$

What you said: $E_{t} L_t(w, d)$

There is huge difference between finite $\mathcal{T}$ and infinite
$\mathcal{T}$.

Important principle? Bayes optimal rule: if we correctly estimate
$\Theta$ (known), then $p(\theta \vert \Theta, D)$ can be used to derive
optimal decision rule.

# [About my confusions]{style="color: red"}

Consider a hierarchical structure with generative model (ground truth)
$T \sim P_T^\ast$ and $D \vert T \sim P_{D \vert T}^\ast$, where
$T \in \mathcal{T}$ and $D \in \mathcal{D}$. In practice $T$ is latent
and only $D$ is observed. Let
$\mathcal{F} = \{f_\theta: \theta \in \Theta\}$ be a family of decision
rules, where each element $f_\theta$ can be evaluated given observed
data $D$ using loss function
$L: (\mathcal{F}, \mathcal{D}) \mapsto \mathbb{R}$. We assume for each
$t \in \mathcal{T}$, there is a unique corresponding best-fitted
$\theta_t$ defined by
$$\theta_t = \underset{\theta \in \Theta}{\arg\min} \, \int_\mathcal{D} L(f_\theta, x)\, p_{D \vert t}^\ast(x \vert t) \, dx$$
This constructs a mapping from $\mathcal{T}$ to $\Theta$, as well as a
corresponding change of measure from $p_T^\ast$ to $p_{\theta_T}^\ast$.
For inference purpose, we introduce hyper-parameter $\psi$ to
approximate $p_{\theta_T}^\ast(\theta) \approx p(\theta \vert \psi)$.
The oracle $\psi^\ast$ is naturally defined by
$$\psi^\ast = \underset{\psi}{\arg\max} \, \int_\Theta p_{\theta_T}^\ast(\theta) \log p(\theta \vert \psi) d\theta$$
First consider the case where $\psi^\ast$ is known, and we are hoping to
recover $\theta_T$, given training data $D_T^\textrm{tr}$.

[below is not finished]{style="color: green"}

Note that for any given $t$, such best-fitted $\theta_t$ might not be
unique, and when that happens, we want to choose[^1] one element
$\theta_t$ from each set $S_t$. Once choices are made, we obtain a
mapping from $\mathcal{T}$ to $\Theta$ as well as a corresponding change
of measure from $p_T^\ast$ to $p_{\theta_T}^\ast$. A convenient way to
construct these choices is using some parametric prior
$p(\theta \vert \psi)$: $$\begin{aligned}
    \underset{\{\theta_t: t \in \mathcal{T}\}, \psi}{\max}\, &\quad \int_\mathcal{T} p_T^\ast(t) \log p(\theta_t \vert \psi)\, dt \\
    \textrm{subject to} &\quad \theta_t \in  S_t \quad \forall t \in \mathcal{T}\end{aligned}$$
We denote the solution of the above optimization as $\psi^\ast$ and
$\theta_t(\psi^\ast)$

With the uniqueness issue resolved, this mapping from $\{T\}$ to
$\{\theta_T\}$ induces a change of measure from $p_\ast(T)$ to
$p_\ast(\theta_T)$. For inference purpose, we parameterize
$p_\ast(\theta_T)$ by $p(\theta_T \vert \Theta)$. Similar to $\theta_T$,
we define the best-fitted $\Theta_\ast$ as
$$\Theta_\ast = \underset{\Theta}{\arg\min} \, \int$$

Once we have data from sufficiently many $T$s ($T_1, T_2, \ldots, T_n$),
then we may get reasonably good estimate for $\Theta$. For example, in
MLE approach:
$$\widehat{\Theta}_n \rightarrow_p \Theta_\ast \quad \textrm{as } n \rightarrow \infty$$
where
$$\Theta_\ast = \underset{\Theta}{\arg\max}\, \int p_\ast(D) \log p(D \vert \Theta)\, dD$$
Let $f$ be some decision rule, and define loss function
$L: (\mathcal{F}, \mathcal{D}) \mapsto \mathbb{R}$. Given data
$D^\textrm{tr}$ (from unknown task $T$), the Bayes rule w.r.t. the
inference model is defined as
$$\hat{f}(D^\textrm{tr}) = \underset{f}{\arg\min}\, \iint L(f, D)\, p(D \vert \theta)\, p(\theta \vert D^\textrm{tr})\, dD d\theta$$

::: {.remark}
**Remark 7**. *Define
$$f_\theta = \underset{f}{\arg\min}\, \int L(f, D)\, p(D \vert \theta)\, dD$$
then in general
$\hat{f}(D^\textrm{tr}) \not\in \{f_\theta: \forall \theta\}$. This can
be seen by $$\begin{aligned}
         \iint L(\hat{f}(D^\textrm{tr}), D)\, p(D \vert \theta)\, p(\theta \vert D^\textrm{tr})\, dD d\theta &= \min_{f} \iint L(f, D)\, p(D \vert \theta)\, p(\theta \vert D^\textrm{tr})\, dD d\theta\\
         &\ge \int \left[\min_{f} \int L(f, D)\, p(D \vert \theta)\, dD\right] p(\theta \vert D^\textrm{tr})\, d\theta\\
         &= \iint L(f_\theta, D)\, p(D \vert \theta)\, p(\theta \vert D^\textrm{tr})\, dD d\theta
     \end{aligned}$$ [This is my hesitation about line 77, because
unless you use some point estimate from posterior (e.g. MAP), you don't
know that $\hat{f}(D^\textrm{tr})$ can be represented by $f_\theta$.
With this obsession in mind, I was not sure what is the family of
decision rules when we talk about "optimal".]{style="color: red"} [Eq 60
is not justified. Need modification.]{style="color: blue"}*
:::

The generalization error (Bayes risk) of $\hat{f}$ can be written as
$$\operatorname{gen}(\hat{f}) = \iiint L(\hat{f}(D^\textrm{tr}), D)\, p_\ast(D \vert T)\, p_\ast(D^\textrm{tr} \vert T)\, p_\ast(T)\, dD dD^\textrm{tr} dT$$

# About last night (5/19)'s discussion

Now we limit ourselves to Gaussian prior. The generative model is
$D \vert \theta \sim P_{D \vert \theta}$ (short notation:
$p(D \vert \theta)$ as density) and
$\theta \vert \mu, \Sigma \sim \mathcal{N}(\mu, \Sigma)$ (short notation
for density: $p(\theta \vert \mu, \Sigma)$. We further denote the ground
truth parameter as $(\mu_\ast, \Sigma_\ast)$.With the inference model
correctly specified, we have
$$L(\mu, \Sigma; D) = \log \int p(D \vert \theta)\, p(\theta \vert \mu, \Sigma)\, d\theta$$
The population version is defined as
$$J(\mu, \Sigma) = \mathbb{E}_\ast L(\mu, \Sigma; D)$$ and we know that
optimality achieves at $(\mu_\ast, \Sigma_\ast)$. Two things are being
considered:

1.  The integral in $L$ is not easy to compute in general.

2.  We care about correct estimation of $\mu$, but not interested in
    $\Sigma$.

Our approach is to find a proxy $\widetilde{L}(\mu; D)$ with
corresponding
$\widetilde{J}(\mu) = \mathbb{E}_\ast \widetilde{L}(\mu; D)$ and
$\widetilde{\mu}_\ast = \arg\max_\mu \widetilde{J}(\mu)$, such that
$\widetilde{\mu}_\ast$ provides a good approximation to $\mu_\ast$ (or
equal, under certain assumptions). Note that we hope $\Sigma_\ast$ to be
very small (so that tasks are close to each other).

## Dirac Delta

Consider $p(\theta \vert \mu, \Sigma) = \delta(\theta - \mu)$, as the
limiting case of $\Sigma \rightarrow 0$. Under this construction
$$\begin{aligned}
     \widetilde{L}(\mu; D) &= \log p(D \vert \mu)\\
     \widetilde{J}(\mu) &= \iint p(D \vert \theta)\, p(\theta \vert \mu_\ast, \Sigma_\ast)\, \log p(D \vert \mu)\, d\theta dD
 \end{aligned}$$ We want
$\widetilde{J}(\mu_\ast) \ge \widetilde{J}(\mu)$, that is
$$\begin{aligned}
    0 &\le \int \left(\int p(D \vert \theta)\, p(\theta \vert \mu_\ast, \Sigma_\ast)\, d\theta \right) \log\frac{p(D \vert \mu_\ast)}{p(D \vert \mu)}\, dD\\
     &= \int \left(\frac{1}{p(D \vert \mu_\ast)}\int p(D \vert \theta)\, p(\theta \vert \mu_\ast, \Sigma_\ast)\, d\theta \right) p(D \vert \mu_\ast) \log\frac{p(D \vert \mu_\ast)}{p(D \vert \mu)}\, dD\end{aligned}$$
A sufficient condition is
$\frac{1}{p(D \vert \mu_\ast)}\int p(D \vert \theta)\, p(\theta \vert \mu_\ast, \Sigma_\ast)\, d\theta$
does not depend on $D$, then the inequality holds by the non-negativity
property of KL divergence.

# Cache: ready to be presented on paper

## About optimality under over-parameterization

### Toy example: simple Gaussian

Consider the following toy example: the generative model is
$X \vert \theta \sim \mathcal{N}(\theta, 1^2)$ and
$\theta \sim \mathcal{N}(u, s^2)$.

### Over-parameteriazation by linear combination ("wall type")

Over-parameterization with $w \in \mathbb{R}^2$. The inference model is
$X \vert w \sim \mathcal{N}(w_1 + w_2, 1^2)$ and
$w \sim \mathcal{N}(\mu, \Sigma)$. Using the L1 approach, we first
compute the log-likelihood function $$\begin{aligned}
    L(\mu, \Sigma; X) &= \log \int p(X \vert w)\, p(w \vert \mu, \Sigma) dw\\
    &= C(X) + \frac{1}{2} \left(\log \lvert A \rvert - \log \lvert \Sigma \rvert + b^T A^{-1} b - \mu^T \Sigma^{-1} \mu\right)\end{aligned}$$
where $$\begin{aligned}
    A^{-1} &= \Sigma^{-1} + 1 1^T\\
    A^{-1} b &= \Sigma^{-1} \mu + X 1\end{aligned}$$ To check the
consistency property, we compute the oracle log-likelihood
$J(\mu, \Sigma) = \mathbb{E}_\ast L(\mu, \Sigma; X)$ and optimize it
over $(\mu, \Sigma)$, giving $$\begin{aligned}
    1^T \mu = u\end{aligned}$$

## About posterior point estimate under Gaussian prior

The generative model is $D \vert \theta \sim P_{D \vert \theta}$ (short
notation: $p(D \vert \theta)$ as density) and
$\theta \vert \mu, \Sigma \sim \mathcal{N}(\mu, \Sigma)$ (short notation
for density: $p(\theta \vert \mu, \Sigma)$. We further denote the ground
truth parameter as $(\mu_\ast, \Sigma_\ast)$.With the inference model
correctly specified, we have
$$L(\mu, \Sigma; D) = \log \int p(D \vert \theta)\, p(\theta \vert \mu, \Sigma)\, d\theta$$
The population version is defined as
$$J(\mu, \Sigma) = \mathbb{E}_\ast L(\mu, \Sigma; D)$$ and we know that
optimality achieves at $(\mu_\ast, \Sigma_\ast)$. Two things are being
considered:

1.  The integral in $L$ is not easy to compute in general.

2.  We care about correct estimation of $\mu$, but not interested in
    $\Sigma$.

Our approach is to find a proxy $\widetilde{L}(\mu; D)$ with
corresponding
$\widetilde{J}(\mu) = \mathbb{E}_\ast \widetilde{L}(\mu; D)$ and
$\widetilde{\mu}_\ast = \arg\max_\mu \widetilde{J}(\mu)$, such that
$\widetilde{\mu}_\ast$ provides a good approximation to $\mu_\ast$ (or
equal, under certain assumptions). Note that we hope $\Sigma_\ast$ to be
very small (so that tasks are close to each other).

## About L1, L2

### Consistency

#### L1

Let $D$ $$L_1(\Theta; D) =$$

#### L2

Now split $D_T = (D_T^{\textrm{tr}}, D_T^{\textrm{val}})$ into
$D_T^{\textrm{tr}} \in \mathcal{D}^k$,
$D_T^{\textrm{val}} \in \mathcal{D}^{m - k}$. By definition,
$$L_2(\Theta; D_T) = \log p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)$$
For a sample containing $n$ tasks:
$$L_2^{(n)}(\Theta) = \frac{1}{n} \sum_{i = 1}^n L_2(\Theta; D_{T_i})$$
Our estimator is provided by (assuming existence and uniqueness)
$$\widehat{\Theta}^{(n)} = \underset{\Theta}{\arg\max}\, L_2^{(n)}(\Theta)$$
Define $$\begin{aligned}
    J(\Theta) &= \mathbb{E}_\ast L_2(\Theta; D_T)\\
    &= \int_{\mathcal{D}^m} p(D_T \vert \Theta_\ast) \log p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta) dD_T\\\end{aligned}$$
where subscript $\ast$ denotes the underlying truth, then
$$\begin{aligned}
    J(\Theta) - J(\Theta_\ast) &= \int_{\mathcal{D}^m} p(D_T \vert \Theta_\ast) \log \frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} dD_T\\
    &= \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \left[\int_{\mathcal{D}^{m - k}} p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast) \log \frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} dD_T^{\textrm{val}} \right] dD_T^{\textrm{tr}}\\
    &\le \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \left[\int_{\mathcal{D}^{m - k}} p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast) \left(\frac{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta)}{p(D_T^{\textrm{val}} \vert D_T^{\textrm{tr}}, \Theta_\ast)} - 1\right) dD_T^{\textrm{val}} \right] dD_T^{\textrm{tr}}\\
    &= \int_{\mathcal{D}^k} p(D_T^{\textrm{tr}} \vert \Theta_\ast) \cdot 0 \cdot dD_T^{\textrm{tr}} = 0\end{aligned}$$
The inequality is due to $\log x \le x - 1$. So we have established that
$\Theta_\ast$ maximizes $J$. To finish the proof, we provide the sketch
as follows:

1.  By law of large numbers (LLN),
    $L_2^{(n)}(\Theta) \rightarrow_p J(\Theta)$ as
    $n \rightarrow \infty$, for all $\Theta$. (Yayi and Xiaoqi have an
    unresolved misunderstanding about $E_\ast$, I will explain and
    justify the use of LLN later.)

2.  $\widehat{\Theta}^{(n)}$ maximizes $L_2^{(n)}(\Theta)$, while
    $\Theta_\ast$ maximizes $J$ (proved above).

3.  By argmax continuous mapping theorem (or the M-estimator theorem, I
    need to check conditions but I am very sure it holds),
    $\widehat{\Theta}^{(n)} \rightarrow_p \Theta_\ast$ as
    $n \rightarrow \infty$.

[^1]: This is feasible, by axiom of choice.


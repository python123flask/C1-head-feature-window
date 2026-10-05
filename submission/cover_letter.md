# Cover letter — Neurocomputing (Type 1: Regular article)

**Manuscript:** Head first, features later: a transient window of classifier-head
de-specialization under uniform label exposure and its effect on task adaptation

**Corresponding author:** Xianzhe Liu, Hunan University (liuxianzhe@hnu.edu.cn)

---

Dear Editor,

We would like to submit our manuscript for consideration as a Type 1 regular
article. It is directly relevant to neural networks and learning systems: it
studies the dynamics of a classifier head and a feature extractor during a
task transition, and derives practical guidance for how long to dwell in a
label-scarce transition.

**What is new.**

1. **A transient mechanism, not a terminal one.** While the literature studies
   loss of plasticity as a terminal state (trapping manifolds, collapsed
   geometry, last-layer resets), we measure the transition itself: within
   65--220 steps of uniform label exposure the classifier head stops reading
   the old task (accuracy <= 0.50) while a linear probe still recovers it from
   the features (>= 0.70) - a timescale separation of roughly an order of
   magnitude, replicated in 5/5 seeds at two levels of feature sharing.

2. **Practical, quantitative guidance.** Read as budget-limited adaptation,
   every exposure beats none for every budget from 50 to 2000 steps (5/5
   seeds), by up to +11.5 accuracy points at a 50-step budget, and an interior
   optimum in exposure length appears on the pre-registered adaptation-speed
   readout. A stopping rule follows directly: switch when the head stops
   reading the old task and before the feature probe begins to fall - a signal
   current-task accuracy cannot provide.

3. **Unusually strong evidence practice.** The study was pre-registered with
   four frozen decision gates and a frozen claim ceiling; every protocol
   deviation is logged with its calibration evidence. We report 207 runs over
   five datasets (CIFAR-10/100, SVHN, Fashion-MNIST, UCI gas-sensor drift),
   three architectures, a five-point target-entropy dose-response, a causal
   cross-transplant of heads and features, five controls (including a
   last-layer-reset baseline), and a deterministic-kernel replication whose
   identical-configuration reruns are bitwise identical (zero process-level
   noise). Code, raw per-evaluation metrics, checkpoints and the analysis
   pipeline are released.

4. **Honest negative results.** Under deterministic kernels the endpoint
   effect is only 0.0021 accuracy points across exposure lengths; we therefore
   claim adaptation-speed gains and explicitly not endpoint gains, and we
   report a benchmark where only half of the mechanism appears.

**Fit to Neurocomputing.** The work combines analysis of network dynamics with
a practical method for non-stationary training, and includes an accessible
theoretical account (a gradient decomposition that predicts crossing times
scaling as 1/learning rate, verified over a 10-fold range).

**Statements.** Single author; no competing interests; this research received
no specific grant funding; the use of an LLM-based AI assistant in manuscript
and code preparation is declared in the manuscript, where all figures are
generated from recorded measurements by released scripts.

Thank you for your consideration.

Yours sincerely,
Xianzhe Liu
Hunan University, Changsha, China
liuxianzhe@hnu.edu.cn

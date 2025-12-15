**Maximum Entropy Tree Search for Autoregressive Models (MaxEnt-TS)**, drawing directly from the principles of Diffusion Tree Sampling (DTS) and the Maximum Entropy RL framework. I will then outline the specific implementation steps for integrating this with OpenTSLM.

***

## 1. Theoretical Mathematical Ground

The proposed method, the **Token-Level Tree Search** (adapted DTS), is mathematically grounded in solving the KL-regularized optimal control problem, with the autoregressive process serving as the Markov chain.

### 1.1 Core Definitions

| Component | Diffusion Tree Sampling (DTS) Context | MaxEnt-TS (OpenTSLM) Context |
| :--- | :--- | :--- |
| **State** ($\mathbf{x}_t$) | Noisy data $\mathbf{x}_t$ at timestep $t$. | Partial sequence $\mathbf{x}_{\le t}$ of length $t$ (prefix). |
| **Transition Kernel** ($p_{\theta}$) | Reverse diffusion step $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$. | Autoregressive generation step $p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})$. |
| **Terminal State** ($\mathbf{x}_0$) | Clean sample at time $t=0$. | Final completed time series sequence $\mathbf{x}$. |
| **Reward** ($r$) | Reward $r(\mathbf{x}_0)$ on the final clean sample. | Reward $r(\mathbf{x})$ (e.g., spectral distance) on the final sequence. |
| **Target Policy** ($\pi^*$) | Optimal reverse diffusion path for sampling $\mathbf{x}_0$. | Optimal sequence generation path for sampling $\mathbf{x}$. |

### 1.2 The Optimal Policy Objective

The goal is to find an optimal policy $\pi$ (the token generation rule) that maximizes the expected final reward $r(\mathbf{x})$ while maintaining closeness to the pretrained OpenTSLM policy $p_{\theta}$ (the prior):

$$\mathbf{\pi^{*}(\mathbf{x}):=\arg\max_{\pi}\mathbb{E}_{\mathbf{x}\sim\pi(\cdot)}[r(\mathbf{x})]-\frac{1}{\lambda}D_{KL}(\pi||p_{\theta})} \quad \text{(Adapted from Eq. 3)}$$

This results in the target distribution $\pi^{*}(\mathbf{x})$:

$$\mathbf{\pi^{*}(\mathbf{x})=\frac{1}{\mathcal{Z}}p_{\theta}(\mathbf{x})\exp(\lambda r(\mathbf{x}))}$$

### 1.3 The Soft Value Function and Bellman Equation (Proof)

The soft value function $V_t(\mathbf{x}_{\le t})$ is defined as the maximum expected exponentiated reward for the remaining path, starting from the prefix $\mathbf{x}_{\le t}$.

**Definition (Soft Value Function):** The soft value function $V_t(\mathbf{x}_{\le t})$ for the sequence prefix $\mathbf{x}_{\le t}$ is:
$$V_t(\mathbf{x}_{\le t}) := \frac{1}{\lambda} \log \mathbb{E}_{p_{\theta}(\mathbf{x}_{t+1:T}|\mathbf{x}_{\le t})}[\exp(\lambda r(\mathbf{x}))]$$
where $p_{\theta}(\mathbf{x}_{t+1:T}|\mathbf{x}_{\le t})$ is the sequence continuation distribution given by OpenTSLM.

**Proposition (Soft Bellman Equation):** The soft value function satisfies the following recursive relation:

$$\mathbf{V_{t}(\mathbf{x}_{\le t}) = \frac{1}{\lambda}\log\mathbb{E}_{p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})}[\exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}))]} \quad \text{(Adapted from Eq. 5)}$$

* **Proof Sketch (Adapted from Appendix D.1):**
    1.  Start with the definition of $V_t(\mathbf{x}_{\le t})$:
        $$V_t(\mathbf{x}_{\le t}) = \frac{1}{\lambda} \log \int p_{\theta}(\mathbf{x}_{t+1:T}|\mathbf{x}_{\le t}) \exp(\lambda r(\mathbf{x})) d\mathbf{x}_{t+1:T}$$
    2.  Use the Markov property for autoregressive generation: $p_{\theta}(\mathbf{x}_{t+1:T}|\mathbf{x}_{\le t}) = p_{\theta}(x_{t+1}|\mathbf{x}_{\le t}) \cdot p_{\theta}(\mathbf{x}_{t+2:T}|\mathbf{x}_{\le t+1})$.
    3.  Separate the integral over the next token $x_{t+1}$ and the remaining tokens $\mathbf{x}_{t+2:T}$:
        $$V_t(\mathbf{x}_{\le t}) = \frac{1}{\lambda} \log \sum_{x_{t+1}} p_{\theta}(x_{t+1}|\mathbf{x}_{\le t}) \left[ \int p_{\theta}(\mathbf{x}_{t+2:T}|\mathbf{x}_{\le t+1}) \exp(\lambda r(\mathbf{x})) d\mathbf{x}_{t+2:T} \right]$$
    4.  Recognize the term in brackets as $\exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}))$.
    5.  Substitute back and rewrite the summation as an expectation:
        $$V_t(\mathbf{x}_{\le t}) = \frac{1}{\lambda} \log \sum_{x_{t+1}} p_{\theta}(x_{t+1}|\mathbf{x}_{\le t}) \exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}))$$
        $$V_t(\mathbf{x}_{\le t}) = \frac{1}{\lambda}\log\mathbb{E}_{p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})}[\exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}))]$$
    This confirms that the DTS backup mechanism is valid for token sequences.

### 1.4 The Optimal Policy (Token Selection Rule)

**Proposition (Optimal Policy):** The optimal policy $\pi^{*}(\mathbf{x}_{\le t+1}|\mathbf{x}_{\le t})$ (i.e., the probability of choosing the next token $x_{t+1}$) is given by:

$$\mathbf{\pi^{*}(\mathbf{x}_{\le t+1}|\mathbf{x}_{\le t}) = \frac{p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})\exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}))}{\sum_{x'_{t+1}} p_{\theta}(x'_{t+1}|\mathbf{x}_{\le t})\exp(\lambda V_{t+1}(\mathbf{x}_{\le t+1}'))}} \quad \text{(Adapted from Eq. 6)}$$

This means the selection of the next token is a **Boltzmann distribution** over the next state's soft value, modulated by the OpenTSLM's original probability $p_{\theta}$. This is the exact formula for the **Selection step** in your tree search.

---

## 2. Implementation Strategy

The implementation adapts Algorithm 1 (DTS/DTS\*) to the token generation loop.

### 2.1 Core Transfer Concepts

The mapping defined by your friend is correct:
* **Tree Search** is over token sequences (paths).
* **Soft Bellman** is over token choices (actions).
* The **Spectral Rewards** act as the final reward $r(\mathbf{x})$ at the terminal sequence state.

### 2.2 Algorithm Steps for Token-Level Tree Search (Adapted DTS)

The search proceeds iteratively for $M$ iterations, with the time index $t$ now representing the sequence length.

#### Step 1: Selection (Exploitation/Exploration)

Start at the current prefix $\mathbf{x}_{\le t}$. Traverse the existing tree by selecting the next token $x_{t+1}$ using one of the following rules:

* **For Sampling (DTS):** Use the Boltzmann distribution (Optimal Policy Eq.):
    $$x_{t+1} \sim \frac{p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})\exp(\lambda\hat{v}(\mathbf{x}_{\le t+1}))}{\sum_{x'_{t+1}} (\cdot)}$$
* **For Search (DTS\*):** Use the **UCT (Upper Confidence bound for Trees)** formula for next token selection:
    $$x_{t+1} = \arg\max_{x'_{t+1}} \left( \hat{v}(\mathbf{x}_{\le t+1}') + c_{uct}\sqrt{\frac{\log N(\mathbf{x}_{\le t})}{N(\mathbf{x}_{\le t+1}')}} \right) \quad \text{(Adapted from Eq. 9)}$$

#### Step 2: Expansion & Rollout (Sequence Continuation)

If the selected node $\mathbf{x}_{\le t}$ has not reached its limit on available next tokens (children), or if the search reaches an unexpanded node:

1.  **Expansion:** Generate a new child node $\mathbf{x}_{\le t+1}$ by selecting a new token $x_{t+1}$ based on $p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})$. Initialize $\hat{v}(\mathbf{x}_{\le t+1})=0$.
2.  **Rollout:** From this new state $\mathbf{x}_{\le t+1}$, complete the sequence to length $T$ by sampling the remaining tokens using the **base OpenTSLM model** $p_{\theta}(\mathbf{x}_{t+2:T}|\mathbf{x}_{\le t+1})$. This yields the final sequence $\mathbf{x}$.

#### Step 3: Backup (Credit Assignment)

1.  **Evaluate Reward:** Calculate the terminal reward $\hat{v}(\mathbf{x}) = r(\mathbf{x})$ using the external spectral reward function.
2.  **Soft Value Update:** Propagate this reward backward along the rollout path, updating the value estimate $\hat{v}(\mathbf{x}_{\le t})$ for all nodes in the path using the Soft Bellman Equation:
    $$\hat{v}(\mathbf{x}_{\le t}) \leftarrow \frac{1}{\lambda}\log\sum_{x_{t+1}\in\mathcal{C}(\mathbf{x}_{\le t})}\exp(\lambda\hat{v}(\mathbf{x}_{\le t+1}))$$
3.  **Update Counts:** Increment the visit count $N(\mathbf{x}_{\le t})$ for all nodes in the path.

### 2.3 Required Implementation Tools

| Tool | Role | Function in MaxEnt-TS |
| :--- | :--- | :--- |
| **OpenTSLM (Frozen)** | Base Policy $\boldsymbol{p_{\theta}}$ | Provides next-token probabilities $p_{\theta}(x_{t+1}|\mathbf{x}_{\le t})$ for Expansion and Rollout steps. |
| **Tree Data Structure** | Memory/State Map | Stores nodes ($\mathbf{x}_{\le t}$), visit counts ($N$), and soft value estimates ($\hat{v}$). |
| **Spectral Reward Model** | External Alignment | Computes the terminal reward $r(\mathbf{x})$ to initiate the Backup step. |
| **GFlowNet (Optional)** | Amortization | Learns the optimal policy $\pi^{*}$ from the high-fidelity samples generated by MaxEnt-TS, enabling faster inference later. |

This adaptation is **valid and will work** because it correctly applies the generalized Maximum Entropy RL framework and the Soft Bellman backup—the core innovation of DTS—to the sequential token generation process.
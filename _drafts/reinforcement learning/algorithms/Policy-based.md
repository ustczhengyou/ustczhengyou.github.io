**TRPO (Trust Region Policy Optimization)**

- **理论基础**

虽然**Causality**和**Baseline**策略一定上降低了Policy Gradient的训练的**high variance** 问题，但是依然面临着训练不稳定的问题。为了解决该问题，**TRPO** 算法提出了每次迭代都将参数更新控制在Trust Region范围内以此来实现有效的Improvement的。

先看一个claim:

**Claim 1**:  $J(\theta)=\eta(\pi')-\eta(\pi)= E_{s \sim \eta_{\pi'}}[\sum_{t=0} r^tA_{\pi}(s_t,a_t)]=\sum_s \rho_{\pi'}(s)\sum_a \pi'(a|s)A_{\pi}(s,a) \tag{1}$

证明见：, 

也就是说新策略 $\theta'$ 相对于旧策略 $\theta$ 的提升就是最大化旧策略的Advantage function，优化目标变成了$J(\theta)$。

将 (1) 公式展开并加入 Importance Sampling 后得到 (2) 式：
$$
\begin{align}
E_{\tau \sim p_{\theta'}(\tau)}[\sum_t\gamma A^{\pi_{\theta}(s_t, a_t)}]=\sum_t E_{s_t \sim p_{\theta'}}[E_{a_t \sim \pi_{\theta'}(a_t|s_t)} [\gamma^t A_{\pi}(a_t|s_t)]] \notag \\
=\sum_t E_{s_t \sim p_{\theta'}}[E_{a_t \sim \pi_{\theta}(a_t|s_t)} [\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} \gamma^t A_{\pi}(a_t|s_t)]] \tag{2} \\
\end{align}
$$
从 (2) 式来看，$s_t$ 的分布是由新的 $\theta'$ 产生的而非策略 $\theta$ 所以不能直接使用梯度上升来优化。

来看另外一个claim:

 **Claim 2**：$p_{\theta}(s_t)$ is *close* to $p_{\theta'}(s_t)$ when $\pi_{\theta}$ is *close* to $\pi_{\theta'}$,  and $\pi_{\theta'}$ is close to $\pi_{\theta}$ if $|\pi_{\theta'}(a_t|s_t)-\pi_{\theta}(a_t|s_t)|\le \epsilon$ for all $s_t$

同时根据 $|\pi_{\theta'}(a_t|s_t)-\pi_{\theta}(a_t|s_t)| \le \sqrt{\frac{1}{2}D_{KL}(\pi_\theta'(a_t|s_t)|\pi_{\theta}(a_t|s_t))}$，使用 $\pi_{\theta'}$和$\pi_{\theta}$的KL Divergence作为新的约束, 即 $D_{KL}(\pi_\theta'(a_t|s_t)|\pi_{\theta}(a_t|s_t)) \le \epsilon$ 。

因此转变带约束的优化问题：
$$
\begin{gather}
\theta' \leftarrow \mathop{\arg\max}_{\theta'} \sum_t E_{s_t \sim p_{\theta}}[E_{a_t \sim \pi_{\theta}(a_t|s_t)}[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}A_{\pi}(a_t|s_t)]] \\
Such \quad that \quad D_{KL}(\pi'(a_t|s_t)|\pi(a_t|s_t)) \le \epsilon
\end{gather}
$$
或者
$$
L(\theta',\lambda)=\sum_t E_{s_t \sim p_{\theta}}[E_{a_t \sim \pi_{\theta}(a_t|s_t)}\frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}A_{\pi}(a_t|s_t)]-\lambda(D_{KL}(\pi'(a_t|s_t)|\pi(a_t|s_t))-\epsilon)
$$
- **实际优化更新**

根据上式，简化为：
$$
maximize_{\theta'}L_{\pi_{\theta}}(\pi_{\theta'})-\beta*KL_{\pi_{\theta'}}(\pi_{\theta})
$$
分为两步：

**Step1**: 将 $L$ 函数和$KL$函数分别用泰勒公式按照一阶和二阶展开得到搜索方向
$$
\mathop{maximize}_{\theta} \quad g^T(\theta' - \theta)-\frac{\beta}{2}(\theta'-\theta)^TF(\theta'-\theta) \\
where \quad g=\frac{\partial}{\partial\theta'}L_{\pi_{\theta}}(\pi_{\theta'})|_{\theta'=\theta}, \quad F=\frac{\partial^2}{\partial^2 \theta'}KL_{\pi_{\theta'}}(\pi_{\theta})|_{\theta'=\theta}
$$
**Step2:** 在1得到的方向上进行线性搜索以保证满足约束要求。
$$
\theta'-\theta = \frac{1}{\beta}F^{-1}g
$$
The **Fisher Information matrix (F)** gives information about how sensitive the probability distribution to different direction in parameter space.

因为FIM(H)的计算成本太高，所以采用 **CG (conjugate gradient)** 来求解 $F.x=g$.

- TRPO和其他算法的关系

  | Linear-quadratic approximation + penalty | **natural gradient**         |
  | ---------------------------------------- | ---------------------------- |
  | No constraint                            | **policy iteration**         |
  | Euclidean penalty instead of KL          | **vanilla policy gradients** |

***

**PPO (Proximal Policy Optimization)** 

TRPO使得训练的稳定性提升，但是面临着CG计算的复杂度高以及和一些网络结构不兼容的问题(例如， dropout策略和共享参数策略)。和TRPO采用KL函数作为约束的策略不同，PPO利用裁剪目标函数的方式来简化目标函数的优化，并保证性能。

其中  $r(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ ， TRPO的目标函数为：
$$
L^{TRPO}(\theta)=E_t[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}A_t]=E_t[r_t(\theta)A_t]
$$
PPO的目标函数为：
$$
L^{PPO}(\theta)=E_t[min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]
$$
从上式可以看到PPO将$r(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]范围内$，只有当$A_t$是负的时候才纳入到目标函数中，忽略掉$A_t$为正使得策略提升的额外部分，从而使得优化方向朝不会为了追求部分step的外额reward方向发展，同时min函数表面$L^{PPO}$是$L^{TRPO}$的下界。

同时，如果PPO采用的是policy和value function共享参数的网络结构时，应该将value functions的lose以及entropy bonus部分一起加入：
$$
L_t^{PPO+VF+S}=E_t[L_t^{PPO}-c_1L_t^{VF}(\theta)+c_2S[\pi_{\theta}](s_t)]
$$
其中: $V_t^{VF}=(V_{\theta}(s_t)-V_t^{target})^2$ 



<img src="/Users/youzheng/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/1023136512/QQ/Temp.db/55C98BB2-0A4E-4EA9-B2D4-18CA24F352C6.png" alt="55C98BB2-0A4E-4EA9-B2D4-18CA24F352C6" style="zoom:60%;" />




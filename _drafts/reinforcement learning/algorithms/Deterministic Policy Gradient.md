##### DPG (Deterministic Policy Gradient)

***

何为Deterministic 呢？这是相对stochastic而言的，所谓的stochastic是指policy输出的是一个action概率分布$\pi_{\theta}(a|s)$, action从该概率分布中进行采样； 而Deterministic policy输出的是一个确定性的action值$\mu_{\theta}(s)$。

为什么在拥有Stochastic Policy方法的情况下还要追求Deterministic policy ？这是因为Stochastic Policy多了一个sample action的步骤，需要更多的样本来覆盖action的空间，如果action维度空间很高，Stochastic Policy就变得非常低效，尤其在面临action为continuous时。

DPG采用的是Actor-Critic框架形式，如下图所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200722234109682.png" alt="image-20200722234109682" style="zoom:45%;" />

和一般的Actor-Critic算法不同在于：Actor的输出会作为Critic部分的输入，存在嵌套关系。

假设: Actor的策略函数为 $\mu_{\theta}(s)$，Critic的Q函数为$Q(s, a)$。因为$\mu_{\theta}(s)$的输出值是唯一的，所以在知道状态 $s$ 的情况下，a也是确定的可以通过 $\mu$ 得到，因此Critic的Q函数可以表示为：$Q^{\mu}(s, \mu_{\theta}(s))$，所以虽然是Actor-Critic的架构，但是本质上在优化Q函数的同时通过求导的链式法则可以优化$\mu$ 函数。

其梯度更新策略为：
$$
\theta^{k+1}=\theta^{k}+E_{s \sim \rho(\mu)}[\bigtriangledown_{\theta}\mu_{\theta}(s)\bigtriangledown_{a} Q^{\mu^k}(s, a)|_{a=\mu_{\theta}(s)}]
$$

$deterministic$ 策略由于每次都只输出一维的确定性的action，所以不具备exploration的能力，为了解决该问题采用off-policy的策略。

***

##### DDPG (Deep DPG)

DDPG是通过将DPG中的Actor和Critic函数通过Neural Network近似发展而来。通过加入Replay Buffer来解决样本直接关联性问题，

- ER (off-policy)
- Target Network soft update
- Noise for exploration
- Batch Normalization

和一般的Actor-Critic架构不同在于，Critic部分依赖actor

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200716221451311.png" alt="image-20200716221451311" style="zoom:50%;" />



***

##### MADDPG (Multi-agent DDPG)



***

##### D4PG (Distributed Distributional DDPG)



***

##### TD3 (Twin Delayed DDPG)

TD3算法引入Double-DQN中的Double Q函数 ($Q_{\theta1}$ 和 $Q_{\theta2}$) 的评估方式来解决overestimation的问题，在进行bellman方程迭代评估target值时采样取最小值的方式：
$$
y = r + \gamma \min_{i=1,2} Q_{\theta_{i}}(s_{t+1},\mu_{\theta'}(s_{t+1}))
$$
同时和DDPG一样：

- target network soft update策略增加算法稳定性
- 在进行action选择的时候加入随机噪声来进行exploration
- 采取batch normalization来平滑效果

算法的整体流程如下所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200723213048531.png" alt="image-20200723213048531" style="zoom:45%;" />

***


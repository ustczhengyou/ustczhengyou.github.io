##### Policy Gradient (PG)



相比于value-based通过评估$(s,a)$的value值作为策略 $\pi$ 依据的方法，PG则是直接显示优化策略 $\pi$。

- **PG的推导**

RL的目标是最大化期望收益，即：
$$
\theta^*=\mathop{\arg\max}_{\theta}E_{\tau\sim p_{\theta}}(\tau)[\sum_tr(s_t,a_t)] \tag{1.1}
$$
影响一条轨迹 $\tau$ 的形成包括两个因素：动作策略$\pi_{\theta}$ 和状态转移概率 $p(s'|s, a)$，，如下(2)式所示
$$
\pi_{\theta}(\tau)=p_{\theta}(s_1,a_1,...,s_T,a_T)=p(s_1)\prod_{i=1}\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t) \tag{1.2}
$$
将(1)式中目标部分写成积分形式：
$$
J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=\int \pi_{\theta}(\tau)r(\tau)d_{\tau}
$$
实际应用中采用无偏采样的形式来近似期望函数，所以上式变换可得：
$$
J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=E_{\tau \sim \pi(\theta)}[\sum_{t}r(s_t,a_t)]\approx \frac{1}{N}\sum_{i}\sum_tr(a_{i,t},s_{i,t}) \tag{1.3}
$$
对$J(\theta)$进行求导，同时根据 $\bigtriangledown_{\theta}\pi_{\theta}(\tau)=\pi_{\theta}(\tau)\frac{\bigtriangledown_{\theta}\pi_{\theta}(\tau)}{\pi_{\theta}(\tau)}=\pi_{\theta}(\tau)\bigtriangledown_{\theta}\log{\pi_{\theta}(\tau)}$ 变换：
$$
\bigtriangledown_{\theta} J(\theta)=\int \bigtriangledown_{\theta}\pi_\theta(\tau)r(\tau)d_{\tau}=\int \pi_{\theta}(\tau) \bigtriangledown \log\pi_{\theta}(\tau)r(\tau)d_{\tau} =E_{\tau \sim \pi_{\theta}(\tau)}[\bigtriangledown\log\pi_{\theta}(\tau)r(\tau)] \tag{1.4}
$$
将(2)式代入(4)式, 同时结合(3) 式得到梯度：
$$
\bigtriangledown_{\theta} J(\theta)\approx \frac{1}{N}\sum_{i=1}^N(\sum_{t=1}^T\bigtriangledown_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t}))(\sum_{t=1}^Tr(s_{i,t},a_{i,t})) \tag{1.5}
$$
和maximum likelihood优化方法的梯度形式相比，PG的梯度函数多了轨迹 $\tau$ 的reward项作为权重，符合直观的认知【添加一下说明】。

***

##### REINFORCE

根据上面的PG推导，可以得到最原始的**PG**算法**REINFORCE**：



1. *Initialize $\theta$ at random*

2. *sample $\{\tau^i\}$ from $\pi_{\theta}(a_t|s_t)$*

3. *$\bigtriangledown_{\theta}J(\theta) \approx \sum_i(\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i))(\sum_tr(s_t^i,a_t^i))$*

4. *$\theta \leftarrow \theta + \alpha \bigtriangledown_{\theta}J(\theta) $*

   

- **Reduce Variance**

  从**REINFORCE** 算法的梯度计算公式可以看出 (1) $\tau$ 的reward期望采用抽样方式进行近似 (2) 使用 完整$\tau$ 的reward值进行加权，来体现对高reward值的 $\tau$ 的偏好。这两点都会导致梯度的High-Variance。因为采样成本或者交互过程中state(action)的探索空间限制，通常情形下采样的 $\pi$ 的数量是不够的，即样本不能很好体现 $\tau$ 的分布情况，这就自然引入了variance。同时，假设存在着reward都为正的情况，那么 $\tau$ 累计的reward都是正的，这样所有的 $\tau$的概率都会上升，那么随着抽样的$\tau$ 增加，相对更好的 $\tau$的概率也会下降。针对以上两点，产生了两种减少variance的方法。

  - Causality

    根据(5)式的梯度公式等于每一条trajectory的reward之和与其完整轨迹的概率乘积，但实际情况是在 $t$ 时刻(step)发生的action并不能影响 $t$ 之前的reward，所以实际计算中应该将 $t$ 之前reward去掉。同时很容易证明一点：当一个分布中的值都减小时，该分布的variance scale将会减小。因此梯度公式变为：
    $$
    \bigtriangledown_{\theta}J(\theta) \approx \sum_i\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i)(\sum_{t'=t}r(s_{t'}^i,a_{t'}^i)) \tag{6}
    $$

  - Baselines

    针对第二个问题，一种自然的想法是使得 $\tau$ 的累积reward有正有负，正的 $\pi$ 概率上升，负的下降。如何操作呢？产生一个baseline作为基准，将所有的 $\pi$ 的reward都减去它。PG公式变为：
    $$
    \bigtriangledown_{\theta}J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)(r(\tau)-b)
    $$
    同时可以证明引入baselines后并没有增加bias(证明:)，其中一个较好的 $b$ 是所有 $\tau$ 的累积reward的算术平均：$b=\frac{1}{N}\sum_{i=1}^Nr(\tau)$ 。使得variance的最小b为：
    $$
    b = \frac{E[\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)^2(\tau)]}{E[\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)^2]}
    $$
    证明见：

    

***

##### Off-Policy Policy Gradient

on-policy的策略存在着样本利用率低的问题(sample inefficient)，为了提升样本的利用率，一种很自然的想法是把之前的采集的样本都能利用起来，即转向off-policy的策略。前面提到了off-policy的概念，即用于采样的策略 (behavior policy)和优化的策略(target policy)不同。根据强化学习的交互性特点，不同的action策略会导致采样结果分布的差异，因此需要方法来修正这种差异。一种方法就是 **importance sampling**：

- **Importance sampling (IS)**

  函数$ f(x) $ 中 $ x$ 服从$p(x)$分布(即：$x\sim p(x)$)，和 $x$ 服从$q(x)$的分布(即：$x\sim q(x)$)，两者之间期望的关系如下：
  $$
  \begin{align}
  E_{x\sim p(x)}[f(x)]=\int p(x)f(x)d_x\\
  =\int \frac{q(x)}{q(x)}p(x)f(x)d_x\\
  =\int \frac{p(x)}{q(x)} q(x)f(x)d_x\\
  =E_{x\sim q(x)}[\frac{q(x)}{p(x)}f(x)]
  \end{align} \tag{3.1}
  $$

- **IS in Policy Gradient**

假设使用策略 $\pi_{\theta}(\tau)$ 采样来更新策略$\pi_{\theta'}(\tau)$ ，根据(2)式：
$$
\frac{\pi_{\theta}({\tau})}{\pi_{\theta'}(\tau)}=\frac{p(s_1)\prod_{t=1}^T\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)}{p(s_1)\prod_{t=1}^T\pi_{\theta'}(a_t|s_t)p(s_{t+1}|s_t,a_t)}=\frac{\prod_{t=1}^T\pi_{\theta}(a_t|s_t)}{\prod_{t=1}{\pi}_{\theta'}(a_t|s_t)} \tag{3.2}
$$
我们将(3.2)式带入到PG的优化目标函数中得到：
$$
J(\theta')=E_{\tau \sim \pi_{\theta}}[\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}r(\tau)] \tag{3.3}
$$
对其进行求导：
$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\frac{\bigtriangledown_{\theta'}\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}r(\tau)]=E_{\tau \sim \pi_{\theta}}[\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau) r(\tau)] \tag{3.4}
$$
根据前面提到的reward计算的Causality以及未来的action并不影响当前的分布差异，对$\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}$ 和 $r(\tau)$进行裁剪得到：
$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\sum_{t=1}^T\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau)(\prod_{t'=1}^t\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)})(\sum_{t'=t}^Tr(s_{t'},a_{t'}))]\tag{3.5}
$$
因为连乘形式( 指数项:$\prod_{t'=1}^t\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}$)的存在很可能会出现值爆炸或者消失的情况，所以需要对该项的计算进行变换。我们变换一种形式，当前是需要对state-action的联合概率求期望，可以写成先对state求期望，然后对action求期望，所以(3.3)可以变换为：
$$
J(\theta')=\sum_{t=1}^TE_{s_t\sim p_{\theta}(s_t)}[\frac{p_{\theta'}(s_t)}{p_{\theta}(s_t)}E_{a_t \sim \pi_{\theta}(a_t|s_t)}[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}r(s_t,a_t)]]
$$
因为$\frac{p_{\theta'}(s_t)}{p_{\theta}(s_t)}$ 未知忽略掉这部分，那么Off-policy的梯度就变成了：
$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\sum_{t=1}^T\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau)\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}(\sum_{t'=t}^Tr(s_{t'},a_{t'}))]
$$


***

##### Actor-Critic

前面提到的REINFORCE算法的梯度计算，$\sum_ir(s_i,a_i)$部分，是基于蒙特卡洛采样方法得到的，需要一条完整轨迹 $\pi$ 后才能。即回合制更新。这种形式容易受到场景限制，影响更新迭代效率和收敛的速度。因此，一种策略是利用value-based方法的思路来近似评估值，即Critic部分。而算法的Actor即原先的PG算法。也就是说Actor-Critic算法引入两组近似函数，即策略函数的近似$\pi_\theta(a|s)$ 和价值函数的近似 $v(s)$ 或 $q(s,a)$。

这就和前面提到的GEI对应起来了，Critic对应policy evaluate，Actor对应policy improvement。 

目前Critic部分的评估有以下几种形式:

- 基于状态价值

$$
\delta(t)=V_w(s_t)
$$

- 基于状态-动作价值

$$
\delta(t)=Q_w(s_t,a_t)
$$

- 基于TD ($\lambda$) 误差

$$
\delta(t)=\sum_{t'=t}^{\lambda} r_{t}+\gamma V_w(s_{t+\lambda})-V_w(s_{t}) \quad or \quad \delta(t)=\sum_{t'=t}^{\lambda} r_{t}+\gamma Q_w(s_{t+\lambda},a_{t+\lambda})-Q_w(s_{t},a_{t})
$$

- 基于Advantage function

$$
\delta(t) = A(s,a,t)=Q_{\theta}(s_t,a_t)-V_{w}(s_t)
$$

梯度更新公式为：
$$
\theta = \theta + \alpha \log \pi_{\theta}(a_t|s_t) \delta(t) \tag{7}
$$
在Actor-Critic 算法中：

**Actor**：根据Critic的结果来更新策略函数 $\pi_{\theta}(a|s)$ 参数 $\theta$

**Critic**：更新value-function的参数$w$，即$Q_w(s, a)$或者$V_w(s)$

基于Q-function的算法示例流程如下：

1. *Initialize $s$, $\theta$, $w$ randomly;  choose action $a$ based $\pi_{\theta}(a|s)$*

2. *For t = 1… T:*

   1. *Get $r_t$ and next state $s'$ based $a$ , get tuple $(s_t, a_t, r_t, s')$* 

   2. *Choose action $a'\sim \pi_{\theta}(a'|s')$* 

   3. *Update policy parameters: $\theta = \theta +\alpha_{\theta}\bigtriangledown_{\theta}\log\pi_{\theta}(a_t|s_t)Q_w(s_t,a_t)$*

   4. *Compute TD error $\delta(t)=r_t+\gamma Q_{w}(s',a')-Q_w(s,a)$ and updete the $w$*

      *$$w←w+\alpha_w\delta(t)\bigtriangledown Q_w(s_t,a_t)$$*

   5. *Update $a\leftarrow a′$ and s←s′*

***

##### A3C (Asynchronous Advantage Actor-Critic)

A3C算法为了提升训练速度采用了异步训练的思想，即利用多个线程，每个线程相当于一个单独的agent和环境进行交互进行探索，各个线程中的actor和critic梯度的计算和参数的更新类似分布式训练中的ps(parameter servers)架构。维护一个global 的actor的参数 $\theta$ 和 critic的参数 $w$， 各个线程定期拷贝 $\theta$ 和 $w$ 作为local $\theta'$ 和 $w'$ 初始值和local的环境进行交互生成样本并计算累积梯度，然后将累积后梯度值同步到global的$\theta$ 和$w$进行参数更新，如此循环。

<img src="https://greentec.github.io/images/rl4_5.png" alt="img" style="zoom:66%;" />

A3C的异步训练方式的不仅增加了采样速度，加快训练和收敛速度，同时相比于DQN采用Replay Buffer存储样本的随机抽样训练方式来弱化样本之间的correlation，A3C通过多个thread独立地和环境交互采样训练同步梯度的方式达到了同样的效果，并更节约存储空间。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200719211338496.png" alt="image-20200719211338496" style="zoom:50%;" />

***

##### A2C (Advantage Actor-Critic)

和A3C区别在于去掉了Asynchronous，即local agent之间和global network 参数更新采用了同步的方式。A3C的各个thread对应的local agent除了定期从global 的parameter中copy参数，相互之间独立，梯度更新和初始policy存在不一致。为了解决不一致问题，A2C中的协调器在更新全局参数之前等待所有并行参与者完成其工作，然后在下一次迭代中并行参与者从同一策略开始。 同步的梯度更新使训练更具凝聚力，并有可能使收敛更快。



![A2C](https://lilianweng.github.io/lil-log/assets/images/A3C_vs_A2C.png)

事实证明，A2C可以更有效地利用GPU，并且在大批量处理时可以更好地工作，同时实现与A3C相同或更好的性能。

***

##### ACER (Actor-Critic with Experience Replay)

DQN算法中的 *Experience replay* 的机制使得其具有弱化样本之间的关联性同时又提升sample efficient 的能力，但是面临着两个问题：1. DQN的确定性优化策略限制了其在对抗领域中的作用；2. 面对高维动作空间场景时贪婪搜索最优动作的属性的成本无疑是巨大的。因此ACER算法将Experience replay的机制融入到A3C算法中形成off-policy 版本的A3C。为了更好地融合引入了3个tricks。下面一一介绍。



***

##### SAC (Soft Actor Critic)

前面提到的A3C，TRPO，PPO算法都是基于on-policy策略，on-policy策略算法的最大问题是sample efficient较低；而采用off-policy的DDPG面临的存在的问题是训练的稳定性不够，收敛性差对超参数比较敏感。因此提出了SAC算法，该算法采用Actor-Critic框架，融合off-policy策略保证sample efficient，同时最大化累积reward的同时最大化策略熵来增加算法exploration的能力提升训练的稳定性。

算法的reward函数为：
$$
\hat{r}(s_t,a_t)=r(s_t,a_t)+\alpha H(\pi(.|s_t))
$$
**Critic**: 该部分使用两个value funtion和两个Q-value function，分别表示为$\psi$ 和 $\psi'$ ，以及$\theta_1$和$\theta_2$。其中作为target value-function的$\psi'$并不参与训练，但是其更新采用和DDPG一样的soft更新策略: $\psi' \leftarrow \xi\psi + (1-\xi)\psi'$。$\psi$ 参与训练，方式和传统的value-function一样最小化估计值和目标值之间的平方差，不一样的地方在于加入前面提到的策略熵即(policy entropy) ，即：
$$
J_v(\psi)=E[\frac{1}{2}(v_{\psi}(s_t)-E[Q_{\theta}(s_t,a_t)-\log\pi_{\pi}(a_t|s_t)]^2)
$$
对于$\theta_1$ 和 $\theta_2$ 同样目标函数为最小化估计值和目标值之间的平方差，注意为了训练稳定性目标值的计算中采用 $\psi'$ 而非 $\psi$ ，即 $\hat{Q}(s_t,a_t)=r(s_t,a_t)+\gamma E[V_{\psi'}(s_{t+1})]$。

**Actor：**这部分的目标函数采用 $\pi$ 和 Q-function之间的KL-divergence，即：
$$
\begin{align}
J_{\pi}(\phi)=E_{s_t\sim D}[D_{KL}(\pi_{\phi}(.|s_t)||\frac{\exp(Q_\theta(s_t,.))}{Z_{\theta}(s_t)})]\\
=E_{s_t\sim D, a_t \sim \pi_{\phi}}[-\log(\frac{\pi_{\phi}(a_t|s_t)}{\exp(\frac{1}{\alpha}Q_{\theta}(s_t,a_t)-\log Z(s_t))})]\\
=E_{s_t\sim D, a_t \sim \pi_{\phi}}[\log \pi_{\phi}(a_t|s_t)-\frac{1}{\alpha}Q_{\theta}(s_t,a_t)+\log Z(s_t)]
\end{align} \tag{8}
$$
单纯通过采样我们并不能覆盖所有的 $a_t$ ，所以通过action产出加入noise生成action的分布值的策略，这里action我们采用reparameterization trick来得到，即：
$$
a_t=f_{\phi}(\xi_t;s_t)=f_{\phi}^{\mu}(s_t)+\xi_t\odot f_{\phi}^{\sigma}(s_t)
$$
$f$ 函数输出平方值和方差，然后$\xi$ 是noise，从$N(\mu;\sigma)$中进行采样。然后将$a_t$带入(8)式中求导。

整个算法如下：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200720204304538.png" alt="image-20200720204304538" style="zoom:50%;" />



***

**ACKTR (Actor Critic Kronecker-Factored Trust Region)**

ACKTR融合了三种不同的技术：同样是以actor-critic算法为基础，加入TRPO来保证稳定性，同时融入了提升样本效率和和可拓展性的分布式Kronecker 因子分解(Kronecker factorization)。



***





***


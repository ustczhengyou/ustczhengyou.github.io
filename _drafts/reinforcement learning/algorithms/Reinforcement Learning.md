### Introduction

*Deepmind* 的 *AlphaGo*在围棋的出色表现，使得强化学习(**Reinforcement Learning** )的名声大噪，随后结合强化学习和深度学习的深度强化学习(**Deep Reinforcement Learning** )被更广泛地研究和发展，并且在游戏、搜索、广告、派单等场景中成功落地。

强化学习要解决的是序列决策问题(**Sequential Decision Making**)，通过代理 (**Agent**) 来不断和环境 (**Environment**)交互(**Action**)来最大化收益(**Reward**) 的过程。其结构及交互过程如下图所示：

<img src="/Users/youzheng/Documents/Blog/_image/rl/rl_process.jpg" style="zoom:45%;" />

对于交互过程中每个step来讲，在 $t$ 时刻：

**角色：Agent**

receive **observation ** $o_t$ =>  extract  **state** $s_t$  =>  choose **action** $a_t$  =>  receive **reward** $r_t$ and  next  **observation** $o_{t+1}$。

**Environment:** 

receive **action** => emits  **observation** $o_{t+1}$ and **reward** $r_t$

**注意：**observation和state是不同概念，observation是指在 $t$ 时刻agent观察environment的信息，但state是指agent在$t$用于决策的信息，有可能包括observation的历史信息或者其他信息，为了简化通常省略observation，只用state表达。

如上过程就形成了一条轨迹 (**trajectory**(或**episode**) $\tau$ ): 
$$
s_1, a_1, r_1, s_2, a_2, r_2,...,s_t,a_t,r_t,...,s_T \tag{1}
$$
RL的目的是学习如何做最优决策 $\pi$ 使得轨迹的累积收益期望最大化 $E_{\pi}[\sum_{t=1}^{T} r_t]$。



### Key Concepts

作为RL的最重要组成部分，一个agent的可能有如下的一个活几个部分组成：

- **Policy**

  用于action的决策，表达state到action的映射关系。根据输出结果的形式，分为确定性策略(Deterministic policy)和随机性策略(Stochastic policy)。

  - **Deterministic policy**: $a=\pi(s)$ ，只输出确定唯一的值
  - **Stochastic policy**：$\pi(a|s)$，输出的是一个概率分布，action的具体值从该分布中抽样。

  根据样本采集的策略(**behavior policy**)和参数优化的策略(**target policy**)是否同一个分为**On-policy**和**Off-policy**，其中On-policy是两者一致，而Off-policy是指两者是不同的策略。

- **Value function** 

  用于表示一个state或者在一个state下action的好坏，通常分布用$V(s)$ 和 $Q(s,a)$ 表示。

  - **Value function**：对于future reward的预测函数，用来评估一个state的好坏。不能用来直接做action的决策。
    $$
    V_{\pi}(s)=E_{\pi}[r_{t+1}+\gamma r_{t+2} +\gamma^2r_{t+3}+...|s_t=s]
    $$

  - **State-action Value function**：同样也是用于future reward的预测函数，用来评估在一个state下采取某个action的好坏，可以直接作为action的选择的依据。
    $$
    Q_{\pi}(s,a)=E_{\pi}[r_{t+1}+\gamma r_{t+2} +\gamma^2r_{t+3}+...|(s_t=s,a)]
    $$

  - **Advantage function**: 就是同一个state下，Q-function和V-function的差值，体现的是这个动作带来的收益的lift。
    $$
    A_{\pi}(s,a)=Q_{\pi}(s,a)-V_{\pi}(s)
    $$

  上述的两式中参数 $\gamma \in [0,1]$ ，是折扣参数(discounting factor)用于惩罚未来的奖励。使用discount的原因：通常情况下，未来收益是具有不确定性的，需要一个权重来调节短期收益和长期收益的权重来满足不同的偏好；同时在无限循环或者step很长的environment中，累积reward的值要能够保证收敛来使得策略评估有效。例如，在单独reward都为正并且轨迹无限长(或循环)的情况下，任何策略的期望收益都趋向于无穷或者一个没有多大区分度下比较大的数。

- **Model** 

  agent表达environment的，用于预测下一个state或reward。根据agent是否知道model分为**model-based**和**model-free**。假设model-based情况下，相应的state转移概率和reward函数如下：
  $$
  \mathcal{P}_{ss'}^a=P[s_{t+1}=s'|s_t=s,a_t=a] \\
  \mathcal{R_s^a}=E[r_{t+1}|s_t=s,a_t=a]
  $$

从RL过程可以看出，agent是可控的，而environment是不可控的，在对问题建模的时候通常是优化可控的部分来实现目标的最大化。因此RL优化算法的类别可以根据agent的类别来分类，类别如下图所示：

<img src="/Users/youzheng/Documents/Blog/_image/rl/rl_algo_classify.jpg" alt="rl_algo_classify" style="zoom:45%;" />

根据model是否已知分为model-based和model-free算法。根据action决策依据的函数类型分为Value-Based，Policy-Based和Actor-Critic三种类型，其中Value-Based基于Value-function即$V(s)$或者$Q(s,a)$，Policy-based基于$\pi(a|s)$函数，而Actor和Critic结合两者。



### Markov Decision Processes (MDPs)

几乎所有的RL问题都可以被定义成MDPs问题。MDP一个重要的性质是马尔可夫性，即影响未来的只和当前的state有关和历史无关。用条件概率分布表示为：
$$
P[s_{t+1}|s_t]=P[s_{t+1}|s_1,...,s_t]
$$
从另外一个方面讲就是当前的state已经包含用于未来决策的所有信息，所以才有和之前的observation概念的区别。

一个MDP过程包含5个要素$\mathcal{M}=<\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma>$，这些要素和前面介绍强化学习中的一些概念对应：

- $\mathcal{S}$ ：状态的集合

- $\mathcal{A}$：动作的集合

- $\mathcal{P}$：状态转移概率矩阵或函数

- $\mathcal{R}$：奖励函数

- $\mathcal{\gamma}$ ：奖励折扣因子

  

### The anatomy of RL algorithms

抛开具体细节，一般的强化学习问题都可以抽象为三大步骤的迭代过程，三大步骤包括generate-samples，policy-evaluation和policy-improvement。其结构和流程如下图所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200728173411765.png" alt="image-20200728173411765" style="zoom:35%;" />

#### Generate-samples

RL问题是序列决策问题，优化目标是寻找最优策略来使得收益最大化。在监督学习，非监督学习的场景中训练样本是一次性采集然后同一批数据进行训练迭代。其假设样本的分布是固定的且服从独立同分布，但是在RL问题中，涉及到一个环境的交互过程，一个trajectory的样本前后相互依赖，同时样本分布随着agent本身的策略改变。这一阶段的优化围绕如何 **加快采样速度，提升样本的利用率，增加样本的覆盖度(exploration)，解决样本的correlation**等问题。主要涉及的技术：

##### Off-policy

前面提到了on-policy和off-policy的概念，所谓的off-policy是指采集样本的策略和优化目标策略不是同一个策略。在某些场景中，agent和environment是交互是需要成本的，或者存在

##### Experience Replay

该通过存储-采样的方法将历史样本和当前样本混合存储并随机抽样。这种经验回放机制主要克服了经验数据的相关性和非平稳分布问题。

###### Vanilla Experience Replay

Experience Replay机制通过将交互采集的样本及附属信息存储在buffer中形成经验池，算法迭代训练从经验池中采样产生训练样本，经验池的样本更新遵循一定的规则(例如，先进先出)。这种机制：

- *sample correlation*：消除了样本之间的关联性，并且使得训练数据分布变得平滑，不容易陷入局部最优。
- *sample efficiency*：增加了样本的利用率，每个样本都可能被多次使用。

该方法在某些方面受到限制，存储样本的内存缓冲区大小总是有限的，并且有新旧样本的替换机制，所以经验池中的样本并不能保留和覆盖真实的样本空间。 一种策略是在有限的空间中仅可能保留重要的样本，即对样本进行加权，所以引出PER (Prioritized Experience Replay)。



###### PER (Prioritized Experience Replay)

既然要给训练样本加权重，那权重(Prioritized) 如何定义呢？一种基本的想法是权重和reward挂钩，即保留轨迹中那些让reward前后步产生明显差异的样本。因此有如下**PER**策略：

- 首先计算 TD-error
  $$
  TD=|Q(s_t,a)-Q(s_{t+1},a)|
  $$

- 得到replay buffer中样本的抽样概率
  $$
  p_i= \frac{(TD_i+\epsilon)^{\alpha}}{\sum_{k=1}^{N}(TD_i+\epsilon)^{\alpha}}
  $$

- 为了克服高 $p_i$的样本多次重复采样带来的overfit 的问题，所以在*training loss*中加入*importance*参数

  $Importance = (\frac{1}{p_i}*\frac{1}{memory\quad size})^{b}$  ，其中 $b$ 由0逐渐到1
  $$
  J=\frac{1}{m}\sum(y-y_{targat})^2*Importance
  $$



###### Ape-X (Distribution Prioritized Experience Replay)

从命名上可以看出，**Ape-X**和PER的区别在于拓展到分布式形式。拓展的理念来自于训练数据的增加有助于模型更有效的学习。个人觉得ApeX是更准确来讲是一种框架，用到了Replay Buffer并且对采用PER策略的算法都可以嵌套入到其中，例如DQN，double-DQN，DPG等。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200721211828488.png" alt="image-20200721211828488" style="zoom:50%;" />

其Ape-X结构如上：采用多个actor，一个learner，一个share memory的Replay Buffer。actor用于和local environment 进行交互产生经验样本，并且定期将样本同步到share replay buffer中和复制learner网络参数值；learner用于从share replay buffer的采样进行训练，并将训练后的参数定期同步给actor。具体流程和逻辑入下图伪码所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200721213309192.png" alt="image-20200721213309192" style="zoom:50%;" />

从试验结果上看Ape-X 的DQN远好于Rainbow，DQN以及Prioritized DQN。其原因在于，分布式的actor采样形式使得采样效率得到很大提升，缩短了采样时间；同时，不同的actor和local environment独立交互的形式使得提升了样本的覆盖度，起到了对样本空间explore的作用，再配合PER策略又提升样本利用率，出现了训练时间缩短并且效果提升的结果。



###### HER (Hindsight Experience Replay)

在有些场景中和环境交互面临着奖励稀疏的问题，例如，在围棋对抗中只有在分出胜负的时候才能知道真正的reward。为了解决奖励稀疏问题，一种方法是采用shaped reward方法重塑reward 函数。HER的提出也是为了解决奖励稀疏问题。

HER的思想很简单：既然reward稀少除了trajectory结束获得收益作为目标外，能否将这中间的某些state(里程碑)也作为我们的目标和参照物呢？就向跑马拉松可以将整个过程划分成好几个阶段，每个阶段都会有标志性的参照物。基于此思想HER会在trajectory完成后，回过头选取某些state作为目标参照物，根据$r(s_t,a_t,g)$ 函数重新得到reward假如到replay buffer中，和一般的ER中存储的$(s_t, a_t, r_t, s_{t+1})$不同，HER中样本形式为$(s_t||g, a_t, r_t, s_{t+1}||g)$。

<img src="/Users/youzheng/Documents/Blog/_image/rl/HER.jpg" alt="image-20200721213309192" style="zoom:50%;" />

**HER** 机制如上图红色方框所示。在第一遍trajectory完成后，第二遍完成**HER**的生成。其中的核心点在于如何采样$g'$ ，以及如何设计$r(s,a,g')$ 函数。

- $g'$ 选择
  - 未来模型(future)：遍历到$s_t$ ,  从后续step中抽取$k$个$s$ 作为目标
  - 回合模式(episode): 遍历到$s_t$ ,  从该episode中抽取$k$个$s$ 作为目标
  - 多回合模式(multi-episode): 遍历到$s_t$ ,  从限定的多个episode中抽取$k$个$s$ 作为目标
  - 最终模式(final)：遍历到$s_t$ ,  以该episode的最后一步作为目标

- $r(s,a,g)$ 设计
  $$
  f_g(s) =[|g-s_object|\le \epsilon] \\
  r(s,a,g)=-[f_g(s')=0]
  $$

##### Distributed

分布式通常指采样多线程或者多进程的方式启动多个agent(或actor)和对应的local environment，agent采样过程相互独立，同时用一个或者learner来管理模型参数，根据梯度计算和模型参数更新方式有如下几种类型：

- 采样

  单个agent只负责采样，并不计算参与梯度计算，agent中策略函数$\pi$ 参数定期从global的actor中拷贝。例如上面提到的Ape-x架构。这种形式的优点：加快了采样的速度，同时增加了样本的多样性，某种程度上起到了exploration的作用；缺点是：负责模型训练和梯度计算的agent可能成为瓶颈。因此，引入GPU加快训练过程是一种比较常用的方案，单GPU+多CPU的形式，GPU负责训练，CPU负责采样。

- 采样+梯度计算

  和上面提到多线程或者多进程agent只负责采样作用不同，local agent增加了本地样本梯度计算的部分并将梯度值同步到global的agent中，global agent按照一定的规则更新后将参数同步回local agent。按照global 和local agent 参数的传递和更新机制分为同步(Synchronous)和异步(Asynchronous)两种。

  在每一批次(batch)的训练中，local agent完成本地梯度计算后将梯度值上传到global agent中，如果需要等待其他local agent完成相关操作后或者global agent的参数后开始下一次batch的迭代，这种属于同步方式，例如A2C算法，异步方式不需要等待其他agent即可获取global agent参数，例如A3C算法。如下图所示：

  <img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200729160121915.png" alt="image-20200729160121915" style="zoom:50%;" />

分布式技术在RL问题中应用的会带来其他问题，例如 采样如何存储共享，local agent的策略和global agent的策略不一致等问题。需要同上面提到的其他技术相结合，例如利用ER解决样本存储和共享问题，利用off-policy解决behavior policy和target policy不一致的问题。



#### Policy-evaluation

Policy-evaluation泛指某种策略 $\pi$ 下对state或者state-action的价值评估，即前面提到的$V_{\pi}(s)$，$Q_{\pi}(s,a)$ 和 $A_{\pi}(s,a)$等价值函数。在state空间或者state-action空间比较小的时候，可以通过table的方式来记录这些价值函数的值，用bellman equation等式进行迭代优化。但是当状态空间或着动作空间足够大时，table形式是不可取的，因而采用函数形式(例如，Neural Network)来近似(approximate)真实值，因此评估问题转化为函数拟合问题，具体点是回归问题。拟合问题通常涉及到两个值：评估值(prediction)和真实值(target)，其中target值获取方式有下面将的MC和TD，而这两种方式都离不开Bellman Equations。

- **Bellman Equations**

  贝尔曼方程式是指将价值函数分解为即时奖励和未来价值折扣的一组方程式。
  $$
  \begin{equation}
  \begin{aligned}
  V(s)&=E[G_t|S_t=s]\\
  &=E[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+...|s_t=t]\\
  &=E[r_{t+1}+\gamma G_{t+1}|s_t=s]\\
  &=E[r_{t+1}+\gamma V(s_{t+1})|s_t=s]
  \end{aligned}
  \end{equation}
  $$
  同样的$Q(s,a)$ :
  $$
  \begin{equation}
  \begin{aligned}
  
  Q(s,a)&=E[r_{t+1}+\gamma V(s_{t+1})|s_t=t, a_t=a]\\
  &=E[r_{t+1}+\gamma E_{a\sim \pi}Q(s_{t+1},a)|s_t=s, a_t=a]\\
  \end{aligned}
  \end{equation}
  $$
  假设是在 $\pi$ 的约束条件下，等式就变成了：
  $$
  \begin{equation}
  \begin{aligned}
  V_{\pi}(s)&=\sum_{a \in \mathcal{A}}\pi(a|s)Q_{\pi}(s,a)\\
  Q_{\pi}(s,a)&=r(s,a)+\gamma \sum_{s'\in \mathcal{S}}P_{ss'}^aV_{\pi}(s')  \\
  V_{\pi}(s) &= \sum_{a \in \mathcal{A}}\pi(a|s)(r(s,a)+\gamma \sum_{s'\in \mathcal{S}}P_{ss'}^aV_{\pi}(s'))\\
  Q_{\pi}(s,a)&=r(s,a)+\gamma\sum_{s'\in \mathcal{S}}p_{ss'}^a\sum_{a \in \mathcal{A}}\pi(a|s')Q_{\pi}(s',a)
  \end{aligned}
  \end{equation}
  $$
  从上面的式子可以看出，单独使用一个函数$V$ 或者 $Q$ 需要知道状态转移概率$P_{ss'}^a$和状态集合$\mathcal{S}$ ，这些条件在model-free的环境中是不可知的。在具体概率未知的情况如何求期望呢？很简单的一种思路是进行采样来进行无偏估计，因此就有了第一种方法：蒙特卡洛采样法。

- **Monte-Carlo Methods(MC)**

  MC的思路很简单，就是采用MC抽样的方式才搜集完整轨迹 $\tau$，如(1)式所示，然后利用这些观测值来计算期望收益。首先可以得到：$G_t=\sum_{k=0}^{T-t-1}\gamma^kr_{t+k+1}$，然后利用$V(s)=E[G_t|s_t=s]$ 关系得到$V(s)$以及后续的$Q(s,a)$，如下所示：
  $$
  V(s)=\frac{\sum_{t=1}^T \rceil[s_t=s]G_t}{\sum_{t=1}^T \rceil[s_t=s]}
  $$
  和
  $$
  Q(s,a)=\frac{\sum_{t=1}^T \rceil[s_t=s,a_t=a]G_t}{\sum_{t=1}^T \rceil[s_t=s, a_t=a]}
  $$
  
- **Temporal-Difference Learning (TD)**

  上面的MC方法估计是无偏的，但是要求每个采样轨迹必须是完整的，这个条件就限制了在某些循环或者trajectory很长的场景中应用。面对高维state或action的场景，采样的数量不够往往会导致high variance的结果。为了应对这个问题，TD方法诞生了。

  TD应用在model-free场景中，对$V(s)$和$Q(s,a)$ 函数的预估并不需要完整的episodes采样，而是基于已有预估值采用Bootstrapping进行预估，每一个step都可以更新。MC的Bootstrapping表示方法为:
  $$
  V(s_t) \leftarrow V(s_t)+\alpha[G_t-V(s_t)]
  $$
  而TD中最简单的TD(0)的V-function预测值为:
  $$
  V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1}+\gamma V(s_{t+1})-V(s_t)]
  $$
  MC是基于完整的trajectory采样，所以是low-bias，但是是high-variance的，而TD是基于已有的预估值基础上的所以是high-bias，但是low-variance的，能否将两者结合呢？这就产生了$TD(\lambda)$，其中$\lambda$ 为抽样的step数。公式为：
  $$
  V(s_t) \leftarrow V(s_t) + \alpha [\sum_{t'=t}^{t+\lambda}r_{t'}+\gamma V(s_{t+\lambda})-V(s_t)]
  $$
  

#### Policy-improvement

策略提升(Policy-improvement)依据算法类型的不同内容有所不同，因此也能看出不同算法类型思路的差异。

- Value-based

基于value-based方法在$Q(s_t,a_t)$或者$A(s_t,a_t)$的已知情况下，最优策略很明显，即：
$$
\pi=\arg \max_{\pi}Q_{\pi}(s,a); \pi=\arg \max_{\pi}A_{\pi}(s,a)
$$
但是只是$V_{\pi}(s)$ 已知，并不能直接得到最优策略，还需要状态转移概率 $p_{ss'}^a$，即model-based环境。因此在model-free环境下，一般采用优化$Q_{\pi}(s,a)$函数的方式来得到最优策略，或者结合$V_{\pi}(s)$作为baseline以降低variance得到$A_{\pi}(s,a)$来实现。因此value-based的策略提升优化是隐私(Implicit)通过优化价值函数来实现的。

- Policy-based

而Policy-based算法的思路则是通过显示(Explicit)优化策略函数 $\pi_{\theta}(a|s)$ 来实现的，通过对函数参数$\theta$求导然后运用梯度上升来优化函数$\pi_{\theta}(a|s)$。当然梯度的计算需要前面的Policy-evaluation过程，即价值函数。

- Actor-Critic

最后Actor-Critic算法是上述两者的结合，在policy improvement阶段的内容和policy-based一致。



### Model-free algorithms

按照前面提到的agent能否对environment的state转移概率建模分为model-based和model-free方法。在很多场景中是无法对environment建模的，因此model-free更具有普遍性，前面的policy-improvement的章节已经提到三种算法类型，分别为Value-based，Policy-based以及两者结合的Actor-Critic。本章先会呈现model-free的算法的关系拓扑图，意在说明算法的演进过程。后面会按类别介绍相关经典的算法。

#### The Taxonomy of Model-free algorithms



#### Value-based algorithms

##### Q-learning

Q-learning基于value-based的思想，通过TD来进行策略评估(policy evaluate)得到$Q(s,a)$的值，然后基于$Q(s,a)$进行策略更新(policy improvement)。$Q(s,a)$值采用table形式进行存储，即策略优化过程中需要维护一张 $Q$-table 表。同时为了增加算法的鲁棒性，避免容易陷入局部最优的情况，在交互过程中加入$\epsilon-greedy$策略对state-action空间进行exploration。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200630190309543.png" alt="image-20200630190309543" style="zoom:70%;" />

***

##### DQN

Q-learning 采用table的形式来记录存储 $Q(s, a)$ 值，当面对对于高维状态空间或者高维动作空间(或者连续动作空间)时，这种精确的记录形式面临数据存储的压力。为了解决该问题，DQN采用Deep Learning中 **Neural Network(NN)** 函数来近似 $Q(s, a)$ 的表示。NN拟合 $Q(s, a)$的方式是通过最小化目标值和评估值的平方差，即回归思想。该算法的前提是样本服从独立同分布，但是依据RL特性step相互之间是关联的，为了消除和弱化这种关联性，DQN引入了Experience Replay机制，同时ER的引入也提升了sample efficient。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200617160406797.png" alt="image-20200617160406797" style="zoom:50%;" />

其中 **equation 3 **为：
$$
\bigtriangledown_{\theta_i}L_i(\theta_i)=\frac{1}{N}\sum_i(r + \gamma max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i))\bigtriangledown_{\theta}Q(s,a;\theta_i)
$$

***

##### DDQN (Double DQN)

DQN算法存在着value函数被高估的问题，假如一个$Q(s_{t+1},a)$被高估，那么根据算法中Bellman方程：
$$
Q(s_t,a_t) = r_t + \max_{a}Q(s_{t+1},a)
$$
中采用 $\arg \max$ 策略引发后面一系列高估的问题。为了解决该问题，提出了Double-DQN的算法，即采样两个Q函数：$Q$ 和 $Q'$ 来消除Q值高估的问题。这和target network策略不一样。
$$
Q(s_t,a_t) = r_t+ Q'(s_{t+1}, \arg \max_{a}Q(s_{t+1},a) )
$$
为什么这样的评估策略能够解决高估问题呢？假设 $Q$ 函数高估了$a$动作，那么$Q$ 函数就会选择动作$a$ ，但是在更新$Q$ 函数的时候需要依赖$Q'$，假如$Q'$能够给出适当的值，那么$Q$函数更新后也会回归到正常值。同样如果$Q'$ 存在高估呢？同样也是需要$Q$存在高估。这种相互纠正的策略能在一定程度上解决DQN高估的问题。

***

##### Dueling DQN

在DQN算法中是通过对Q值的预估来实现策略提升，其中Q值可以分解为state的value值和action的advantage值。如果算法能够将这两个值的分离评估，就能更准确学习和刻画Q值两部分的影响比重，到底是state本身价值还是action选择带来价值的提升。基于此在进行action选择的时候就更具鲁棒性和算法的稳定性。Dueling DQN就是基于此思想，通过优化神经网络(Q函数)的结构来优化算法。

Dueling DQN通过Q函数分成两部分：和只和state相关的Value函数 $V(s;\theta,\beta)$，还有和具体action相关的Advantage函数$A(s,a;\theta,\alpha)$，所以Q函数为：
$$
Q(s,a;\theta,\alpha,\beta) =V(s;\theta,\beta)+A(s,a;\theta,\alpha)
$$
网络结构如下图所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200724115220981.png" alt="image-20200724115220981" style="zoom:50%;" />

图中上部网络为正常DQN结构，下部Dueling-DQN，可以看到有两个子网络结构，分别对应Value函数和Advantage函数，最后输出为两部分的线性组合。由于最终输出还是还是无法区分$V$函数和$A$函数的作用，为了体现这种可辨识性(identifiability)，对$A$ 函数进行去中性化的处理，即$A$值剪去$A$均值，最终Q函数的形式如下：
$$
Q(s,a;\theta,\alpha,\beta) =V(s;\theta,\beta)+A(s,a;\theta,\alpha)-\frac{1}{\mathcal{A}}\sum_{d' \in \mathcal{A}}A(s,a',\theta,\alpha)
$$
Dueling DQN的其他流程和其他DQN基本一致。

***

#### Noisy Net (Noisy DQN)

DQN的，所以有必要增加训练样本的exploration。

其中一种方式是在action选择的时候采用 $\epsilon - greedy$ 的方式。还有另外一种方式是Noisy Net。

There are two “classic” approaches to this task:

1. **epsilon-greedy**:  with some probability epsilon (given as hyperparameter) your agent takes a random step instead of acting according to policy it has learned. It’s a common practice to have this epsilon equal to 1 in the beginning of training and slowly decrease to some small value, like 0.1 or 0.02.
2. **entropy regularisation**:  is used in policy gradient methods when we’re adding entropy of our policy to the loss function, punishing our model for being too certain in it’s actions.

Noisy Net的原理：

*The method basically consists of adding the gaussian noise to the last (fully-connected) layers of the network. Parameters of this noise can be adjusted by the model during training, which allows the agent to decide when and in what proportion it wants to introduce the uncertanty in it’s weights.*

***



#### Policy-based algorithms



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

#### Actor-Critic algorithms

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



#### Value Function 算法

##### SARSA

其命令“SARSA”来源于$...,S_{t},A_t,R_{t+1},S_{t+1},A_{t+1},...$ 序列，步骤如下：

**Policy optimization**

- policy iteration methods
  - which alternate between estimating the value function under the current policy and improving the policy
- policy gradient methods
  - which use an estimator of the gradient of the expected return (total reward) obtained from sample trajectories
- derivative-free optimization
  - such as the cross-entropy method (CEM) and covariance matrix adaptation (CMA), which treat the return as a black box function to be optimized in terms of the policy parameters

#### Policy Gradient (PG)算法

相比于前面通过计算Value function来作为策略提升(Policy Improvemnt)的方式，PG则是更显示地直接来优化策略 $\pi$。

##### PG的推导

RL的目标是最大化交互轨迹的reward，即：

$\pi_{\theta}(\tau)=p_{\theta}(s_1,a_1,...,s_T,a_T)=p(s_1)\prod_{i=1}\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)$

$\theta^*=argmax_{\theta}E_{\tau\sim p_{\theta}}(\tau)[\sum_tr(s_t,a_t)]$

可以推导出其目标函数：

$J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=E_{\tau \sim \pi(\theta)}[\sum_{t}r(s_t,a_t)]\approx \frac{1}{N}\sum_{i}\sum_tr(s_{i,t}, a_{i,t})$

将期望形式变成积分形式：

$J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=\int \pi_{\theta}(\tau)r(\tau)d_{\tau}$

然后对$J(\theta)$ 求关于$\theta$ 的导数：

$\bigtriangledown J(\theta)=\int \bigtriangledown_{\theta}\pi_\theta(\tau)r(\tau)d_{\tau}=\int \pi_{\theta}(\tau) \bigtriangledown log\pi_{\theta}(\tau)r(\tau)d_{\tau}$ 

根据 () 式，$log\pi_{\theta}(\tau)=logp(s_1)+\sum_{i} p(s_{t+1}|s_t,a_t)+\sum_i log \pi_{\theta}(a_t|s_t)$

所以得到：

$\bigtriangledown J(\theta)=E_{\tau \sim \pi_{\theta}(\tau)}[(\sum_{t=1}^T\bigtriangledown_{\theta}log \pi_{\theta}(a_t|s_t))(\sum_{t=1}^T(r(a_t,s_t)))]$

$=\underbrace{\frac{1}{N}\sum_{i=1}^N }_{sample}(\sum_{t=1}^T\bigtriangledown_{\theta}log \pi_{\theta}(a_t|s_t))(\underbrace{\sum_{t=1}^T(r(a_t,s_t))}_{policy\;evalution})$

$\underbrace{\theta \gets \theta + \alpha \bigtriangledown_{\theta}J(\theta)}_{policy \; improvement}$

和maximum likelihood的思想对比，目标函数多了trajectory 的reward作为权重，所以优化的目标是reward大的trajectory增大了可能性。



##### REINFORCE

根据上面的PG推导，可以得到最原始的PG算法REINFORCE：

***

1. Initialize $\theta$ at random
2. sample $\{\tau^i\}$ from $\pi_{\theta}(a_t|s_t)$
3. $\bigtriangledown_{\theta}J(\theta) \approx \sum_i(\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i))(\sum_tr(s_t^i,a_t^i))$
4. $\theta \leftarrow \theta + \alpha \bigtriangledown_{\theta}J(\theta) $

***

- REINFORCE算法存在的问题 -> high variance

  ---------- high variance 说明 --------------

  两个思路：

  - Causality

    `poliy at time t' cannot affect reward at time t when t<t' `

    $\bigtriangledown_{\theta}J(\theta) \approx \sum_i(\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i))(\sum_tr(s_t^i,a_t^i))=\sum_i\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i)(\sum_tr(s_t^i,a_t^i))$

    根据 $\bigtriangledown_{\theta}J(\theta) \approx \sum_i(\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i))(\sum_tr(s_t^i,a_t^i))$ 公式，在计算过程中是每一条trajectory和其完整的概率乘积，但实际情况是在 $t$ 时刻发生的action并不能改变之前的reward，所以实际计算中应该将 $t$ 之前reward去掉。

    $\bigtriangledown_{\theta}J(\theta) \approx \sum_i\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i)(\sum_{t'=t}r(s_{t'}^i,a_{t'}^i))$

  - Baselines

##### Actor-Critic

根据Causality的策略，将 $\sum_{t'=t}r(s_{t'}^i,a_{t'}^i)$ 替换成 $Q(s_{t}^i,a_{t}^i)$

$\bigtriangledown_{\theta}J(\theta) \approx \sum_i\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i)Q(s_{t}^i,a_{t}^i) $

***

1. Initialize $s$ , $\theta$, $w$ at random; sample $a \sim \pi_{\theta}(a|s)$

2. For t = 1... T:

   1. Get $r_t \sim R(s, a)$  and next state $s'$ from environment

   2. Then sample the next action $a' \sim \pi_{\theta(a'|s')}$

   3. update the policy parameters: $\theta \leftarrow \theta + \alpha_{\theta}* Q_w(s, a)log \pi_{\theta}(a|s)$

   4. Compute the correaction(TD error) for action-value at time t:

      $\beta = r_t+\gamma Q_w(s',a')-Q_w(s, a)$

      and use it to update the parameters of action-value function:

      $w \leftarrow w + \alpha_{w}\bigtriangledown_{w}Q_w(s,a)$ 

   5. Update $a \leftarrow a'$ and $s \leftarrow s'$.

***

##### Importance Sampling

REINFORCE 和 Actor-Critic 算法都是on-policy的，on-policy会存在样本利用率低的问题。所以很多算法加入**importance sampling (IS)** 策略来将采样与训练变成off-policy。

$E_{x \sim p(x)}[f(x)]=\int p(x)f(x)dx=\int \frac{q(x)}{q(x)}p(x)f(x)dx=\int q(x)\frac{p(x)}{q(x)}f(x)dx=E_{x \sim q(x)}[\frac{p(x)}{q(x)}f(x)]$

如果用 $q$ 策略分布来采样来训练 $p$ 策略，那么样本要增加 $\frac{p(x)}{q(x)}$ 来进行加权变换。



****

**Q-learning: Off-policy TD control**

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200630190309543.png" alt="image-20200630190309543" style="zoom:70%;" />



***

**DQN** 

Q-learning 采用table的形式来记录存储 $Q(s, a)$ 值，对于高维$  S$空间或者高维$A$ 来讲，这种精确的记录形式无法满足需求，所以引入了Deep Learning中Neural Network 来近似 $Q(s, a)$ 函数，这便有了Deep Q-learning (DQN) 的诞生。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200617160406797.png" alt="image-20200617160406797" style="zoom:50%;" />

其中 **equation 3 **为：

$\bigtriangledown_{\theta_i}L_i(\theta_i)=\frac{1}{N}\sum_i(r + \gamma max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i))\bigtriangledown_{\theta}Q(s,a;\theta_i)$

除了将NN用于近似$Q(s,a)$之外，DQN 还引入了 **Experience Replay** 机制。

##### Experience Replay

在DQN中使用Replay buffer来实现 **Experience Replay**，具备的优点：

- 消除了样本之间的correlation，并且使得训练数据分布变得平滑，使得policy不容易陷入local optimization。
- 增加了样本的利用率，每个样本都可能被多次使用。

在Replay Buffer存在：

*This approach is in some respects limited since the memory buffer does not differentiate important transitions and always overwrites with recent transitions due to the finite memory size N.* 

所以对样本进行加权，所以引出PER (Prioritized Experience Replay)。

because it has two important limitations. First, the deterministic nature of the optimal policy limits its use in adversarial domains. Second, finding the greedy action with respect to the Q function is costly for large action spaces.

***

**PER (Prioritized Experience Replay)**

既然要给训练样本加权重，那权重(Prioritized) 如何定义呢？

- 首先计算 TD-error

  $TD=|Q(s_t,a)-Q(s_{t+1},a)|$

- 得到replay buffer中样本的抽样概率

  $p_i= \frac{(TD_i+\epsilon)^{\alpha}}{\sum_{k=1}^{N}(TD_i+\epsilon)^{\alpha}}$

- 为了克服高 $p_i$的样本多次重复采样带来的overfit 的问题，所以在training loss中加入importance参数

  $Importance = (\frac{1}{p_i}*\frac{1}{memory\quad size})^{b}$  其中b由0逐渐到1

   $J=\frac{1}{m}\sum(y-y_{targat})^2*Importance$

***

**APX (Distribution Prioritized Experience Replay)**



***

**HER (Hindsight Experience Replay)**



***

**Double DQN**

DQN存在着overestimate的问题，所以采用将estimate action-value和select action的函数分离的策略。采用两个Q网络，select action网络不断更新，用于estimate action-value的网络定期copy select action网络的参数。

***

**Noisy Net (Noisy DQN)** 

DQN的Improvement是greedy max形式，所以有必要增加training 样本的exploration。

其中一种方式是在action选择的时候采用 $\epsilon - greedy$ 的方式。还有另外一种方式是Noisy Net。

There are two “classic” approaches to this task:

1. **epsilon-greedy**:  with some probability epsilon (given as hyperparameter) your agent takes a random step instead of acting according to policy it has learned. It’s a common practice to have this epsilon equal to 1 in the beginning of training and slowly decrease to some small value, like 0.1 or 0.02.
2. **entropy regularisation**:  is used in policy gradient methods when we’re adding entropy of our policy to the loss function, punishing our model for being too certain in it’s actions.

Noisy Net的原理：

*The method basically consists of adding the gaussian noise to the last (fully-connected) layers of the network. Parameters of this noise can be adjusted by the model during training, which allows the agent to decide when and in what proportion it wants to introduce the uncertanty in it’s weights.*

***

**Dueling DQN**



***

**DPG (Deterministic Policy Gradient)**

DQN可以解决action space为离散(discrete)的RL问题，但是如果action为连续(continious)的呢? DQN基于bellman的policy improvement无法枚举所有的action值，所以就引出了一个直接基于state给出action的确定策略 (Deterministic policy)，即: $a=\mu_{\theta}(s)$。那如何迭代优化该函数呢？

一个基本思想，函数$\mu_{\theta}(s)$ 能够使得state的期望收益最大化，即：

$J(\theta)=\int_{S}\sigma^{\mu}Q(s,\mu_{\theta}(s))$

**Deterministic policy gradient theorem**: Now it is the time to compute the gradient! According to the chain rule, we first take the gradient of the action $a$ and then take the gradient of the deterministic policy function $μ$.

*In the off-policy approach with a stochastic policy, importance sampling is often used to correct the mismatch between behavior and target policies, as what we have described above. However, because the deterministic policy gradient removes the integral over actions, we can avoid importance sampling.*

***

**DPG (Deep Deterministic Policy Gradient)**

DDPG将DQN和DPG结合



***

**Advanced Policy Gradients**

- Policy gradient is a type of policy iteration

  - Understand the **policy iteration** view of **policy gradient**

  1. Why does policy gradient work?

  根据**REINFORCE**其中的loss：$\bigtriangledown_{\theta}\sim \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\bigtriangledown_{\theta} log\pi_{\theta}(a_{i,t}|s_{i,t})A_{i,j}^{\pi}$

  步骤：1. 根据当前的策略 $\pi$ 来评估 $A_{i,j}^{\pi}(s_t, a_t)$ ; 2. 使用$A_{i,j}^{\pi}(s_t, a_t)$ 来得到新的(improved) 策略$\pi'$ 。这就是policy iteration algorithms。

  **申明1：**$J(\theta')-J(\theta)=E_{\tau \sim p_{\theta'}(\tau)}[\sum_t\gamma^tA^{\pi_{\theta}}(s_t,a_t)]$

  The difference between $J(\theta')$ and $J(\theta)$ is the expected advantage. All you have to do is for all the time steps for all states is **maximize the advantage the old policy** and if you can do that you can realize the difference the impovement from new policy minis the old one. 

  **解释：**依据旧策略$\pi$的 advantage function, 新的策略是尽可能让新策略形成的轨迹的advantage期望值近可能大，所以就变成了最大化旧策略评估下的Advantage，也就是REINFORCE的原理。但是为什么就变成了申明1的形式呢？一种解释，每次迭代的轨迹在策略 $\pi$ 确定后也基本确定，所以Advantage每次评估都是在$\pi$的基础上评估的，也就是原先的Value值变(Q-function的期望值分布在变)了。策略$\pi$ 的改变 --> Q(s, a)值的分布在变 --> V(s) 在变 ---> A(s, a)在变 。

  **如何求解呢？** 最大化申明1中的式子呢？

  策略提升的表示方式

  <img src="/Users/youzheng/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/1023136512/QQ/Temp.db/ACFC8296-F95C-43B5-A17E-AAD41B50AB80.png" alt="ACFC8296-F95C-43B5-A17E-AAD41B50AB80" style="zoom:40%;" />

  **申明2：**$p_{\theta}(s_t)$ is ***close*** to $p_{\theta'}(s_t)$  when $\pi_{\theta}$ is *close* to $\pi_{\theta'}$ 。 

- Policy gradient as a constrained optimization

  - Understand how to analyze **policy gradient improvement**

  <img src="/Users/youzheng/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/1023136512/QQ/Temp.db/345D7BE3-64FE-4A6A-8461-9BD8DA58E5DF.png" alt="345D7BE3-64FE-4A6A-8461-9BD8DA58E5DF" style="zoom:40%;" />

  <img src="/Users/youzheng/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/1023136512/QQ/Temp.db/88D3FAF6-6517-4612-9788-3B6155141948.png" alt="88D3FAF6-6517-4612-9788-3B6155141948" style="zoom:40%;" />

- From constrained optimization to **natural gradient**

  - Understand what natural gradient does and how to use it

  <img src="/Users/youzheng/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/1023136512/QQ/Temp.db/160B7803-0DC5-4B6B-A9B6-5B597C1B4F69.png" alt="160B7803-0DC5-4B6B-A9B6-5B597C1B4F69" style="zoom:45%;" />

- Natural gradients and trust regions

**Review**

- Policy gradient = policy iteration
  - Evaluate advantage of **old policy**
  - Maximize advantage w.r.t. (with respect to) **new policy**
- Correct thing to do is **optimize expected advantage** under **new policy state distribution**
- Doing this under old policy state distribution optimizes a bound, *if* the policies are close enough
- Results in ***constrained* optimization** problem

***

Two Limitation of  "Vanilla" Policy Gradient Methods

- Hard to choose stepsizes
  - input data is nonstationary due to changing policy:  observation and reward distribution change
  - Bad step is more damaging than in unsupervised learning, since it affects visitation distribution
- Sample efficiency
  - only one gradient step per environment sample
  - dependent on scaling of coordinates

**Trust Region Policy Optimization: Pseudo code**

For iteration=1, 2, ..., do:

​	Run policy for T times or N trajectories:

​	Estimate advantage funcion at all  timesteps

​	maxmize :  $\sum_{i=0}^N \frac{\pi_{\theta}(a_n|s_n)}{\pi_{\theta_{old}}(a_n|s_n)}A_{i} $

​	subject to: $KL_{\pi_{\theta_{old}}}(\pi_{\theta})\le \gamma$

**Solving KL penalized problem:**

Maximize $L_{\pi_{\theta_{old}}}(\pi_{\theta})-\beta*KL{\pi_{\theta_{old}}}(\pi_{\theta})$

Make linear approximation to $L_{\pi_{\theta_{old}}}$ and quadratic approximation to KL term:

***






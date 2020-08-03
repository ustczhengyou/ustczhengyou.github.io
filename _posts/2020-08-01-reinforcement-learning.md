---
layout: post
title:  "reinforcement learning"
tags:   [reinforcement learning, policy-based, value-based]
date:   2020-08-01
toc: true
typora-root-url: ..
---



## Introduction

Deepmind 的 AlphaGo在围棋的出色表现，使得强化学习(Reinforcement Learning，RL)的名声大噪，随后结合强化学习和深度学习的深度强化学习(Deep Reinforcement Learning，DRL )被更广泛地研究和发展，并且在游戏、搜索、广告、派单等场景中成功落地。

强化学习要解决的是序列决策问题(Sequential Decision Making，SDM)，通过代理 (Agent) 来不断和环境 (Environment)交互(Action)来最大化收益(Reward) 的过程。其结构及交互过程如下图所示：

<div  align="center"> <img src="/assets/images/rl/rl_process.jpg" width="50%" height="50%" alt="RL框架"/> </div>

<center>图 1.1 强化学习中agent同environment交互流程图，图片来源[3]</center>

对于交互过程中每个step来讲，在 $$t$$ 时刻：

**Agent:**

receive observation( $o_t$) $\rightarrow$  extract  state($s_t$)   $\rightarrow$  choose action( $a_t$)  $\rightarrow$  receive reward( $r_t$)  and  next  observation( $o_{t+1}$) $\rightarrow$ extract  state($s_{t+1}$) 。

**Environment:** 

receive action($a_t$) $\rightarrow$ emits  observation( $o_{t+1}$) and reward( $r_t$) $\rightarrow$ receive action($a_{t+1}$)

*注意：*observation和state是不同概念，observation是指在 $t$ 时刻agent观察environment的信息，但state是指agent在$t$用于决策的信息，有可能包括observation的历史信息或者其他信息，为了简化通常省略observation，只用state表达。

如上过程就形成了一条轨迹 (trajectory |episode， $\tau$ ): 


$$
s_1, a_1, r_1, s_2, a_2, r_2,...,s_t,a_t,r_t,...,s_T \tag{1}
$$


RL的目的是学习如何做最优决策 $\pi$ 使得轨迹的累积收益期望最大化 $E_{\pi}[\sum_{t=1}^{T} r_t]$。



## Key Concepts

作为RL的最重要组成部分，一个agent的组成可能包括：用Policy做action决策， 用Value-function评估期望收益，用Model对环境建模。下面来逐个介绍。

### Policy

所谓的策略(Policy)表达的是state到action的映射关系，作为action选择的决策依据。根据输出结果的形式，分为确定性策略(Deterministic policy，DP)和随机性策略(Stochastic policy，SP)。

- **Deterministic policy**

  策略函数只输出确定唯一的值，即 $$a=\pi(s)$$ 。优点：由于action的输出唯一值，因此所需要覆盖action空间的样本变小，能加快收敛速度和稳定性，尤其是面对action为高维离散或者是连续空间的场景。缺点：由于输出单一值缺乏action空间探索，而RL算法的基于迭代优化的特性很容易陷入局部优化的陷阱中，因此需要加强exploration机制。

- **Stochastic policy**

   策略函数输出的是一个概率分布 $\pi(a\|s)$，action的决策值从该分布中进行抽样，因此带有一定的随机性。这种随机性方式本身带有exploration的功能，因此不容易陷于局部最优解，但是在面对高维离散或连续空间时表达能力有限，需要大量的样本覆盖action的空间来进行优化。

根据用于训练样本采集的策略(behavior policy)和优化策略(target policy)是否是同一个策略，将agent的采样方法分为在线策略(On-policy)和离线策略(Off-policy)。On-policy和Off-policy的概念在后面介绍的基于Policy Gradient算法中体现的尤为明显。

- **On-policy**

  behavior policy和target policy是同一个策略。在使用机器学习方法对策略函数(对应策略函数$\pi$)进行优化的时候，其中最基本的假设是训练样本来自同一个分布。由于RL问题的特殊性，样本的分布和$$\pi_{\theta}$$直接相关，不同的策略(即参数$$\theta$$)下采集的样本可能不属于同一分布，因此策略函数的每一轮迭代优化都只能运用对应策略采集的样本来进行优化。

- **Off-policy**

  behavior policy和target policy可以是不同的策略。在很多环境中agent同environment交互成本很高，为了提升样本的利用率，即样本的重复利用；以及后面提到采用分布式采样来提升采样效率，提出了Off-policy的采样策略。Off-policy 需要解决的一个最重要问题是如何消除不同分布样本带来的训练差异，这就引出了重要性采样(Importance Sample，IS)方法。详细见：

  

### Value function

值函数 (value-funtion) 用于表示一个state或者在一个state下action的好坏，通常对应用$V(s)$ 和 $Q(s,a)$ 表示。在state或action的空间为低维的时候可以采样table的形式表示，但是在高维的场景中通常用函数(例如：neural network)来近似。

$V(s)$：对于future reward的预测函数，用来评估一个state的好坏。不能用来直接做action的决策。

$$
V_{\pi}(s)=E_{\pi}[r_{t+1}+\gamma r_{t+2} +\gamma^2r_{t+3}+...|s_t=s]
$$

$Q(s,a)$：同样也是用于future reward的预测函数，用来评估在一个state下采取某个action的好坏，可以直接作为action的选择的依据。

$$
Q_{\pi}(s,a)=E_{\pi}[r_{t+1}+\gamma r_{t+2} +\gamma^2r_{t+3}+...|(s_t=s,a)]
$$

$A(s,a)$: 就是同一个state下，Q-function和V-function的差值，体现的是这个动作带来的收益的lift。

$$
A_{\pi}(s,a)=Q_{\pi}(s,a)-V_{\pi}(s)
$$


上述的三式中都出现了参数 $\gamma \in [0,1]$ ，表示折扣参数(discounting factor)用于调节当前收益和未来收益的权重。使用discount的原因：通常情况下，未来收益是具有不确定性的，需要一个权重来调节短期收益和长期收益的权重来满足不同的偏好；同时在无限循环或者step很长的environment中，累积reward的值要能够保证收敛来使得策略评估有效。例如，在单独reward都为正并且轨迹无限长(或循环)的情况下，任何策略的期望收益都趋向于无穷或者一个没有多大区分度下比较大的数。

### Model 

Model是指agent用来表达environment的信息，包括预测下一个state、reward的概率或值等。根据agent是否知道model分为model-based和model-free。假设model-based情况下，相应的state转移概率和reward函数如下：


$$
\mathcal{P}_{ss'}^a=P[s_{t+1}=s'|s_t=s,a_t=a] \\
\mathcal{R_s^a}=E[r_{t+1}|s_t=s,a_t=a] 
$$


从RL过程可以看出，agent是可控的，而environment是不可控的，在对问题建模的时候通常是优化可控的部分来实现目标的最大化。因此RL优化算法的类别可以根据agent的类别来分类，类别如下图所示：

<div  align="center"> <img src="/assets/images/rl/rl_algo_classify.jpg" width="60%" height="60%" alt="RL算法分类"/> </div>

<center>图 3.1 强化学习算法分类图, 图片来源[3]</center>

如上图所示：根据model是否已知分为model-based和model-free算法；根据action决策依据的函数类型分为value-based，policy-based和actor-critic三种类型，其中value-based基于value-function即$$V(s)$$或者$$Q(s,a)$$，policy-based基于 $\pi (a\|s)$​ 函数，而actor-critic结合两者。



## Markov Decision Processes 

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

  

## The anatomy of RL algorithms

抛开具体细节，一般的强化学习问题都可以抽象为三大步骤的迭代过程，三大步骤包括generate-samples，policy-evaluation和policy-improvement。其结构和流程如下图所示：

<div  align="center"> <img src="/assets/images/rl/general-process.jpg" width="50%" height="50%" alt="RL算法优化流程框架图"/> </div>

<center>图 4.1 强化学习算法优化结构流程图</center>

### Generate-samples

RL问题是序列决策问题，优化目标是寻找最优策略来使得收益最大化。在监督学习，非监督学习的场景中训练样本是一次性采集然后同一批数据进行训练迭代。其假设样本的分布是固定的且服从独立同分布，但是在RL问题中，涉及到一个环境的交互过程，一个trajectory的样本前后相互依赖，同时样本分布随着agent本身的策略改变。这一阶段的优化围绕如何 加快采样速度，提升样本的利用率，增加样本的覆盖度(exploration)，解决样本的correlation等问题。主要涉及的技术：

#### Off-policy

前面提到了on-policy和off-policy的概念，所谓的off-policy是指采集样本的策略和优化目标策略不是同一个策略。详细说明见：Off-policy

#### Experience Replay

经验重放(Experience Replay, ER)是指通过存储-采样的方法将历史样本和当前样本混合存储并在训练过程中从持中随机抽样的过程。ER机制主要克服了经验数据的关性(sample efficiency)和训练样本的非平稳分布问题。

- **Vanilla Experience Replay**

Experience Replay机制通过将交互采集的样本及附属信息存储在内存的buffer中形成经验池，每一轮算法迭代训练从经验池中采样产生训练样本，经验池的样本更新遵循一定的规则(例如，先进先出)。这种机制从一定程度上解决了sample correlation问题以及提升了sample efficiency：

​	*sample correlation*：消除样本之间的关联性，并使得训练数据分布变得平滑，训练过程稳定且不容易陷入局部最优。

​	*sample efficiency*：一个样本可能被多次利用，因此增加了样本的利用率。

该方法在某些方面受到限制，存储样本的内存缓冲区大小总是有限的，并且有新旧样本的替换机制，所以经验池中的样本并不能保留和覆盖真实的样本空间。 一种策略是在有限的空间中仅可能保留重要的样本，即对样本进行加权，所以引出PER (Prioritized Experience Replay)。

- **PER (Prioritized Experience Replay)**

既然要给训练样本加权重，那权重(Prioritized) 如何定义呢？一种基本的想法是权重和reward挂钩，即保留轨迹中那些让reward前后步产生明显差异的样本。PER的大致流程如下：

*Step1:* 计算 TD-error

$$
TD=|Q(s_t,a)-Q(s_{t+1},a)|
$$

*Step2:* 得到replay buffer中样本的抽样概率

$$
p_i= \frac{(TD_i+\epsilon)^{\alpha}}{\sum_{k=1}^{N}(TD_i+\epsilon)^{\alpha}}
$$

*Step3:* 为了克服高 $p_i$的样本多次重复采样带来的overfit的问题，所以在*training loss*中加入*importance*参数$I$，



$$I = (\frac{1}{p_i}*\frac{1}{memory\quad size})^{b}$$ 



其中 $b$ 由0逐渐到1，最终得到：


$$
J=\frac{1}{m}\sum(y-y_{targat})^2*I
$$


- **Ape-X (Distribution Prioritized Experience Replay)**

从命名上可以看出，Ape-X和PER的区别在于拓展到分布式形式。拓展的理念来自于训练数据的增加有助于模型更有效的学习。个人觉得ApeX是更准确来讲是一种框架，用到了Replay Buffer并且对采用PER策略的算法都可以嵌套入到其中，例如DQN，Double-DQN，DPG等。

<div  align="center"> <img src="/assets/images/rl/apex_architecture.jpg" width="75%" height="75%" alt="Ape-X算法框架结构图"/> </div>

<center>图 4.2 Ape-X结构框架图，图片来源[19]</center>

其Ape-X结构如上：采用多个actor，一个learner，一个share memory的Replay Buffer。actor用于和local environment 进行交互产生经验样本，并且定期将样本同步到share replay buffer中和复制learner网络参数值；learner用于从share replay buffer的采样进行训练，并将训练后的参数定期同步给actor。具体流程和逻辑入下图伪码所示：

<div  align="center"> <img src="/assets/images/rl/apex_algo.jpg" width="80%" height="80%" alt="Ape-X框架算法伪码图"/> </div>

<center>图 4.3 Ape-X框架Actor和Learner算法伪码，图片来源[19]</center>

从试验结果上看Ape-X 的DQN远好于Rainbow，DQN以及Prioritized DQN。其原因在于，分布式的actor采样形式使得采样效率得到很大提升，缩短了采样时间；同时，不同的actor和local environment独立交互的形式使得提升了样本的覆盖度，起到了对样本空间explore的作用，再配合PER策略又提升样本利用率，出现了训练时间缩短并且效果提升的结果。



- **HER (Hindsight Experience Replay)** 

在有些场景中和环境交互面临着奖励稀疏的问题，例如，在围棋对抗中只有在分出胜负的时候才能知道真正的reward。为了解决奖励稀疏问题，一种方法是采用shaped reward方法重塑reward 函数。HER的提出也是为了解决奖励稀疏问题。

HER的思想很简单：既然reward稀少除了trajectory结束获得收益作为目标外，能否将这中间的某些state(里程碑)也作为我们的目标和参照物呢？就向跑马拉松可以将整个过程划分成好几个阶段，每个阶段都会有标志性的参照物。基于此思想HER会在trajectory完成后，回过头选取某些state作为目标参照物，根据$r(s_t,a_t,g)$ 函数重新得到reward假如到replay buffer中，和一般的ER中存储的$$(s_t, a_t, r_t, s_{t+1})$$不同，HER中样本形式为$(s_t\|g, a_t, r_t, s_{t+1}\|g)$。

<div  align="center"> <img src="/assets/images/rl/HER.jpg" width="80%" height="80%" alt="HER机制算法伪码"/> </div>

<center>图 4.4 HER伪码; 图片来源[20]</center>

HER机制如上图红色方框所示。在第一遍trajectory完成后，第二遍完成HER的生成。其中的核心点在于如何采样$g'$ ，以及如何设计$r(s,a,g')$ 函数。

- $g'$ 选择

  - 未来模型(future)：遍历到$s_t$ ,  从后续step中抽取$k$个$s$ 作为目标
  - 回合模式(episode): 遍历到$s_t$ ,  从该episode中抽取$k$个$s$ 作为目标
  - 多回合模式(multi-episode): 遍历到$s_t$ ,  从限定的多个episode中抽取$k$个$s$ 作为目标
  - 最终模式(final)：遍历到$s_t$ ,  以该episode的最后一步作为目标

- $r(s,a,g)$ 设计
  
  
  $$
  f_g(s) =[|g-s_object|\le \epsilon] \\
  
  r(s,a,g)=-[f_g(s')=0
  $$
  
  $g'$ 和 $r(s,a,g)$不同组合方式形成不同的HER策略。

#### Distributed

分布式通常指采样多线程或者多进程的方式启动多个agent(或actor)和对应的local environment，agent采样过程相互独立，同时用一个或者learner来管理模型参数，根据梯度计算和模型参数更新方式有如下几种类型：

- **采样**

  单个agent只负责采样，并不计算参与梯度计算，agent中策略函数$\pi$ 参数定期从global的actor中拷贝。例如上面提到的Ape-x架构。这种形式的优点：加快了采样的速度，同时增加了样本的多样性，某种程度上起到了exploration的作用；缺点是：负责模型训练和梯度计算的agent可能成为瓶颈。因此，引入GPU加快训练过程是一种比较常用的方案，单GPU+多CPU的形式，GPU负责训练，CPU负责采样。

- **采样+梯度计算**

  和上面提到多线程或者多进程agent只负责采样作用不同，local agent增加了本地样本梯度计算的部分并将梯度值同步到global的agent中，global agent按照一定的规则更新后将参数同步回local agent。按照global 和local agent 参数的传递和更新机制分为同步(Synchronous)和异步(Asynchronous)两种。

  在每一批次(batch)的训练中，local agent完成本地梯度计算后将梯度值上传到global agent中，如果需要等待其他local agent完成相关操作后或者global agent的参数后开始下一次batch的迭代，这种属于同步方式，例如A2C算法；异步方式不需要等待其他agent即可获取global agent参数，例如A3C算法。A2C和A3C的形式如下图所示：

  <div  align="center"> <img src="/assets/images/rl/a2c_a3c.jpg" width="90%" height="90%" alt="AC算法同步和异步形式对比图"/> </div>

<center>图 4.5 AC算法同步和异步结构对比图; 图片来源[2]</center>

分布式技术在RL问题中应用的会带来其他问题，例如 采样如何存储共享，local agent的策略和global agent的策略不一致等问题。需要同上面提到的其他技术相结合，例如利用ER解决样本存储和共享问题，利用off-policy解决behavior policy和target policy不一致的问题。



### Policy-evaluation

policy-evaluation泛指某种策略 $\pi$ 下对state或者state-action的价值评估，即前面提到的$V_{\pi}(s)$，$Q_{\pi}(s,a)$ 和 $A_{\pi}(s,a)$等价值函数。在state空间或者state-action空间比较小的时候，可以通过table的方式来记录这些价值函数的值，用Bellman Equations等式进行迭代优化。但是当状态空间或着动作空间足够大时，table形式是不可取的，因而采用函数形式(例如，Neural Network)来近似(approximate)真实值，因此评估问题转化为函数拟合问题，具体点是回归问题。拟合问题通常涉及到两个值：评估值(prediction)和真实值(target)，其中target值获取方式有下面将的MC和TD，而这两种方式都离不开Bellman Equations。

#### Bellman Equations

贝尔曼方程式是指将价值函数分解为即时奖励和未来价值折扣的一组方程式。

$V(s)$的Bellman Equations形式：


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

#### MC (Monte-Carlo Methods)

MC的思路很简单：采用MC抽样的方式搜集完整轨迹 $\tau$，然后利用整条序列的观测值来计算期望收益。首先可以得到：$G_t=\sum_{k=0}^{T-t-1}\gamma^kr_{t+k+1}$，然后利用$V(s)=E[G_t\|s_t=s]$ 关系得到$V(s)$以及后续的$$Q(s,a)$$，如下所示：


$$
V(s)=\frac{\sum_{t=1}^T \rceil[s_t=s]G_t}{\sum_{t=1}^T \rceil[s_t=s]}
$$


和


$$
Q(s,a)=\frac{\sum_{t=1}^T \rceil[s_t=s,a_t=a]G_t}{\sum_{t=1}^T \rceil[s_t=s, a_t=a]}
$$


#### TD (Temporal-Difference Learning )

上面的MC方法估计是无偏的，但是要求每个采样轨迹必须是完整的，这个条件就限制了在某些循环或者trajectory很长的场景中应用。面对高维state或action的场景，采样的数量不够往往会导致high variance的结果。为了应对这个问题，TD方法诞生了。

在model-free场景，采样TD方法对$V(s)$和$Q(s,a)$ 函数的预估并不需要完整的episodes采样，而是基于已有预估值采用Bootstrapping进行预估，每一个step都可以更新。MC的Bootstrapping表示方法为:


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


### Policy-improvement

策略提升(Policy-improvement)依据算法类型的不同内容有所不同，因此也能看出不同算法类型思路的差异。

- **Value-based**

  基于value-based方法在$Q(s_t,a_t)$或者$A(s_t,a_t)$的已知情况下，最优策略很明显，即：

$$
\pi=\arg \max_{\pi}Q_{\pi}(s,a); \pi=\arg \max_{\pi}A_{\pi}(s,a)
$$
​	但是只是$V_{\pi}(s)$ 已知，并不能直接得到最优策略，还需要状态转移概率 $p_{ss'}^a$，即model-based环境。因此在	model-free环境下，一般采用优化$Q_{\pi}(s,a)$函数的方式来得到最优策略，或者结合$V_{\pi}(s)$作为baseline以降低variance得到$A_{\pi}(s,a)$来实现。因此value-based的策略提升优化是隐私(Implicit)通过优化价值函数来实现的。

- **Policy-based**

  而Policy-based算法的思路则是基于Policy Gradient，通过显示(Explicit)优化策略函数 $\pi_{\theta}(a\|s)$ 来实现的，通过对函数参数 $\theta$ 求导然后运用梯度上升来优化函数$\pi_{\theta}(a\|s)$。当然梯度的计算需要前面的Policy-evaluation过程，即价值函数。

- **Actor-Critic**

  最后Actor-Critic算法是上述两者的结合，在policy improvement阶段的内容和policy-based一致。



##  Model-free algorithms

按照前面提到的agent能否对environment的state转移概率建模分为model-based和model-free方法。在很多场景中是无法对environment建模的，因此model-free更具有普遍性，前面的policy-improvement的章节已经提到三种算法类型，分别为Value-based，Policy-based以及两者结合的Actor-Critic。本章先会呈现model-free的算法的关系拓扑图，意在说明算法的演进过程。后面会按类别介绍相关经典的算法。

### The Taxonomy of Model-free algorithms

Model-free算法演化路径有如下几个主轴：

- 如何提升值函数($Q(s,a)$或$V(s)$)的评估的准确性？
- 如何解决Policy Gradient算法High-Variance问题和迭代稳定性问题？
- 如何提升样本的利用率？
- 样本的采样效率？工程化角度？

Model-free算法演化关系如下所示：

<div  align="center"> <img src="/assets/images/rl/model-free.jpg" width="90%" height="90%" alt="q-learning算法伪码"/> </div>

<center>图 5.1 model-free部分算法演化图</center>

### Value-based algorithms

#### Q-Learning

Q-learning基于value-based的思想，通过TD来进行策略评估(policy evaluate)得到$Q(s,a)$的值，然后基于$Q(s,a)$进行策略更新(policy improvement)。$Q(s,a)$值采用table形式进行存储，即策略优化过程中需要维护一张 $Q$-table 表。同时为了增加算法的鲁棒性，避免容易陷入局部最优的情况，在交互过程中加入$\epsilon-greedy $策略对state-action空间进行exploration。

<div  align="center"> <img src="/assets/images/rl/q-learning.jpg" width="90%" height="90%" alt="q-learning算法伪码"/> </div>

<center>图 5.2 Q-learning 算法的off-policy TD版本; 图片来源[1]</center>



#### DQN 

Q-learning 采用table的形式来记录存储 $Q(s, a)$ 值，当面对对于高维状态空间或者高维动作空间(或者连续动作空间)时，这种精确的记录形式面临数据存储的压力。为了解决该问题，DQN采用Deep Learning中 **Neural Network(NN)** 函数来近似 $Q(s, a)$ 的表示。NN拟合 $Q(s, a)$的方式是通过最小化目标值和评估值的平方差，即回归思想。该算法的前提是样本服从独立同分布，但是依据RL特性step相互之间是关联的，为了消除和弱化这种关联性，DQN引入了Experience Replay机制，同时ER的引入也提升了sample efficient。

<div  align="center"> <img src="/assets/images/rl/DQN.jpg" width="90%" height="90%" alt="DQN算法伪码"/> </div>

<center>图 5.3 DQN算法伪码; 图片来源[5]</center>

其中 equation 3为：

$$
\bigtriangledown_{\theta_i}L_i(\theta_i)=\frac{1}{N}\sum_i(r + \gamma max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i))\bigtriangledown_{\theta}Q(s,a;\theta_i)
$$





#### Double DQN 

DQN算法存在着value函数被高估的问题，假如一个$Q(s_{t+1},a)$被高估，那么根据算法中Bellman方程：
$$
Q(s_t,a_t) = r_t + \max_{a}Q(s_{t+1},a)
$$
中采用 $\arg \max$ 策略引发后面一系列高估的问题。为了解决该问题，提出了Double-DQN的算法，即采样两个Q函数：$Q$ 和 $Q'$ 来消除Q值高估的问题。这和target network策略不一样。
$$
Q(s_t,a_t) = r_t+ Q'(s_{t+1}, \arg \max_{a}Q(s_{t+1},a) )
$$
为什么这样的评估策略能够解决高估问题呢？假设 $Q$ 函数高估了$a$动作，那么$Q$ 函数就会选择动作$a$ ，但是在更新$Q$ 函数的时候需要依赖$Q'$，假如$Q'$能够给出适当的值，那么$Q$函数更新后也会回归到正常值。同样如果$Q'$ 存在高估呢？同样也是需要$Q$存在高估。这种相互纠正的策略能在一定程度上解决DQN高估的问题。



#### Dueling DQN

在DQN算法中是通过对Q值的预估来实现策略提升，其中Q值可以分解为state的value值和action的advantage值。如果算法能够将这两个值的分离评估，就能更准确学习和刻画Q值两部分的影响比重，到底是state本身价值还是action选择带来价值的提升。基于此在进行action选择的时候就更具鲁棒性和算法的稳定性。Dueling DQN就是基于此思想，通过优化神经网络(Q函数)的结构来优化算法。

Dueling DQN通过Q函数分成两部分：和只和state相关的Value函数 $V(s;\theta,\beta)$，还有和具体action相关的Advantage函数$A(s,a;\theta,\alpha)$，所以Q函数为：
$$
Q(s,a;\theta,\alpha,\beta) =V(s;\theta,\beta)+A(s,a;\theta,\alpha)
$$
网络结构如下图所示：

<div  align="center"> <img src="/assets/images/rl/dueling-dqn.jpg" width="60%" height="60%" alt="Dueling-DQN模型结构图"/> </div>

<center>图 5.4 DQN和Dueling-DQN结构对比图; 图片来源[8]</center>

图中上部网络为正常DQN结构，下部Dueling-DQN，可以看到有两个子网络结构，分别对应Value函数和Advantage函数，最后输出为两部分的线性组合。由于最终输出还是还是无法区分$V$函数和$A$函数的作用，为了体现这种可辨识性(identifiability)，对$A$ 函数进行去中性化的处理，即$A$值剪去$A$均值，最终Q函数的形式如下：
$$
Q(s,a;\theta,\alpha,\beta) =V(s;\theta,\beta)+A(s,a;\theta,\alpha)-\frac{1}{\mathcal{A}}\sum_{d' \in \mathcal{A}}A(s,a',\theta,\alpha)
$$
Dueling DQN的其他流程和其他DQN基本一致。



#### NoisyNet DQN

DQN的迭代是基于贪婪搜索策略，因此需要增加exploration策略来提升模型的泛化能力，平衡exploration和exploitation。其中一种方式是在action选择的时候采用 $\epsilon - greedy$ 的方式；另外一种是：entropy regularisation，在PG的方法中，将policy的熵添加到损失函数中时以惩罚模型容易产生action唯一性的；除了上述两种方法，深度学习中Noisy Net也被引入RL的Policy 函数中。

**基本原理：**

通过在网络的最后一层(通常fully-connected层)添加高斯噪声(gaussian noise)，该噪声的参数可以在训练过程中由模型进行调整，这使agent可以决定何时以及以什么权重来引入输出的不确定性。

Noisy DQN就是引入了Noisy Net的一种DQN算法，其他的RL算法，例如Actor-Critic中。



### Policy-based algorithms

#### PG (Policy Gradient )

相比于value-based通过评估$(s,a)$的value值作为策略 $\pi$ 依据的方法，PG则是直接显示优化策略 $\pi$。

- PG的推导

RL的目标是最大化期望收益，即：


$$
\theta^*=\mathop{\arg\max}_{\theta}E_{\tau\sim p_{\theta}}(\tau)[\sum_tr(s_t,a_t)]
$$


影响一条轨迹 $\tau$ 的形成包括两个因素：动作策略$\pi_{\theta}$ 和状态转移概率 $p(s'|s, a)$，，如下(2)式所示


$$
\pi_{\theta}(\tau)=p_{\theta}(s_1,a_1,...,s_T,a_T)=p(s_1)\prod_{i=1}\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t) \tag{2}
$$


将(1)式中目标部分写成积分形式：


$$
J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=\int \pi_{\theta}(\tau)r(\tau)d_{\tau} \tag{3}
$$


实际应用中采用无偏采样的形式来近似期望函数，所以上式变换可得：


$$
J(\theta)=E_{\tau \sim \pi(\theta)}[r(\tau)]=E_{\tau \sim \pi(\theta)}[\sum_{t}r(s_t,a_t)]\approx \frac{1}{N}\sum_{i}\sum_tr(a_{i,t},s_{i,t})
$$


对$J(\theta)$进行求导，同时根据 $\bigtriangledown_{\theta}\pi_{\theta}(\tau)=\pi_{\theta}(\tau)\frac{\bigtriangledown_{\theta}\pi_{\theta}(\tau)}{\pi_{\theta}(\tau)}=\pi_{\theta}(\tau)\bigtriangledown_{\theta}\log{\pi_{\theta}(\tau)}$ 变换：


$$
\bigtriangledown_{\theta} J(\theta)=\int \bigtriangledown_{\theta}\pi_\theta(\tau)r(\tau)d_{\tau}=\int \pi_{\theta}(\tau) \bigtriangledown \log\pi_{\theta}(\tau)r(\tau)d_{\tau} =E_{\tau \sim \pi_{\theta}(\tau)}[\bigtriangledown\log\pi_{\theta}(\tau)r(\tau)] \tag{4}
$$


将(2)式代入(4)式, 同时结合(3) 式得到梯度：


$$
\bigtriangledown_{\theta} J(\theta)\approx \frac{1}{N}\sum_{i=1}^N(\sum_{t=1}^T\bigtriangledown_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t}))(\sum_{t=1}^Tr(s_{i,t},a_{i,t})) \tag{5}
$$

和maximum likelihood优化方法的梯度形式相比，PG的梯度函数多了轨迹 $\tau$ 的reward项作为权重，符合直观的认知。



#### REINFORCE

根据上面的PG推导，可以得到最原始的PG算法REINFORCE：

<div  align="center"> <img src="/assets/images/rl/reinforce.jpg" width="80%" height="80%" alt="REINFORCE算法"/> </div>

<center>图 5.5 REINFORCE算法的MC版本伪码; 图片来源[1]</center>

- **Reduce Variance**

  从REINFORCE 算法的梯度计算公式可以看出 (1) 采用MC方法来得到reward的期望值 (2) 使用 完整轨迹$\tau$ 的reward值进行加权，来体现对高reward值的 $\tau$ 的偏好。这两点都会导致梯度的High-Variance。因为采样成本或者交互过程中state(action)的探索空间限制，通常情形下采样的 $\pi$ 的数量是不够的，即样本不能很好体现 $\tau$ 的分布情况，这就自然引入了variance。同时，假设存在着reward都为正的情况，那么 $\tau$ 累计的reward都是正的，这样所有的 $\tau$的概率都会上升，那么随着抽样的$\tau$ 增加，相对更好的 $\tau$的概率也会下降。针对以上两点，产生了两种减少variance的方法。

  - **Causality**

    根据(5)式的梯度公式等于每一条trajectory的reward之和与其完整轨迹的概率乘积，但实际情况是在 $t$ 时刻(step)发生的action并不能影响 $t$ 之前的reward，所以实际计算中应该将 $t$ 之前reward去掉。同时很容易证明一点：当一个分布中的值都减小时，该分布的variance scale将会减小。因此梯度公式变为：
    $$
    \bigtriangledown_{\theta}J(\theta) \approx \sum_i\sum_t\bigtriangledown_{\theta}log(\pi_{\theta}(a_{t}^i|s_t^i)(\sum_{t'=t}r(s_{t'}^i,a_{t'}^i)) \tag{6}
    $$

  - **Baselines**

    针对第二个问题，一种自然的想法是使得 $\tau$ 的累积reward有正有负，正的 $\pi$ 概率上升，负的下降。如何操作呢？产生一个baseline作为基准，将所有的 $\pi$ 的reward都减去它。PG公式变为：
    
    
    $$
    \bigtriangledown_{\theta}J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)(r(\tau)-b)
    $$
    
    
    同时可以证明引入baselines后并没有增加bias(证明:)，其中一个较好的 $b$ 是所有 $\tau$ 的累积reward的算术平均：$b=\frac{1}{N}\sum_{i=1}^Nr(\tau)$ 。使得variance的最小b为：

    
    $$
    b = \frac{E[\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)^2(\tau)]}{E[\bigtriangledown_{\theta}\log\pi_{\theta}(\tau)^2]}
    $$
    
    
    下图就是REINFORCE采用State的值函数作为baseline的实例：
    
    <div  align="center"> <img src="/assets/images/rl/reinforce-baseline.jpg" width="85%" height="85%" alt="REINFORCE算法"/> </div>
    
    <center>图 5.6 REINFORCE算法的Baseline版本伪码; 图片来源[1]</center>



#### Off-Policy PG

on-policy的策略存在着样本利用率低的问题(sample inefficient)，为了提升样本的利用率，一种很自然的想法是把之前的采集的样本都能利用起来，即转向off-policy的策略。前面提到了off-policy的概念，即用于采样的策略和优化的策略不同。根据强化学习的交互性特点，不同的action策略会导致采样结果分布的差异，因此需要方法来修正这种差异。一种方法就是前面提到的重要性采样：

- **Importance sampling (IS)**

  函数$ f(x) $ 中 $ x$ 服从$p(x)$分布(即：$x\sim p(x)$)，和 $x$ 服从$q(x)$的分布(即：$x\sim q(x)$)，两者之间期望的关系如下：
  $$
  \begin{align}
  E_{x\sim p(x)}[f(x)]=\int p(x)f(x)d_x\\
  =\int \frac{q(x)}{q(x)}p(x)f(x)d_x\\
  =\int \frac{p(x)}{q(x)} q(x)f(x)d_x\\
  =E_{x\sim q(x)}[\frac{q(x)}{p(x)}f(x)]
  \end{align}
  $$
  
- **IS in Policy Gradient**

假设使用策略 $\pi_{\theta}(\tau)$ 采样来更新策略$\pi_{\theta'}(\tau)$ ，根据(2)式：


$$
\frac{\pi_{\theta}({\tau})}{\pi_{\theta'}(\tau)}=\frac{p(s_1)\prod_{t=1}^T\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)}{p(s_1)\prod_{t=1}^T\pi_{\theta'}(a_t|s_t)p(s_{t+1}|s_t,a_t)}=\frac{\prod_{t=1}^T\pi_{\theta}(a_t|s_t)}{\prod_{t=1}{\pi}_{\theta'}(a_t|s_t)}
$$


我们将(3.2)式带入到PG的优化目标函数中得到：


$$
J(\theta')=E_{\tau \sim \pi_{\theta}}[\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}r(\tau)]
$$


对其进行求导：


$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\frac{\bigtriangledown_{\theta'}\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}r(\tau)]=E_{\tau \sim \pi_{\theta}}[\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau) r(\tau)]
$$


根据前面提到的reward计算的Causality以及未来的action并不影响当前的分布差异，对$\frac{\pi_{\theta'}(\tau)}{\pi_{\theta}(\tau)}$ 和 $r(\tau)$进行裁剪得到：


$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\sum_{t=1}^T\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau)(\prod_{t'=1}^t\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)})(\sum_{t'=t}^Tr(s_{t'},a_{t'}))]
$$


因为连乘形式(指数项:$\prod_{t'=1}^t\frac{\pi_{\theta'}(a_t\|s_t)}{\pi_{\theta}(a_t\|s_t)}$)的存在很可能会出现值爆炸或者消失的情况，所以需要对该项的计算进行变换。我们变换一种形式，当前是需要对state-action的联合概率求期望，可以写成先对state求期望，然后对action求期望，所以(3.3)可以变换为：


$$
J(\theta')=\sum_{t=1}^TE_{s_t\sim p_{\theta}(s_t)}[\frac{p_{\theta'}(s_t)}{p_{\theta}(s_t)}E_{a_t \sim \pi_{\theta}(a_t|s_t)}[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}r(s_t,a_t)]]
$$


因为$\frac{p_{\theta'}(s_t)}{p_{\theta}(s_t)}$ 未知忽略掉这部分，那么Off-policy的梯度就变成了：


$$
\bigtriangledown_{\theta'}J(\theta')=E_{\tau \sim \pi_{\theta}}[\sum_{t=1}^T\bigtriangledown_{\theta'}\log \pi_{\theta'}(\tau)\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}(\sum_{t'=t}^Tr(s_{t'},a_{t'}))]
$$



#### TRPO (Trust Region Policy Optimization)

- **理论基础**

虽然前面提到的Causality和Baseline策略一定上降低了Policy Gradient的训练的high variance问题，但是依然面临着训练不稳定的问题。为了解决该问题，**TRPO** 算法提出了每次迭代都将参数更新控制在Trust Region范围内以此来实现有效的Improvement的。

先看一个一个引理，**Claim 1**:



 $$J(\theta)=\eta(\pi')-\eta(\pi)= E_{s \sim \eta_{\pi'}}[\sum_{t=0} r^tA_{\pi}(s_t,a_t)]=\sum_s \rho_{\pi'}(s)\sum_a \pi'(a|s)A_{\pi}(s,a) \tag{7}$$



从(7)式可以看出新策略 $\theta'$ 相对于旧策略 $\theta$ 的提升就是最大化旧策略的Advantage function，于是优化目标可以转变为优化函数$J(\theta)$。

将 (7) 公式展开并加入 Importance Sampling 后得到 (8) 式：


$$
\begin{align}
E_{\tau \sim p_{\theta'}(\tau)}[\sum_t\gamma A^{\pi_{\theta}(s_t, a_t)}] &=\sum_t E_{s_t \sim p_{\theta'}}[E_{a_t \sim \pi_{\theta'}(a_t|s_t)} [\gamma^t A_{\pi}(a_t|s_t)]] \notag \\
&=\sum_t E_{s_t \sim p_{\theta'}}[E_{a_t \sim \pi_{\theta}(a_t|s_t)} [\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} \gamma^t A_{\pi}(a_t|s_t)]] \tag{8} \\
\end{align}
$$


从 (8) 式来看，$s_t$ 的分布是由新的 $\theta'$ 产生的而非策略 $\theta$ 所以不能直接使用梯度上升来优化。

来看另外一个引理 **Claim 2**:

 **Claim 2**：$p_{\theta}(s_t)$ is *close* to $p_{\theta'}(s_t)$ when $\pi_{\theta}$ is *close* to $\pi_{\theta'}$,  and $\pi_{\theta'}$ is close to $\pi_{\theta}$ if $\|\pi_{\theta'}(a_t\|s_t)-\pi_{\theta}(a_t\|s_t)\|\le \epsilon$ for all $s_t$

同时根据KL函数：
$$
\|\pi_{\theta'}(a_t\|s_t)-\pi_{\theta}(a_t\|s_t)\| \le \sqrt{\frac{1}{2}D_{KL}(\pi_\theta'(a_t\|s_t)\|\pi_{\theta}(a_t\|s_t))}
$$


使用 $\pi_{\theta'}$和$\pi_{\theta}$的KL Divergence作为新的约束, 即 :$D_{KL}(\pi_\theta'(a_t\|s_t)\|\pi_{\theta}(a_t\|s_t)) \le \epsilon$ 。

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
\mathop{maximize}_{\theta'}L_{\pi_{\theta}}(\pi_{\theta'})-\beta*KL_{\pi_{\theta'}}(\pi_{\theta})
$$


分为两步：

*Step1:* 将 $L$ 函数和$KL$函数分别用泰勒公式按照一阶和二阶展开得到搜索方向:

$$
\begin{align}
\mathop{maximize}_{\theta} \quad g^T(\theta' - \theta)-\frac{\beta}{2}(\theta'-\theta)^TF(\theta'-\theta) \\
where \quad g=\frac{\partial}{\partial\theta'}L_{\pi_{\theta}}(\pi_{\theta'})|_{\theta'=\theta}, \quad F=\frac{\partial^2}{\partial^2 \theta'}KL_{\pi_{\theta'}}(\pi_{\theta})|_{\theta'=\theta}
\end{align}
$$

*Step2:* 在step1得到的方向上进行线性搜索以保证满足约束要求。

$$
\theta'-\theta = \frac{1}{\beta}F^{-1}g
$$

因为FIM(H)的计算成本太高，所以采用共轭梯度 (Conjugate Gradient，CG) 来求解 $F.x=g$.



#### PPO (Proximal Policy Optimization)

TRPO使得训练的稳定性提升，但是面临着CG计算的复杂度高以及和一些网络结构不兼容的问题(例如， dropout策略和共享参数策略)。和TRPO采用KL函数作为约束的策略不同，PPO利用裁剪目标函数的方式来简化目标函数的优化，并保证性能。

其中  $r(\theta)=\frac{\pi_{\theta}(a_t\|s_t)}{\pi_{\theta_{old}}(a_t\|s_t)}$ ， TRPO的目标函数为：


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

<img src="/assets/images/rl/ppo.jpg" alt="55C98BB2-0A4E-4EA9-B2D4-18CA24F352C6" style="zoom:50%;" />

<center>图 5.7 PPO算法伪码; 图片来源[12]</center>



### Actor-Critic algorithms

#### AC (Actor-Critic)

前面提到的REINFORCE算法的梯度计算，$G_t=\sum_ir(s_i,a_i)$部分，是基于蒙特卡洛采样方法得到的，需要一条完整轨迹 $\pi$ 后才能。即回合制更新。这种形式容易受到场景限制，影响更新迭代效率和收敛的速度。因此，一种策略是利用value-based方法的思路来近似评估值，即Critic部分。而算法的Actor即原先的PG算法。也就是说Actor-Critic算法引入两组近似函数，即策略函数的近似 $\pi_\theta(a\|s)$ 和价值函数的近似 $v(s)$ 或 $q(s,a)$。

这就和前面提到的GEI对应起来了: Critic对应policy evaluate，Actor对应policy improvement。 

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

- 基于优势函数

$$
\delta(t) = A(s,a,t)=Q_{\theta}(s_t,a_t)-V_{w}(s_t)
$$

梯度更新公式为：
$$
\theta = \theta + \alpha \log \pi_{\theta}(a_t|s_t) \delta(t)
$$
在 Actor-Critic 算法中：

**Actor**：根据Critic的结果来更新策略函数 $\pi_{\theta}(a\|s)$ 参数 $\theta$

**Critic**：更新value-function的参数$w$，即 $Q_w(s, a)$ 或者 $V_w(s)$

Actor-Critic算法架构如下图所示：

<div  align="center"> <img src="/assets/images/rl/ac_architecture.jpg" width="50%" height="50%" alt="REINFORCE算法"/> </div>

<center>图 5.8 Actor-Critic算法框架图; 图片来源[1]</center>

#### A3C (Asynchronous Advantage Actor-Critic)

A3C算法为了提升训练速度采用了异步训练的思想，即利用多个线程，每个线程相当于一个单独的agent和环境进行交互进行探索，各个线程中的actor和critic梯度的计算和参数的更新类似分布式训练中的ps(parameter servers)架构。维护一个global 的actor的参数 $\theta$ 和 critic的参数 $w$， 各个线程定期拷贝 $\theta$ 和 $w$ 作为local $\theta'$ 和 $w'$ 初始值和local的环境进行交互生成样本并计算累积梯度，然后将累积后梯度值同步到global的$\theta$ 和$w$进行参数更新，如此循环。

<div  align="center"> <img src="/assets/images/rl/a3c_architecture.jpg" width="75%" height="75%" alt="REINFORCE算法"/> </div>

<center>图 5.9 A3C算法框架图; 图片来源[17]</center>

A3C的异步训练方式的不仅增加了采样速度，加快训练和收敛速度，同时相比于DQN采用Replay Buffer存储样本的随机抽样训练方式来弱化样本之间的correlation，A3C通过多个thread独立地和环境交互采样训练同步梯度的方式达到了同样的效果，并更节约存储空间。A3C具体算法如下所示：

<div  align="center"> <img src="/assets/images/rl/a3c_algorithms.jpg" width="70%" height="70%" alt="REINFORCE算法"/> </div>

<center>图 5.10 A3C算法伪码; 图片来源[17]</center>



#### A2C (Advantage Actor-Critic)

A2C和A3C区别在于去掉了Asynchronous，即local agent之间和global network 参数更新采用了同步的方式。A3C的各个thread对应的local agent除了定期从global 的parameter中copy参数，相互之间独立，梯度更新和初始policy存在不一致。为了解决不一致问题，A2C中的协调器在更新全局参数之前等待所有并行参与者完成其工作，然后在下一次迭代中并行参与者从同一策略开始。 同步的梯度更新使训练更具凝聚力，并有可能使收敛更快。

<div  align="center"> <img src="/assets/images/rl/a3c_vs_a2c.jpg" width="75%" height="75%" alt="REINFORCE算法"/> </div>

<center>图 5.11 A3C和A2C结构对比图; 图片来源[4]</center>

事实证明，A2C可以更有效地利用GPU，并且在大批量处理时可以更好地工作，同时实现与A3C相同或更好的性能。



#### SAC (Soft Actor Critic)

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
\end{align}
$$


单纯通过采样我们并不能覆盖所有的 $a_t$ ，所以通过action产出加入noise生成action的分布值的策略，这里action我们采用reparameterization trick来得到，即：


$$
a_t=f_{\phi}(\xi_t;s_t)=f_{\phi}^{\mu}(s_t)+\xi_t\odot f_{\phi}^{\sigma}(s_t)
$$


$f$ 函数输出平方值和方差，然后$\xi$ 是noise，从$N(\mu;\sigma)$中进行采样。然后将$a_t$带入(8)式中求导。

整个算法如下：

<div  align="center"> <img src="/assets/images/rl/sac.jpg" width="60%" height="60%" alt="SAC算法"/> </div>

<center>图 4.12 SAC算法伪码; 图片来源[21]</center>



#### DPG (Deterministic Policy Gradient)



何为Deterministic 呢？这是相对stochastic而言的，所谓的stochastic是指policy输出的是一个action概率分布$\pi_{\theta}(a\|s)$, action从该概率分布中进行采样； 而Deterministic policy输出的是一个确定性的action值$\mu_{\theta}(s)$。

为什么在拥有Stochastic Policy方法的情况下还要追求Deterministic policy ？这是因为Stochastic Policy多了一个sample action的步骤，需要更多的样本来覆盖action的空间，如果action维度空间很高，Stochastic Policy就变得非常低效，尤其在面临action为continuous时。

DPG采用的是Actor-Critic框架形式，如下图所示：

<div  align="center"> <img src="/assets/images/rl/dpg_ac.jpg" width="45%" height="45%" alt="REINFORCE算法"/> </div>

<center>图 5.13 DPG算法结构流程图 </center>

和一般的Actor-Critic算法不同在于：Actor的输出会作为Critic部分的输入，存在嵌套关系。

假设: Actor的策略函数为 $\mu_{\theta}(s)$，Critic的Q函数为$Q(s, a)$。因为$\mu_{\theta}(s)$的输出值是唯一的，所以在知道状态 $s$ 的情况下，a也是确定的可以通过 $\mu$ 得到，因此Critic的Q函数可以表示为：$Q^{\mu}(s, \mu_{\theta}(s))$，所以虽然是Actor-Critic的架构，但是本质上在优化Q函数的同时通过求导的链式法则可以优化$\mu$ 函数。

其梯度更新策略为：


$$
\theta^{k+1}=\theta^{k}+E_{s \sim \rho(\mu)}[\bigtriangledown_{\theta}\mu_{\theta}(s)\bigtriangledown_{a} Q^{\mu^k}(s, a)|_{a=\mu_{\theta}(s)}]
$$



DPG由于每次都只输出一维的确定性的action，所以不具备exploration的能力，为了解决该问题需要配合off-policy的策略。



#### DDPG (Deep DPG)

DDPG是通过将DPG中的Actor和Critic函数通过Neural Network近似发展而来。通过加入Replay Buffer来解决样本直接关联性问题，

- ER (off-policy)
- Target Network soft update
- Noise for exploration
- Batch Normalization

和一般的Actor-Critic架构不同在于，Critic部分依赖actor

<div  align="center"> <img src="/assets/images/rl/ddpg.jpg" width="80%" height="80%" alt="REINFORCE算法"/> </div>

<center>图 5.14 DDPG算法伪码; 图片来源[14]</center>



#### TD3 (Twin Delayed DDPG)

TD3算法引入Double-DQN中的Double Q函数 ($Q_{\theta1}$ 和 $Q_{\theta2}$) 的评估方式来解决overestimation的问题，在进行bellman方程迭代评估target值时采样取最小值的方式：


$$
y = r + \gamma \min_{i=1,2} Q_{\theta_{i}}(s_{t+1},\mu_{\theta'}(s_{t+1}))
$$


同时和DDPG一样：

- target network soft update策略增加算法稳定性
- 在进行action选择的时候加入随机噪声来进行exploration
- 采取batch normalization来平滑效果

算法的整体流程如下所示：

<div  align="center"> <img src="/assets/images/rl/TD3.jpg" width="60%" height="60%" alt="REINFORCE算法"/> </div>

<center>图 5.15 TD3算法伪码; 图片来源[15]</center>





<p> </p>



***

<p></p>

<p></p>

```
博文转载请注明出处：
作者：郑春荟
题目：reinforcement learning
链接：https://ustczhengyou.github.io
```



**参考文献：**

[1] [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/bookdraft2017nov5.pdf).

[2] [CS294-112: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/)

[3]  [Introduction to Reinforcement Learning; David Silver](https://www.davidsilver.uk/)

[4] [lilianweng's blog](https://lilianweng.github.io)

[5] [DQN: Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[6] [DQN: Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

[7] [Double-DQN:Deep Reinforcement Learning with Double Q-learning ](https://arxiv.org/abs/1509.06461)

[8] [Dueling-DQN: Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

[9] [NoisyNet: Noisy Networks for Exploration ](https://arxiv.org/abs/1706.10295)

[10] [Policy Gradient: Policy Gradient Methods for Reinforcement Learning with Function Approximation ](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

[11] [TRPO:Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

[12] [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[13] [DPG: Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)

[14] [DDPG: Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

[15] [TD3:Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

[16] [AC: Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)

[17] [A3C: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

[18] [ACER: sample effcient actor-critic with experience replay](https://arxiv.org/abs/1611.01224)

[19] [Ape-X: Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)

[20] [HER: Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

[21] [SAC:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
**Q-learning: Off-policy TD control**

Q-learning基于value-based的思想，通过TD来进行策略评估(policy evaluate)得到$Q(s,a)$的值，然后基于$Q(s,a)$进行策略更新(policy improvement)。$Q(s,a)$值采用table形式进行存储，即策略优化过程中需要维护一张 $Q$-table 表。同时为了增加算法的鲁棒性，避免容易陷入局部最优的情况，在交互过程中加入$\epsilon-greedy$策略对state-action空间进行exploration。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200630190309543.png" alt="image-20200630190309543" style="zoom:70%;" />

***

**DQN** 

Q-learning 采用table的形式来记录存储 $Q(s, a)$ 值，当面对对于高维状态空间或者高维动作空间(或者连续动作空间)时，这种精确的记录形式面临数据存储的压力。为了解决该问题，DQN采用Deep Learning中 **Neural Network(NN)** 函数来近似 $Q(s, a)$ 的表示。NN拟合 $Q(s, a)$的方式是通过最小化目标值和评估值的平方差，即回归思想。该算法的前提是样本服从独立同分布，但是依据RL特性step相互之间是关联的，为了消除和弱化这种关联性，DQN引入了Experience Replay机制，同时ER的引入也提升了sample efficient。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200617160406797.png" alt="image-20200617160406797" style="zoom:50%;" />

其中 **equation 3 **为：
$$
\bigtriangledown_{\theta_i}L_i(\theta_i)=\frac{1}{N}\sum_i(r + \gamma max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i))\bigtriangledown_{\theta}Q(s,a;\theta_i)
$$

***

##### Experience Replay

Experience Replay机制通过将交互采集的样本及附属信息存储在buffer中形成经验池，算法迭代训练从经验池中采样产生训练样本，经验池的样本更新遵循一定的规则(例如，先进先出)。这种机制：

- *sample correlation*：消除了样本之间的关联性，并且使得训练数据分布变得平滑，不容易陷入局部最优。
- *sample efficiency*：增加了样本的利用率，每个样本都可能被多次使用。

该方法在某些方面受到限制，存储样本的内存缓冲区大小总是有限的，并且有新旧样本的替换机制，所以经验池中的样本并不能保留和覆盖真实的样本空间。 一种策略是在有限的空间中仅可能保留重要的样本，即对样本进行加权，所以引出PER (Prioritized Experience Replay)。

***

**PER (Prioritized Experience Replay)**

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
  


***

**ApeX (Distribution Prioritized Experience Replay)**

从命名上可以看出，ApeX和PER的区别在于拓展到分布式形式。拓展的理念来自于训练数据的增加有助于模型更有效的学习。个人觉得ApeX是更准确来讲是一种框架，用到了Replay Buffer并且对采用PER策略的算法都可以嵌套入到其中，例如DQN，double-DQN，DPG等。

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200721211828488.png" alt="image-20200721211828488" style="zoom:50%;" />

其ApeX结构如上：采用多个actor，一个learner，一个share memory的Replay Buffer。actor用于和local environment 进行交互产生经验样本，并且定期将样本同步到share replay buffer中和复制learner网络参数值；learner用于从share replay buffer的采样进行训练，并将训练后的参数定期同步给actor。具体流程和逻辑入下图伪码所示：

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200721213309192.png" alt="image-20200721213309192" style="zoom:50%;" />

从试验结果上看Ape-X 的DQN远好于Rainbow，DQN以及Prioritized DQN。其原因在于，分布式的actor采样形式使得采样效率得到很大提升，缩短了采样时间；同时，不同的actor和local environment独立交互的形式使得提升了样本的覆盖度，起到了对样本空间explore的作用，再配合PER策略又提升样本利用率，出现了训练时间缩短并且效果提升的结果。

***

**HER (Hindsight Experience Replay)**

在有些场景中和环境交互面临着奖励稀疏的问题，例如，在围棋对抗中只有在分出胜负的时候才能知道真正的reward。为了解决奖励稀疏问题，一种方法是采用shaped reward方法重塑reward 函数。HER的提出也是为了解决奖励稀疏问题。

HER的思想很简单：既然reward稀少除了trajectory结束获得收益作为目标外，能否将这中间的某些state(里程碑)也作为我们的目标和参照物呢？就向跑马拉松可以将整个过程划分成好几个阶段，每个阶段都会有标志性的参照物。基于此思想HER会在trajectory完成后，回过头选取某些state作为目标参照物，根据$r(s_t,a_t,g)$ 函数重新得到reward假如到replay buffer中，和一般的ER中存储的$(s_t, a_t, r_t, s_{t+1})$不同，HER中样本形式为$(s_t||g, a_t, r_t, s_{t+1}||g)$。

<img src="/Users/youzheng/Documents/Blog/_image/rl/HER.jpg" alt="image-20200721213309192" style="zoom:50%;" />

HER机制如上图红色方框所示。在第一遍trajectory完成后，第二遍完成HER的生成。其中的核心点在于如何采样$g'$ ，以及如何设计$r(s,a,g')$ 函数。

- $g'$ 选择
  - 未来模型(future)：遍历到$s_t$ ,  从后续step中抽取$k$个$s$ 作为目标
  - 回合模式(episode): 遍历到$s_t$ ,  从该episode中抽取$k$个$s$ 作为目标
  - 多回合模式(multi-episode): 遍历到$s_t$ ,  从限定的多个episode中抽取$k$个$s$ 作为目标
  - 最终模式(final)：遍历到$s_t$ ,  以该episode的最后一步作为目标

有如下四种模型：		

- $r(s,a,g)$ 设计
  $$
  f_g(s) =[|g-s_object|\le \epsilon] \\
  r(s,a,g)=-[f_g(s')=0]
  $$
  

***

**DDQN (Double DQN)**

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

**Dueling DQN**

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

**Noisy Net (Noisy DQN)**

DQN的，所以有必要增加训练样本的exploration。

其中一种方式是在action选择的时候采用 $\epsilon - greedy$ 的方式。还有另外一种方式是Noisy Net。

There are two “classic” approaches to this task:

1. **epsilon-greedy**:  with some probability epsilon (given as hyperparameter) your agent takes a random step instead of acting according to policy it has learned. It’s a common practice to have this epsilon equal to 1 in the beginning of training and slowly decrease to some small value, like 0.1 or 0.02.
2. **entropy regularisation**:  is used in policy gradient methods when we’re adding entropy of our policy to the loss function, punishing our model for being too certain in it’s actions.

Noisy Net的原理：

*The method basically consists of adding the gaussian noise to the last (fully-connected) layers of the network. Parameters of this noise can be adjusted by the model during training, which allows the agent to decide when and in what proportion it wants to introduce the uncertanty in it’s weights.*

***




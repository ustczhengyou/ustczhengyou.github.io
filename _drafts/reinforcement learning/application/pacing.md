### 业务意义

如何对广告主的budget进行合理的花费的过程。



在**广义第二高价扣费逻辑**下，原先按照ecpm排序的无干预的逻辑对于平台方和广告主来讲都是suboptimal的。

- 平台方(媒体)

  - 增加广告主预算周期内的竞价激烈程度来提升平台的收益。

    **解析：**如果是按照广义二价的扣费逻辑，每次赢得竞价的广告主只要支付第二高价的价格，一些出价比较高的广告主如果在周期内比较早的阶段花完预算。

- 广告主

  - 帮助广告主触达更多人群，提升广告主的ROI
- 对于按照CPM或者Brand campaign来讲，广告主触达更多人群
  
  ​	  **解析：** 在CPC或者CPM等不考虑转化扣费的模式下，Pacing使得广告主的预算不容易短期内预算消耗完毕而退出竞价，从而触达更多的人群及易转化人群来达到提升转化率降低ROI的作用。

### 形式

一般以天为单位(windows)

- 均匀花费(Even pacing)
- 基于流量(Traffic-based pacing)

### 评估效果

- 平台收入提升
- 



### 解决思路

一般有两种思路：Probabilistic throttling 和 Bid modification, 其中Probabilistic throttling是通过一个概率值来决定是否参与竞价来控制预算话费速率，而Bid modification则通过直接改价的方式来控制花费速率。



### 算法

#### 符号声明

- $B_d$：当日总预算

- $B_a$：当日剩下的预算
- $T_a$：假设广告主没有预算限制的情况下，当日剩余时间可能产生的最大消耗预算
- $F_{\theta,a}(\mu)$：$F$ 是累积分布函数，满足 $\theta(i)\le \mu$ 的情况下，当日剩余时间的流量比例
- $R_{\theta,a}(\mu)=1-$$F_{\theta,a}(\mu)$
- 

#### Vanilla Probabilistic Throttling (VPT)

**业务问题：**解决广告主预算平稳消耗的问题

**基本思想：**随机抽样一定比例的流量，这部分流量的预算消耗应该等于抽样比例和总流量预算消耗的乘积。

**实时统计:**  $B_a$

**预测参数:**  $T_a$

**难点：**如何预测无预算限制情况下的总消耗呢？

*For each arriving query $q$：*

​	*For each budget constrained advertiser $a$：*

​		*Flip  a coin with $P[Heads] = B_a/T_a$*

​		*If heads, a participates in the auction*



#### Optimized Throttling (OTP)

**业务问题：**在解决广告主预算消耗平滑的同时，满足一些其他优化条件，例如：ctr， clicks，conversions 等。

**基本思想：**在VPT的基础上再按照目标值 $\theta(i)$ 再进行细分。使用的是累积分布概率来计算流量比例(反向)，同时为了满足需要选择的是 $R_{\theta,a}(\mu)$ 而非 $F_{\theta,a}(\mu)$。

**实时统计:**  $B_a$

**预测参数:**  $T_a$， $R_{\theta,a}(\mu)$

**难点：**如何预测无预算限制情况下的总消耗呢？以及满足特定条件的流量比例情况

*For each arriving query $q$：*

​	*For each budget constrained advertiser $a$：*

​		*if $R_{\theta,a}(\mu) \le B_a/T_a$，then $a$ participates in the auction*


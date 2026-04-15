# CS 185/285 Practice Questions 逐题详解

基于 [Practice Questions.pdf](/mnt/d/CS285/midterm/Practice%20Questions.pdf) 与 [Practice Questions (Answers).pdf](/mnt/d/CS285/midterm/Practice%20Questions%20(Answers).pdf) 整理。以下默认答案页是正确的，我的任务是解释为什么这些答案正确。

## 总答案

- `1.1 = e`
- `1.2 = c`
- `1.3 = b`
- `2.1 = d`
- `2.2 = d`
- `3 = b`
- `4 = a, g`
- `5.1 = d`
- `5.2 = f`
- `5.3 = c`
- `6.1 = a`
- `6.2 = b`
- `7.1 = a`
- `7.2 = a, b, c, d`
- `8 = b, d, f, j, m`
- `9.1 = (≥, ?)`
- `9.2 = (=, =)`
- `10 = d`

## 1. Distribution Shift for Success/Failure Tasks

### 1.1 为什么选 `e`

题目定义“成功/失败”只看最后一个状态 `s_H`。因此

```math
p_\theta(\tau \text{ failed})
= \Pr_{\tau \sim \pi_\theta}(s_H \notin \Omega)
= \sum_{s_H} p_{\pi_\theta}(s_H)\,\mathbf 1(s_H \notin \Omega).
```

这正是选项 `e`。

其余选项为什么错：

- `a`、`d` 都没有对 `s_H` 求和，所以不是完整概率。
- `b`、`c` 用的是训练分布 `p_train`，而我们要的是当前策略 `\pi_\theta` 的失败概率。
- `c`、`f` 还是成功指示函数，不是失败指示函数。

### 1.2 为什么选 `c`

从 `1.1` 出发：

```math
p_\theta(\tau \text{ failed})
= \sum_{s_H} p_{\pi_\theta}(s_H)\mathbf 1(s_H \notin \Omega).
```

加上减去 `p_train(s_H)`：

```math
= \sum_{s_H}\big(p_{\pi_\theta}(s_H)-p_{train}(s_H)\big)\mathbf 1(s_H \notin \Omega)
+ \sum_{s_H} p_{train}(s_H)\mathbf 1(s_H \notin \Omega).
```

对第一项用三角不等式：

```math
p_\theta(\tau \text{ failed})
\le
\sum_{s_H}\left|p_{\pi_\theta}(s_H)-p_{train}(s_H)\right|
+ \sum_{s_H} p_{train}(s_H)\mathbf 1(s_H \notin \Omega).
```

这就是选项 `c`。

接着利用题设“专家策略总是成功”，所以训练分布下末状态一定在 `\Omega` 中：

```math
\sum_{s_H} p_{train}(s_H)\mathbf 1(s_H \notin \Omega)=0.
```

于是

```math
p_\theta(\tau \text{ failed})
\le
\sum_{s_H}\left|p_{\pi_\theta}(s_H)-p_{train}(s_H)\right|.
```

而 lecture 里已经给了

```math
\sum_{s_t}\left|p_{\pi_\theta}(s_t)-p_{train}(s_t)\right|\le \epsilon t.
```

在 `t=H` 处得到

```math
p_\theta(\tau \text{ failed}) \le \epsilon H.
```

### 1.3 为什么选 `b`

上一问已经推出

```math
p_\theta(\tau \text{ failed}) \le \epsilon H.
```

在给出的候选上界里，最紧的是 `b`。

补充一句：更细一点的分析还可以得到

```math
p_\theta(\tau \text{ failed})
\le 1-(1-\epsilon)^H
\le \epsilon H,
```

但这个更紧的式子不在选项里。

## 2. Open Loop Control

### 2.1 为什么选 `d`

题目问“哪一个 **不真**”。

#### `a` 为真

stationary open-loop 本质上是“每一步都执行同一个固定动作”。最优的 open-loop 完全可能是时变的。

反例：必须先执行 `x` 再执行 `y` 才能到达目标，而 `x,x,x,...` 或 `y,y,y,...` 都到不了目标。

#### `b` 为真

若初始状态固定、动力学确定，则任意 closed-loop policy 从 `s_0` 出发都会产生唯一轨迹，也就对应唯一动作序列。于是一个 open-loop action sequence 就能完全复现该轨迹，因此

```math
\sup_A J(A)=\sup_\pi J(\pi).
```

#### `c` 为真

构造一个状态连续的 MDP：

- 初始状态 `x ~ Unif[0,1]`
- 动作空间也是 `[0,1]`
- 若第一步选 `a=x`，则得到最终回报 1，否则回报 0

closed-loop policy 可以观察 `x` 后选 `a=x`，所以回报 1。open-loop 必须先承诺一个固定常数 `a_0`，而连续分布下 `Pr(x=a_0)=0`，所以回报 0。

#### `d` 为假

题目说：任意 open-loop policy `A`，都存在一个 deterministic Markovian closed-loop policy `\pi` 使得 `p_A = p_\pi`。这不对。

反例：

- 状态 `s` 和吸收态 `g`
- 在 `s` 里动作 `wait` 会留在 `s`
- 动作 `quit` 会进入 `g`

考虑 open-loop 序列：

- `t=0` 选 `wait`
- `t\ge 1` 全部选 `quit`

那么它的 occupancy 是

```math
p_A(s)=1+\gamma,\qquad p_A(g)=\frac{\gamma^2}{1-\gamma}.
```

但 deterministic Markovian policy 只有两种：

- 永远 `wait`
- 永远 `quit`

它们对应的 occupancy 都和上面不同。所以 `d` 不真，因此 `2.1` 选 `d`。

### 2.2 为什么选 `d`

#### `a` 为假

CEM 没有“一定几乎处处收敛到全局最优”的一般保证。它可能由于初始高斯分布和 elite 筛选机制，早早收缩到局部峰附近。

#### `b` 为假

random shooting 是从固定分布独立采样然后直接取最优；CEM 会根据 elite 更新采样分布。只有“第一轮、且完全不更新分布”时才像 random shooting。

#### `c` 为假

“only converge when ...” 太强。即使不依赖 elites，若每轮都从一个对最优序列赋正概率的固定分布独立采样，那么 best-so-far 也会以概率 1  eventually hit 最优序列。

#### `d` 为真

CEM 是典型的 gradient-free / derivative-free 方法。它只需要 rollout 和比较回报，不需要通过模型反传梯度。

#### `e` 为假

CEM 既可用于离散动作，也常用于连续动作。题里 `(a)` 自己都提到了 Gaussian sampling。

## 3. Entropy Regularized Policy Gradient

正确答案：`b`。

目标函数是

```math
J(\theta)
= \mathbb E_{s\sim p(s),a\sim\pi_\theta}[r(s,a)] + H(\pi_\theta(\cdot|s)).
```

因为

```math
H(\pi_\theta(\cdot|s))
= -\mathbb E_{a\sim\pi_\theta}[\log \pi_\theta(a|s)],
```

所以

```math
J(\theta)
= \mathbb E_{s,a}\big[r(s,a)-\log \pi_\theta(a|s)\big].
```

对 `\mathbb E_a[f_\theta(a)]` 用 score-function 恒等式：

```math
\nabla_\theta \mathbb E_{a\sim\pi_\theta}[f_\theta(a)]
= \mathbb E_{a\sim\pi_\theta}
\big[\nabla_\theta \log\pi_\theta(a)\,f_\theta(a)+\nabla_\theta f_\theta(a)\big].
```

这里

```math
f_\theta(a)=r(s,a)-\log \pi_\theta(a|s),
\qquad
\nabla_\theta f_\theta(a)=-\nabla_\theta \log \pi_\theta(a|s).
```

于是

```math
\nabla_\theta J(\theta)
= \mathbb E_{s,a}
\Big[
\nabla_\theta\log\pi_\theta(a|s)\big(r(s,a)-\log\pi_\theta(a|s)\big)
-\nabla_\theta\log\pi_\theta(a|s)
\Big].
```

整理成

```math
\nabla_\theta J(\theta)
= \mathbb E_{s,a}
\Big[
\nabla_\theta\log\pi_\theta(a|s)\big(r(s,a)-\log\pi_\theta(a|s)-1\big)
\Big].
```

但

```math
\mathbb E_{a\sim\pi_\theta}[\nabla_\theta \log\pi_\theta(a|s)]
= \nabla_\theta \sum_a \pi_\theta(a|s)=0,
```

所以常数 `-1` 可以并入 baseline `b`。最终无偏估计器是

```math
\nabla_\theta J(\theta)\approx
\frac1N\sum_{i=1}^N
\nabla_\theta\log\pi_\theta(a_i|s_i)
\Big[r(s_i,a_i)-\log\pi_\theta(a_i|s_i)-b\Big].
```

这就是选项 `b`。

## 4. Learning in Simulation

正确答案：`a, g`。

### 第一步：把 KL 变成每步转移的 TV 上界

题目给

```math
D_{KL}(P(\cdot|s,a)\|f(\cdot|s,a))<\epsilon^2.
```

由 Pinsker，

```math
\|P(\cdot|s,a)-f(\cdot|s,a)\|_1 \le \sqrt{2}\epsilon,
```

所以

```math
\mathrm{TV}(P(\cdot|s,a),f(\cdot|s,a))
\le \frac{\epsilon}{\sqrt 2}.
```

记这个上界为 `\delta = \epsilon/\sqrt2`。

### 第二步：固定任意策略 `\pi`，比较 `M` 和 `F` 下的 value

记

```math
\Delta_\pi(s)=V_M^\pi(s)-V_F^\pi(s).
```

两边 reward 相同，只是转移不同，于是

```math
\Delta_\pi(s)
=
\gamma\,\mathbb E_{a\sim\pi(\cdot|s)}
\Big[\mathbb E_P[V_M^\pi(s')] - \mathbb E_f[V_F^\pi(s')]\Big].
```

加减 `\mathbb E_P[V_F^\pi(s')]`：

```math
\Delta_\pi(s)
=
\gamma\,\mathbb E_a
\Big[
\mathbb E_P[\Delta_\pi(s')]
+(\mathbb E_P-\mathbb E_f)V_F^\pi(s')
\Big].
```

因为 `0\le r\le 1`，

```math
0\le V_F^\pi(s)\le \frac1{1-\gamma}.
```

于是

```math
\mathrm{span}(V_F^\pi)\le \frac1{1-\gamma}.
```

又有

```math
\left| \mathbb E_p[g]-\mathbb E_q[g]\right|
\le \mathrm{TV}(p,q)\,\mathrm{span}(g),
```

所以

```math
\left|(\mathbb E_P-\mathbb E_f)V_F^\pi\right|
\le \delta\cdot \frac1{1-\gamma}.
```

从而

```math
\|\Delta_\pi\|_\infty
\le
\gamma\|\Delta_\pi\|_\infty
+\frac{\gamma\delta}{1-\gamma},
```

即

```math
\|\Delta_\pi\|_\infty
\le
\frac{\gamma\delta}{(1-\gamma)^2}.
```

代入 `\delta=\epsilon/\sqrt2`：

```math
|J_M(\pi)-J_F(\pi)|
\le
\frac{\gamma\epsilon}{\sqrt2(1-\gamma)^2}.
```

### 第三步：上界 regret

```math
\widetilde R_M(\pi_F^*)
= J_M(\pi_M^*)-J_M(\pi_F^*).
```

加减 `J_F`：

```math
\widetilde R_M(\pi_F^*)
=
\big(J_M(\pi_M^*)-J_F(\pi_M^*)\big)
+\big(J_F(\pi_M^*)-J_F(\pi_F^*)\big)
+\big(J_F(\pi_F^*)-J_M(\pi_F^*)\big).
```

中间项因为 `\pi_F^*` 在 `F` 里最优，所以 `\le 0`。于是

```math
\widetilde R_M(\pi_F^*)
\le
|J_M(\pi_M^*)-J_F(\pi_M^*)|
+|J_M(\pi_F^*)-J_F(\pi_F^*)|.
```

因此

```math
\widetilde R_M(\pi_F^*)
\le
2\cdot \frac{\gamma\epsilon}{\sqrt2(1-\gamma)^2}
= \epsilon\gamma\sqrt2\,(1-\gamma)^{-2}.
```

这正是选项 `a`。

选项 `g` 也是对的，因为它更松。只要验证

```math
\epsilon\gamma\sqrt2(1-\gamma)^{-2}
\le
\epsilon\sqrt2(1-\gamma)^{-5/2},
```

也就是

```math
\gamma\sqrt{1-\gamma}\le 1,
```

而这在 `0<\gamma<1` 时总成立。

## 5. Computing the Policy Gradient

正确答案：`5.1 = d`，`5.2 = f`，`5.3 = c`。

### 先把 MDP 读清楚

从图上读出转移：

- `s_0 \xrightarrow{a_1,\; r=1} s_3`
- `s_3 \xrightarrow{a_1,a_2,\; r=0} s_1`
- `s_1 \xrightarrow{a_1,a_2,\; r=0} s_0`

这是左边一个长度 3 的回环，第 1 步拿奖励 1。

- `s_0 \xrightarrow{a_2,\; r=0} s_2`
- `s_2 \xrightarrow{a_1,\; r=0} s_1`
- `s_2 \xrightarrow{a_2,\; r=0} s_4`
- `s_4 \xrightarrow{a_1,a_2,\; r=2/\gamma} s_0`

这是右边一个长度 3 的回环，只有走到 `s_4\to s_0` 才拿奖励 `2/\gamma`。

政策参数为

```math
\pi_\theta(a_1|s)=1-\theta,\qquad \pi_\theta(a_2|s)=\theta.
```

### 5.1 先写出 `J(\theta)` 再求导

令 `V_0 = J(\theta)`。我们直接从 `s_0` 分两种情况讨论：

#### 走左环：先选 `a_1`

概率 `1-\theta`。立即得到奖励 1，然后两步后确定回到 `s_0`，所以这条分支的 return 是

```math
1+\gamma^3 V_0.
```

#### 走右环：先选 `a_2`

概率 `\theta`。先到 `s_2`。

在 `s_2`：

- 以概率 `1-\theta` 选 `a_1`，走 `s_2\to s_1\to s_0`，整个分支没有即时奖励，贡献 `\gamma^3V_0`
- 以概率 `\theta` 选 `a_2`，走 `s_2\to s_4\to s_0`，在 `s_4\to s_0` 时得到 `2/\gamma`

所以右分支的额外奖励贡献是

```math
\gamma^2\cdot \frac{2}{\gamma}=2\gamma,
```

但这件事只在 `s_2` 再次选中 `a_2` 时发生，所以右环总回报是

```math
2\gamma\theta + \gamma^3V_0.
```

于是

```math
V_0
=(1-\theta)(1+\gamma^3V_0)+\theta(2\gamma\theta+\gamma^3V_0).
```

整理：

```math
V_0 = 1-\theta + 2\gamma\theta^2 + \gamma^3V_0.
```

因此

```math
J(\theta)=V_0=\frac{1-\theta+2\gamma\theta^2}{1-\gamma^3}.
```

求导：

```math
\nabla_\theta J(\theta)
=\frac{-1+4\gamma\theta}{1-\gamma^3}
=\frac{4\theta\gamma-1}{1-\gamma^3}.
```

所以 `5.1` 选 `d`。

### 5.2 最大 achievable return

只需最大化分子

```math
N(\theta)=1-\theta+2\gamma\theta^2.
```

因为

```math
N''(\theta)=4\gamma>0,
```

这是个凸函数，所以区间 `[0,1]` 上最大值必在端点。

比较端点：

```math
N(0)=1,\qquad N(1)=2\gamma.
```

题目给 `\gamma>1/2`，所以 `2\gamma>1`。因此最优点是 `\theta^*=1`。

于是

```math
J(\theta^*)
= J(1)
= \frac{2\gamma}{1-\gamma^3}.
```

所以 `5.2` 选 `f`。

### 5.3 哪些初值会收敛到最优策略

取 `\gamma=3/4`：

```math
\nabla_\theta J(\theta)
= \frac{3\theta-1}{1-27/64}.
```

分母为正，因此梯度方向完全由 `3\theta-1` 决定：

- 若 `\theta>1/3`，梯度为正，更新 `\theta \leftarrow \theta + \alpha \nabla J(\theta)` 会让 `\theta` 继续增大，最终走向最优端点 `\theta=1`
- 若 `\theta<1/3`，梯度为负，会往左走，最终趋向 `\theta=0`
- 若 `\theta=1/3`，梯度刚好 0，会卡在非最优驻点

所以能恢复最优策略的初值集合是

```math
\theta>1/3,
```

即 `5.3` 选 `c`。

## 6. Imitation Learning Analysis

正确答案：`6.1 = a`，`6.2 = b`。

定义每一步在专家分布下的错误率

```math
\epsilon_t
:=
\mathbb E_{s_t\sim p_{\pi^*}(s_t)}
\big[\Pr(\pi_\theta(s_t)\neq \pi^*(s_t)\mid s_t)\big].
```

题目给的是平均错误率

```math
\frac1T\sum_{t=1}^T \epsilon_t \le \epsilon,
```

因此

```math
\sum_{t=1}^T \epsilon_t \le T\epsilon.
```

### 关键耦合论证

把专家轨迹和 learner 轨迹做 coupling。只要在前 `t-1` 步从未出错，那么第 `t` 步状态分布就仍和专家相同。因此

```math
\Pr(\text{到第 }t\text{ 步前已经偏离})
\le
\sum_{i=1}^{t-1}\epsilon_i.
```

于是

```math
\mathrm{TV}\big(p_{\pi_\theta}(s_t),p_{\pi^*}(s_t)\big)
\le
\sum_{i=1}^{t-1}\epsilon_i.
```

又因为 `|r(s_t)|\le 1`，对任意两个分布 `p,q`，

```math
\left|\mathbb E_p[r]-\mathbb E_q[r]\right|
\le 2\,\mathrm{TV}(p,q).
```

### 6.1 一般 state-only reward

```math
J(\pi^*)-J(\pi_\theta)
\le
2\sum_{t=1}^T \mathrm{TV}\big(p_{\pi_\theta}(s_t),p_{\pi^*}(s_t)\big)
\le
2\sum_{t=1}^T\sum_{i=1}^{t-1}\epsilon_i.
```

交换求和：

```math
=
2\sum_{i=1}^{T-1}(T-i)\epsilon_i
\le
2T\sum_{i=1}^T \epsilon_i
\le
2T^2\epsilon.
```

所以

```math
J(\pi^*)-J(\pi_\theta)=O(T^2\epsilon).
```

因此 `6.1` 选 `a`。

### 6.2 如果只有末状态有奖励

这时只看 `t=T`：

```math
J(\pi^*)-J(\pi_\theta)
\le
2\,\mathrm{TV}\big(p_{\pi^*}(s_T),p_{\pi_\theta}(s_T)\big)
\le
2\sum_{i=1}^{T-1}\epsilon_i
\le
2T\epsilon.
```

因此

```math
J(\pi^*)-J(\pi_\theta)=O(T\epsilon),
```

所以 `6.2` 选 `b`。

## 7. IQL Variants

正确答案：`7.1 = a`，`7.2 = a,b,c,d`。

### expectile loss 的两个核心性质

```math
\ell_\tau^2(x)=|1(x>0)-\tau|x^2.
```

- 当 `\tau=0.5` 时，`ell` 退化成普通平方误差，符号方向不重要。
- 当 `\tau\to 1` 时：
  - 最小化 `E[\ell_\tau^2(v-X)]` 会把 `v` 推向 `X` 的上 expectile，极限趋向上界
  - 最小化 `E[\ell_\tau^2(X-v)]` 会把 `v` 推向下 expectile，极限趋向下界

### 7.1 为什么只有 `a` 对 `Q^*` 无偏

#### `a` 正确

```math
L_\psi = \mathbb E_{s,a\sim D}[\ell_\tau^2(V(s)-Q(s,a))].
```

当 `\tau\to 1` 时，`V(s)` 逼近 `\max_a Q(s,a)`。于是 Q update 的目标就是

```math
r(s,a)+\gamma \max_{a'}Q(s',a'),
```

这正是 Bellman optimality backup。

#### `b` 错

这里是

```math
\ell_\tau^2(Q(s,a)-V(s)),
```

方向反了。当 `\tau\to 1` 时，`V(s)` 会更像 `\min_a Q(s,a)`，不是 `\max_a Q(s,a)`。

#### `c` 错

这里直接对

```math
Y=r(s,a)+\gamma Q(s',a')
```

做 expectile regression。若环境随机，`Y` 本身是随机变量。上 expectile 的极限会偏向 `\sup Y`，而最优 Q 需要的是

```math
\mathbb E_{s'}[\max_{a'}Q(s',a')\mid s,a],
```

两者一般不相等。

#### `d` 错

同理，只不过方向相反，会偏向 `\inf Y`。

### 7.2 为什么四个在 `\tau=0.5` 都对 `Q^{\pi_\beta}` 无偏

当 `\tau=0.5` 时，全部都退化成平方误差。

#### `a`、`b`

这时

```math
V(s)=\mathbb E_{a\sim\pi_\beta(\cdot|s)}[Q(s,a)],
```

所以

```math
Q(s,a)=\mathbb E\big[r+\gamma \mathbb E_{a'\sim\pi_\beta}[Q(s',a')]\mid s,a\big]
= \mathcal T^{\pi_\beta}Q(s,a).
```

#### `c`、`d`

这时 loss 就是普通 TD 平方误差，target 样本是

```math
r+\gamma Q(s',a'),
```

而 `a'` 本来就是数据策略 `\pi_\beta` 产生的，所以其条件期望就是 `\mathcal T^{\pi_\beta}`。因此也无偏。

## 8. LLM MDPs

正确答案：`b, d, f, j, m`。

设最终 token 序列为 `X_{1:H}`。由于状态是前缀、动作是下一个 token，且转移是确定拼接：

```math
p_\pi(x_{1:H})=\prod_{t=1}^H \pi(x_t\mid x_{<t}).
```

下面逐项解释。

### `a` 错

baseline 当然有用。REINFORCE 中

```math
\nabla \log \pi(a_t|s_t)\,(G_t-b(s_t))
```

仍是无偏的，但方差会降低。LLM 序列回报极稀疏，baseline 反而更重要。

### `b` 对

环境动力学已经完全已知：

```math
s' = s \oplus a.
```

再学一个 dynamics model 没有原则性收益。

### `c` 错

训练 value function 很难，但不可能说“不可能”。这里

```math
V^\pi(s)=\Pr_\pi(\text{从前缀 }s\text{ 出发最终答对}\mid s)
```

是完全良定义的。

### `d` 对

因为转移确定，

```math
Q^\pi(s,a)=r(s,a)+V^\pi(s\oplus a).
```

也就是说，有了 `V` 以后，scalar `Q(s,a)` 并不携带额外信息。

### `e` 错

policy gradient 依然有 bias-variance tradeoff。MC return、bootstrapping、GAE 都还在。

### `f` 对

control-as-inference 不是“必须用 variational inference”。在这里，状态和转移都显式、确定、无 latent dynamics；VI 只是一般情形下的近似工具，不是必需步骤。

### `g` 错

DQN 对大离散动作空间很不方便，但并非“不兼容”。词表 `W` 仍是离散动作空间。

### `h` 错

TRPO 也兼容离散动作策略。计算贵，不等于理论不兼容。

### `i` 错

offline RL 完全可以在静态 prompt-response 数据集上做。

### `j` 对

next-token prediction 就是 behavioral cloning：

- state = prefix
- action = next token
- 最大化 expert token 的条件 log-likelihood

这就是模仿学习。

### `k` 错

value iteration 和 policy iteration 仍是不同算法。一个是反复 Bellman optimality backup，一个是“策略评估 + 策略改进”交替。

### `l` 错

policy iteration 不保证“一次 policy improvement 就到最优”。很多 MDP 都需要多轮迭代。

### `m` 对

最终状态分布就是整段 token 序列分布，因此

```math
H(p_\pi(s_H))
= H(X_{1:H})
= \sum_{t=1}^H H(X_t\mid X_{<t})
= \sum_{t=1}^H \mathbb E_{s_t\sim p_\pi}[H(\pi(\cdot|s_t))].
```

所以如果每个状态都把动作分布熵最大化，那么最终序列分布熵也最大。

## 9. Multi-Step Dynamics Models

正确答案：`9.1 = (≥, ?)`，`9.2 = (=, =)`。

### 9.1 为什么第一个符号是 `≥`

任取一个 2-step chunk policy `\pi^{(2)}`。我们都可以在原始 1-step MDP 里实现同样的行为：

- 在偶数时刻看到 `s_t`
- 一次采样出 `(a_t,a_{t+1})`
- 第 `t` 步执行 `a_t`
- 第 `t+1` 步执行预先承诺的 `a_{t+1}`

这样得到的 primitive policy 在原 MDP 中的 trajectory distribution 与 `M^{(2)}` 中完全一致，因此 return 完全一样。

所以任何 2-chunk policy 的值都不可能超过原始 1-step MDP 的最优值：

```math
J^{(1)}(\pi_1^*) \ge J^{(2)}(\pi_2^*).
```

这里 even stochastic chunk policy 也没用，因为它虽然可以在 chunk 起点随机抽整个 action chunk，但仍然不能在 chunk 内看到中间随机状态后再改动作。

### 9.1 为什么第二个符号是 `?`

要证明 `?`，必须给出两个方向的反例。

#### 反例 A：`J^{(2)} > J^{(3)}`

设 `H=6`。让关键状态在 `t=2` 才被随机揭示，而第 2 步动作必须和该状态匹配才能得 1 分。

- 2-chunk policy 在 `t=2` 正好重新决策，所以能看到状态再选动作，得到 1。
- 3-chunk policy 的第一块覆盖 `t=0,1,2`，因此第 2 步动作必须在 `t=0` 就承诺，最多拿 `1/2`。

所以 `J^{(2)} > J^{(3)}` 可以发生。

#### 反例 B：`J^{(2)} < J^{(3)}`

还是 `H=6`。这次把关键随机状态放到 `t=3` 才揭示，而第 3 步动作要匹配该状态才能得 1 分。

- 3-chunk policy 的第二块从 `t=3` 开始，所以能看到状态再选动作，得到 1。
- 2-chunk policy 的第二块是 `t=2,3`，因此第 3 步动作必须在 `t=2` 就和第 2 步一起承诺，最多拿 `1/2`。

所以 `J^{(2)} < J^{(3)}` 也可以发生。

因此两者没有统一顺序，选 `?`。

### 9.2 为什么 deterministic dynamics 时两个都是 `=`

如果动力学是 deterministic，那么在 chunk 起点看到 `s_t` 后，只要知道接下来准备执行的动作序列，中间状态就都能被唯一推出来。于是任意最优 1-step policy 都可以无损地“编译”成任意 `n`-step chunk policy。

更具体地说，设 `\pi_1^*` 是一个最优的 primitive policy。对于某个 chunk 起点状态 `s_t`，我们可以按以下递推提前算出接下来 `n` 步它会做什么：

```math
a_t=\pi_1^*(s_t),\quad
s_{t+1}=f(s_t,a_t),\quad
a_{t+1}=\pi_1^*(s_{t+1}),\dots
```

这样就得到整个 chunk

```math
(a_t,\dots,a_{t+n-1}),
```

并且执行结果与原始 1-step policy 完全相同。

所以

```math
J^{(n)}(\pi_n^*) \ge J^{(1)}(\pi_1^*).
```

另一方面，chunk policy 本来就是原 MDP 中的一类受限策略，所以

```math
J^{(1)}(\pi_1^*) \ge J^{(n)}(\pi_n^*).
```

两边合并，得到

```math
J^{(1)}(\pi_1^*) = J^{(2)}(\pi_2^*) = J^{(3)}(\pi_3^*).
```

## 10. Managing Distribution Shift in Offline RL

正确答案：`d`。

我们希望把 KL 约束

```math
D(\pi,\pi_\beta)
=
\mathbb E_{s\sim p_\pi}
\Big[D_{KL}\big(\pi(\cdot|s)\|\pi_\beta(\cdot|s)\big)\Big]
```

通过 reward bonus 写成 Lagrangian 形式：

```math
J(\pi)-\lambda D(\pi,\pi_\beta).
```

因此我们希望 bonus 满足

```math
\mathbb E_{s\sim p_\pi,a\sim\pi}[b(s,a)]
= -D(\pi,\pi_\beta).
```

取

```math
b(s,a)=\log \pi_\beta(a|s)-\log \pi(a|s).
```

则

```math
\mathbb E[b(s,a)]
=
\mathbb E_{s\sim p_\pi}
\left[
\sum_a \pi(a|s)\big(\log \pi_\beta(a|s)-\log \pi(a|s)\big)
\right]
```

```math
=
-\mathbb E_{s\sim p_\pi}
\left[
\sum_a \pi(a|s)\log \frac{\pi(a|s)}{\pi_\beta(a|s)}
\right]
```

```math
= -\mathbb E_{s\sim p_\pi}
\Big[D_{KL}(\pi(\cdot|s)\|\pi_\beta(\cdot|s))\Big]
= -D(\pi,\pi_\beta).
```

这正是我们要的形式，因此选 `d`。

其余选项为什么错：

- `a: -\log \pi(a|s)` 的期望是熵奖励，不涉及 `\pi_\beta`
- `b: \log \pi_\beta(a|s)` 只是一半 cross-entropy 项，缺 `-\log \pi(a|s)`
- `c: \log \pi(a|s)-\log \pi_\beta(a|s)` 的期望是 `+KL`，符号反了，会鼓励远离行为策略

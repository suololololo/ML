# 朴素贝叶斯
what?
<br>
朴素贝叶斯是基于贝叶斯定理和特征条件独立假设的分类方法，利用后验概率最大化和期望风险最小化来进行分类,输入空间$X\subseteq R^n$, 输出空间$Y =\{c_1, c_2, ..., c_k\} $ <br>
后验概率为
$
P(Y = c_k | X = x) = {P(Y = c_k, X = x) \over P(X = x)} \tag{1}
$
后验概率最大化时，所取得类别$c_k$为预测类别,目标函数为
$
f(x) = {\underset{c_k} {\operatorname {arg\,max}}} P(Y = c_k | X = x) \tag{2}
$
然而如何获得后验概率呢？目前我们可以训练数据中知道的信息只有先验概率分布
$
P(Y = c_k)   k= 1,2,...,K \tag{3}
$
$
P(X = x | Y = c_k) = P(X^{(1)} = x^{(1)}, ..., X^{n} = x^{(n)} | Y = c_k) \tag{4}
$
公式4中含有指数量级的参数，假设$x^{(j)}$有$S_j$种取值，总参数就为$K \prod_{j=1}^{n}  S_j$个，参数量非常多。

因此，**朴素贝叶斯做了一个强力的假设，假设特征变量的特征在类确定的情况下是相互独立(这是朴素的原因)**，反应为下式
$
P(X = x | Y = c_k) = P(X^{(1)} = x^{(1)}, ..., X^{n} = x^{(n)} | Y = c_k) \\= P(X^{(1)} = x^{(1)}| Y = c_k),...,P(X^{n} = x^{(n)} | Y = c_k) \\ =
\prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) \tag{5}
$
代入公式1得
$
P(Y = c_k | X = x) = {P(Y = c_k, X = x) \over P(X = x)} \\ = {P(X = x | Y = c_k)P(Y = c_k)\over \sum_{k=1}^k P(X = x, Y = c_k)} \\= {P(Y = c_k)\prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) \over \sum_{k=1}^k P(X = x | Y = c_k) P(Y =c_k)} \\ = {P(Y = c_k)\prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) \over \sum_{k=1}^k \prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) P(Y =c_k)} \tag{6}
$
因此，预测函数为
$
f(x) = {\underset{c_k} {\operatorname {arg\,max}}} {P(Y = c_k)\prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) \over \sum_{k=1}^k \prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) P(Y =c_k)} \tag{7}
$
由于分母对于所有$c_k$是相同的，因此
$
f(x) = {\underset{c_k} {\operatorname {arg\,max}}} P(Y = c_k)\prod_{j = 1}^{n} P(X^{(j)} = x^{(j)}| Y = c_k) \tag{8}
$
上式子表明后验概率最大化

<h2>后验概率最大化与期望风险最小化</h2>
假设选择0-1损失函数(为什么选择这个?)

$
   L(Y, f(x))=\left\{\begin{matrix}1，Y\neq f(x)\\0，Y= f(x) \end{matrix}\right. \tag{8}
$
损失函数的数学期望(期望风险函数)
$
R_{exp}(f) = E[L(Y,f(x))] \\=
E_x\sum_{k = 1}^{K} [L(c_k,f(x))P(c_k|X=x)] \tag{9}
$
为了使得期望风险最小化，只需要使得每个求和项最小

$
f(x) = {\underset{c_k} {\operatorname {arg\,min}}} \sum_{k = 1}^{K} [L(c_k,f(x))P(c_k|X=x)]  \\= {\underset{c_k} {\operatorname {arg\,min}}}  \sum_{k=1}^{K}P(c_k \neq f(x)|X=x) \\= {\underset{c_k} {\operatorname {arg\,min}}}  (1 - P(c_k = f(x)|X=x)) \\={\underset{c_k} {\operatorname {arg\,max}}} P(c_k = f(x)|X=x) \tag{10}
$
这样一来就得到了后验概率最大化
$
f(x) ={\underset{c_k} {\operatorname {arg\,max}}} P(c_k = f(x)|X=x) \tag{11}
$

<h2>那么朴素贝叶斯怎么学习呢？</h2>

根据上面的公式推导可知，要估计后验概率，只需要估计出先验概率$P(Y =c_k) $和$P(X^{(j)}=x ^{(j)}|Y =c_k)$ <br>
容易得到
$
P(Y = c_k) = {\sum_{i = 1} ^{N} I(y_i = c_k) \over N}  \tag{12}
$
$
P(X^{(j)}=x ^{(j)}|Y =c_k) = {P(X^{(j)}=x ^{(j)},Y =c_k) \over P(Y =c_k)} \\= {{\sum_{i = 1} ^{N} I(y_i = c_k, x_i^j = x^{(j)} ) \over N}  \over{\sum_{i = 1} ^{N} I(y_i = c_k) \over N} }
\\= {\sum_{i = 1} ^{N} I(y_i = c_k, x_i^j = x^{(j)} ) \over \sum_{i = 1} ^{N} I(y_i = c_k)}
$

$x_i^j$是第i个样本的j个特征， $x^{(j)}$第j个特征的取值,
计算出先验概率就可以进行预测

选择0-1损失函数的原因<br>
简单容易使用，直观容易理解，但是0-1损失函数对所有错误分类的点同样看待，即使有可能有些错误离正确分类很近，对错的比较离谱的点没有惩罚
Loss函数不可导，不连续，非凸

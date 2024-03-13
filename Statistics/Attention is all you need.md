---
date: 2017/06
link: https://arxiv.org/pdf/1706.03762.pdf
tags:
  - Transformer
  - Attention
cssclasses:
  - width-100
---
## Keywords
- [[Attention is all you need#Effect of Multi-Head Attention|Effect of Multi-Head Attention]]


## 0. Abstract
Extended neural GPU, ByteNet, ConvS2S와 같은 convolution network의 경우, input과 output의 길이차이가 많아짐에 따라 연산 횟수가 점점 증가하게 된다. 연산횟수는 model의 구조에 따라 달라지지만, 대부분의 경우 linear하게 혹은 logarithmic하게 증가하게된다. 하지만, <span style="background:#fff88f">Transformer</span>에서는 <span style="background:#fff88f">constant number</span>로 결정된다. [[Attention is all you need#1. 1. Attention|Attention]]의 averaging attention-weighted positions로 인해 유효해상도(Effective resolution)이 감소하지만, 이는 Multi-Head Attention을 통해 대응할 수 있다.
또한, NLP에서 문장 (sequence)에서 단어간의 연관성이 있는데, 모든 <span style="background:#fff88f">단어들 간의 관계성을 파악하기 위해서 각 단어에 대한 전파력이 앞으로만 전달되는 것이 아닌 전체적으로 전파되도록 Attention</span>을 활용하는 것이 효과적이다.
# 1. Model Architechture

![center|500](image_20240311111549.png)


모델의 전체적인 구조는 endcoder와 decoder로 분리되어있다. encoder부분은 sentence를 입력받아 context로 변환하는 구조를 가지고 있으면 <font color="#00b050">endcoder</font>는 여러 <font color="#00b050">encoder</font> 블럭을 serial 하게 연결하여 구성되어있다. 해당 논문은 Transformer구조를 하나의 encoder 블럭으로 설정하였다. Decoder는 output sentence를 하나씩 shift하여 다음 단어를 예측하는 방식의 input을 채택하거나, encoder의 output인 context를 input으로 활용하여 output인 sentence를 결정한다.
Figure 1.에서 [[Attention is all you need]]는 stacked self-attention, point-wise, fully connected layer를 활용하였으며, 크게 다를 것 없지만, self-attention 매커니즘을 적용하였다.

## 1. 1. Attention
Attention 함수의 기본적인 구조는 <span style="background:#fff88f"><font color="#ff0000">query, key-value pair를 output과 mapping</font></span>하는 것에 있다. Output은 value들의 weighted sum으로 해당 wieght들은 query와 key에 특정 compatibility function을 적용한 결과이다.
$$ Attention(Q, K, V) = softmax(\frac {QK^T}{\sqrt{d_k}}) \cdot V$$
$$ d_k : dim\ of\ Query\  \& \ Key$$
*Attention* function을 살펴보면, Q와 K를 matmul하여 softmax라는 compatibilitiy funcion을 적용한 후, V와 matmul하였다. 대체로, 2가지 방식의 attention이 사용되었는데, <font color="#0070c0">Additive attention</font>과 <font color="#0070c0">Dot-product attention</font>이 있다. Dot-product attention은 Additive와 이론적으로 동일한 complexity를 가지지만, <span style="background:rgba(255, 183, 139, 0.55)">dot-product attention이 실험에서 훨씬 빠르고, space efficiency가 높고, matrix multiplication code로 최적화가 잘된다는 장점이 있다.</span> 
##### Effect of $d_k$
*softmax function*을 보면 내부에 $1 / \sqrt{d_k}$ 가 scaling factor로 사용된다. $d_k$ 가 작은 값을 때는 dot-product attention과 additive가 비슷하게 작동한다. 하지만, $d_k$ 가 큰 값을 가질 때, 즉, Query와 Key matrix가 높은 dimension을 가질 때는 softmax가 대부분이 절댓값이 큰값을 가질 확률이 높아져 <font color="#de7802">softmax가 extremely small gradient</font>값을 가지게 된다. 이에 대응하기 위해 dot product를 $1 / d_k$ 로 scaling 해준다. 이 연산으로 인해 값들이 큰값이 아닌 0 근처, 즉 <font color="#de7802">softmax함수의 gradient가 0이 아닌 값을 가지는 부분으로 모아주기 때문에 학습에 용이해지는 효과를 가질 수 있다.</font>
[참고 : Gradient of Softmax](https://velog.io/@hjk1996/Cross-Entropy%EC%99%80-Softmax%EC%9D%98-%EB%AF%B8%EB%B6%84)
##### <span style="background:rgba(173, 239, 239, 0.55)">왜 </span>$d_k$ <span style="background:rgba(173, 239, 239, 0.55)">가 크면, dot product의 결과도 큰 값을 가질까?</span>
이 논문에서 Query와 Key의 (예를 들어, 열벡터나 행벡터) *q*와 *k*의 각 요소들을 independent random variable ~ $N (0, 1)$ 로 가정하였다. 따라서 $q \cdot k = \sum\limits_{i=1}^{d_k} q_i k_i$ 는 $N(0, d_k)$ 분포를 따른다. 따라서 $d_k$ 가 커짐에 따라 값들의 분포가 넓어져 큰 값을 가질 수 있는 가능성이 생기게 된다.


## 1. 2. Multi-Head Attention
[[Attention is all you need#1. 1. Attention|Attention]] 은 single attention function으로 $d_{model}$ 의 차원을 가진다. 하지만 이 대신, V, K, Q에 학습된 Weight들을 곱하여 linearly projection하게 된다. 대신, $d_{model} / h$ 로 header의 수만큼 각각의 Dot-Product Attention의 computational cost를 Single attention function을 수행한 것과 유사하게 맞춰줄 수 있다.
![center|400](image_20240313155043.png)
##### Equation Analysis

$$MultiHead(Q, K,V) = Concat(head_1, head_2, \cdots , head_h) \cdot W^o \  \ \cdots \ (eq\ 1.)
$$
 $${where} \ \  head_i = Attention(Q\cdot W_i^Q, K\cdot W_i^K, V \cdot W_i^V) \ \ \cdots \ \ (eq 2.)$$

eq 1. 은 위의 Fig에서 마지막 Concat과 Linear에 관련된 수식으로, multi-head 의 output을 concatenation하여 linear projection하였다. 또한, 각각의 header는 Q, K, V를 linear projection하고, Attention operation을 수행한다. (eq 2.) 이는 위의 Fig에서 아랫부분에 해당한다.

##### Effect of Multi-Head Attention
- Parallelization (계산 효율성 ⬆️)
- <span style="background:rgba(255, 183, 139, 0.55)">Jointly combine infor. from 각각의 subspaces (linear projection)의 다른 representation</span>


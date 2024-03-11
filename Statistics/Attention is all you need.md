---
date: 2017/06
link: https://arxiv.org/pdf/1706.03762.pdf
tags:
  - Transformer
  - Attention
cssclasses:
  - width-100
---

## 0. Abstract
Extended neural GPU, ByteNet, ConvS2S와 같은 convolution network의 경우, input과 output의 길이차이가 많아짐에 따라 연산 횟수가 점점 증가하게 된다. 연산횟수는 model의 구조에 따라 달라지지만, 대부분의 경우 linear하게 혹은 logarithmic하게 증가하게된다. 하지만, <span style="background:#fff88f">Transformer</span>에서는 <span style="background:#fff88f">constant number</span>로 결정된다. [[Attention is all you need#1. 1. Attention|Attention]]의 averaging attention-weighted positions로 인해 유효해상도(Effective resolution)이 감소하지만, 이는 Multi-Head Attention을 통해 대응할 수 있다.

# 1. Model Architechture

![center|500](image_20240311111549.png)


모델의 전체적인 구조는 endcoder와 decoder로 분리되어있다. encoder부분은 sentence를 입력받아 context로 변환하는 구조를 가지고 있으면 <font color="#00b050">endcoder</font>는 여러 <font color="#00b050">encoder</font> 블럭을 serial 하게 연결하여 구성되어있다. 해당 논문은 Transformer구조를 하나의 encoder 블럭으로 설정하였다. Decoder는 output sentence를 하나씩 shift하여 다음 단어를 예측하는 방식의 input을 채택하거나, encoder의 output인 context를 input으로 활용하여 output인 sentence를 결정한다.
Figure 1.에서 [[Attention is all you need]]는 stacked self-attention, point-wise, fully connected layer를 활용하였으며, 크게 다를 것 없지만, self-attention 매커니즘을 적용하였다.

## 1. 1. Attention
Attention 함수의 기본적인 구조는 <span style="background:#fff88f"><font color="#ff0000">query, key-value pair를 output과 mapping</font></span>하는 것에 있다. Output은 value들의 weighted sum으로 해당 wieght들은 query와 key에 특정 compatibility function을 적용한 결과이다. 해당 내용은 



## Hovers

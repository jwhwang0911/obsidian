---
cssclasses:
  - width-100
  - col-lines
  - row-alt
  - row-lines
  - table-small
tags:
  - Attention
---
## Preliminary - 기존의 Attention
$$  Attention(Q, K, V) = softmax(\frac {QK^T}{\sqrt{d_k}}) \cdot V $$
- Basic Attention : $softmax$를 사용
	 Value 앞에 붙는 부분 => Weight, 따라서 weighted sum이라고 생각할 수 있음.
	 Constriant : $softmax$때문에 양의 weight들만 적용 가능
- Linear Attention


## 📝cosFormer : Rethinking SOFTMAX in Attention
### 1.1 Analysis of SOFTMAX Attention


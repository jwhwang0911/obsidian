---
cssclasses:
  - col-lines
  - row-alt
---
# 1. Generalized Least Square

# 3. Inverse Covariance Matrix
- Zero-order Regression (<font color="#0070c0">GLS</font>)
> \[Properties]
> - Invertible : Full rank matrix
> - Symmetric :$Q=Q^T$
> - Positive Definite : $\forall x \in ℝ^{N \times 1},\\ x^{T}Qx > 0$ (non-zero igenvalue) 

다음과 같은 성질을 만족하는 matrix를 생성하여 이를 optimal parameter로 설정하여 covariance를 고려한 neural network를 구축할 수 있다.

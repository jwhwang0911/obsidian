---
tags:
  - Adaptive_Sampling
date:
---


>Monte Carlo Rendering => Consistent (=Inf Sample이면 correct solution), unbias
>Bidirectional Also.
>하지만,
>Variance가 높아 이를 해결하기위한 Variance Reduction이나 Adaptive Sampling이 많이 등장

## 1. Priori Method
	Light Transport Eq. 기반으로 Analyticaly Analysis
	 따라서 몇몇 Scene Parameter (BRDF, 3D geometry) 에 접근해야함


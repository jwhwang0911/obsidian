---
tags:
  - Diffusion
  - Vision
date: 2020/11
cssclasses:
  - width-100
---
# 1. Diffusion Process
![center](image_20240322170201.png)   
모든 step을 Markov process로 정의함.
📢 Markov process : 다음 step의 random variable은 이전 step의 random variable에만 영향을 받음. 
#### <span style="background:#fff88f">Forward Process</span> : Noise를 더해가는 과정
- 정의 : $q(\vec{x_{t}}|{\vec{x}_{t-1}})=N(\vec{x}_{t}; \sqrt{1-\beta_{t}} \vec{x}_{t-1}, \beta_{t} \cdot \vec{I})$ 
- $\beta_t$ : 이미지가 매 step마다 망가지는 정도 (hyperparameter) - \[0.02 ~ 0.0004], linear schedule
#### <span style="background:#fff88f">Backward Process</span> : 이미지를 복원하는 과정
- 정의 : $p(\vec{x_{t-1}}|{\vec{x}_{t-1}})=N(\vec{x}_{t}; \vec{\mu}_{\theta}(\vec{x}_{t}, t), \Sigma_{\theta}(\vec{x}_{t}, t))$ 

## Objective Function

---
tags:
  - Diffusion
cssclasses:
  - width-100
---
# 1. Sampling for DDRM
#### 1.1 Linear System
$$\begin{aligned}
y& = H \cdot x + z \\
y& : \text{noisy image} \\
H& : \text{Transform Matirx s.t. for Tasks} \\
x& : \text{Reference image} \\
z& : \text{nosie that added from reference } x
\end{aligned}$$
#### 1.2 Sampling from DDRM
$$\begin{aligned}
H& = U \Sigma V^T\\
\bar{x}_t& = V^T x_t \\
\bar{y}& =\Sigma^{\dagger}U^T y= (\Sigma^T \Sigma)^{-1}\Sigma U^Ty
\end{aligned}
$$
- 여기서 Denoising task에 적용하는 Sigma는 결국 정사각대각행렬의 각 대각 원소 값 = 1이고 U 와 V는 모두 Identity Matrix
$$\begin{aligned}
p_\theta^{T} (\bar{x}_T^{(i)}|y) &= \begin{cases}
									N(\bar{y}^{(i)}, \sigma^2_T-\frac{\sigma_y^2}{s_i^2}), \quad\;if\;s_i>0\quad (1)\\
									N(0, \sigma^2_T),\quad \quad \quad \quad\ \;if\;s_i=0 \quad(2)
									\end{cases} \\
p_\theta^{T} (\bar{x}_t^{(i)}|x_{t+1}, x_0, y) &= \begin{cases} 
												N(\bar{x}_{\theta, t}^{(i)} + \sqrt{1-\eta^2}\frac{\sigma_t}{\sigma_{t+1}}(\bar{x}_{t+1}^{(i)}-\bar{x}_{\theta, t}^{(i)}), \ \eta^2\sigma^2_t) \quad\; if\;s_i = 0 \quad \quad \ (3) \\
												N(\bar{x}_{\theta, t}^{(i)} + \sqrt{1-\eta^2}\frac{\sigma_t s_i}{\sigma_{y}}(\bar{y}^{(i)}-\bar{x}_{\theta, t}^{(i)}), \ \eta^2\sigma^2_t) \quad\; \ \ if\;s_i \sigma_t < \sigma_y \quad (4) \\
												N((1-\eta_b)\bar{x}_{\theta, t}^{(i)} + \eta_b \bar{y}^{(i)}, \sigma_t^2-\frac{\sigma_y^2}{s_i^2}\eta_b^2) \; \quad \quad \quad \quad if\; s_i\sigma_t \geq \sigma_y \quad(5)
\end{cases}
\end{aligned}
$$
- Corresponing $\eta = 1$ and $\eta_b = \frac{2\sigma_t^2}{\sigma_t^2 + \sigma^2_y/s_i^2}$ 
##### 1.3 Denoising task + Heterogeneous noise *Z*

$$\begin{aligned}
p_\theta^{T} ({x}_T^{(i)}|y) &= \begin{cases}
									N({y}^{(i)}, \sigma^2_T-{\sigma_y^2}), \quad\;if\;s_i>0\quad (1)\\
									\end{cases} \\
p_\theta^{T} ({x}_t^{(i)}|x_{t+1}, x_0, y) &= \begin{cases} 
												N({x}_{\theta, t}^{(i)} + \sqrt{1-\eta^2}\frac{\sigma_t}{\sigma_{y^{(i)}}}(\bar{y}^{(i)}-{x}_{\theta, t}^{(i)}), \ \eta^2\sigma^2_t) \quad\; \ \ if\; \sigma_t < \sigma_{y^{(i)}} \quad (2) \\
												N((1-\eta_b){x}_{\theta, t}^{(i)} + \eta_b \bar{y}^{(i)}, \sigma_t^2-\sigma_{y^{(i)}}\eta_b^2) \; \quad \quad \quad \quad if\; \sigma_t \geq \sigma_{y^{(i)}} \quad(3)
\end{cases}
\end{aligned}
$$
- Corresponing $\eta = 1$ and $\eta_b^{(i)} = \frac{2\sigma_t^2}{\sigma_t^2 + \sigma^2_{y^{(i)}}}$ 

# 2. Code Review
```python
class Denoising(H_functions):

	def __init__(self, channels, img_dim, device):
		self._singulars = torch.ones(channels * img_dim**2, device=device)  
	def V(self, vec):
		return vec.clone().reshape(vec.shape[0], -1)
	def Vt(self, vec):
		return vec.clone().reshape(vec.shape[0], -1)
	def U(self, vec):
		return vec.clone().reshape(vec.shape[0], -1)
	def Ut(self, vec):
		return vec.clone().reshape(vec.shape[0], -1)
	def singulars(self):
		return self._singulars
	def add_zeros(self, vec):
		return vec.clone().reshape(vec.shape[0], -1)
```
```python
def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
singulars = H_funcs.singulars()
Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
Sigma[:singulars.shape[0]] = singulars
U_t_y = H_funcs.Ut(y_0)
Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]
```
singulars : $\Sigma$ , 1D vector \[Size : res_x * res_y * 3 (rgb)]
Sigma -> res_x * res_y * 3
Sigma를 초기화 => singulars를 대입
`{python icon} Sigma` = singulars : 1D vector
`{python icon } Sig_inv_U_t_y` = Sigma inverse * U^T * y
```python ln:7
#initialize x_T as given in the paper
largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)
```
$$
\begin{aligned}
q(x_{t-1}| x_{t}, x_0) &= N(x_{t-1}; \tilde{\mu_t}(x_t. x_0), \tilde{\beta_{t}} \bf{\text{I}}) \\
\tilde{\beta_{t}} &= \frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}} \beta_t \\
& \text{DDPM Equation}
\end{aligned}
$$
`{python icon }largest_alphas` = $\bar{\alpha_T}$ 들 중 가장 마지막, 즉 T가 가장 큰 $\bar{\alpha}$를 계산

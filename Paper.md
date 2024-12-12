---
banner: "https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2790&q=80"
banner_x: 0.5
banner_y: 0.05
banner_icon: 📚
cssclasses:
  - row-alt
  - table-small
  - col-lines
  - row-lines
sticker: emoji//1f4d3
---
# BackGround Study
- [ ] Monte Carlo and Quasi-Monte Carlo Sampling
	- [x] Monte Carlo Method
	- [x] Pseudorandom Number Generators
	- [x] Variance Reduction
	- [ ] Quasi-Monte Carlo Construction
		- [x] Introduction
		- [x] Main Constructions : basic principles
		- [x] Lattices
		- [ ] Digital nets and sequences - Remain Later
			- [x] Introduction
			- [x] Sobol' Sequence
			- [ ] Faure sequence
			- [ ] Niederreiter sequences
			- [ ] Improvements to the original constructions of Halton, Sobol', Niederreiter, and Faure
			- [ ] Digital net constructions and extensions
		- [ ] Recurrence-based point sets
		- [ ] Quality measures
			- [ ] Discrepancy and related measures
			- [ ] Criteria based on Fourier and Walsh decompositions
			- [ ] Motivation for going beyond error bounds
	- [x] Markov Chain Monte Carlo (MCMC) - 대강 살펴봄
- [ ] Metropolis-Hastings algorithm \[PBRT]
- [ ] "Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks", **SIG24**
- [ ] MLT (metroplis Light transport) ; Correlated Image

# Research


## 🎯 Statistics & Deep Learning
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Statistics"
SORT date
```

## 🟦 Rendering Core

```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Rendering_Core"
SORT date
```

## 🟧 Vision
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Vision"
WHERE contains(tags, "Vision")
SORT date
```

## 🟨 Diffusion
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Vision"
WHERE contains(tags, "Diffusion")
SORT date
```


## 🟩 Realtime Denoising
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Realtime_Denoising"
SORT date
WHERE file.name != "Realtime_Denoising"
```

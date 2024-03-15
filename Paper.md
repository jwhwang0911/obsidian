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
## <font color="#ff0000">⚠논문 읽을 때, 식의 의미를 파악하면서 읽자⚠️ </font>
잘 했으면 서 왜 안해!


# BackGround Study

- [ ] generalized least squares (GLS) <font color="#ff0000">[나중에 더 살펴봐야할 부분이 있음]</font>
- [ ] Self-Attention
	- [ ] Attention is all you need
	- [ ] Uformer : A general u-shaped transformer for image restoration
	- [ ] Show, attend and tell: Neural image caption generation with visual attention
	- [x] Monte carlo denoising via auxiliary feature guided self-attention
	- [ ] Restormer: Efficient trnasformer for high-resolution image restoration
	- [ ] Swinir: Image restoration using swin transformer
	- [ ] Swin transformer: Hierarchical vision transformer using shifted windows
	- [ ] Scaling local self-attention for parameter efficient visual backbones
	- [ ] Pre-trained image processing transformer
- [ ] Joint Self-Attention for MC denoising
- [ ] Deep combiner
- [ ] Nvidia ReSTIR
- [ ] Shift mapping
- [ ] CRN (참고용)


> 대충 내가 이해한 주제는
> "Independent Path tracing이 아니라 (pixel이 서로 independent) shift mapping 같이 correlation을 줘서 variance를 낮춘 scene에서 denoising 하는건 어떰?"
> "현재 Attention에서는 softmax function을 사용해서 Negative correlation의 경우 고려하지 못하는 경우도 있음. 만약 Positive Corr있으면 Weight이 양수이면 안될 것 같은데"

1. 단순하게 correlation map을 사용하면? => 너무 무거움


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

## 🟧 Inverse Rendering
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Inverse_Rendering"
SORT date
```

## 🟨 Offline Denoising
```dataview
TABLE 
	without id
	file.link as "논문", date, tags
FROM "Offline Denoising"
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

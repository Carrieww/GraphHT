---
layout: default
title: GraphHT
---

<div style="text-align: center;">
  <h1>GraphHT: A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs</h1>
  <p>Yun Wang, Chrysanthi Kosyfaki, Sihem Amer-Yahia, Reynold Cheng</p>
  <p><strong><a href="#">[Paper]</a></strong> <strong><a href="https://github.com/Carrieww/GraphHT">[Code]</a></strong></p>
</div>

![Framework Overview](img/framework.png)

## Abstract

Hypothesis testing is a statistical method used to draw conclusions about populations from sample data, typically represented in tables. With the prevalence of graph representations in real-life applications, hypothesis testing on graphs is gaining importance. In this work, we formalize node, edge, and path hypotheses on attributed graphs. We develop a sampling-based hypothesis testing framework, which can accommodate existing hypothesis-agnostic graph sampling methods. To achieve accurate and time-efficient sampling, we then propose a Path-Hypothesis-Aware SamplEr, PHASE, an $m$-dimensional random walk that accounts for the paths specified in the hypothesis. We further optimize its time efficiency and propose $\text{PHASE}_{\text{opt}}$. Experiments on three real datasets demonstrate the ability of our framework to leverage common graph sampling methods for hypothesis testing, and the superiority of hypothesis-aware sampling methods in terms of accuracy and time efficiency. 

## Key Features

- **Efficient Sampling**: Our framework provides efficient sampling methods for large attributed graphs
- **Hypothesis Testing**: Support for various hypothesis testing scenarios on nodes, edges, and paths
- **PHASE Sampler**: A novel Path-Hypothesis-Aware SamplEr using m-dimensional random walks
- **Optimized Implementation**: Includes both basic and optimized versions (PHASE<sub>opt</sub>) of the sampler

## Experimental Results

Our comprehensive experiments on MovieLens, DBLP, and Yelp datasets demonstrate the effectiveness of our approach:

### DBLP Dataset
![DBLP Results](img/DBLP.png)

### Yelp Dataset
![Yelp Results](img/Yelp.png)

$\text{PHASE}_{\text{opt}}$ consistently outperforms the top 5 hypothesis-agnostic samplers across most hypotheses and sampling proportions. Its advantage is less pronounced for easier hypotheses with abundant relevant structures in $\mathcal{G}$, where hypothesis-agnostic samplers perform comparably. However, for harder hypotheses (e.g., with **(hard)** tag), $\text{PHASE}_{\text{opt}}$ shows clear superiority, achieving the highest accuracy at all sampling proportions.

## Citations

```bibtex
@article{wang2024sampling,
  title={A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs},
  author={Wang, Yun and Kosyfaki, Chrysanthi and Amer-Yahia, Sihem and Cheng, Reynold},
  journal={[Journal Name]},
  year={2024}
}
```

## Contact

For questions and feedback, please contact [carrie07@connect.hku.hk](mailto:carrie07@connect.hku.hk) 

.wrapper {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
} 
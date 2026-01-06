---
layout: default
title: GraphHT
---

<!-- 引入 MathJax -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">

<div style="text-align: center;">
  <h1>GraphHT: A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs</h1>
  <p><strong>Yun Wang, Chrysanthi Kosyfaki, Sihem Amer-Yahia, Reynold Cheng</strong></p>
  <p><strong><a href="https://arxiv.org/pdf/2403.13286">[Paper]</a></strong> &nbsp;&nbsp; <strong><a href="https://github.com/Carrieww/GraphHT">[Code]</a></strong></p>
</div>

<p align="center">
  <img src="img/framework.png" alt="Framework Overview" style="max-width: 100%;">
</p>

<h2>Abstract</h2>

<div style="text-align: justify;">
<p>Hypothesis testing is a statistical method used to draw conclusions about populations from sample data, typically represented in tables. With the prevalence of graph representations in real-life applications, hypothesis testing on graphs is gaining importance. In this work, we formalize node, edge, and path hypotheses on attributed graphs. We develop a sampling-based hypothesis testing framework, which can accommodate existing hypothesis-agnostic graph sampling methods. To achieve accurate and time-efficient sampling, we then propose a <strong>Path-Hypothesis-Aware SamplEr</strong>, PHASE, an <em>m</em>-dimensional random walk that accounts for the paths specified in the hypothesis. We further optimize its time efficiency and propose \(\text{PHASE}_{\text{opt}}\). Experiments on three real datasets demonstrate the ability of our framework to leverage common graph sampling methods for hypothesis testing, and the superiority of hypothesis-aware sampling methods in terms of accuracy and time efficiency.</p>
</div>

<h2>Key Features</h2>

<div style="text-align: justify;">
<ul>
  <li><strong>Efficient Sampling</strong>: Our framework provides efficient sampling methods for large attributed graphs</li>
  <li><strong>Hypothesis Testing</strong>: Support for various hypothesis testing scenarios on nodes, edges, and paths</li>
  <li><strong>PHASE Sampler</strong>: A novel Path-Hypothesis-Aware SamplEr using <em>m</em>-dimensional random walks</li>
  <li><strong>Optimized Implementation</strong>: Includes both basic and optimized versions (\(\text{PHASE}_{\text{opt}}\)) of the sampler</li>
</ul>
</div>

<h2>Experimental Results</h2>

<div style="text-align: justify;">
<p>Our comprehensive experiments on MovieLens, DBLP, and Yelp datasets demonstrate the effectiveness of our approach:</p>

<h3>DBLP Dataset</h3>
<p align="center">
  <img src="img/DBLP.png" alt="DBLP Results" style="max-width: 100%;">
</p>

<h3>Yelp Dataset</h3>
<p align="center">
  <img src="img/Yelp.png" alt="Yelp Results" style="max-width: 100%;">
</p>

<p>
\(\text{PHASE}_{\text{opt}}\) consistently outperforms the top 5 hypothesis-agnostic samplers across most hypotheses and sampling proportions. Its advantage is less pronounced for easier hypotheses with abundant relevant structures in \(\mathcal{G}\), where hypothesis-agnostic samplers perform comparably. However, for harder hypotheses (e.g., with <strong>(hard)</strong> tag), \(\text{PHASE}_{\text{opt}}\) shows clear superiority, achieving the highest accuracy at all sampling proportions.
</p>
</div>

<h2>Citation</h2>

<pre><code>@article{wang2024sampling,
  title={A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs},
  author={Wang, Yun and Kosyfaki, Chrysanthi and Amer-Yahia, Sihem and Cheng, Reynold},
  journal={[Journal Name]},
  year={2024}
}
</code></pre>

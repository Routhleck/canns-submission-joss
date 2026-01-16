---
title: "CANNs: Continuous Attractor Neural Networks Toolkit with ASA for Attractor Structure Analysis"
tags:
  - Python
  - continuous attractor neural networks
  - computational neuroscience
  - topological data analysis
authors:
  - name: Sichao He
    orcid: 0009-0009-4564-1753
    affiliation: 1
  - name: Aiersi Tuerhong
    orcid: 0009-0006-4603-4734
    affiliation: 2
  - name: Shangjun She
    orcid: 0009-0008-4490-7612
    affiliation: 3
  - name: Tianhao Chu  
    orcid: 0000-0001-9910-9361
    affiliation: 3
  - name: Yuling Wu
    orcid: 0009-0001-5303-5413
    affiliation: 1
  - name: Junfeng Zuo
    affiliation: 1
  - name: Si Wu
    orcid: 0000-0001-9650-6935
    corresponding: true
    affiliation: "1, 3, 4, 5, 6"
      
affiliations:
  - name: Peking-Tsinghua Center for Life Sciences, Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China
    index: 1
  - name: College of Mathematics and Statistics, Chongqing University, Chongqing, China
    index: 2
  - name: School of Psychological and Cognitive Sciences, Peking University, Beijing, China
    index: 3
  - name: School of Psychology and Cognitive Sciences, Peking University, Beijing, China
    index: 4
  - name: PKU-IDG/McGovern Institute for Brain Research, Peking University, Beijing, China
    index: 5
  - name: Center of Quantitative Biology, Peking University, Beijing, China
    index: 6

date: 16 January 2026
bibliography: paper.bib
---
# Summary

CANNs (Continuous Attractor Neural Networks toolkit) is a Python library built on BrainPy, a powerful framework for brain dynamics 
programming. It streamlines experimentation with continuous attractor neural networks and related brain-inspired models. The library 
delivers ready-to-use models, task generators, analysis tools, and pipelines—enabling neuroscience and AI researchers to move quickly 
from ideas to reproducible simulations.


# Statement of need

Continuous Attractor Neural Networks (CANNs) provide a theoretical framework for understanding how the brain encodes continuous variables through stable neural activity patterns. Despite their importance in computational neuroscience, CANN research suffers from fragmentation: researchers implement models from scratch using incompatible codebases, creating reproducibility barriers and steep learning curves.

CANNs addresses this gap by providing a unified Python toolkit built on BrainPy [@wang2023brainpy]. It delivers standardized CANN implementations, integrated task generation and analysis pipelines, and high-performance computation via JAX JIT compilation with optional Rust acceleration.

# Software design

![Layer hierarchy of the CANNs library showing five levels: Application (Pipeline orchestration), Functional (Task, Trainer, Analyzer, Utils modules), Core Models (CANN implementations), Foundation (BrainPy/JAX and Rust FFI backends), and Hardware (CPU/GPU/TPU support).\label{fig:architecture}](img/architecture.png)

The CANNs library follows a modular architecture (\autoref{fig:architecture}) with five independent modules: Models, Tasks, Analyzers, Trainers, and Pipeline. This separation ensures maintainability and extensibility through abstract base classes that define standard interfaces.

The library supports CANN modeling and simulation, experimental data analysis, brain-inspired learning, and automated parameter sweeps. All models inherit from BrainPy's `DynamicalSystem` base class [@wang2023brainpy], leveraging JAX JIT compilation for GPU/TPU acceleration. A companion Rust library provides optional accelerated backends for performance-critical operations.

# Related Works

While general-purpose neural network simulators like NEST [@Gewaltig:NEST], Brian 2 [@stimberg2019brian], and NEURON [@hines1997neuron] exist, they lack specialized support for continuous attractor dynamics. Existing CANN implementations remain fragmented, lab-specific codebases without standardized APIs or comprehensive tooling.

CANNs builds upon BrainPy [@wang2023brainpy], a modern brain dynamics framework leveraging JAX [@jax2018github] for JIT compilation and GPU/TPU acceleration. CANNs extends BrainPy with CANN-specific abstractions: standardized model implementations, task-generation APIs, analysis pipelines, and optional Rust-accelerated backends for performance-critical operations.

# AI usage disclosure

AI-assisted tools were used during the development of this software in limited, auxiliary capacities. Sourcery AI was employed for
automated code quality reviews and refactoring suggestions in pull requests. Large language models (including Claude and ChatGPT) assisted
with documentation writing, including docstring generation, tutorial content drafting, and technical writing refinement. However, all core
library code—including model implementations, task generators, analyzers, and algorithmic components—was written by human developers. All
AI-generated content was reviewed, validated, and edited by the authors to ensure technical accuracy and consistency with project standards.
The software architecture, design decisions, and scientific contributions represent original human intellectual work.

# Acknowledgements

We thank Aiersi Tuerhong and Shangjun She for their collaborative development contributions to the library implementation. We are grateful
to Tianhao Chu, Yuling Wu, and Junfeng Zuo for valuable discussions, feedback, and guidance throughout the project development. We especially
thank Si Wu for overall scientific guidance and mentorship. We acknowledge the BrainPy development team for providing the foundational
framework upon which this library is built, and the broader open-source community for tools and libraries that enabled this work.

# References
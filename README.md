# Adversarial Uncertainty Quantification in Physics-Informed Neural Networks
We present a deep learning framework for quantifying and propagating uncertainty in systems governed by non-linear differential equations using physics-informed neural networks. Specifically, we employ latent variable models to construct probabilistic representations for the system states, and put forth an adversarial inference procedure for training them on data, while constraining their predictions to satisfy given physical laws expressed by partial differential equations. Such physics-informed constraints provide a regularization mechanism for effectively training deep generative models as surrogates of physical systems in which the cost of data acquisition is high, and training data-sets are typically small. This provides a flexible framework for characterizing  uncertainty in the outputs of physical systems due to randomness in their inputs or noise in their observations that entirely bypasses the need for repeatedly sampling expensive experiments or numerical simulators. We demonstrate the effectiveness of our approach through a series of examples involving uncertainty propagation in non-linear conservation laws, and the discovery of constitutive laws for flow through porous media directly from noisy data.

This paper is published on Journal of Computational Physics.

- Yibo Yang, Paris Perdikaris,
Adversarial uncertainty quantification in physics-informed neural networks,
Journal of Computational Physics,
2019,
ISSN 0021-9991,
https://doi.org/10.1016/j.jcp.2019.05.027.


## Citation
```
@article{yang2019adversarial,
  title={Adversarial uncertainty quantification in physics-informed neural networks},
  author={Yang, Yibo and Perdikaris, Paris},
  journal={Journal of Computational Physics},
  year={2019},
  publisher={Elsevier}
}
```

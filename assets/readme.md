# SA-SOR

## Difference Between SASOR(Original) and SASOR(All)

The original SASOR implementation from **"[Instance-Level Relative Saliency Ranking with Graph Reasoning](https://github.com/dragonlee258079/Saliency-Ranking)"** ignores images that contain only one instance. There are two possible reasons:

1. It is not meaningful to discuss relative saliency relationships when only one instance exists.
2. The Spearman (or Pearson) correlation coefficient implementation (see [code](../tools/evaluation/spearman_correlation.py#L124)) produces **NaN** when the denominator becomes 0, since the variance is zero for one instance.

SASOR(All) includes images that contain only one instance. To make this feasible, we must explicitly handle NaN cases (as done in this [project](../tools/evaluation/spearman_correlation.py#L125-L126)), or add a very small constant to the denominator to avoid division by zero. Regardless of the implementation detail, the SASOR value for a single-instance image will always be 0.

Since a successful ranking should produce a positive SASOR value, including many zero-score images will inevitably reduce the overall average.

Therefore:

> **SASOR(All) is always smaller than SASOR(Original).**

In this project:

- Comment out [these lines](../tools/evaluation/spearman_correlation.py#L89-L91) to use **SASOR(All)**  
- Keep them to use **SASOR(Original)**

## Difference Between SASOR and Norm. SASOR

SASOR is a correlation-based metric with range $[-1, 1]$ where 1 represents perfect positive correlation and 0 represents no correlation. To make the metric more intuitive and easier to interpret as a "score", we normalize it to: $\text{Norm.\ SASOR} = \frac{\text{SASOR} + 1}{2}$. This converts it into a percentage-like scale in the range $[0, 1]$, which is more intuitive for reporting.

## Which Version Did the Authors Use in their paper, SASOR(Reported)?

Unfortunately, it is unclear which version was used in the original paper.

Therefore, this project reports all variants for transparency and comparison.


# MAE

## MAE(SOD)

Both this project and the baseline **"[Instance-Level Relative Saliency Ranking with Graph Reasoning](https://github.com/dragonlee258079/Saliency-Ranking)"** treat MAE as a Salient Object Detection (SOD) evaluation. By rank_map[segmaps[i] >0.5] = 1, all detected salient regions are binarized and treated as foreground. MAE is then computed in the standard SOD manner.

This MAE evaluates saliency detection quality and does **not** evaluate ranking performance.

## MAE(SOR)

In RVSOD, pixel values represent ranking relationships rather than absolute saliency confidence. To evaluate ranking-sensitive MAE:

1. Each GT instance is reassigned a grayscale value in [0, 255] at equal intervals according to its rank.
2. The prediction image is assigned values using the same rule.
3. MAE is computed directly between the two grayscale maps.

This version of MAE is affected by ranking quality.

## Which did the authors use in their paper, MAE(Reported)?

MAE(SOD) is more consistent with the original baseline implementation and is likely the metric reported in the paper.


# Contact

For further discussion, please open an issue or contact:

**ruisong8-c@my.cityu.edu.hk**

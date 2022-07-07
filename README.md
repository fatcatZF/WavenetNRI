# WavenetNRI

**Abstract:** Group detection from spatio-temporal data is helpful in many applications, such as automatic driving and social sciences. Most previous works in this domain are based on conventional machine learning methods with feature engineering; only few works are based on deep learning. We proposed a graph neural network (GNN) based method for group detection. Our method is an extension of neural relational inference (NRI) [[1]](#1). We made the following changes to the original NRI: (1) We applied symmetric edge features with symmetric edge updating processes to output symmetric edge representations corresponding to the symmetric binary group relationships. (2 )Inspired by Wavenet [[2]](#2), we applied a gated dilated residual causal convolutional block to capture both short and long dependency of the sequences of edge features. We name our method ``WavenetNRI''. Our experiments compare our method with several baselines, including the original NRI on two types of data sets: (1) six spring simulation data sets; (2) five pedestrian data sets. Experimental results show that on the spring simulation data sets, NRI and WavenetNRI with supervised training outperform all other baselines, and NRI performs slightly better than WavenetNRI. On the pedestrian data sets, our method WavenetNRI with supervised training outperforms other pairwise classification-based baselines. However, it cannot compete against the clustering-based methods. In the ablation study, we study the effects of our changes to NRI on group detection. We find that on the pedestrian data sets, the symmetric edge features with the symmetric edge updating processes can significantly improve the performance of NRI. On the spring simulation data sets, the gated dilated residual causal convolutional block can slightly improve the performance of NRI. 


## Data sets
We used six spring simulation data sets and five pedestrian data sets in our experiments.
### Spring simulation data sets

### Pedestrian data sets

## Run experiments


## References
<a id="1">[1]</a>
Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). Neural relational inference for interacting systems. In International Conference on Machine Learning (pp. 2688-2697). PMLR.

<a id="2">[2]</a>
Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.

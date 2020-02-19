# [ICLR'20] How to 0wn NAS in Your Spare Time

This repository contains the code for the paper [_"How to 0wn NAS in Your Spare Time"_](https://arxiv.org/abs/2002.06776) <br>
Published at the [_International Conference on Learning Representation (ICLR) 2020_](https://icml.cc/Conferences/2020), Addis Ababa, Ethoipia.

**Authors:** [Sanghyun Hong](http://sanghyun-hong.com), Michael Davinroy, [Yigitcan Kaya](http://www.cs.umd.edu/~yigitcan), [Dana Dachman-Soled](https://user.eng.umd.edu/~danadach/), and [Tudor Dumitras](http://users.umiacs.umd.edu/~tdumitra/) <br>
**Contact:** [Sanghyun Hong](mailto:shhong@cs.umd.edu)


## About

Our study presents an algorithm that reconstructs the key components of a novel deep learning systems—_i.e.,_ a novel data pre-preprocessing pipeline and a neural network architecture---by exploiting information leakage from a cache side-channel attack, Flush+Reload. Based on the trace of computations and the timing for each computation observed by Flush+Reload, we generate candidate computational graphs from the trace and eliminate incompatible candidates through a parameter estimation process. We demonstrate experimentally that we can reconstruct [_MalConv_](https://github.com/endgameinc/ember/tree/master/malconv), a novel data pre-processing pipeline for malware detection, and [_ProxylessNAS-CPU_](https://github.com/mit-han-lab/ProxylessNAS), a novel network architecture for the ImageNet classification optimized to run on CPUs, without knowing the architecture family. This repository contains the traces that we observed by the side-channel attack and the scripts for reconstructing victim architectures.

**Note:** this repository currently includes the code for the ToyNet and MalConv reconstructions.



## Install Dependencies

You can install the required Python packages by running the following command:

```
  $ pip install -r requirements.txt
```



## Run (MalConv)

To run the script for reconstructing the MalConv architecture:

```
  $ ./reconstruct_malconv.sh
```

The reconstruction results are stored under the ``results/reconstruct/<victim>`` folder.

- ``computational_graphs``: contains the computational graphs reconstructed from a trace.
- ``architecture_candidates``: contains the candidate architecture reconstructed by pruning.
- ``architectures``: contains the final architecture after removing unrealistic candidates.

[This PDF](./results/reconstruct/malconv/architectures/architecture_0.pdf) shows the final architecture from this reconstruction.



## Traces from Flush+Reload

You can see the traces observed from the cache side-channel attack (Flush+Reload) in the ``traces/<victim>`` folder. We use the [Mastik toolkit](https://github.com/Sanghyun-Hong/Mastik) to extract those traces. If you're interested in this process, you can refer to [this repository for our previous project](https://github.com/Sanghyun-Hong/DeepRecon).

- ``raw``: contains the raw traces observed by the side-channel attacker.
- ``processed``: contains the traces processed offline, used as an input to the reconstruction algorithm.



## Cite This Work

You are encouraged to cite our paper if you use this code for academic research.

```
@inproceedings{Hong200wn,
  author    = {Sanghyun Hong and
               Michael Davinroy and
               Yigitcan Kaya and
               Dana Dachman{-}Soled and
               Tudor Dumitras},
  title     = {How to 0wn NAS in Your Spare Time},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  url       = {https://arxiv.org/pdf/2002.06776.pdf},
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


**Fin.**

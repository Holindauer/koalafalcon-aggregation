# KoalaFalcon

This repository contains two preliminary draft papers and supporting code for research I conducted in the summer of 2026 as part of the 12-week Ethereum Foundation internship program 2026. The project specification is listed [here](https://esp.ethereum.foundation/funded-projects) under the name *Internship Program 2026 | Poseidon*. 

I was initially intending to shape this work up further for publication in a journal. However, I recently had a change in perspective. With the advent of auto research as a genuine tool for progressing the state of the art in science, I realize that more frequent partial results are likely to progress the state of science faster than infrequent polished publications. For that reason, I've decided to collect the work that has been done up to now and put it on GitHub as a means of sharing these results early with any auto-researchers currently working on related problems. With that said, if you find any genuine issues with the work here, please leave a GitHub issue and I will address them and alert others.

The contents of this repository are split into three components.

- KoalaFalcon: a modification of Falcon to support the KoalaBear prime modulus and Poseidon
- KoalaFalcon supporting code
- Virtualizing the LogUp* Pushforward: an optimization to LogUp* to reduce online commitment cost associated with the challenge dependent pushforward


*Note: the current KoalaFalcon paper and code intentionally omits a few sections which I am waiting to put here until another project concludes. This will be updated once possible to include those sections and code.*

# Citation

If you find this work useful in your own research, please cite using the following bibtex citations. 

For the *KoalaFalcon: Lattice Signature Aggregation with SNARKs over KoalaBear* paper, use

    @misc{lindauer2026koalafalcon,
      author = {Hunter Lindauer},
      title  = {KoalaFalcon: Lattice Signature Aggregation with SNARKs over KoalaBear},
      year   = {2026},
      note   = {Unpublished manuscript, implementation},
      url    = {https://github.com/Holindauer/koalafalcon-aggregation/tree/main/koalafalcon-paper}
    }
    
For *Virtualization of the LogUp\* Pushforward*, use

    @misc{lindauer2026virtuallogup,
      author = {Hunter Lindauer},
      title  = {Virtualizing the LogUp* Pushforward},
      year   = {2026},
      note   = {Unpublished manuscript},
      url    = {https://github.com/Holindauer/koalafalcon-aggregation/blob/main/logupstar-tensor-paper/logupstar_tensor.pdf}
    }

## License

The software under `koalafalcon-impl/` is licensed under the MIT License.

The papers under `koalafalcon-paper/` and `logupstar-tensor-paper/` are
licensed under the Creative Commons Attribution 4.0 International License.


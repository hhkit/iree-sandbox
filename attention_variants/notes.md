- to look up:
    - pytorch inductor
        - cannot generate cuda, too verbose
    - tileir is a block programming language like triton

- can't write FA in pytorch
- can write FA in block programming languages
    - Q: what is missing in tensor programming languages?
    - If the ONNX is written in a certain way, IREE can consume it and achieve FA-like optimizations


Title: **Enabling block-level compiler optimizations in tensor programming languages**

Abstract:
When attention is all you need, all attention is on attention. Various attention variants have risen in order to reduce the memory or computation complexity of the attention mechanism, adopting online methods, virtual paging, and many other techniques. However, while tensor programming languages are much easier for machine learning engineers and mathematicians to reason with, they turn out to be too far from the hardware to capture these variants, and a performance engineer needs to be brought in to maximize hardware usage using block-level programming paradigms. We demonstrate that, by describing tensor operations in a particular way, automatic compiler optimizations and rewrites can be more easily applied, achieving the high parallelizability and low memory cost that online algorithms afford while retaining the flexibility that tensor languages afford.

Problem:
- Flash attention cannot be written in tensor programming languages like Torch.
- FA can be written in block programming languages, because BPLs let us describe problems in terms of work groups, leaving the compiler to determine how distribute within the WG.
- But most ML research engineers would rather write in tensor PLs like PyTorch
    - eg. If you introduce quantization into a FA model, you will need to rewrite the underlying triton kernel in pytorch
- Enabling FA's loop fusion through compiler rewrites is non-trivial, and adding additional optimizations such as quantization results in needing to rewrite the kernel
    - TODO: why is the loop fusion of attention difficult to begin with, and what difference does the alternative way make?

Contribution:
- We can achieve flash attention by doing loop fusion on a tensor program
    - Note: the tensor program is not the original attention program, which is just `softmax (Q @ K.T) @ V`. In PyTorch, an optimized intrinsic `torch.nn.functional.scaled_dot_product_attention` is usually used, but its kernel is hard-coded to FA2 with a hook provided for element-wise operations.
    - This allows ML engineers to further experiment with the attention mechanism, adding quantization and other optimizations without needing to drop to a block-level programming language like Triton
        - Dealing with blocks/workgroups is not their domain 
- To accomplish this, the tensor language must be expressed in a specific way ala linalg (Kunwar Slide 29)
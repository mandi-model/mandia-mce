# Architecture and Conditional Preservation

[Français](architecture.md)

The causal core has 12 layers, width 320, eight heads, a SwiGLU feed-forward dimension of 832, RMSNorm, RoPE, an 8,192-token BPE vocabulary, and maximum context length 512. Input embeddings and the output head share weights. It contains 17,129,280 parameters. This adaptation study uses context length 256.

## Residual MCE banks

An adapter follows each core block:

```text
h' = h + W_up · Dropout(GELU(W_down · RMSNorm(h)))
```

The up projection starts at zero. A separate bank is created for each domain, with one adapter per core block. Bottleneck 256 adds 1,969,920 parameters per domain; bottleneck 64 adds 495,360. When a domain is added, the core and previous banks are frozen. The new domain is trained through the frozen core and its own bank, without consuming previous banks.

## Independent LoRA banks

QKV, attention-output, and feed-forward gate/up/down projections receive separate low-rank updates per domain. The compared ranks are 8 and 32, with alpha/rank = 2 and dropout 0.1, adding 516,096 and 2,064,384 parameters per domain. Updates remain unmerged. A `finally` block restores the previously selected route after each forward call. Concurrent calls on the same model instance are unsupported.

## Preservation conditions

A previously selected route computes the same function if its parameters, buffers, operators, routing selection, and numerical inference conditions remain unchanged. This is a consequence of freezing computational paths, not a new general theorem of continual learning. Weight hashes and probe logits verify this property in the experiment. Independent LoRA banks share it.

Preservation does not imply that a new expert learns successfully or that a router selects the appropriate expert. Expert cooperation and fusion are absent. Storage increases with the number of domains; only one bank is active per forward pass.

## Related work

[Progressive Neural Networks](https://arxiv.org/abs/1606.04671) use frozen columns and lateral connections; this MCE implementation has no lateral transfer from previous experts. [NLP adapters](https://proceedings.mlr.press/v97/houlsby19a.html) and [LoRA](https://arxiv.org/abs/2106.09685) are established parameter-efficient adaptation methods. The contribution here is the small-model empirical comparison and its preservation, capacity, order, and routing measurements; general architectural novelty remains unestablished.

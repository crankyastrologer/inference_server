Hello everyone I have been working on understanding and optimizing inference for some time. But with the age of AI agents you work does gets faster but somethings that can only know by working handson can be missed. Now I wouldn't like to penalize my organization just because I feel I want to know speculative decoding better.
Hence this is my approach to understanding the end to end infernce for the models. It is somewhat similar to the approach I took for my GPT 2 from scratch project ie writing the whole architecture whereever feasible from a low enough abstraction level.
Here I have mostly chosen python as I feel I know the lower layers well enough thanks to that project.
I would use this document to basically explain how each function I have written works to make sure I am actually understanding what I am creating.

## Memory

### kv_cache

Here is the implementation of our kv cache:

```python
class ContiguousKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.length = 0
        shape = (1, num_kv_heads, max_len, head_dim)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
```

- num_layers: The no of Attention Layers in the model
- num_kv_heads: The total no of key value heads that model has basically no of independent attention heads that don't share same kv cache
- head_dim: The dimension of te acutal head basically howm many dimensions the head works on from the embedding dimension
- max_len: maximum sequence length allowed

## Model

### loader
Here we load the model in memory and if needed download it from the source

### norm
This is pretty simple function to normalize the values between layers we are using Root mean squared normalization with a small value added to make sure the cases with 0 value doesnt result in indefinite


## Engine

### Generate.py

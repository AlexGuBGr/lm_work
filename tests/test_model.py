import torch

from model import GPT, ModelConfig, apply_rope, rope_cache

torch.set_num_threads(1)


def test_default_param_budget():
    # the requirement is about 1M parameters
    n = GPT(ModelConfig()).num_params()
    assert 0.9e6 <= n <= 1.1e6, n


def test_weight_tying():
    m = GPT(ModelConfig())
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_forward_shapes_and_loss():
    cfg = ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, mlp_hidden=64)
    m = GPT(cfg)
    torch.manual_seed(0)
    x = torch.randint(0, 20, (3, 16))
    y = torch.randint(0, 20, (3, 16))
    logits, loss = m(x, y)
    assert logits.shape == (3, 16, 20)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # random initialisation should give a loss close to ln(vocab)
    assert abs(loss.item() - torch.log(torch.tensor(20.0)).item()) < 0.5


def test_causal():
    # changing a later token must not affect the logits at earlier positions
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, mlp_hidden=64)
    m = GPT(cfg).eval()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 20
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_rope_preserves_norm_and_relative_position():
    cos, sin = rope_cache(8, 32, 10000.0)
    q = torch.randn(1, 1, 32, 8)
    k = torch.randn(1, 1, 32, 8)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # the same q/k vectors at (i, j) and (i+s, j+s) give the same dot product
    qv, kv = q[0, 0, 0], k[0, 0, 0]
    qq = qv.expand(1, 1, 32, 8).clone()
    kk = kv.expand(1, 1, 32, 8).clone()
    rq, rk = apply_rope(qq, cos, sin)[0, 0], apply_rope(kk, cos, sin)[0, 0]
    assert torch.allclose(rq[5] @ rk[2], rq[20] @ rk[17], atol=1e-4)


def test_generate_respects_context_and_stop():
    cfg = ModelConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16, mlp_hidden=32)
    m = GPT(cfg).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)
    assert int(out.max()) < 10

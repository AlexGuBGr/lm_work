// Character-level Llama-style GPT in plain JavaScript (no libraries).
// Mirrors model.py: RMSNorm, RoPE, causal multi-head attention, SwiGLU, tied embeddings.
// Uses a key/value cache, so every new character costs one forward step instead of a full re-run.

export class CharGPT {
  constructor(meta, buffer) {
    this.meta = meta;
    const c = meta.config;
    Object.assign(this, { V: c.vocab_size, C: c.n_embd, H: c.n_head, L: c.n_layer, F: c.mlp_hidden,
                          T: c.block_size, eps: c.norm_eps });
    this.hd = this.C / this.H;
    const all = new Float32Array(buffer);
    this.w = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.w[name] = all.subarray(t.offset, t.offset + n);
    }
    // RoPE tables (T x hd/2)
    const half = this.hd / 2;
    this.cos = new Float32Array(this.T * half);
    this.sin = new Float32Array(this.T * half);
    for (let p = 0; p < this.T; p++)
      for (let i = 0; i < half; i++) {
        const f = p / Math.pow(c.rope_theta, (2 * i) / this.hd);
        this.cos[p * half + i] = Math.cos(f);
        this.sin[p * half + i] = Math.sin(f);
      }
    this.stoi = new Map(meta.itos.map((ch, i) => [ch, i]).filter(([, i]) => i >= 2));
    this.reset();
  }

  static async load(base = ".") {
    const [meta, buf] = await Promise.all([
      fetch(`${base}/model.json`).then((r) => r.json()),
      fetch(`${base}/model.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new CharGPT(meta, buf);
  }

  encode(text) { return [...text].map((ch) => this.stoi.get(ch) ?? this.meta.unk_id); }
  decode(id) { return id < 2 ? "" : this.meta.itos[id]; }

  reset() {
    const size = this.T * this.C;
    this.kCache = Array.from({ length: this.L }, () => new Float32Array(size));
    this.vCache = Array.from({ length: this.L }, () => new Float32Array(size));
    this.pos = 0;
    this.history = [];
  }

  // y = W x, W stored as (out, in) row-major
  matvec(W, x, out, rows, cols) {
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o = r * cols;
      for (let k = 0; k < cols; k++) s += W[o + k] * x[k];
      out[r] = s;
    }
    return out;
  }

  rmsnorm(x, w, out) {
    let ss = 0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const inv = 1 / Math.sqrt(ss / x.length + this.eps);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
    return out;
  }

  rope(vec, offset, p) {  // rotate pairs (2i, 2i+1) of one head in place
    const half = this.hd / 2;
    for (let i = 0; i < half; i++) {
      const a = vec[offset + 2 * i], b = vec[offset + 2 * i + 1];
      const c = this.cos[p * half + i], s = this.sin[p * half + i];
      vec[offset + 2 * i] = a * c - b * s;
      vec[offset + 2 * i + 1] = a * s + b * c;
    }
  }

  // feed one token; returns logits (Float32Array of size V) for the next token
  step(token) {
    if (this.pos >= this.T) this.slide();
    const { C, H, hd, F, L, w } = this;
    const p = this.pos;
    const x = Float32Array.from(w["tok_emb.weight"].subarray(token * C, token * C + C));
    const h = new Float32Array(C), qkv = new Float32Array(3 * C), att = new Float32Array(C);
    const tmp = new Float32Array(C), g = new Float32Array(F), u = new Float32Array(F);
    const scores = new Float32Array(p + 1);
    for (let l = 0; l < L; l++) {
      const pre = `blocks.${l}.`;
      this.rmsnorm(x, w[pre + "attn_norm.weight"], h);
      this.matvec(w[pre + "attn.qkv.weight"], h, qkv, 3 * C, C);
      for (let hh = 0; hh < H; hh++) {
        this.rope(qkv, hh * hd, p);            // q
        this.rope(qkv, C + hh * hd, p);        // k
      }
      this.kCache[l].set(qkv.subarray(C, 2 * C), p * C);
      this.vCache[l].set(qkv.subarray(2 * C, 3 * C), p * C);
      const K = this.kCache[l], Vc = this.vCache[l], scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd;
        let max = -Infinity;
        for (let t = 0; t <= p; t++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += qkv[qo + i] * K[t * C + qo + i];
          scores[t] = s * scale;
          if (scores[t] > max) max = scores[t];
        }
        let sum = 0;
        for (let t = 0; t <= p; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
        for (let i = 0; i < hd; i++) {
          let s = 0;
          for (let t = 0; t <= p; t++) s += scores[t] * Vc[t * C + qo + i];
          att[qo + i] = s / sum;
        }
      }
      this.matvec(w[pre + "attn.proj.weight"], att, tmp, C, C);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      this.rmsnorm(x, w[pre + "mlp_norm.weight"], h);
      this.matvec(w[pre + "mlp.gate.weight"], h, g, F, C);
      this.matvec(w[pre + "mlp.up.weight"], h, u, F, C);
      for (let i = 0; i < F; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];   // SiLU(gate) * up
      this.matvec(w[pre + "mlp.down.weight"], g, tmp, C, F);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    this.rmsnorm(x, w["norm.weight"], h);
    const logits = this.matvec(w["tok_emb.weight"], h, new Float32Array(this.V), this.V, C);
    this.pos++;
    this.history.push(token);
    return logits;
  }

  // context window full: keep the last half and recompute the cache (positions restart at 0,
  // exactly like the Python model, which always sees at most the last 256 characters)
  slide() {
    const keep = this.history.slice(-this.T / 2);
    this.reset();
    for (const t of keep) this.step(t);
  }

  // feed a prompt; returns logits after the last character
  prefill(tokens) {
    let logits = null;
    for (const t of tokens) logits = this.step(t);
    return logits;
  }

  static softmax(logits, temperature = 1) {
    const p = new Float32Array(logits.length);
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - max) / temperature); sum += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  sample(logits, temperature = 0.8, topK = 0, rand = Math.random) {
    let p = CharGPT.softmax(logits, temperature);
    if (topK > 0 && topK < p.length) {
      const cut = [...p].sort((a, b) => b - a)[topK - 1];
      let sum = 0;
      for (let i = 0; i < p.length; i++) { if (p[i] < cut) p[i] = 0; sum += p[i]; }
      for (let i = 0; i < p.length; i++) p[i] /= sum;
    }
    let r = rand(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) return i; }
    return p.length - 1;
  }
}

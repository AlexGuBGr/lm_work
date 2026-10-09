// Run the JavaScript model on a prompt and print its next-character logits (used by test_demo.py).
import { readFileSync } from "node:fs";
import { CharGPT } from "../docs/demo/model.js";

const dir = new URL("../docs/demo/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("model.json", dir), "utf8"));
const buf = readFileSync(new URL("model.bin", dir));
const m = new CharGPT(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const prompt = process.argv[2];
const ids = [meta.eot_id, ...m.encode(prompt)];
const t0 = performance.now();
const logits = m.prefill(ids);
const ms = (performance.now() - t0) / ids.length;
console.log(JSON.stringify({ logits: Array.from(logits), ms_per_char: ms, pos: m.pos }));

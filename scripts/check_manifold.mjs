import { readFileSync } from "node:fs";

// Load MC_CASE_TABLE from src/mc_case_table.ts by extracting the Int32Array literal.
const src = readFileSync(new URL("../src/marching-cubes/mc_case_table.ts", import.meta.url), "utf8");
const match = src.match(/new Int32Array\(\[([\s\S]*?)\]\)/);
if (!match) throw new Error("could not parse MC_CASE_TABLE");
const table = eval("[" + match[1] + "]"); // numbers
if (table.length !== 256 * 16) throw new Error("unexpected table length " + table.length);

// Cube corners 0..7 and the 12 cube edges (same as compute_vertices.wgsl).
const EDGE = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [6, 5], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

// Each face: corner set (for "is this edge on this face") and the 4 cube-edge
// indices that lie on it.
const FACES = [
  { corners: [0, 1, 2, 3], edges: [0, 1, 2, 3] }, // z=0
  { corners: [4, 5, 6, 7], edges: [4, 5, 6, 7] }, // z=1
  { corners: [0, 1, 5, 4], edges: [0, 9, 4, 8] }, // y=0
  { corners: [2, 3, 7, 6], edges: [2, 11, 6, 10] }, // y=1
  { corners: [0, 3, 7, 4], edges: [3, 11, 7, 8] }, // x=0
  { corners: [1, 2, 6, 5], edges: [1, 10, 5, 9] }, // x=1
];

const edgeOnFace = (e, f) =>
  f.corners.includes(EDGE[e][0]) && f.corners.includes(EDGE[e][1]);

function trianglesOf(caseIndex) {
  const tris = [];
  const base = caseIndex * 16;
  for (let i = 0; i < 16 && table[base + i] !== -1; i += 3) {
    tris.push([table[base + i], table[base + i + 1], table[base + i + 2]]);
  }
  return tris;
}

// 1) Crack-freedom: the set of cube-edges used on each face must depend only on
//    that face's corner signs, so neighbouring cells always agree.
let inconsistent = 0;
for (const face of FACES) {
  const seen = new Map(); // face corner-sign key -> canonical used-edge string
  for (let c = 0; c < 256; c++) {
    const key = face.corners.map((k) => (c >> k) & 1).join("");
    const used = new Set();
    for (const tri of trianglesOf(c)) {
      for (const e of tri) if (edgeOnFace(e, face)) used.add(e);
    }
    const sig = [...used].sort((a, b) => a - b).join(",");
    if (seen.has(key) && seen.get(key) !== sig) {
      inconsistent++;
    } else {
      seen.set(key, sig);
    }
  }
}

// 2) Edge manifoldness within each cube: every unordered pair of cube-edges
//    spanned by a triangle must occur exactly twice (closed surface patch).
let badMultiplicity = 0;
let maxEdgeUse = 0;
for (let c = 0; c < 256; c++) {
  const count = new Map();
  for (const tri of trianglesOf(c)) {
    for (const [a, b] of [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]) {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      count.set(k, (count.get(k) || 0) + 1);
    }
  }
  for (const [k, n] of count) {
    const [a, b] = k.split(",").map(Number);
    const onFace = FACES.some((f) => edgeOnFace(a, f) && edgeOnFace(b, f));
    // face segments: completed by neighbour (expect 1); internal: expect 2
    const expect = onFace ? 1 : 2;
    if (n !== expect) {
      badMultiplicity++;
      if (c < 32) console.log(`case ${c}: edge {${k}} used ${n}x, expected ${expect}`);
    }
    maxEdgeUse = Math.max(maxEdgeUse, n);
  }
}

// 3) Vertex manifoldness check is not meaningful pre-welding, but count
//    non-manifold "pinch" candidates is out of scope here.

console.log("faces with inconsistent cross-cell triangulations:", inconsistent);
console.log("cube-edge segments with wrong multiplicity:", badMultiplicity);
console.log("max times any segment is used:", maxEdgeUse);

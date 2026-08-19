// Verify src/molecular/mrc.ts against generated MRC/CCP4 files.
// Transpiles the TS source with the project's TypeScript compiler, then
// round-trips hand-built maps (little/big-endian, int8, downsampling, axis
// reorder) through parseMrc().
//
// Run with: node scripts/check_mrc.mjs

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import ts from "typescript";

const src = readFileSync(new URL("../src/molecular/mrc.ts", import.meta.url), "utf8");
const out = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }
}).outputText;
const module_ = { exports: {} };
const require_ = createRequire(import.meta.url);
new Function("module", "exports", "require", out)(module_, module_.exports, require_);
const { parseMrc } = module_.exports;

let failures = 0;
const close = (a, b, tol = 1e-4) => Math.abs(a - b) <= tol;

function check(name, cond, detail = "") {
    if (cond) {
        console.log(`ok   ${name}`);
    } else {
        failures++;
        console.log(`FAIL ${name} ${detail}`);
    }
}

function makeHeader({ nc, nr, ns, mode, cell = [0, 0, 0],
                      mapc = 1, mapr = 2, maps = 3, nx, ny, nz, little = true }) {
    const buf = Buffer.alloc(1024);
    const i32 = little ? (o, v) => buf.writeInt32LE(v, o) : (o, v) => buf.writeInt32BE(v, o);
    const f32 = little ? (o, v) => buf.writeFloatLE(v, o) : (o, v) => buf.writeFloatBE(v, o);
    i32(0, nc); i32(4, nr); i32(8, ns); i32(12, mode);
    i32(28, nx ?? nc); i32(32, ny ?? nr); i32(36, nz ?? ns);
    f32(40, cell[0]); f32(44, cell[1]); f32(48, cell[2]);
    i32(64, mapc); i32(68, mapr); i32(72, maps);
    buf.write("MAP ", 208, 4, "latin1");
    buf[212] = little ? 0x44 : 0x11;
    buf[213] = little ? 0x41 : 0x11;
    return buf;
}

function floatData(values, little) {
    const buf = Buffer.alloc(values.length * 4);
    for (let i = 0; i < values.length; ++i) {
        little ? buf.writeFloatLE(values[i], i * 4) : buf.writeFloatBE(values[i], i * 4);
    }
    return buf;
}

// Concatenate Buffers into a fresh, zero-offset ArrayBuffer (Buffer.buffer may
// point into Node's pooled ArrayBuffer with a non-zero byteOffset).
function assemble(...parts) {
    const total = parts.reduce((n, b) => n + b.byteLength, 0);
    const out = new ArrayBuffer(total);
    const view = new Uint8Array(out);
    let off = 0;
    for (const p of parts) {
        view.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), off);
        off += p.byteLength;
    }
    return out;
}

// --- Test 1: float32 little-endian, gradient values, cell spacing ---
{
    const nc = 4, nr = 5, ns = 6;
    const values = [];
    for (let z = 0; z < ns; ++z)
        for (let y = 0; y < nr; ++y)
            for (let x = 0; x < nc; ++x)
                values.push(x + 10 * y + 100 * z);
    const header = makeHeader({ nc, nr, ns, mode: 2, cell: [40, 50, 60] });
    const map = parseMrc(assemble(header, floatData(values, true)));

    check("t1 dims", map.nx === nc && map.ny === nr && map.nz === ns, `${map.nx}x${map.ny}x${map.nz}`);
    check("t1 spacing", close(map.spacing[0], 10) && close(map.spacing[1], 10) && close(map.spacing[2], 10),
        JSON.stringify(map.spacing));
    check("t1 data[0]", map.data[0] === 0);
    check("t1 data[last]", map.data[nc * nr * ns - 1] === 543);
    check("t1 data[5]", map.data[5] === 11, `got ${map.data[5]}`);   // x=1,y=1,z=0 => 1+10
    check("t1 min/max", close(map.min, 0) && close(map.max, 543), `${map.min}/${map.max}`);
    check("t1 mean", close(map.mean, 271.5), `got ${map.mean}`);
    check("t1 not downsampled", map.downsampled === false);
}

// --- Test 2: big-endian float32, machine stamp 0x11 0x11 ---
{
    const nc = 3, nr = 4, ns = 5;
    const values = [];
    for (let z = 0; z < ns; ++z)
        for (let y = 0; y < nr; ++y)
            for (let x = 0; x < nc; ++x)
                values.push(x + 1 + 100 * z);
    const header = makeHeader({ nc, nr, ns, mode: 2, cell: [30, 40, 50], little: false });
    const map = parseMrc(assemble(header, floatData(values, false)));

    check("t2 dims", map.nx === nc && map.ny === nr && map.nz === ns);
    check("t2 spacing", close(map.spacing[0], 10) && close(map.spacing[1], 10) && close(map.spacing[2], 10),
        JSON.stringify(map.spacing));
    check("t2 data[0]", map.data[0] === 1, `got ${map.data[0]}`);
    check("t2 data[last]", map.data[nc * nr * ns - 1] === 403, `got ${map.data[nc * nr * ns - 1]}`);
}

// --- Test 3: mode 0 (int8) ---
{
    const nc = 4, nr = 4, nz = 4;
    const values = [];
    for (let i = 0; i < nc * nr * nz; ++i) values.push(i % 10);
    const header = makeHeader({ nc, nr, ns: nz, mode: 0 });
    const body = Buffer.from(values.map(v => v & 0xff));
    const map = parseMrc(assemble(header, body));
    check("t3 dims", map.nx === nc && map.ny === nr && map.nz === nz);
    check("t3 data[7]", map.data[7] === 7, `got ${map.data[7]}`);
    check("t3 min/max", map.min === 0 && map.max === 9, `${map.min}/${map.max}`);
}

// --- Test 4: downsampling to maxDim ---
{
    const nx = 300, ny = 4, nz = 4;
    const values = new Array(nx * ny * nz).fill(3.5);   // constant map
    const header = makeHeader({ nc: nx, nr: ny, ns: nz, mode: 2, cell: [300, 40, 40] });
    const map = parseMrc(assemble(header, floatData(values, true)), { maxDim: 256 });
    check("t4 downsampled", map.downsampled === true);
    check("t4 nx", map.nx === 150, `got ${map.nx}`);
    check("t4 ny/nz unchanged", map.ny === 4 && map.nz === 4, `${map.ny}/${map.nz}`);
    check("t4 constant value", map.data[0] === 3.5, `got ${map.data[0]}`);
    check("t4 spacing preserved", close(map.spacing[0], 1) && close(map.spacing[1], 10) && close(map.spacing[2], 10),
        JSON.stringify(map.spacing));
}

// --- Test 5: axis reorder (MAPC=2, MAPR=1, MAPS=3) ---
{
    // Stored (c=column fastest) data where value = c + 100*r + 10000*s.
    // With MAPC=2: column axis is crystal-y; MAPR=1: row axis is crystal-x.
    // So output (x=crystal x, y=crystal y, z=crystal z) picks src c=y, r=x.
    const nc = 3, nr = 4, ns = 2;
    const srcV = [];
    for (let s = 0; s < ns; ++s)
        for (let r = 0; r < nr; ++r)
            for (let c = 0; c < nc; ++c)
                srcV.push(c + 100 * r + 10000 * s);
    const header = makeHeader({ nc, nr, ns, mode: 2, mapc: 2, mapr: 1, maps: 3 });
    const map = parseMrc(assemble(header, floatData(srcV, true)));

    // Output index (x + nx*(y + ny*z)) should hold src[y + nc*(x + nr*z)].
    const outV = map.data;
    const expected = [];
    for (let z = 0; z < ns; ++z)
        for (let y = 0; y < nr; ++y)
            for (let x = 0; x < nc; ++x)
                expected.push(srcV[y + nc * (x + nr * z)]);
    check("t5 reorder", outV.every((v, i) => v === expected[i]),
        `first diff at ${outV.findIndex((v, i) => v !== expected[i])}`);
    check("t5 dims preserved", map.nx === nc && map.ny === nr && map.nz === ns);
}

// --- Test 6: unsupported mode rejected ---
{
    const header = makeHeader({ nc: 2, nr: 2, ns: 2, mode: 99 });
    let threw = false;
    try {
        parseMrc(assemble(header, Buffer.alloc(8)));
    } catch (err) {
        threw = /unsupported data mode 99/.test(String(err));
    }
    check("t6 unsupported mode", threw);
}

// --- Test 8: full resolution (maxDim <= 0 disables downsampling) ---
{
    const nx = 300, ny = 4, nz = 4;
    const values = [];
    for (let i = 0; i < nx * ny * nz; ++i) values.push(i);
    const header = makeHeader({ nc: nx, nr: ny, ns: nz, mode: 2, cell: [300, 40, 40] });
    const map = parseMrc(assemble(header, floatData(values, true)), { maxDim: 0 });
    check("t8 full-res not downsampled", map.downsampled === false);
    check("t8 full-res dims", map.nx === nx && map.data.length === nx * ny * nz, `nx=${map.nx}`);
    check("t8 full-res value", map.data[nx * ny * nz - 1] === nx * ny * nz - 1,
        `got ${map.data[nx * ny * nz - 1]}`);
}

// --- Test 7: real cryo-EM map (skipped if not present locally) ---
{
    const path = "/home/clement/Projects/Moorhen/baby-gru/tests/test_data/low-res-emd-32143.map";
    try {
        const { statSync } = await import("node:fs");
        statSync(path);
        const buf = readFileSync(path);
        const ab = new ArrayBuffer(buf.byteLength);
        new Uint8Array(ab).set(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
        const map = parseMrc(ab, { maxDim: 256 });
        const n = map.nx * map.ny * map.nz;
        check("t7 dims", n === map.data.length, `${map.nx}x${map.ny}x${map.nz}`);
        check("t7 finite stats",
            Number.isFinite(map.mean) && Number.isFinite(map.rms) && map.rms > 0,
            `mean=${map.mean} rms=${map.rms}`);
        check("t7 min<max", map.min < map.max, `${map.min}/${map.max}`);
        check("t7 spacing positive",
            map.spacing.every(s => s > 0), JSON.stringify(map.spacing));
        console.log(`     emd-32143: ${map.nx}x${map.ny}x${map.nz} @ ` +
            `${map.spacing.map(s => s.toFixed(2)).join("x")} Å, ` +
            `mean=${map.mean.toFixed(2)}, rms=${map.rms.toFixed(2)}, ` +
            `[${map.min.toFixed(2)}, ${map.max.toFixed(2)}], downsampled=${map.downsampled}`);
    } catch (err) {
        console.log(`skip t7 (real map not available: ${err.code ?? err})`);
    }
}

console.log(failures === 0 ? "\nAll MRC parser checks passed." : `\n${failures} check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);

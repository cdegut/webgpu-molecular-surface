// Native TypeScript parser for MRC/CCP4 electron-density maps.
//
// No gemmi / DOM / WebGPU dependencies, so this module can be unit-tested in
// isolation or reused elsewhere (e.g. a future Moorhen integration). It reads
// the 1024-byte CCP4/MRC header, converts the stored values to float32, applies
// the axis order from MAPC/MAPR/MAPS, optionally downsamples, and returns a
// scalar field compatible with the marching-cubes `Volume.fromFloat32` route.

export interface MrcGrid
{
    nx: number;
    ny: number;
    nz: number;
    spacing: [number, number, number];   // Angstroms per voxel per axis (x,y,z)
    data: Float32Array;                  // length nx*ny*nz, x fastest
    mean: number;                        // mean density
    rms: number;                         // standard deviation (sigma) of density
    min: number;
    max: number;
    downsampled: boolean;                // true if data was reduced to fit maxDim
}

export interface MrcOptions
{
    maxDim?: number;   // longest axis in voxels after downsampling, default 256
}

// CCP4/MRC header word offsets (32-bit words).
const W_NC = 0;       // columns (x)
const W_NR = 1;       // rows (y)
const W_NS = 2;       // sections (z)
const W_MODE = 3;     // data type
const W_NX = 7;       // cell: intervals along x
const W_NY = 8;
const W_NZ = 9;
const W_CELL_A = 10;  // cell length a (float)
const W_CELL_B = 11;
const W_CELL_C = 12;
const W_MAPC = 16;    // axis order: data axis holding crystal x/y/z
const W_MAPR = 17;
const W_MAPS = 18;
const W_NSYMBT = 23;  // bytes of symmetry records following the header

const HEADER_BYTES = 1024;

// Detect the file's byte order from the machine stamp (word 53, bytes 212-215).
// Little-endian convention: 0x44 0x41 0x00 0x00 ("DA\0\0"); big-endian: 0x11 0x11 ...
function detectLittleEndian(dv: DataView): boolean
{
    const s0 = dv.getUint8(212);
    const s1 = dv.getUint8(213);
    if (s0 === 0x44 && s1 === 0x41) {
        return true;   // little-endian stamp
    }
    if (s0 === 0x11 && s1 === 0x11) {
        return false;  // big-endian stamp
    }
    // No / unknown stamp: prefer the byte order giving a plausible NC.
    const ok = (n: number) => Number.isInteger(n) && n > 0 && n <= 100000;
    if (ok(dv.getInt32(0, true))) {
        return true;
    }
    if (ok(dv.getInt32(0, false))) {
        return false;
    }
    throw new Error("MRC: unrecognised file (no machine stamp / valid grid size)");
}

// IEEE 754 half-precision -> float (avoids relying on DataView.getFloat16).
function halfToFloat(h: number): number
{
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) {
        return sign * frac * 2 ** -24;
    }
    if (exp === 0x1f) {
        return frac ? NaN : sign * Infinity;
    }
    return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

// Read all samples into a Float32Array, converting the stored mode.
function readSamples(
    dv: DataView, little: boolean, mode: number, count: number, offset: number): Float32Array
{
    const out = new Float32Array(count);
    switch (mode) {
        case 0: // int8
            for (let i = 0; i < count; ++i) {
                out[i] = dv.getInt8(offset + i);
            }
            break;
        case 1: // int16
            for (let i = 0; i < count; ++i) {
                out[i] = dv.getInt16(offset + i * 2, little);
            }
            break;
        case 2: // float32
            for (let i = 0; i < count; ++i) {
                out[i] = dv.getFloat32(offset + i * 4, little);
            }
            break;
        case 3: // complex int16 -> magnitude
            for (let i = 0; i < count; ++i) {
                const re = dv.getInt16(offset + i * 4, little);
                const im = dv.getInt16(offset + i * 4 + 2, little);
                out[i] = Math.hypot(re, im);
            }
            break;
        case 4: // complex int32 -> magnitude
            for (let i = 0; i < count; ++i) {
                const re = dv.getInt32(offset + i * 8, little);
                const im = dv.getInt32(offset + i * 8 + 4, little);
                out[i] = Math.hypot(re, im);
            }
            break;
        case 6: // uint16
            for (let i = 0; i < count; ++i) {
                out[i] = dv.getUint16(offset + i * 2, little);
            }
            break;
        case 12: // float16
            for (let i = 0; i < count; ++i) {
                out[i] = halfToFloat(dv.getUint16(offset + i * 2, little));
            }
            break;
        default:
            throw new Error(`MRC: unsupported data mode ${mode}`);
    }
    return out;
}

// Reorder data from (c, r, s) storage order to (x, y, z) = crystal (a, b, c),
// using the MAPC/MAPR/MAPS header fields. x is fastest in both layouts.
function reorderAxes(
    src: Float32Array, nc: number, nr: number, ns: number,
    mapc: number, mapr: number, maps: number): Float32Array
{
    // Identity (the overwhelmingly common case) — no work needed.
    if (mapc === 1 && mapr === 2 && maps === 3) {
        return src;
    }
    const out = new Float32Array(src.length);
    const pick = (axis: number, x: number, y: number, z: number) =>
        axis === 1 ? x : (axis === 2 ? y : z);
    let o = 0;
    for (let z = 0; z < ns; ++z) {
        for (let y = 0; y < nr; ++y) {
            for (let x = 0; x < nc; ++x, ++o) {
                const c = pick(mapc, x, y, z);
                const r = pick(mapr, x, y, z);
                const s = pick(maps, x, y, z);
                out[o] = src[c + nc * (r + nr * s)];
            }
        }
    }
    return out;
}

// Box-average downsample so no axis exceeds maxDim. Returns new dims + data.
function downsample(
    data: Float32Array, nx: number, ny: number, nz: number, maxDim: number):
    { data: Float32Array; nx: number; ny: number; nz: number; downsampled: boolean }
{
    const fx = Math.max(1, Math.ceil(nx / maxDim));
    const fy = Math.max(1, Math.ceil(ny / maxDim));
    const fz = Math.max(1, Math.ceil(nz / maxDim));
    if (fx === 1 && fy === 1 && fz === 1) {
        return {data, nx, ny, nz, downsampled: false};
    }
    const tnx = Math.ceil(nx / fx);
    const tny = Math.ceil(ny / fy);
    const tnz = Math.ceil(nz / fz);
    const out = new Float32Array(tnx * tny * tnz);
    for (let tz = 0; tz < tnz; ++tz) {
        const sz0 = tz * fz, sz1 = Math.min(nz, sz0 + fz);
        for (let ty = 0; ty < tny; ++ty) {
            const sy0 = ty * fy, sy1 = Math.min(ny, sy0 + fy);
            for (let tx = 0; tx < tnx; ++tx) {
                const sx0 = tx * fx, sx1 = Math.min(nx, sx0 + fx);
                let sum = 0;
                const count = (sx1 - sx0) * (sy1 - sy0) * (sz1 - sz0);
                for (let sz = sz0; sz < sz1; ++sz) {
                    for (let sy = sy0; sy < sy1; ++sy) {
                        const row = sx0 + nx * (sy + ny * sz);
                        for (let sx = sx0; sx < sx1; ++sx) {
                            sum += data[row + (sx - sx0)];
                        }
                    }
                }
                out[tx + tnx * (ty + tny * tz)] = sum / count;
            }
        }
    }
    return {data: out, nx: tnx, ny: tny, nz: tnz, downsampled: true};
}

function computeStats(data: Float32Array): { mean: number; rms: number; min: number; max: number }
{
    let min = Infinity, max = -Infinity;
    let sum = 0, sumsq = 0;
    for (let i = 0; i < data.length; ++i) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumsq += v * v;
    }
    const mean = sum / data.length;
    const variance = Math.max(0, sumsq / data.length - mean * mean);
    return {mean, rms: Math.sqrt(variance), min, max};
}

export function parseMrc(buffer: ArrayBuffer, options: MrcOptions = {}): MrcGrid
{
    // Non-positive maxDim disables downsampling ("full resolution").
    const maxDim = options.maxDim != null && options.maxDim > 0 ? options.maxDim : Infinity;
    if (buffer.byteLength < HEADER_BYTES) {
        throw new Error("MRC: file too small to contain a 1024-byte header");
    }
    const dv = new DataView(buffer);
    const little = detectLittleEndian(dv);
    const i32 = (w: number) => dv.getInt32(w * 4, little);

    const nc = i32(W_NC);
    const nr = i32(W_NR);
    const ns = i32(W_NS);
    const mode = i32(W_MODE);
    if (nc <= 0 || nr <= 0 || ns <= 0) {
        throw new Error(`MRC: invalid grid dimensions ${nc} x ${nr} x ${ns}`);
    }

    const cellN = [i32(W_NX), i32(W_NY), i32(W_NZ)];
    const cell = [dv.getFloat32(W_CELL_A * 4, little),
                  dv.getFloat32(W_CELL_B * 4, little),
                  dv.getFloat32(W_CELL_C * 4, little)];
    // Some cryo-EM maps omit cell dimensions / cell grid; fall back to 1 Å.
    const spacing: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < 3; ++i) {
        const nCell = cellN[i] > 0 ? cellN[i] : [nc, nr, ns][i];
        spacing[i] = cell[i] > 0 && nCell > 0 ? cell[i] / nCell : 1.0;
    }

    // Sanitize axis order (1/2/3); anything else treated as identity.
    const axis = [i32(W_MAPC), i32(W_MAPR), i32(W_MAPS)];
    for (let i = 0; i < 3; ++i) {
        if (axis[i] < 1 || axis[i] > 3) {
            axis[i] = i + 1;
        }
    }

    const nsymbt = Math.max(0, i32(W_NSYMBT));
    const dataOffset = HEADER_BYTES + nsymbt;
    const count = nc * nr * ns;

    let src: Float32Array;
    // Fast path: mode 2 (float32), little-endian file on little-endian host.
    const hostLittle = true;  // all supported web platforms are little-endian
    if (mode === 2 && little === hostLittle) {
        src = new Float32Array(dv.buffer, dataOffset, count).slice();
    } else {
        src = readSamples(dv, little, mode, count, dataOffset);
    }

    const reordered = reorderAxes(src, nc, nr, ns, axis[0], axis[1], axis[2]);
    const ds = downsample(reordered, nc, nr, ns, maxDim);
    const stats = computeStats(ds.data);

    return {
        nx: ds.nx,
        ny: ds.ny,
        nz: ds.nz,
        spacing,
        data: ds.data,
        mean: stats.mean,
        rms: stats.rms,
        min: stats.min,
        max: stats.max,
        downsampled: ds.downsampled,
    };
}

// Pure TypeScript computation of a molecular-surface scalar field on a
// uniform 3D grid. No DOM / WebGPU / gemmi dependencies, so this module can
// later be dropped into Moorhen or unit-tested in isolation.
//
// Both surface modes produce a smooth field in [0, 1] whose 0.5 level-set is
// the surface, matching the marching-cubes isovalue convention:
//   * vdw : union of per-atom van der Waals spheres (smooth soft edge)
//   * ses : solvent-excluded surface via the classic CNS / gemmi SolventMasker
//           algorithm (extended spheres + shrink + soft edge)

export type SurfaceMode = "ses" | "vdw";

export interface SurfaceAtom
{
    x: number;
    y: number;
    z: number;
    r: number;      // van der Waals radius in Angstroms
    element: string;
}

export interface SurfaceGrid
{
    nx: number;
    ny: number;
    nz: number;
    spacing: number;                       // Angstroms per voxel (uniform/cubic)
    origin: [number, number, number];      // world position of voxel (0,0,0)
    data: Float32Array;                    // length nx*ny*nz, Fortran-ish order x fastest
}

export interface SurfaceOptions
{
    mode: SurfaceMode;
    maxDim?: number;     // longest grid axis in voxels (resolution), default 128
    probe?: number;      // SES solvent probe radius (Angstroms), default 1.4
    blur?: number;       // soft-edge width (Angstroms), default 0.6
}

const VDW_RADII: Record<string, number> = {
    H: 1.10, HE: 1.40, LI: 1.82, BE: 1.53, B: 1.92, C: 1.70, N: 1.55, O: 1.52,
    F: 1.47, NE: 1.54, NA: 2.27, MG: 1.73, AL: 1.84, SI: 2.10, P: 1.80, S: 1.80,
    CL: 1.75, AR: 1.88, K: 2.75, CA: 2.31, SC: 2.15, TI: 2.11, V: 2.07, CR: 2.06,
    MN: 2.05, FE: 2.04, CO: 2.00, NI: 1.97, CU: 1.96, ZN: 2.01, GA: 1.87, GE: 2.11,
    AS: 1.85, SE: 1.90, BR: 1.85, KR: 2.02, RB: 3.03, SR: 2.49, Y: 2.32, ZR: 2.23,
    NB: 2.18, MO: 2.17, TC: 2.16, RU: 2.13, RH: 2.10, PD: 2.10, AG: 2.11, CD: 2.18,
    IN: 1.93, SN: 2.17, SB: 2.06, TE: 2.06, I: 1.98, XE: 2.16, CS: 3.43, BA: 2.68,
    LA: 2.43, CE: 2.42, PR: 2.40, ND: 2.39, SM: 2.36, EU: 2.35, GD: 2.34, TB: 2.33,
    DY: 2.31, HO: 2.30, ER: 2.29, TM: 2.27, YB: 2.26, LU: 2.24, HF: 2.23, TA: 2.22,
    W: 2.18, RE: 2.16, OS: 2.16, IR: 2.13, PT: 2.13, AU: 2.14, HG: 2.23, TL: 1.96,
    PB: 2.02, BI: 2.07, PO: 1.97, AT: 2.02, RN: 2.20, FR: 3.48, RA: 2.83, AC: 2.47,
    TH: 2.45, U: 2.41,
};

export function vdwRadius(element: string): number
{
    const sym = element.replace(/[^a-zA-Z]/g, "").toUpperCase();
    const r = VDW_RADII[sym];
    return r !== undefined ? r : 1.70;
}

function clamp01(x: number): number
{
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

interface Bin
{
    // atom indices
    ids: number[];
}

// Bin atoms (and voxels) on a uniform world-space grid so each voxel only
// examines atoms in the 3x3x3 cells around it.
function buildAtomBins(
    atoms: SurfaceAtom[],
    origin: [number, number, number],
    cellSize: number,
    nx: number, ny: number, nz: number, spacing: number,
    cellVox: number): { bins: Bin[]; ncx: number; ncy: number; ncz: number }
{
    const ncx = Math.max(1, Math.ceil(nx / cellVox) + 2);
    const ncy = Math.max(1, Math.ceil(ny / cellVox) + 2);
    const ncz = Math.max(1, Math.ceil(nz / cellVox) + 2);
    const bins: Bin[] = new Array(ncx * ncy * ncz);
    for (let i = 0; i < bins.length; ++i) {
        bins[i] = {ids: []};
    }

    for (let a = 0; a < atoms.length; ++a) {
        const cx = Math.floor((atoms[a].x - origin[0]) / cellSize);
        const cy = Math.floor((atoms[a].y - origin[1]) / cellSize);
        const cz = Math.floor((atoms[a].z - origin[2]) / cellSize);
        if (cx < 0 || cy < 0 || cz < 0 || cx >= ncx || cy >= ncy || cz >= ncz) {
            continue;
        }
        bins[cx + ncx * (cy + ncy * cz)].ids.push(a);
    }
    return {bins, ncx, ncy, ncz};
}

// Approximate Euclidean distance transform (3-4-5 chamfer, two passes).
// set[i] == 1 marks seeds; returns distance to nearest seed in VOXEL units.
function chamferDistanceToSet(set: Uint8Array, nx: number, ny: number, nz: number): Float32Array
{
    const n = nx * ny * nz;
    const INF = 1e9;
    const d = new Float32Array(n);
    for (let i = 0; i < n; ++i) {
        d[i] = set[i] ? 0.0 : INF;
    }

    const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z);

    // Offsets: weight = 3 (face), 4 (edge), 5 (corner).
    const offsets: Array<[number, number, number, number]> = [];
    const add = (dx: number, dy: number, dz: number) => {
        const w = (dx !== 0 ? 1 : 0) + (dy !== 0 ? 1 : 0) + (dz !== 0 ? 1 : 0);
        offsets.push([dx, dy, dz, [0, 3, 4, 5][w]]);
    };

    // Forward pass: visit voxels in increasing index order, only pulling from
    // already-visited neighbours.
    offsets.length = 0;
    for (const dz of [-1, 0]) {
        for (const dy of [-1, 0, 1]) {
            for (const dx of [-1, 0, 1]) {
                if (dz === 0 && dy === 0 && dx >= 0) continue;
                add(dx, dy, dz);
            }
        }
    }
    for (let z = 0; z < nz; ++z) {
        for (let y = 0; y < ny; ++y) {
            for (let x = 0; x < nx; ++x) {
                const i = idx(x, y, z);
                let best = d[i];
                for (const [dx, dy, dz, w] of offsets) {
                    const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                    if (nx2 < 0 || ny2 < 0 || nz2 < 0 || nx2 >= nx || ny2 >= ny || nz2 >= nz) {
                        continue;
                    }
                    const v = d[idx(nx2, ny2, nz2)] + w;
                    if (v < best) best = v;
                }
                d[i] = best;
            }
        }
    }

    // Backward pass: decreasing index order, pulling from higher neighbours.
    offsets.length = 0;
    for (const dz of [1, 0]) {
        for (const dy of [-1, 0, 1]) {
            for (const dx of [-1, 0, 1]) {
                if (dz === 0 && dy === 0 && dx <= 0) continue;
                add(dx, dy, dz);
            }
        }
    }
    for (let z = nz - 1; z >= 0; --z) {
        for (let y = ny - 1; y >= 0; --y) {
            for (let x = nx - 1; x >= 0; --x) {
                const i = idx(x, y, z);
                let best = d[i];
                for (const [dx, dy, dz, w] of offsets) {
                    const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                    if (nx2 < 0 || ny2 < 0 || nz2 < 0 || nx2 >= nx || ny2 >= ny || nz2 >= nz) {
                        continue;
                    }
                    const v = d[idx(nx2, ny2, nz2)] + w;
                    if (v < best) best = v;
                }
                d[i] = best;
            }
        }
    }

    // Chamfer weights are in units where one voxel step == 3.
    for (let i = 0; i < n; ++i) {
        d[i] /= 3.0;
    }
    return d;
}

export function buildSurfaceGrid(
    atomsIn: SurfaceAtom[],
    options: SurfaceOptions): SurfaceGrid
{
    const maxDim = options.maxDim ?? 128;
    const probe = options.probe ?? 1.4;
    const blur = options.blur ?? 0.6;
    const mode = options.mode;

    const atoms = atomsIn.filter(a =>
        Number.isFinite(a.x) && Number.isFinite(a.y) && Number.isFinite(a.z)
        && Number.isFinite(a.r) && a.r > 0.0);
    if (atoms.length === 0) {
        throw new Error("No atoms with valid coordinates and radii");
    }

    let maxR = 0.0;
    for (const a of atoms) {
        if (a.r > maxR) maxR = a.r;
    }
    // How far an atom can influence the field from its centre.
    const extent = (mode === "ses" ? maxR + probe : maxR + 2.0 * blur);
    const pad = 1.0;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const a of atoms) {
        if (a.x < minX) minX = a.x;
        if (a.y < minY) minY = a.y;
        if (a.z < minZ) minZ = a.z;
        if (a.x > maxX) maxX = a.x;
        if (a.y > maxY) maxY = a.y;
        if (a.z > maxZ) maxZ = a.z;
    }

    const origin: [number, number, number] = [
        minX - extent - pad, minY - extent - pad, minZ - extent - pad];
    const sizeX = (maxX - minX) + 2.0 * (extent + pad);
    const sizeY = (maxY - minY) + 2.0 * (extent + pad);
    const sizeZ = (maxZ - minZ) + 2.0 * (extent + pad);

    // Cubic voxels: spacing is chosen so the longest axis fits in maxDim.
    let spacing = Math.max(sizeX, sizeY, sizeZ) / (maxDim - 1);
    let nx = Math.ceil(sizeX / spacing) + 1;
    let ny = Math.ceil(sizeY / spacing) + 1;
    let nz = Math.ceil(sizeZ / spacing) + 1;

    // Hard safety clamp: keep the grid small enough to be fast in JS and to
    // fit comfortably inside a WebGPU 3D texture.
    const MAX_DIM = 320;
    if (nx > MAX_DIM || ny > MAX_DIM || nz > MAX_DIM) {
        const grow = Math.max(nx / MAX_DIM, ny / MAX_DIM, nz / MAX_DIM, 1.0);
        spacing *= grow;
        nx = Math.ceil(sizeX / spacing) + 1;
        ny = Math.ceil(sizeY / spacing) + 1;
        nz = Math.ceil(sizeZ / spacing) + 1;
    }
    nx = Math.max(nx, 8); ny = Math.max(ny, 8); nz = Math.max(nz, 8);

    const n = nx * ny * nz;
    const data = new Float32Array(n);
    const cellSize = extent;
    const cellVox = Math.max(1.0, cellSize / spacing);
    const {bins, ncx, ncy, ncz} =
        buildAtomBins(atoms, origin, cellSize, nx, ny, nz, spacing, cellVox);

    const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z);

    if (mode === "vdw") {
        for (let z = 0; z < nz; ++z) {
            const pz = origin[2] + z * spacing;
            for (let y = 0; y < ny; ++y) {
                const py = origin[1] + y * spacing;
                for (let x = 0; x < nx; ++x) {
                    const px = origin[0] + x * spacing;
                    const cx = Math.floor(x / cellVox);
                    const cy = Math.floor(y / cellVox);
                    const cz = Math.floor(z / cellVox);
                    let best = 0.0;
                    outer:
                    for (let bz = cz - 1; bz <= cz + 1; ++bz) {
                        for (let by = cy - 1; by <= cy + 1; ++by) {
                            for (let bx = cx - 1; bx <= cx + 1; ++bx) {
                                if (bx < 0 || by < 0 || bz < 0 || bx >= ncx || by >= ncy || bz >= ncz) {
                                    continue;
                                }
                                const bin = bins[bx + ncx * (by + ncy * bz)];
                                for (let ai = 0; ai < bin.ids.length; ++ai) {
                                    const a = atoms[bin.ids[ai]];
                                    const dx = px - a.x, dy = py - a.y, dz = pz - a.z;
                                    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                                    const v = clamp01(0.5 + (a.r - d) / (2.0 * blur));
                                    if (v > best) best = v;
                                    if (best >= 1.0) break outer;
                                }
                            }
                        }
                    }
                    data[idx(x, y, z)] = best;
                }
            }
        }
    } else {
        // SES: pass A - mark extended spheres (vdw + probe) as inside (0).
        const mask = new Uint8Array(n).fill(1);
        for (let z = 0; z < nz; ++z) {
            const pz = origin[2] + z * spacing;
            for (let y = 0; y < ny; ++y) {
                const py = origin[1] + y * spacing;
                for (let x = 0; x < nx; ++x) {
                    const px = origin[0] + x * spacing;
                    const cx = Math.floor(x / cellVox);
                    const cy = Math.floor(y / cellVox);
                    const cz = Math.floor(z / cellVox);
                    let inside = false;
                    outer2:
                    for (let bz = cz - 1; bz <= cz + 1; ++bz) {
                        for (let by = cy - 1; by <= cy + 1; ++by) {
                            for (let bx = cx - 1; bx <= cx + 1; ++bx) {
                                if (bx < 0 || by < 0 || bz < 0 || bx >= ncx || by >= ncy || bz >= ncz) {
                                    continue;
                                }
                                const bin = bins[bx + ncx * (by + ncy * bz)];
                                for (let ai = 0; ai < bin.ids.length; ++ai) {
                                    const a = atoms[bin.ids[ai]];
                                    const dx = px - a.x, dy = py - a.y, dz = pz - a.z;
                                    const lim = a.r + probe;
                                    if (dx * dx + dy * dy + dz * dz < lim * lim) {
                                        inside = true;
                                        break outer2;
                                    }
                                }
                            }
                        }
                    }
                    if (inside) {
                        mask[idx(x, y, z)] = 0;
                    }
                }
            }
        }

        // Pass B: shrink the solvent region by `probe` (erode the inside region).
        // Distance to nearest solvent voxel -> smooth soft edge field.
        const distVox = chamferDistanceToSet(mask, nx, ny, nz);
        const shrink = probe;
        for (let i = 0; i < n; ++i) {
            const distA = distVox[i] * spacing;
            data[i] = clamp01(0.5 + (distA - shrink) / (2.0 * blur));
        }
    }

    return {nx, ny, nz, spacing, origin, data};
}

// Adapter around Moorhen's standalone gemmi WASM module (gemmi.js + gemmi.wasm).
// It is an Emscripten UMD that defines the global `createGemmiModule` when
// loaded via a <script> tag. This file keeps all gemmi specifics in one place
// so the rest of the app (and a future Moorhen integration) can stay agnostic.

import {SurfaceAtom, SurfaceGrid, SurfaceOptions, buildSurfaceGrid, vdwRadius} from "./surface";

export interface GemmiAtomInfo
{
    x: number;
    y: number;
    z: number;
    charge: number;
    element: string;
    tempFactor: number;
    serial: number;
    occupancy: number;
    name: string;
    has_altloc: boolean;
    alt_loc?: string;
    mol_name: string;
    chain_id: string;
    res_no: string;
    res_name: string;
}

interface GemmiAtomInfoVector
{
    size(): number;
    get(i: number): GemmiAtomInfo;
    delete(): void;
}

export interface GemmiModule
{
    read_structure_from_string(data: string, path: string): GemmiStructure;
    get_atom_info_for_selection(structure: GemmiStructure, cid: string, altloc: string): GemmiAtomInfoVector;
}

export interface GemmiStructure
{
    delete(): void;
}

interface GemmiFactory
{
    (options?: {print?: (text: string) => void; printErr?: (text: string) => void;
        locateFile?: (path: string) => string}): Promise<GemmiModule>;
}

declare global
{
    interface Window
    {
        createGemmiModule?: GemmiFactory;
    }
}

function loadScript(src: string): Promise<void>
{
    return new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[src="${src}"]`) as HTMLScriptElement | null;
        if (existing) {
            if (existing.dataset.loaded === "true") {
                resolve();
                return;
            }
            existing.addEventListener("load", () => resolve());
            existing.addEventListener("error", () => reject(new Error(`Failed to load script: ${src}`)));
            return;
        }
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.onload = () => {script.dataset.loaded = "true"; resolve();};
        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
        document.head.appendChild(script);
    });
}

let modulePromise: Promise<GemmiModule> | null = null;

export function loadGemmi(wasmDir = "/wasm/"): Promise<GemmiModule>
{
    if (modulePromise) {
        return modulePromise;
    }
    modulePromise = (async () => {
        await loadScript(`${wasmDir}gemmi.js`);
        const factory = window.createGemmiModule;
        if (!factory) {
            throw new Error("gemmi.js loaded but did not define createGemmiModule");
        }
        const mod = await factory({
            locateFile: (path: string) => `${wasmDir}${path}`,
        });
        return mod;
    })();
    return modulePromise;
}

// Extract atoms (position + element) from a gemmi Structure. Uses the
// all-atoms CID selection "*".
export function structureToAtoms(mod: GemmiModule, structure: GemmiStructure): SurfaceAtom[]
{
    const vec = mod.get_atom_info_for_selection(structure, "*", "");
    const atoms: SurfaceAtom[] = [];
    try {
        const n = vec.size();
        for (let i = 0; i < n; ++i) {
            const a = vec.get(i);
            const el = (a.element || "").trim();
            // Solvent masks conventionally ignore hydrogen; VDW radii for H are
            // noisy anyway, so skip them by default.
            const up = el.toUpperCase();
            if (up === "H" || up === "D") {
                continue;
            }
            if (!Number.isFinite(a.x) || !Number.isFinite(a.y) || !Number.isFinite(a.z)) {
                continue;
            }
            atoms.push({x: a.x, y: a.y, z: a.z, r: vdwRadius(el), element: el});
        }
    } finally {
        vec.delete();
    }
    return atoms;
}

// Read a coordinate file (PDB / mmCIF / mmJSON text) and produce the surface
// scalar field for it.
export async function structureTextToSurfaceGrid(
    mod: GemmiModule,
    text: string,
    path: string,
    options: SurfaceOptions): Promise<SurfaceGrid>
{
    const structure = mod.read_structure_from_string(text, path);
    let atoms: SurfaceAtom[];
    try {
        atoms = structureToAtoms(mod, structure);
    } finally {
        structure.delete();
    }
    return buildSurfaceGrid(atoms, options);
}

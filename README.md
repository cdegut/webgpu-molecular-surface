# webgpu-molecular-surface

In-browser molecular surface rendering with WebGPU.

Parses PDB / mmCIF / mmJSON coordinates with
[gemmi](https://github.com/project-gemmi/gemmi) compiled to WebAssembly, builds a
solvent-excluded (SES) or van der Waals (VDW) scalar field, extracts the surface
on the GPU with marching cubes, and renders it with smooth shading. Atoms can
also be shown as ray-traced CPK spheres.

## Features

- SES and VDW molecular surfaces (float32 scalar field, GPU marching cubes)
- Smooth shading from field-gradient normals (crack-free, edge-manifold mesh)
- Atom spheres rendered as instanced billboard impostors with CPK colours
- In-browser parsing via gemmi WASM — no server, local file upload only
- Standalone proof-of-concept intended for later integration into Moorhen

## Layout

- `src/marching-cubes/` — GPU marching cubes core, adapted from
  [Twinklebear/webgpu-marching-cubes](https://github.com/Twinklebear/webgpu-marching-cubes)
- `src/molecular/` — gemmi WASM adapter, surface-field computation, atom rendering
- `src/app.ts` — application entry point
- `public/wasm/` — gemmi WASM runtime (`gemmi.js`, `gemmi.wasm`)
- `scripts/check_manifold.mjs` — verifies the marching-cubes case table is
  crack-free and edge-manifold

## Run

```sh
npm install
npm run serve    # webpack-dev-server at http://localhost:8080
npm run build    # production-ish bundle into dist/
```

## License

MIT. Includes GPU marching-cubes code from
[Twinklebear/webgpu-marching-cubes](https://github.com/Twinklebear/webgpu-marching-cubes).

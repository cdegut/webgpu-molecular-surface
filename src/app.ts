import {ArcballCamera} from "arcball_camera";
import {Controller} from "ez_canvas_controller";
import {mat4, vec3} from "gl-matrix";

import {Volume} from "./marching-cubes/volume";
import {MarchingCubes, MarchingCubesResult} from "./marching-cubes/marching_cubes";
import {loadGemmi, structureToAtoms} from "./molecular/gemmi";
import {SurfaceMode, SurfaceAtom, buildSurfaceGrid} from "./molecular/surface";
import {cpkColor} from "./molecular/cpk";
import renderMeshShaders from "./marching-cubes/render_mesh.wgsl";
import renderSpheresWgsl from "./molecular/render_spheres.wgsl";
import {compileShader} from "./marching-cubes/util";

(async () =>
{
    const showNoWebgpu = (title: string, hint?: string) => {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        document.getElementById("no-webgpu-title").innerHTML = title;
        const hintEl = document.getElementById("no-webgpu-hint");
        hintEl.innerHTML = hint ?? "";
    };

    if (navigator.gpu === undefined) {
        showNoWebgpu("WebGPU is not available in this browser",
            "Use a recent Chromium-based browser (Chrome/Edge 113+) or Firefox with WebGPU enabled.");
        return;
    }

    // Get a GPU device to render with. Retry with different power preferences:
    // on some Linux setups the default request returns null even though a
    // (software or discrete) adapter exists.
    const adapterOptions: GPURequestAdapterOptions[] = [
        {},
        {powerPreference: "high-performance"},
        {powerPreference: "low-power"},
        // Last resort: the software (SwiftShader) adapter, which works even
        // without Vulkan. Slow, but proves the pipeline works.
        {forceFallbackAdapter: true},
    ];
    let adapter: GPUAdapter = null;
    let usedFallback = false;
    for (const options of adapterOptions) {
        try {
            adapter = await navigator.gpu.requestAdapter(options);
        } catch (err) {
            console.warn("requestAdapter failed", options, err);
        }
        if (adapter) {
            usedFallback = options.forceFallbackAdapter === true;
            break;
        }
    }
    if (!adapter) {
        console.log("WebGPU adapter request returned null for all options");
        showNoWebgpu("No WebGPU adapter found",
            "Your Chrome build has WebGPU but no adapter could be created. " +
            "On Linux try relaunching with: <code>--enable-features=Vulkan</code>, " +
            "<code>--use-angle=vulkan</code> and <code>--enable-unsafe-webgpu</code> " +
            "(add <code>--ignore-gpu-blocklist</code> if needed), then check " +
            "<code>chrome://gpu</code> for a WebGPU entry.");
        return;
    }
    console.log("Using " + (usedFallback ? "software (fallback) " : "") + "WebGPU adapter");
    console.log(adapter.limits);

    let deviceRequiredFeatures: GPUFeatureName[] = [];
    const timestampSupport = adapter.features.has("timestamp-query");
    // Enable timestamp queries if the device supports them
    if (timestampSupport) {
        deviceRequiredFeatures.push("timestamp-query");
    } else {
        console.log("Device does not support timestamp queries");
    }
    let deviceDescriptor = {
        requiredFeatures: deviceRequiredFeatures,
        requiredLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        }
    };

    let device = await adapter.requestDevice(deviceDescriptor);

    // Get a context to display our rendered image on the canvas
    let canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    let context = canvas.getContext("webgpu");

    let fileInput = document.getElementById("structureFile") as HTMLInputElement;
    let modeSelect = document.getElementById("surfaceMode") as HTMLSelectElement;
    let resolutionSelect = document.getElementById("gridResolution") as HTMLSelectElement;

    let isovalueSlider = document.getElementById("isovalueSlider") as HTMLInputElement;
    // Force computing the surface on the initial load
    let currentIsovalue = -1;

    let perfDisplay = document.getElementById("stats") as HTMLElement;
    let timestampDisplay = document.getElementById("timestamp-stats") as HTMLElement;
    let statusText = document.getElementById("loadingText") as HTMLElement;

    let volume: Volume = null;
    let mc: MarchingCubes = null;
    let isosurface: MarchingCubesResult = null;
    let spacing = 1.0;
    let rebuildRequested = false;
    let busy = false;

    // Atom-sphere rendering state
    let atomBuffer: GPUBuffer = null;
    let sphereShaderModule = await compileShader(device, renderSpheresWgsl, "renderSpheresWgsl");
    let atomCount = 0;
    let sphereBindGroup: GPUBindGroup = null;
    let spherePipeline: GPURenderPipeline = null;
    let sphereParamsBuffer: GPUBuffer = null;
    let sphereUploadBuffer: GPUBuffer = null;
    let sphereBindGroupLayout: GPUBindGroupLayout = null;
    let showAtomsCheckbox = document.getElementById("showAtoms") as HTMLInputElement;
    let showSurfaceCheckbox = document.getElementById("showSurface") as HTMLInputElement;

    // Setup shader modules
    let shaderModule = await compileShader(device, renderMeshShaders, "renderMeshShaders");

    // Vertex attribute state and shader stage
    let vertexState = {
        // Shader stage info
        module: shaderModule,
        entryPoint: "vertex_main",
        // Vertex buffer info
        buffers: [{
            arrayStride: 8 * 4,
            attributes: [
                {format: "float32x4" as GPUVertexFormat, offset: 0, shaderLocation: 0},
                {format: "float32x4" as GPUVertexFormat, offset: 4 * 4, shaderLocation: 1}
            ]
        }]
    };

    // Setup render outputs
    let swapChainFormat = "bgra8unorm" as GPUTextureFormat;
    context.configure(
        {device: device, format: swapChainFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT});

    let depthFormat = "depth24plus-stencil8" as GPUTextureFormat;
    let depthTexture = device.createTexture({
        size: {width: canvas.width, height: canvas.height, depthOrArrayLayers: 1},
        format: depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    let fragmentState = {
        // Shader info
        module: shaderModule,
        entryPoint: "fragment_main",
        // Output render target info
        targets: [{format: swapChainFormat}]
    };

    let bindGroupLayout = device.createBindGroupLayout({
        entries: [{binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}}]
    });

    // Create render pipeline
    let layout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

    let renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertex: vertexState,
        fragment: fragmentState,
        depthStencil: {format: depthFormat, depthWriteEnabled: true, depthCompare: "less"}
    });

    let renderPassDesc = {
        colorAttachments: [{
            view: null as GPUTextureView,
            loadOp: "clear" as GPULoadOp,
            clearValue: [0.3, 0.3, 0.3, 1],
            storeOp: "store" as GPUStoreOp
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear" as GPULoadOp,
            depthClearValue: 1.0,
            depthStoreOp: "store" as GPUStoreOp,
            stencilLoadOp: "clear" as GPULoadOp,
            stencilClearValue: 0,
            stencilStoreOp: "store" as GPUStoreOp
        }
    };

    let viewParamsBuffer = device.createBuffer({
        size: (4 * 4 + 4 + 4 + 4) * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
    });

    let uploadBuffer = device.createBuffer({
        size: viewParamsBuffer.size,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false,
    });

    let bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{binding: 0, resource: {buffer: viewParamsBuffer}}]
    });

    // --- Atom spheres: instanced billboard quads ray-traced in the fragment shader ---
    sphereParamsBuffer = device.createBuffer({
        size: 128,  // view (mat4) + proj (mat4)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    sphereUploadBuffer = device.createBuffer({
        size: sphereParamsBuffer.size,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });

    sphereBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}}
        ]
    });

    spherePipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [sphereBindGroupLayout]}),
        vertex: {
            module: sphereShaderModule,
            entryPoint: "vs_main",
            // Geometry is generated from vertex_index / instance_index.
            buffers: []
        },
        fragment: {
            module: sphereShaderModule,
            entryPoint: "fs_main",
            targets: [{format: swapChainFormat}]
        },
        depthStencil: {format: depthFormat, depthWriteEnabled: true, depthCompare: "less"}
    });

    // Setup camera and camera controls. Positions are in Angstroms, with the
    // surface centred at the origin (see render_mesh.wgsl).
    const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
    const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
    let camera: any = null;
    let proj = mat4.perspective(
        mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 10000);
    let projView = mat4.create();

    // (Re)build the volume, marching-cubes pipeline and camera for the current
    // file / mode / resolution selection.
    const rebuild = async () => {
        if (busy) {
            rebuildRequested = true;
            return;
        }
        const file = fileInput.files[0];
        if (!file) {
            return;
        }
        busy = true;
        try {
            statusText.innerHTML = "Parsing structure and building surface field...";
            const mod = await loadGemmi();
            const text = await file.text();
            const mode = modeSelect.value as SurfaceMode;
            const maxDim = parseInt(resolutionSelect.value, 10);

            // Parse once and reuse the atoms for both the surface field and the
            // atom spheres.
            const structure = mod.read_structure_from_string(text, file.name);
            let parsedAtoms: SurfaceAtom[];
            try {
                parsedAtoms = structureToAtoms(mod, structure);
            } finally {
                structure.delete();
            }
            const grid = buildSurfaceGrid(parsedAtoms, {mode, maxDim});

            volume = await Volume.fromFloat32([grid.nx, grid.ny, grid.nz], grid.data, device);
            mc = await MarchingCubes.create(volume, device);
            spacing = grid.spacing;
            currentIsovalue = -1;

            // Centre the atoms on the grid centre so they share the surface's
            // world space (render_mesh.wgsl maps grid indices to a world space
            // centred at the origin).
            const gc: [number, number, number] = [
                grid.origin[0] + grid.nx * spacing * 0.5,
                grid.origin[1] + grid.ny * spacing * 0.5,
                grid.origin[2] + grid.nz * spacing * 0.5,
            ];
            atomCount = parsedAtoms.length;
            const atomData = new Float32Array(atomCount * 8);
            for (let i = 0; i < atomCount; ++i) {
                const a = parsedAtoms[i];
                const c = cpkColor(a.element);
                atomData[i * 8 + 0] = a.x - gc[0];
                atomData[i * 8 + 1] = a.y - gc[1];
                atomData[i * 8 + 2] = a.z - gc[2];
                atomData[i * 8 + 3] = a.r;
                atomData[i * 8 + 4] = c[0];
                atomData[i * 8 + 5] = c[1];
                atomData[i * 8 + 6] = c[2];
                atomData[i * 8 + 7] = 1.0;
            }
            if (atomBuffer) {
                atomBuffer.destroy();
            }
            atomBuffer = device.createBuffer({
                size: atomData.byteLength,
                usage: GPUBufferUsage.STORAGE,
                mappedAtCreation: true,
            });
            new Float32Array(atomBuffer.getMappedRange()).set(atomData);
            atomBuffer.unmap();

            sphereBindGroup = device.createBindGroup({
                layout: sphereBindGroupLayout,
                entries: [
                    {binding: 0, resource: {buffer: sphereParamsBuffer}},
                    {binding: 1, resource: {buffer: atomBuffer}},
                ]
            });

            const physX = grid.nx * spacing, physY = grid.ny * spacing, physZ = grid.nz * spacing;
            const diagonal = Math.sqrt(physX * physX + physY * physY + physZ * physZ);
            const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, physZ * 0.75);
            camera = new ArcballCamera(defaultEye, center, up, diagonal * 0.1,
                [canvas.width, canvas.height]);

            statusText.innerHTML =
                `${file.name} — ${grid.nx}×${grid.ny}×${grid.nz} @ ${spacing.toFixed(3)} Å, ` +
                `${atomCount} atoms`;
        } catch (err) {
            console.error(err);
            statusText.innerHTML = `Error: ${err.message ?? err}`;
        } finally {
            busy = false;
        }
    };

    fileInput.addEventListener("change", () => {rebuildRequested = true;});
    modeSelect.addEventListener("change", () => {rebuildRequested = true;});
    resolutionSelect.addEventListener("change", () => {rebuildRequested = true;});

    // Register mouse and touch listeners
    var controller = new Controller();
    controller.mousemove = function (prev: Array<number>, cur: Array<number>, evt: MouseEvent)
    {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function (amt: number)
    {
        camera.zoom(amt);
    };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function (drag: number)
    {
        camera.pan(drag);
    };
    controller.registerForCanvas(canvas);

    // Render!
    const render = async () => {
        if (rebuildRequested) {
            rebuildRequested = false;
            await rebuild();
        }

        if (!volume || !mc || !camera) {
            requestAnimationFrame(render);
            return;
        }

        let sliderValue = parseFloat(isovalueSlider.value);
        let recomputeSurface = sliderValue != currentIsovalue;

        if (recomputeSurface) {
            if (isosurface && isosurface.buffer) {
                isosurface.buffer.destroy();
            }

            currentIsovalue = sliderValue;
            let start = performance.now();
            isosurface = await mc.computeSurface(currentIsovalue);
            let end = performance.now();

            perfDisplay.innerHTML =
                `<p>Compute Time: ${(end - start).toFixed((2))}ms<br/># Triangles: ${isosurface.count / 3}</p>`

            timestampDisplay.innerHTML =
                `<h4>Timing Breakdown</h4>
                    <p>Note: if timestamp-query is not supported, -1 is shown for kernel times</p>

                    Compute Active Voxels: ${mc.computeActiveVoxelsTime.toFixed(2)}ms
                    <ul>
                    <li>
                    Mark Active Voxels Kernel: ${mc.markActiveVoxelsKernelTime.toFixed(2)}ms
                    </li>
                    <li>
                    Stream Compact: ${mc.computeActiveVoxelsCompactTime.toFixed(2)}ms
                    </li>
                    </ul>

                    Compute Vertex Offsets: ${mc.computeVertexOffsetsTime.toFixed(2)}ms
                    <ul>
                    <li>
                    Compute # of Vertices Kernel: ${mc.computeNumVertsKernelTime.toFixed(2)}ms
                    </li>
                    </ul>

                    Compute Vertices: ${mc.computeVerticesTime.toFixed(2)}ms
                    <ul>
                    <li>
                    Compute Vertices Kernel: ${mc.computeVerticesKernelTime.toFixed(2)}ms
                    </li>
                    </ul>`;
        }

        projView = mat4.mul(projView, proj, camera.camera);
        {
            await uploadBuffer.mapAsync(GPUMapMode.WRITE);
            let map = uploadBuffer.getMappedRange();
            new Float32Array(map).set(projView);
            new Uint32Array(map, 16 * 4, 4).set(volume.dims);
            new Float32Array(map, 20 * 4, 1).set([spacing]);
            // Camera position (world space) = translation of the inverse view matrix.
            const invView = mat4.invert(mat4.create(), camera.camera);
            new Float32Array(map, 24 * 4, 3).set([invView[12], invView[13], invView[14]]);
            uploadBuffer.unmap();
        }
        {
            await sphereUploadBuffer.mapAsync(GPUMapMode.WRITE);
            let map = sphereUploadBuffer.getMappedRange();
            new Float32Array(map, 0, 16).set(camera.camera);
            new Float32Array(map, 64, 16).set(proj);
            sphereUploadBuffer.unmap();
        }

        renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

        let commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(
            uploadBuffer, 0, viewParamsBuffer, 0, viewParamsBuffer.size);
        commandEncoder.copyBufferToBuffer(
            sphereUploadBuffer, 0, sphereParamsBuffer, 0, sphereParamsBuffer.size);

        let renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        if (showSurfaceCheckbox.checked && isosurface && isosurface.count > 0) {
            renderPass.setBindGroup(0, bindGroup);
            renderPass.setPipeline(renderPipeline);
            renderPass.setVertexBuffer(0, isosurface.buffer);
            renderPass.draw(isosurface.count, 1, 0, 0);
        }

        if (showAtomsCheckbox.checked && atomCount > 0 && sphereBindGroup) {
            renderPass.setPipeline(spherePipeline);
            renderPass.setBindGroup(0, sphereBindGroup);
            renderPass.draw(6, atomCount, 0, 0);
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
})();

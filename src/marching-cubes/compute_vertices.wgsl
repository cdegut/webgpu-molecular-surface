// include of compute_voxel_values.wgsl is inserted here

@id(0) override WORKGROUP_SIZE: u32 = 32;

@group(1) @binding(0)
var<storage> case_table: array<i32>;

@group(1) @binding(1)
var<storage, read_write> active_voxel_ids: array<u32>;

@group(1) @binding(2)
var<storage, read_write> voxel_vertex_offsets: array<u32>;

@group(1) @binding(3)
var<storage, read_write> vertices: array<float4>;

struct PushConstants {
    group_id_offset: u32,
    total_workgoups: u32,
    total_elements: u32
};

@group(2) @binding(0)
var<uniform> push_constants: PushConstants;

const EDGE_VERTICES: array<uint2, 12> = array<uint2, 12>(
    uint2(0, 1),
    uint2(1, 2),
    uint2(2, 3),
    uint2(3, 0),
    uint2(4, 5),
    uint2(6, 5),
    uint2(6, 7),
    uint2(7, 4),
    uint2(0, 4),
    uint2(1, 5),
    uint2(2, 6),
    uint2(3, 7)
);

fn lerp_verts(va: int3, vb: int3, fa: f32, fb: f32) -> float3
{
    var t: f32 = 0.0;
    if (abs(fa - fb) >= 0.001) {
        t = (volume_info.isovalue - fa) / (fb - fa);
    }
    return mix(float3(va), float3(vb), t);
}

// Analytic gradient of the trilinearly-interpolated scalar field, evaluated at
// the cell-local position p (each component in [0, 1]). Using the gradient of
// the interpolation (rather than per-face screen derivatives) gives a normal
// that is continuous across cell faces, i.e. smooth shading.
fn field_value(p: int3) -> f32
{
    return textureLoad(volume, p, 0).x;
}

fn node_gradient(p: int3) -> float3
{
    let dx = field_value(p + int3(1, 0, 0)) - field_value(p - int3(1, 0, 0));
    let dy = field_value(p + int3(0, 1, 0)) - field_value(p - int3(0, 1, 0));
    let dz = field_value(p + int3(0, 0, 1)) - field_value(p - int3(0, 0, 1));
    return float3(dx, dy, dz);
}

// Gradient of the scalar field at cell-local position p, computed by
// central-differencing the field at each of the 8 surrounding grid nodes and
// trilinearly interpolating those node gradients. Node gradients are shared
// between adjacent cells, so this field is continuous across cell faces —
// unlike the per-cell analytic gradient, which is not. Duplicated vertices on
// shared edges therefore get identical normals and the surface shades smoothly.
fn trilinear_gradient(voxel: uint3, p: float3) -> float3
{
    var g: array<float3, 8>;
    for (var i = 0; i < 8; i++) {
        g[i] = node_gradient(int3(voxel) + INDEX_TO_VERTEX[i]);
    }

    let x = p.x;
    let y = p.y;
    let z = p.z;

    let g00 = g[0] * (1.0 - x) + g[1] * x;
    let g10 = g[3] * (1.0 - x) + g[2] * x;
    let g01 = g[4] * (1.0 - x) + g[5] * x;
    let g11 = g[7] * (1.0 - x) + g[6] * x;

    let g0 = g00 * (1.0 - y) + g10 * y;
    let g1 = g01 * (1.0 - y) + g11 * y;

    return g0 * (1.0 - z) + g1 * z;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: uint3)
{
    // Skip out of bounds threads
    let work_item = global_id.x + push_constants.group_id_offset * WORKGROUP_SIZE;
    if (work_item >= push_constants.total_elements) {
        return;
    }

    let voxel_id = active_voxel_ids[work_item];
    var values: array<f32, 8>;
    compute_voxel_values(voxel_id_to_pos(voxel_id), &values);

    var case_index = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (values[i] <= volume_info.isovalue) {
            case_index |= 1u << i;
        }
    }

    let voxel_pos = voxel_id_to_pos(voxel_id);
    let vertex_offset = voxel_vertex_offsets[work_item];
    // Now we can finally compute and output the vertices
    for (var i = 0u; i < MC_CASE_ELEMENTS && case_table[case_index * MC_CASE_ELEMENTS + i] != -1; i++)
    {
        let edge = case_table[case_index * MC_CASE_ELEMENTS + i];
        let v0 = EDGE_VERTICES[edge].x;
        let v1 = EDGE_VERTICES[edge].y;

        // Compute the interpolated vertex for this edge within the unit cell.
        // `local_pos` is the cell-local position in [0, 1]^3 before offsetting.
        let local_pos = lerp_verts(INDEX_TO_VERTEX[v0], INDEX_TO_VERTEX[v1], values[v0], values[v1]);

        // Smooth surface normal from the field gradient at the vertex.
        var n = normalize(trilinear_gradient(voxel_pos, local_pos));
        if (length(n) < 0.0001) {
            n = float3(0.0, 0.0, 1.0);
        }

        // Offset the vertex into the global volume grid
        let v = local_pos + float3(voxel_pos) + 0.5;

        // Interleaved output: [position, normal] per vertex (two float4s each).
        let base = vertex_offset * 2 + i * 2;
        vertices[base] = float4(v, 1.0);
        vertices[base + 1] = float4(n, 0.0);
    }
}



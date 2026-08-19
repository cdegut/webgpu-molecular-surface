// This type definition is just to make typing a bit easier
alias float3 = vec3<f32>;
alias float4 = vec4<f32>;

alias uint4 = vec4<u32>;

struct VertexInput {
    @location(0) position: float4,
    @location(1) normal: float4,
};

struct VertexOutput {
    // This is the equivalent of gl_Position in GLSL
    @builtin(position) position: float4,
    @location(0) world_pos: float3,
    @location(1) normal: float3,
};

struct ViewParams {
    proj_view: mat4x4<f32>,
    volume_dims: uint4,
    voxel_size: vec3<f32>,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> params: ViewParams;

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = (vert.position.xyz - float3(params.volume_dims.xyz) / 2.0) * params.voxel_size;
    out.position = params.proj_view * float4(pos, 1.0);
    out.world_pos = pos;
    out.normal = vert.normal.xyz;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) float4 {
    // Per-vertex normals from the field gradient, interpolated across the
    // triangle, give smooth shading. Normalize to undo interpolation shrinkage.
    var n = normalize(in.normal);

    // Orient the normal toward the camera so lighting is consistent regardless
    // of triangle winding.
    let view_dir = normalize(params.camera_pos - in.world_pos);
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    // Simple two-term Blinn-Phong: one key light + ambient + view specular.
    let light_dir = normalize(vec3<f32>(-0.45, 0.85, 0.65));
    let half_vec = normalize(light_dir + view_dir);

    let base = vec3<f32>(0.22, 0.56, 0.78);
    let ambient = 0.30;
    let diffuse = 0.70 * max(dot(n, light_dir), 0.0);
    let specular = 0.50 * pow(max(dot(n, half_vec), 0.0), 48.0);

    let col = base * (ambient + diffuse) + vec3<f32>(1.0, 1.0, 1.0) * specular;
    return float4(col, 1.0);
}

// Atom spheres rendered as instanced billboard quads, with the sphere resolved
// analytically (ray-sphere intersection) in the fragment shader. This gives
// perfectly smooth spheres and correct depth with only 6 vertices per atom.

struct SphereParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

struct Atom {
    center: vec4<f32>,  // xyz = world position, w = radius (Angstroms)
    color: vec4<f32>,   // rgb
};

@group(0) @binding(0)
var<uniform> params: SphereParams;

@group(0) @binding(1)
var<storage, read> atoms: array<Atom>;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) pos_view: vec3<f32>,     // quad corner in view space
    @location(1) center_view: vec3<f32>,  // sphere centre in view space
    @location(2) radius: f32,
    @location(3) color: vec3<f32>,
};

struct FSOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

const CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut
{
    var out: VSOut;
    let atom = atoms[ii];
    let c_view = (params.view * vec4<f32>(atom.center.xyz, 1.0)).xyz;
    let r = atom.center.w;
    let d = -c_view.z;  // view-space depth of the sphere centre

    // Kill the instance if the centre is behind the camera (zero-area triangle).
    if (c_view.z >= 0.0) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.pos_view = vec3<f32>(0.0, 0.0, 1.0);
        out.center_view = c_view;
        out.radius = r;
        out.color = atom.color.rgb;
        return out;
    }

    // Half-size of the quad in view space that exactly bounds the sphere's
    // silhouette under perspective.
    var h = r;
    if (d > r) {
        h = r * d / sqrt(d * d - r * r);
    } else {
        // Camera inside / very near the sphere: use a generous fallback.
        h = r * 4.0;
    }

    let corner = CORNERS[vi % 6u];
    let pos_view = vec3<f32>(c_view.xy + corner * h, c_view.z);

    out.position = params.proj * vec4<f32>(pos_view, 1.0);
    out.pos_view = pos_view;
    out.center_view = c_view;
    out.radius = r;
    out.color = atom.color.rgb;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> FSOut
{
    var out: FSOut;

    // Ray-sphere intersection in view space (camera at origin).
    let ray = normalize(in.pos_view);
    let oc = -in.center_view;
    let b = dot(ray, oc);
    let c = dot(oc, oc) - in.radius * in.radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        discard;
    }
    // Near intersection: with oc = -center the roots are -b ± sqrt(disc),
    // and the smaller (nearer) root is -b - sqrt(disc).
    let t = -b - sqrt(disc);
    if (t <= 0.0) {
        discard;
    }

    let p = ray * t;
    let clip = params.proj * vec4<f32>(p, 1.0);
    out.depth = (clip.z / clip.w) * 0.5 + 0.5;

    var n = normalize(p - in.center_view);
    let view_dir = -normalize(p);
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    // Same key light as the surface shader, transformed into view space.
    let light_view = normalize((params.view * vec4<f32>(-0.45, 0.85, 0.65, 0.0)).xyz);
    let half_vec = normalize(light_view + view_dir);

    let base = in.color;
    let ambient = 0.30;
    let diffuse = 0.70 * max(dot(n, light_view), 0.0);
    let specular = 0.50 * pow(max(dot(n, half_vec), 0.0), 48.0);

    out.color = vec4<f32>(base * (ambient + diffuse) + vec3<f32>(1.0) * specular, 1.0);
    return out;
}

// CPK colouring for atoms. Returns [r, g, b] in [0, 1].
const CPK: Record<string, [number, number, number]> = {
    H: [1.00, 1.00, 1.00],
    C: [0.35, 0.35, 0.35],
    N: [0.12, 0.31, 0.98],
    O: [1.00, 0.10, 0.10],
    F: [0.55, 0.90, 0.35],
    CL: [0.15, 0.80, 0.20],
    BR: [0.65, 0.16, 0.16],
    I: [0.58, 0.00, 0.58],
    P: [1.00, 0.55, 0.00],
    S: [1.00, 0.85, 0.15],
    FE: [0.88, 0.40, 0.20],
    NA: [0.67, 0.36, 0.95],
    K: [0.56, 0.25, 0.83],
    MG: [0.14, 1.00, 0.16],
    CA: [0.24, 1.00, 0.00],
    ZN: [0.49, 0.50, 0.69],
    CU: [0.78, 0.50, 0.20],
    MN: [0.61, 0.48, 0.78],
    CO: [0.94, 0.56, 0.63],
    NI: [0.31, 0.82, 0.31],
    SE: [1.00, 0.63, 0.00],
};

export function cpkColor(element: string): [number, number, number]
{
    const sym = (element || "").replace(/[^a-zA-Z]/g, "").toUpperCase();
    return CPK[sym] ?? [1.00, 0.60, 0.70];
}

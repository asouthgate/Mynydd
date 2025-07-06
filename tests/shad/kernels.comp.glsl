// Constants
const float PI = 3.14159265359;

// Normalization factor
float cubic_spline_2d_fac(float h) {
    return 10.0 / (7.0 * PI * (h * h));
}

// Cubic spline kernel
float cubic_spline_2d_kernel(float r, float h) {
    float norm = cubic_spline_2d_fac(h);
    float q = r / h;

    float fq;
    if (q > 2.0) {
        fq = 0.0;
    } else if (q > 1.0) {
        fq = 0.25 * pow(2.0 - q, 3.0);
    } else {
        fq = 1.0 - 1.5 * q * q * (1.0 - 0.5 * q);
    }

    return norm * fq;
}
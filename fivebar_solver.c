#include <math.h>
#include <stddef.h>

#define STEP_WIDTH 16

static void mat_identity(double m[9]) {
    for (int i = 0; i < 9; ++i) {
        m[i] = 0.0;
    }
    m[0] = 1.0;
    m[4] = 1.0;
    m[8] = 1.0;
}

static void mat_mul(const double a[9], const double b[9], double out[9]) {
    double r[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            r[row * 3 + col] =
                a[row * 3 + 0] * b[0 * 3 + col] +
                a[row * 3 + 1] * b[1 * 3 + col] +
                a[row * 3 + 2] * b[2 * 3 + col];
        }
    }
    for (int i = 0; i < 9; ++i) {
        out[i] = r[i];
    }
}

static void mat_vec(const double m[9], const double v[3], double out[3]) {
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

static void axis_angle(const double axis_in[3], double angle, double out[9]) {
    double x = axis_in[0];
    double y = axis_in[1];
    double z = axis_in[2];
    double n = sqrt(x * x + y * y + z * z);
    if (n < 1e-12) {
        x = 0.0;
        y = 0.0;
        z = 1.0;
    } else {
        x /= n;
        y /= n;
        z /= n;
    }
    double c = cos(angle);
    double s = sin(angle);
    double C = 1.0 - c;
    out[0] = c + x * x * C;
    out[1] = x * y * C - z * s;
    out[2] = x * z * C + y * s;
    out[3] = y * x * C + z * s;
    out[4] = c + y * y * C;
    out[5] = y * z * C - x * s;
    out[6] = z * x * C - y * s;
    out[7] = z * y * C + x * s;
    out[8] = c + z * z * C;
}

static int angle_for_index(
    int index,
    const double *angles,
    int angle_count,
    int passive_a,
    int passive_b,
    double candidate_a,
    double candidate_b,
    double *out
) {
    if (index == passive_a) {
        *out = candidate_a;
        return 1;
    }
    if (index == passive_b) {
        *out = candidate_b;
        return 1;
    }
    if (index >= 0 && index < angle_count) {
        *out = angles[index];
        return 1;
    }
    return 0;
}

static void fk(
    const double *chain,
    int count,
    const double *angles,
    int angle_count,
    int passive_a,
    int passive_b,
    double candidate_a,
    double candidate_b,
    double origin[3],
    double rotation[9]
) {
    origin[0] = 0.0;
    origin[1] = 0.0;
    origin[2] = 0.0;
    mat_identity(rotation);

    for (int i = 0; i < count; ++i) {
        const double *step = chain + (size_t)i * STEP_WIDTH;
        double offset[3] = {step[0], step[1], step[2]};
        double world_offset[3];
        mat_vec(rotation, offset, world_offset);
        origin[0] += world_offset[0];
        origin[1] += world_offset[1];
        origin[2] += world_offset[2];

        double origin_rot[9];
        for (int j = 0; j < 9; ++j) {
            origin_rot[j] = step[3 + j];
        }
        mat_mul(rotation, origin_rot, rotation);

        int angle_index = (int)step[15];
        if (angle_index >= 0) {
            double angle = 0.0;
            if (angle_for_index(angle_index, angles, angle_count, passive_a, passive_b, candidate_a, candidate_b, &angle)) {
                double joint_rot[9];
                axis_angle(step + 12, angle, joint_rot);
                mat_mul(rotation, joint_rot, rotation);
            }
        }
    }
}

static double closure_error(
    const double *wheel_chain,
    int wheel_count,
    const double *branch_chain,
    int branch_count,
    const double *angles,
    int angle_count,
    int passive_a,
    int passive_b,
    const double loop_origin[3],
    double candidate_a,
    double candidate_b
) {
    double wheel_origin[3], branch_origin[3], wheel_rot[9], branch_rot[9];
    fk(wheel_chain, wheel_count, angles, angle_count, passive_a, passive_b, candidate_a, candidate_b, wheel_origin, wheel_rot);
    fk(branch_chain, branch_count, angles, angle_count, passive_a, passive_b, candidate_a, candidate_b, branch_origin, branch_rot);

    double branch_tip_offset[3];
    mat_vec(branch_rot, loop_origin, branch_tip_offset);
    double dx = wheel_origin[0] - (branch_origin[0] + branch_tip_offset[0]);
    double dy = wheel_origin[1] - (branch_origin[1] + branch_tip_offset[1]);
    double dz = wheel_origin[2] - (branch_origin[2] + branch_tip_offset[2]);
    return sqrt(dx * dx + dy * dy + dz * dz);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int solve_passive_pair_c(
    const double *wheel_chain,
    int wheel_count,
    const double *branch_chain,
    int branch_count,
    const double *angles,
    int angle_count,
    int passive_a,
    int passive_b,
    const double *loop_origin,
    double initial_a,
    double initial_b,
    double lower,
    double upper,
    double *out
) {
    if (!wheel_chain || !branch_chain || !angles || !loop_origin || !out || wheel_count <= 0 || branch_count <= 0) {
        return 0;
    }

    int sample_counts[4] = {25, 21, 17, 13};
    double best_a = initial_a;
    double best_b = initial_b;
    double best_error = HUGE_VAL;
    double center_a = initial_a;
    double center_b = initial_b;
    double radius = (upper - lower) / 2.0;

    for (int level = 0; level < 4; ++level) {
        int samples = sample_counts[level];
        double a0 = fmax(lower, center_a - radius);
        double a1 = fmin(upper, center_a + radius);
        double b0 = fmax(lower, center_b - radius);
        double b1 = fmin(upper, center_b + radius);

        for (int ia = 0; ia < samples; ++ia) {
            double a = samples == 1 ? a0 : a0 + (a1 - a0) * (double)ia / (double)(samples - 1);
            for (int ib = 0; ib < samples; ++ib) {
                double b = samples == 1 ? b0 : b0 + (b1 - b0) * (double)ib / (double)(samples - 1);
                double error = closure_error(
                    wheel_chain,
                    wheel_count,
                    branch_chain,
                    branch_count,
                    angles,
                    angle_count,
                    passive_a,
                    passive_b,
                    loop_origin,
                    a,
                    b
                );
                if (error < best_error) {
                    best_error = error;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        center_a = best_a;
        center_b = best_b;
        radius *= 0.28;
    }

    out[0] = best_a;
    out[1] = best_b;
    out[2] = best_error;
    return isfinite(best_error) ? 1 : 0;
}

#include <math.h>
#include <stddef.h>

#define STEP_WIDTH 16
#define TREE_STEP_WIDTH 13
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void axis_angle(const double axis_in[3], double angle, double out[9]);

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

static double vec_norm(const double v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void rotation_between_vectors(const double from_in[3], const double to_in[3], double out[9]) {
    double from[3] = {from_in[0], from_in[1], from_in[2]};
    double to[3] = {to_in[0], to_in[1], to_in[2]};
    double from_n = vec_norm(from);
    double to_n = vec_norm(to);
    if (from_n < 1e-12 || to_n < 1e-12) {
        mat_identity(out);
        return;
    }
    for (int i = 0; i < 3; ++i) {
        from[i] /= from_n;
        to[i] /= to_n;
    }

    double dot = from[0] * to[0] + from[1] * to[1] + from[2] * to[2];
    if (dot > 0.999999) {
        mat_identity(out);
        return;
    }
    if (dot < -0.999999) {
        double axis[3] = {1.0, 0.0, 0.0};
        if (fabs(from[0]) > 0.9) {
            axis[0] = 0.0;
            axis[1] = 1.0;
        }
        double cross[3] = {
            from[1] * axis[2] - from[2] * axis[1],
            from[2] * axis[0] - from[0] * axis[2],
            from[0] * axis[1] - from[1] * axis[0],
        };
        axis_angle(cross, M_PI, out);
        return;
    }

    double axis[3] = {
        from[1] * to[2] - from[2] * to[1],
        from[2] * to[0] - from[0] * to[2],
        from[0] * to[1] - from[1] * to[0],
    };
    axis_angle(axis, acos(dot), out);
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

static void rot_x(double a, double out[9]) {
    double c = cos(a);
    double s = sin(a);
    out[0] = 1.0; out[1] = 0.0; out[2] = 0.0;
    out[3] = 0.0; out[4] = c;   out[5] = -s;
    out[6] = 0.0; out[7] = s;   out[8] = c;
}

static void rot_y(double a, double out[9]) {
    double c = cos(a);
    double s = sin(a);
    out[0] = c;   out[1] = 0.0; out[2] = s;
    out[3] = 0.0; out[4] = 1.0; out[5] = 0.0;
    out[6] = -s;  out[7] = 0.0; out[8] = c;
}

static void rot_z(double a, double out[9]) {
    double c = cos(a);
    double s = sin(a);
    out[0] = c;   out[1] = -s;  out[2] = 0.0;
    out[3] = s;   out[4] = c;   out[5] = 0.0;
    out[6] = 0.0; out[7] = 0.0; out[8] = 1.0;
}

static void rpy_matrix(const double rpy[3], double out[9]) {
    double rx[9], ry[9], rz[9], zy[9];
    rot_x(rpy[0], rx);
    rot_y(rpy[1], ry);
    rot_z(rpy[2], rz);
    mat_mul(rz, ry, zy);
    mat_mul(zy, rx, out);
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

    int sample_counts[4] = {17, 13, 11, 9};
    double best_a = initial_a;
    double best_b = initial_b;
    double best_error = HUGE_VAL;
    double best_score = HUGE_VAL;
    double center_a = initial_a;
    double center_b = initial_b;
    const double branch_window = 0.55;
    double search_lower_a = fmax(lower, initial_a - branch_window);
    double search_upper_a = fmin(upper, initial_a + branch_window);
    double search_lower_b = fmax(lower, initial_b - branch_window);
    double search_upper_b = fmin(upper, initial_b + branch_window);
    double radius = fmin((upper - lower) / 2.0, branch_window);
    const double continuity_weight = 0.015;

    for (int level = 0; level < 4; ++level) {
        int samples = sample_counts[level];
        double a0 = fmax(search_lower_a, center_a - radius);
        double a1 = fmin(search_upper_a, center_a + radius);
        double b0 = fmax(search_lower_b, center_b - radius);
        double b1 = fmin(search_upper_b, center_b + radius);

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
                double da = a - initial_a;
                double db = b - initial_b;
                double score = error + continuity_weight * (da * da + db * db);
                if (score < best_score) {
                    best_score = score;
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

static int compute_tree_transforms(
    const double *tree_steps,
    int step_count,
    const double *angles,
    int angle_count,
    const double *root_origin,
    const double *root_rotation,
    double scale,
    int link_count,
    double *out_origins,
    double *out_rotations
) {
    if (!tree_steps || !angles || !root_origin || !root_rotation || !out_origins || !out_rotations || link_count <= 0) {
        return 0;
    }

    for (int i = 0; i < link_count; ++i) {
        out_origins[i * 3 + 0] = 0.0;
        out_origins[i * 3 + 1] = 0.0;
        out_origins[i * 3 + 2] = 0.0;
        mat_identity(out_rotations + (size_t)i * 9);
    }

    out_origins[0] = root_origin[0];
    out_origins[1] = root_origin[1];
    out_origins[2] = root_origin[2];
    for (int j = 0; j < 9; ++j) {
        out_rotations[j] = root_rotation[j];
    }

    for (int i = 0; i < step_count; ++i) {
        const double *step = tree_steps + (size_t)i * TREE_STEP_WIDTH;
        int parent = (int)step[0];
        int child = (int)step[1];
        if (parent < 0 || parent >= link_count || child < 0 || child >= link_count) {
            continue;
        }

        const double *parent_origin = out_origins + (size_t)parent * 3;
        const double *parent_rot = out_rotations + (size_t)parent * 9;
        double local_origin[3] = {step[2] * scale, step[3] * scale, step[4] * scale};
        double world_offset[3];
        mat_vec(parent_rot, local_origin, world_offset);

        double *child_origin = out_origins + (size_t)child * 3;
        child_origin[0] = parent_origin[0] + world_offset[0];
        child_origin[1] = parent_origin[1] + world_offset[1];
        child_origin[2] = parent_origin[2] + world_offset[2];

        double origin_rpy[3] = {step[5], step[6], step[7]};
        double joint_rot[9];
        rpy_matrix(origin_rpy, joint_rot);

        double *child_rot = out_rotations + (size_t)child * 9;
        mat_mul(parent_rot, joint_rot, child_rot);

        int angle_index = (int)step[11];
        int movable = (int)step[12];
        if (movable && angle_index >= 0 && angle_index < angle_count) {
            double angle_rot[9];
            axis_angle(step + 8, angles[angle_index], angle_rot);
            mat_mul(child_rot, angle_rot, child_rot);
        }
    }

    return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int compute_link_transforms_c(
    const double *tree_steps,
    int step_count,
    const double *angles,
    int angle_count,
    const double *root_origin,
    const double *root_rotation,
    double scale,
    int link_count,
    double *out_origins,
    double *out_rotations
) {
    return compute_tree_transforms(
        tree_steps,
        step_count,
        angles,
        angle_count,
        root_origin,
        root_rotation,
        scale,
        link_count,
        out_origins,
        out_rotations
    );
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int compute_supported_link_transforms_c(
    const double *tree_steps,
    int step_count,
    const double *angles,
    int angle_count,
    const double *support_origin,
    const double *requested_root_rotation,
    double scale,
    int link_count,
    int wheel_left_index,
    int wheel_right_index,
    double wheel_radius,
    double *out_origins,
    double *out_rotations
) {
    if (
        !tree_steps || !angles || !support_origin || !requested_root_rotation || !out_origins || !out_rotations ||
        link_count <= 0 || wheel_left_index < 0 || wheel_left_index >= link_count ||
        wheel_right_index < 0 || wheel_right_index >= link_count
    ) {
        return 0;
    }

    double zero_origin[3] = {0.0, 0.0, 0.0};
    if (!compute_tree_transforms(
        tree_steps,
        step_count,
        angles,
        angle_count,
        zero_origin,
        requested_root_rotation,
        scale,
        link_count,
        out_origins,
        out_rotations
    )) {
        return 0;
    }

    double *left = out_origins + (size_t)wheel_left_index * 3;
    double *right = out_origins + (size_t)wheel_right_index * 3;
    double wheel_line[3] = {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
    double horizontal_line[3] = {wheel_line[0], wheel_line[1], 0.0};
    double aligned_root_rotation[9];
    if (vec_norm(horizontal_line) > 1e-12) {
        double correction[9];
        rotation_between_vectors(wheel_line, horizontal_line, correction);
        mat_mul(correction, requested_root_rotation, aligned_root_rotation);
    } else {
        for (int i = 0; i < 9; ++i) {
            aligned_root_rotation[i] = requested_root_rotation[i];
        }
    }

    if (!compute_tree_transforms(
        tree_steps,
        step_count,
        angles,
        angle_count,
        zero_origin,
        aligned_root_rotation,
        scale,
        link_count,
        out_origins,
        out_rotations
    )) {
        return 0;
    }

    left = out_origins + (size_t)wheel_left_index * 3;
    right = out_origins + (size_t)wheel_right_index * 3;
    double support_center[3] = {
        (left[0] + right[0]) * 0.5,
        (left[1] + right[1]) * 0.5,
        (left[2] + right[2]) * 0.5,
    };
    double anchored_root_origin[3] = {
        support_origin[0] - support_center[0],
        support_origin[1] - support_center[1],
        support_origin[2] + wheel_radius - support_center[2],
    };

    return compute_tree_transforms(
        tree_steps,
        step_count,
        angles,
        angle_count,
        anchored_root_origin,
        aligned_root_rotation,
        scale,
        link_count,
        out_origins,
        out_rotations
    );
}

import math

import halide as hl

from utils import DENOISE_PASSES, TONE_MAP_PASSES, SHARPEN_STRENGTH


def black_white_level(input, black_point, white_point):
    output = hl.Func("black_white_level_output")

    x, y = hl.Var("x"), hl.Var("y")

    white_factor = 65535 / (white_point - black_point)

    output[x, y] = hl.u16_sat((hl.i32(input[x, y]) - black_point) * white_factor)

    return output


def white_balance(input, width, height, white_balance_r, white_balance_g0, white_balance_g1, white_balance_b):
    output = hl.Func("white_balance_output")

    print(width, height, white_balance_r, white_balance_g0, white_balance_g1, white_balance_b)

    x, y = hl.Var("x"), hl.Var("y")

    rdom = hl.RDom([(0, width / 2), (0, height / 2)])

    output[x, y] = hl.u16(0)

    output[rdom.x * 2, rdom.y * 2] = hl.u16_sat(white_balance_r * hl.f32(input[rdom.x * 2, rdom.y * 2]))
    output[rdom.x * 2 + 1, rdom.y * 2] = hl.u16_sat(white_balance_g0 * hl.f32(input[rdom.x * 2 + 1, rdom.y * 2]))
    output[rdom.x * 2, rdom.y * 2 + 1] = hl.u16_sat(white_balance_g1 * hl.f32(input[rdom.x * 2, rdom.y * 2 + 1]))
    output[rdom.x * 2 + 1, rdom.y * 2 + 1] = hl.u16_sat(white_balance_b * hl.f32(input[rdom.x * 2 + 1, rdom.y * 2 + 1]))

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(rdom.y)
    output.update(1).parallel(rdom.y)
    output.update(2).parallel(rdom.y)
    output.update(3).parallel(rdom.y)

    return output


def demosaic(input, width, height):
    print(f'width: {width}, height: {height}')

    f0 = hl.Buffer(hl.Int(32), [5, 5], "demosaic_f0")
    f1 = hl.Buffer(hl.Int(32), [5, 5], "demosaic_f1")
    f2 = hl.Buffer(hl.Int(32), [5, 5], "demosaic_f2")
    f3 = hl.Buffer(hl.Int(32), [5, 5], "demosaic_f3")

    f0.translate([-2, -2])
    f1.translate([-2, -2])
    f2.translate([-2, -2])
    f3.translate([-2, -2])

    d0 = hl.Func("demosaic_0")
    d1 = hl.Func("demosaic_1")
    d2 = hl.Func("demosaic_2")
    d3 = hl.Func("demosaic_3")

    output = hl.Func("demosaic_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
    rdom0 = hl.RDom([(-2, 5), (-2, 5)])
    # rdom1 = hl.RDom([(0, width / 2), (0, height / 2)])

    input_mirror = hl.BoundaryConditions.mirror_interior(input, [(0, width), (0, height)])

    f0.fill(0)
    f1.fill(0)
    f2.fill(0)
    f3.fill(0)

    f0_sum = 8
    f1_sum = 16
    f2_sum = 16
    f3_sum = 16

    f0[0, -2] = -1
    f0[0, -1] = 2
    f0[-2, 0] = -1
    f0[-1, 0] = 2
    f0[0, 0] = 4
    f0[1, 0] = 2
    f0[2, 0] = -1
    f0[0, 1] = 2
    f0[0, 2] = -1

    f1[0, -2] = 1
    f1[-1, -1] = -2
    f1[1, -1] = -2
    f1[-2, 0] = -2
    f1[-1, 0] = 8
    f1[0, 0] = 10
    f1[1, 0] = 8
    f1[2, 0] = -2
    f1[-1, 1] = -2
    f1[1, 1] = -2
    f1[0, 2] = 1

    f2[0, -2] = -2
    f2[-1, -1] = -2
    f2[0, -1] = 8
    f2[1, -1] = -2
    f2[-2, 0] = 1
    f2[0, 0] = 10
    f2[2, 0] = 1
    f2[-1, 1] = -2
    f2[0, 1] = 8
    f2[1, 1] = -2
    f2[0, 2] = -2

    f3[0, -2] = -3
    f3[-1, -1] = 4
    f3[1, -1] = 4
    f3[-2, 0] = -3
    f3[0, 0] = 12
    f3[2, 0] = -3
    f3[-1, 1] = 4
    f3[1, 1] = 4
    f3[0, 2] = -3

    d0[x, y] = hl.u16_sat(hl.sum(hl.i32(input_mirror[x + rdom0.x, y + rdom0.y]) * f0[rdom0.x, rdom0.y]) / f0_sum)
    d1[x, y] = hl.u16_sat(hl.sum(hl.i32(input_mirror[x + rdom0.x, y + rdom0.y]) * f1[rdom0.x, rdom0.y]) / f1_sum)
    d2[x, y] = hl.u16_sat(hl.sum(hl.i32(input_mirror[x + rdom0.x, y + rdom0.y]) * f2[rdom0.x, rdom0.y]) / f2_sum)
    d3[x, y] = hl.u16_sat(hl.sum(hl.i32(input_mirror[x + rdom0.x, y + rdom0.y]) * f3[rdom0.x, rdom0.y]) / f3_sum)

    R_row = y % 2 == 0
    B_row = y % 2 != 0
    R_col = x % 2 == 0
    B_col = x % 2 != 0
    at_R = c == 0
    at_G = c == 1
    at_B = c == 2

    output[x, y, c] = hl.select(at_R & R_row & B_col, d1[x, y],
                                at_R & B_row & R_col, d2[x, y],
                                at_R & B_row & B_col, d3[x, y],
                                at_G & R_row & R_col, d0[x, y],
                                at_G & B_row & B_col, d0[x, y],
                                at_B & B_row & R_col, d1[x, y],
                                at_B & R_row & B_col, d2[x, y],
                                at_B & R_row & R_col, d3[x, y],
                                input[x, y])

    d0.compute_root().parallel(y).vectorize(x, 16)
    d1.compute_root().parallel(y).vectorize(x, 16)
    d2.compute_root().parallel(y).vectorize(x, 16)
    d3.compute_root().parallel(y).vectorize(x, 16)

    output.compute_root().parallel(y).align_bounds(x, 2).unroll(x, 2).align_bounds(y, 2).unroll(y, 2).vectorize(x, 16)

    return output


def rgb_to_yuv(input):
    print('    rgb_to_yuv')

    output = hl.Func("rgb_to_yuv_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    rdom = input[x, y, 0]
    g = input[x, y, 1]
    b = input[x, y, 2]

    output[x, y, c] = hl.f32(0)

    output[x, y, 0] = 0.2989 * rdom + 0.587 * g + 0.114 * b
    output[x, y, 1] = -0.168935 * rdom - 0.331655 * g + 0.50059 * b
    output[x, y, 2] = 0.499813 * rdom - 0.418531 * g + - 0.081282 * b

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(y).vectorize(x, 16)
    output.update(1).parallel(y).vectorize(x, 16)
    output.update(2).parallel(y).vectorize(x, 16)

    return output


def bilateral_filter(input, width, height):
    print('    bilateral_filter')

    k = hl.Buffer(hl.Float(32), [7, 7], "gauss_kernel")
    k.translate([-3, -3])

    weights = hl.Func("bilateral_weights")
    total_weights = hl.Func("bilateral_total_weights")
    bilateral = hl.Func("bilateral")
    output = hl.Func("bilateral_filter_output")

    x, y, dx, dy, c = hl.Var("x"), hl.Var("y"), hl.Var("dx"), hl.Var("dy"), hl.Var("c")
    rdom = hl.RDom([(-3, 7), (-3, 7)])

    k.fill(0)
    k[-3, -3] = 0.000690
    k[-2, -3] = 0.002646
    k[-1, -3] = 0.005923
    k[0, -3] = 0.007748
    k[1, -3] = 0.005923
    k[2, -3] = 0.002646
    k[3, -3] = 0.000690
    k[-3, -2] = 0.002646
    k[-2, -2] = 0.010149
    k[-1, -2] = 0.022718
    k[0, -2] = 0.029715
    k[1, -2] = 0.022718
    k[2, -2] = 0.010149
    k[3, -2] = 0.002646
    k[-3, -1] = 0.005923
    k[-2, -1] = 0.022718
    k[-1, -1] = 0.050855
    k[0, -1] = 0.066517
    k[1, -1] = 0.050855
    k[2, -1] = 0.022718
    k[3, -1] = 0.005923
    k[-3, 0] = 0.007748
    k[-2, 0] = 0.029715
    k[-1, 0] = 0.066517
    k[0, 0] = 0.087001
    k[1, 0] = 0.066517
    k[2, 0] = 0.029715
    k[3, 0] = 0.007748
    k[-3, 1] = 0.005923
    k[-2, 1] = 0.022718
    k[-1, 1] = 0.050855
    k[0, 1] = 0.066517
    k[1, 1] = 0.050855
    k[2, 1] = 0.022718
    k[3, 1] = 0.005923
    k[-3, 2] = 0.002646
    k[-2, 2] = 0.010149
    k[-1, 2] = 0.022718
    k[0, 2] = 0.029715
    k[1, 2] = 0.022718
    k[2, 2] = 0.010149
    k[3, 2] = 0.002646
    k[-3, 3] = 0.000690
    k[-2, 3] = 0.002646
    k[-1, 3] = 0.005923
    k[0, 3] = 0.007748
    k[1, 3] = 0.005923
    k[2, 3] = 0.002646
    k[3, 3] = 0.000690

    input_mirror = hl.BoundaryConditions.mirror_interior(input, [(0, width), (0, height)])

    dist = hl.cast(hl.Float(32),
                   hl.cast(hl.Int(32), input_mirror[x, y, c]) - hl.cast(hl.Int(32), input_mirror[x + dx, y + dy, c]))

    sig2 = 100

    threshold = 25000

    score = hl.select(hl.abs(input_mirror[x + dx, y + dy, c]) > threshold, 0, hl.exp(-dist * dist / sig2))

    weights[dx, dy, x, y, c] = k[dx, dy] * score

    total_weights[x, y, c] = hl.sum(weights[rdom.x, rdom.y, x, y, c])

    bilateral[x, y, c] = hl.sum(input_mirror[x + rdom.x, y + rdom.y, c] * weights[rdom.x, rdom.y, x, y, c]) / \
                         total_weights[x, y, c]

    output[x, y, c] = hl.cast(hl.Float(32), input[x, y, c])

    output[x, y, 1] = bilateral[x, y, 1]
    output[x, y, 2] = bilateral[x, y, 2]

    weights.compute_at(output, y).vectorize(x, 16)

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(y).vectorize(x, 16)
    output.update(1).parallel(y).vectorize(x, 16)

    return output


def gauss_15x15(input, name):
    print('        gauss_15x15')

    k = hl.Buffer(hl.Float(32), [15], "gauss_15x15")
    k.translate([-7])

    rdom = hl.RDom([(-7, 15)])

    k.fill(0)
    k[-7] = 0.004961
    k[-6] = 0.012246
    k[-5] = 0.026304
    k[-4] = 0.049165
    k[-3] = 0.079968
    k[-2] = 0.113193
    k[-1] = 0.139431
    k[0] = 0.149464
    k[7] = 0.004961
    k[6] = 0.012246
    k[5] = 0.026304
    k[4] = 0.049165
    k[3] = 0.079968
    k[2] = 0.113193
    k[1] = 0.139431

    return gauss(input, k, rdom, name)


def desaturate_noise(input, width, height):
    print('    desaturate_noise')

    output = hl.Func("desaturate_noise_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    input_mirror = hl.BoundaryConditions.mirror_image(input, [(0, width), (0, height)])

    blur = gauss_15x15(gauss_15x15(input_mirror, "desaturate_noise_blur1"), "desaturate_noise_blur_2")

    factor = 1.4

    threshold = 25000

    output[x, y, c] = input[x, y, c]

    output[x, y, 1] = hl.select((hl.abs(blur[x, y, 1]) / hl.abs(input[x, y, 1]) < factor) &
                                (hl.abs(input[x, y, 1]) < threshold) & (hl.abs(blur[x, y, 1]) < threshold),
                                0.7 * blur[x, y, 1] + 0.3 * input[x, y, 1], input[x, y, 1])

    output[x, y, 2] = hl.select((hl.abs(blur[x, y, 2]) / hl.abs(input[x, y, 2]) < factor) &
                                (hl.abs(input[x, y, 2]) < threshold) & (hl.abs(blur[x, y, 2]) < threshold),
                                0.7 * blur[x, y, 2] + 0.3 * input[x, y, 2], input[x, y, 2])

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def increase_saturation(input, strength):
    print('    increase saturation')

    output = hl.Func("increase_saturation_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    output[x, y, c] = strength * input[x, y, c]
    output[x, y, 0] = input[x, y, 0]

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def yuv_to_rgb(input):
    print('    yuv_to_rgb')

    output = hl.Func("yuv_to_rgb_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    Y = input[x, y, 0]
    U = input[x, y, 1]
    V = input[x, y, 2]

    output[x, y, c] = hl.cast(hl.UInt(16), 0)

    output[x, y, 0] = hl.u16_sat(Y + 1.403 * V)
    output[x, y, 1] = hl.u16_sat(Y - 0.344 * U - 0.714 * V)
    output[x, y, 2] = hl.u16_sat(Y + 1.77 * U)

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(y).vectorize(x, 16)
    output.update(1).parallel(y).vectorize(x, 16)
    output.update(2).parallel(y).vectorize(x, 16)

    return output


def chroma_denoise(input, width, height, denoise_passes):
    print(f'width: {width}, height: {height}, passes: {denoise_passes}')

    output = rgb_to_yuv(input)

    p = 0

    if denoise_passes > 0:
        output = bilateral_filter(output, width, height)
    p += 1

    while p < denoise_passes:
        output = desaturate_noise(output, width, height)
        p += 1

    if denoise_passes > 2:
        output = increase_saturation(output, 1.1)

    return yuv_to_rgb(output)


def srgb(input, ccm):
    srgb_matrix = hl.Func("srgb_matrix")
    output = hl.Func("srgb_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    rdom = hl.RDom([(0, 3)])

    srgb_matrix[x, y] = hl.f32(0)

    srgb_matrix[0, 0] = hl.f32(ccm[0][0])
    srgb_matrix[1, 0] = hl.f32(ccm[0][1])
    srgb_matrix[2, 0] = hl.f32(ccm[0][2])
    srgb_matrix[0, 1] = hl.f32(ccm[1][0])
    srgb_matrix[1, 1] = hl.f32(ccm[1][1])
    srgb_matrix[2, 1] = hl.f32(ccm[1][2])
    srgb_matrix[0, 2] = hl.f32(ccm[2][0])
    srgb_matrix[1, 2] = hl.f32(ccm[2][1])
    srgb_matrix[2, 2] = hl.f32(ccm[2][2])

    output[x, y, c] = hl.u16_sat(hl.sum(srgb_matrix[rdom, c] * input[x, y, rdom]))

    return output


def gauss(input, k, rdom, name):
    blur_x = hl.Func(name + "_x")
    output = hl.Func(name)

    x, y, c, xi, yi = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("xi"), hl.Var("yi")

    val = hl.Expr("val")

    if input.dimensions() == 2:
        blur_x[x, y] = hl.sum(input[x + rdom, y] * k[rdom])
        val = hl.sum(blur_x[x, y + rdom] * k[rdom])
        if input.output_types()[0] == hl.UInt(16):
            val = hl.u16(val)
        output[x, y] = val
    else:
        blur_x[x, y, c] = hl.sum(input[x + rdom, y, c] * k[rdom])
        val = hl.sum(blur_x[x, y + rdom, c] * k[rdom])
        if input.output_types()[0] == hl.UInt(16):
            val = hl.u16(val)
        output[x, y, c] = val

    blur_x.compute_at(output, x).vectorize(x, 16)

    output.compute_root().tile(x, y, xi, yi, 256, 128).vectorize(xi, 16).parallel(y)

    return output


def gauss_7x7(input, name):
    k = hl.Buffer(hl.Float(32), [7], "gauss_7x7_kernel")
    k.translate([-3])

    rdom = hl.RDom([(-3, 7)])

    k.fill(0)
    k[-3] = 0.026267
    k[-2] = 0.100742
    k[-1] = 0.225511
    k[0] = 0.29496
    k[1] = 0.225511
    k[2] = 0.100742
    k[3] = 0.026267

    return gauss(input, k, rdom, name)


def diff(im1, im2, name):
    output = hl.Func(name)

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    if im1.dimensions() == 2:
        output[x, y] = hl.i32(im1[x, y]) - hl.i32(im2[x, y])
    else:
        output[x, y, c] = hl.i32(im1[x, y, c]) - hl.i32(im2[x, y, c])

    return output


def combine(im1, im2, width, height, dist):
    init_mask1 = hl.Func("mask1_layer_0")
    init_mask2 = hl.Func("mask2_layer_0")
    accumulator = hl.Func("combine_accumulator")
    output = hl.Func("combine_output")

    x, y = hl.Var("x"), hl.Var("y")

    im1_mirror = hl.BoundaryConditions.repeat_edge(im1, [(0, width), (0, height)])
    im2_mirror = hl.BoundaryConditions.repeat_edge(im2, [(0, width), (0, height)])

    unblurred1 = im1_mirror
    unblurred2 = im2_mirror

    blurred1 = gauss_7x7(im1_mirror, "img1_layer_0")
    blurred2 = gauss_7x7(im2_mirror, "img2_layer_0")

    weight1 = hl.f32(dist[im1_mirror[x, y]])
    weight2 = hl.f32(dist[im2_mirror[x, y]])

    init_mask1[x, y] = weight1 / (weight1 + weight2)
    init_mask2[x, y] = 1 - init_mask1[x, y]

    mask1 = init_mask1
    mask2 = init_mask2

    num_layers = 2

    accumulator[x, y] = hl.i32(0)

    for i in range(1, num_layers):
        print('        layer', i)

        prev_layer_str = str(i - 1)
        layer_str = str(i)

        laplace1 = diff(unblurred1, blurred1, "laplace1_layer_" + prev_layer_str)
        laplace2 = diff(unblurred2, blurred2, "laplace2_layer_" + layer_str)

        accumulator[x, y] += hl.i32(laplace1[x, y] * mask1[x, y]) + hl.i32(laplace2[x, y] * mask2[x, y])

        unblurred1 = blurred1
        unblurred2 = blurred2

        blurred1 = gauss_7x7(blurred1, "img1_layer_" + layer_str)
        blurred2 = gauss_7x7(blurred2, "img2_layer_" + layer_str)

        mask1 = gauss_7x7(mask1, "mask1_layer_" + layer_str)
        mask2 = gauss_7x7(mask2, "mask2_layer_" + layer_str)

    accumulator[x, y] += hl.i32(blurred1[x, y] * mask1[x, y]) + hl.i32(blurred2[x, y] * mask2[x, y])

    output[x, y] = hl.u16_sat(accumulator[x, y])

    init_mask1.compute_root().parallel(y).vectorize(x, 16)

    accumulator.compute_root().parallel(y).vectorize(x, 16)

    for i in range(num_layers):
        accumulator.update(i).parallel(y).vectorize(x, 16)

    return output


def combine2(im1, im2, width, height, dist):
    init_mask1 = hl.Func("mask1_layer_0")
    init_mask2 = hl.Func("mask2_layer_0")
    accumulator = hl.Func("combine_accumulator")
    output = hl.Func("combine_output")

    x, y = hl.Var("x"), hl.Var("y")

    im1_mirror = hl.BoundaryConditions.repeat_edge(im1, [(0, width), (0, height)])
    im2_mirror = hl.BoundaryConditions.repeat_edge(im2, [(0, width), (0, height)])

    weight1 = hl.f32(dist[im1_mirror[x, y]])
    weight2 = hl.f32(dist[im2_mirror[x, y]])

    init_mask1[x, y] = weight1 / (weight1 + weight2)
    init_mask2[x, y] = 1 - init_mask1[x, y]

    mask1 = init_mask1
    mask2 = init_mask2

    accumulator[x, y] = hl.i32(0)

    accumulator[x, y] += hl.i32(im1_mirror[x, y] * mask1[x, y]) + hl.i32(im2_mirror[x, y] * mask2[x, y])

    output[x, y] = hl.u16_sat(accumulator[x, y])

    init_mask1.compute_root().parallel(y).vectorize(x, 16)

    accumulator.compute_root().parallel(y).vectorize(x, 16)

    accumulator.update(0).parallel(y).vectorize(x, 16)

    return output


def brighten(input, gain):
    output = hl.Func("brighten_output")

    x, y = hl.Var("x"), hl.Var("y")

    output[x, y] = hl.u16_sat(gain * hl.u32(input[x, y]))

    return output


def gamma_correct(input):
    output = hl.Func("gamma_correct_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    cutoff = 200
    gamma_toe = 12.92
    gamma_pow = 0.416667
    gamma_fac = 680.552897
    gamma_con = -3604.425

    if input.dimensions() == 2:
        output[x, y] = hl.u16(hl.select(input[x, y] < cutoff,
                                        gamma_toe * input[x, y],
                                        gamma_fac * hl.pow(input[x, y], gamma_pow) + gamma_con))
    else:
        output[x, y, c] = hl.u16(hl.select(input[x, y, c] < cutoff,
                                           gamma_toe * input[x, y, c],
                                           gamma_fac * hl.pow(input[x, y, c], gamma_pow) + gamma_con))

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def gamma_inverse(input):
    output = hl.Func("gamma_inverse_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    cutoff = 2575
    gamma_toe = 0.0774
    gamma_pow = 2.4
    gamma_fac = 57632.49226
    gamma_con = 0.055

    if input.dimensions() == 2:
        output[x, y] = hl.u16(hl.select(input[x, y] < cutoff,
                                        gamma_toe * input[x, y],
                                        hl.pow(hl.f32(input[x, y]) / 65535 + gamma_con, gamma_pow) * gamma_fac))
    else:
        output[x, y, c] = hl.u16(hl.select(input[x, y, c] < cutoff,
                                           gamma_toe * input[x, y, c],
                                           hl.pow(hl.f32(input[x, y, c]) / 65535 + gamma_con, gamma_pow) * gamma_fac))

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def tone_map(input, width, height, compression, gain):
    print(f'Compression: {compression}, gain: {gain}')

    normal_dist = hl.Func("luma_weight_distribution")
    grayscale = hl.Func("grayscale")
    output = hl.Func("tone_map_output")

    x, y, c, v = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("v")

    rdom = hl.RDom([(0, 3)])

    normal_dist[v] = hl.f32(hl.exp(-12.5 * hl.pow(hl.f32(v) / 65535 - 0.5, 2)))

    grayscale[x, y] = hl.u16(hl.sum(hl.u32(input[x, y, rdom])) / 3)

    dark = grayscale

    comp_const = 1
    gain_const = 1

    comp_slope = (compression - comp_const) / (TONE_MAP_PASSES)
    gain_slope = (gain - gain_const) / (TONE_MAP_PASSES)

    for i in range(TONE_MAP_PASSES):
        print('    pass', i)

        norm_comp = i * comp_slope + comp_const
        norm_gain = i * gain_slope + gain_const

        bright = brighten(dark, norm_comp)

        dark_gamma = gamma_correct(dark)
        bright_gamma = gamma_correct(bright)

        dark_gamma = combine2(dark_gamma, bright_gamma, width, height, normal_dist)

        dark = brighten(gamma_inverse(dark_gamma), norm_gain)

    output[x, y, c] = hl.u16_sat(hl.u32(input[x, y, c]) * hl.u32(dark[x, y]) / hl.u32(hl.max(1, grayscale[x, y])))

    grayscale.compute_root().parallel(y).vectorize(x, 16)

    normal_dist.compute_root().vectorize(v, 16)

    return output


def shift_bayer_to_rggb(input, cfa_pattern):
    print(f'cfa_pattern: {cfa_pattern}')
    output = hl.Func("rggb_input")
    x, y = hl.Var("x"), hl.Var("y")

    cfa = hl.u16(cfa_pattern)

    output[x, y] = hl.select(cfa == hl.u16(1), input[x, y],
                             cfa == hl.u16(2), input[x + 1, y],
                             cfa == hl.u16(4), input[x, y + 1],
                             cfa == hl.u16(3), input[x + 1, y + 1],
                             0)
    return output


def contrast(input, strength, black_point):
    output = hl.Func("contrast_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    scale = strength

    inner_constant = math.pi / (2 * scale)
    sin_constant = hl.sin(inner_constant)
    slope = 65535 / (2 * sin_constant)
    constant = slope * sin_constant
    factor = math.pi / (scale * 65535)

    val = factor * hl.cast(hl.Float(32), input[x, y, c])

    output[x, y, c] = hl.u16_sat(slope * hl.sin(val - inner_constant) + constant)

    white_scale = 65535 / (65535 - black_point)

    output[x, y, c] = hl.u16_sat((hl.cast(hl.Int(32), output[x, y, c]) - black_point) * white_scale)

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def sharpen(input, strength):
    output_yuv = hl.Func("sharpen_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    yuv_input = rgb_to_yuv(input)

    small_blurred = gauss_7x7(yuv_input, "unsharp_small_blur")
    large_blurred = gauss_7x7(small_blurred, "unsharp_large_blur")

    difference_of_gauss = diff(small_blurred, large_blurred, "unsharp_DoG")

    output_yuv[x, y, c] = yuv_input[x, y, c]
    output_yuv[x, y, 0] = yuv_input[x, y, 0] + strength * difference_of_gauss[x, y, 0]

    output = yuv_to_rgb(output_yuv)

    output_yuv.compute_root().parallel(y).vectorize(x, 16)

    return output


def u8bit_interleave(input):
    output = hl.Func("8bit_interleaved_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    output[x, y, c] = hl.u8_sat(input[x, y, c] / 256)

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


'''
Step 3 of HDR+ pipeline: finish

image : numpy ndarray
    The merged image to be finished

Returns: numpy ndarray (finished image)
'''


def finish_image(imgs, width, height, black_point, white_point, white_balance_r, white_balance_g0, white_balance_g1,
                 white_balance_b, compression, gain, contrast_strength, cfa_pattern, ccm):
    print(black_point, white_point, white_balance_r, white_balance_g0, white_balance_g1,
          white_balance_b, compression, gain)

    print("bayer_to_rggb")
    bayer_shifted = shift_bayer_to_rggb(imgs, cfa_pattern)

    print("black_white_level")
    black_white_level_output = black_white_level(bayer_shifted, black_point, white_point)

    print("white_balance")
    white_balance_output = white_balance(black_white_level_output, width, height, white_balance_r, white_balance_g0,
                                         white_balance_g1, white_balance_b)

    print("demosaic")
    demosaic_output = demosaic(white_balance_output, width, height)

    print('chroma_denoise')
    chroma_denoised_output = chroma_denoise(demosaic_output, width, height, DENOISE_PASSES)

    print("srgb")
    srgb_output = srgb(chroma_denoised_output, ccm)

    print("tone_map")
    tone_map_output = tone_map(srgb_output, width, height, compression, gain)

    print("gamma_correct")
    gamma_correct_output = gamma_correct(tone_map_output)

    print('contrast')
    contrast_output = contrast(gamma_correct_output, contrast_strength, black_point)

    print('sharpen')
    sharpen_output = sharpen(contrast_output, SHARPEN_STRENGTH)

    print('u8bit_interleave')
    u8bit_interleave_output = u8bit_interleave(sharpen_output)

    return u8bit_interleave_output

import cv2 as cv
import numpy as np
import halide as hl

from datetime import datetime

from utils import time_diff

DENOISE_PASSES = 1
CONTRAST_STRENGTH = 5
BLACK_LEVEL = 2000
SHARPEN_STRENGTH = 2


def black_white_level(input, black_point, white_point):
    output = hl.Func("black_white_level_output")

    x, y = hl.Var("x"), hl.Var("y")

    white_factor = 65535 / (white_point - black_point)

    output[x, y] = hl.u16_sat((hl.cast(hl.Int(32), input[x, y]) - black_point) * white_factor)

    return output


def white_balance(input, width, height, white_balance_r, white_balance_g0, white_balance_g1, white_balance_b):
    output = hl.Func("white_balance_output")

    print('here')
    print(width, height, white_balance_r, white_balance_g0, white_balance_g1, white_balance_b)

    x, y = hl.Var("x"), hl.Var("y")

    r = hl.RDom([(0, width / 2), (0, height / 2)])

    output[x, y] = hl.u16(0)

    output[r.x * 2    , r.y * 2    ] = hl.u16_sat(white_balance_r  * hl.cast(hl.Float(32), input[r.x * 2    , r.y * 2    ]))
    output[r.x * 2 + 1, r.y * 2    ] = hl.u16_sat(white_balance_g0 * hl.cast(hl.Float(32), input[r.x * 2 + 1, r.y * 2    ]))
    output[r.x * 2    , r.y * 2 + 1] = hl.u16_sat(white_balance_g1 * hl.cast(hl.Float(32), input[r.x * 2    , r.y * 2 + 1]))
    output[r.x * 2 + 1, r.y * 2 + 1] = hl.u16_sat(white_balance_b  * hl.cast(hl.Float(32), input[r.x * 2 + 1, r.y * 2 + 1]))

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(r.y)
    output.update(1).parallel(r.y)
    output.update(2).parallel(r.y)
    output.update(3).parallel(r.y)

    return output


def demosaic(input, width, height):
    f0 = hl.Func("demosaic_f0")
    f1 = hl.Func("demosaic_f1")
    f2 = hl.Func("demosaic_f2")
    f3 = hl.Func("demosaic_f3")
    d0 = hl.Func("demosaic_0")
    d1 = hl.Func("demosaic_1")
    d2 = hl.Func("demosaic_2")
    d3 = hl.Func("demosaic_3")

    output = hl.Func("demosaic_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
    r0 = hl.RDom([(-2, 5), (-2, 5)])
    r1 = hl.RDom([(0, width / 2), (0, height / 2)])

    input_mirror = hl.BoundaryConditions.mirror_interior(input, [(0, width), (0, height)])

    f0[x, y] = 0
    f1[x, y] = 0
    f2[x, y] = 0
    f3[x, y] = 0

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

    d0[x, y] = hl.u16_sat(hl.sum(hl.cast(hl.Int(32), (input_mirror[x + r0.x, y + r0.y])) * f0[r0.x, r0.y]) / f0_sum)
    d1[x, y] = hl.u16_sat(hl.sum(hl.cast(hl.Int(32), (input_mirror[x + r0.x, y + r0.y])) * f1[r0.x, r0.y]) / f1_sum)
    d2[x, y] = hl.u16_sat(hl.sum(hl.cast(hl.Int(32), (input_mirror[x + r0.x, y + r0.y])) * f2[r0.x, r0.y]) / f2_sum)
    d3[x, y] = hl.u16_sat(hl.sum(hl.cast(hl.Int(32), (input_mirror[x + r0.x, y + r0.y])) * f3[r0.x, r0.y]) / f3_sum)

    output[x, y, c] = input[x, y]

    output[r1.x * 2 + 1, r1.y * 2, 0] = d1[r1.x * 2 + 1, r1.y * 2]
    output[r1.x * 2, r1.y * 2 + 1, 0] = d2[r1.x * 2, r1.y * 2 + 1]
    output[r1.x * 2 + 1, r1.y * 2 + 1, 0] = d3[r1.x * 2 + 1, r1.y * 2 + 1]

    output[r1.x * 2, r1.y * 2, 1] = d0[r1.x * 2, r1.y * 2]
    output[r1.x * 2 + 1, r1.y * 2 + 1, 1] = d0[r1.x * 2 + 1, r1.y * 2 + 1]

    output[r1.x * 2, r1.y * 2 + 1, 2] = d1[r1.x * 2, r1.y * 2 + 1]
    output[r1.x * 2 + 1, r1.y * 2, 2] = d2[r1.x * 2 + 1, r1.y * 2]
    output[r1.x * 2, r1.y * 2, 2] = d3[r1.x * 2, r1.y * 2]

    f0.compute_root().parallel(y).parallel(x)
    f1.compute_root().parallel(y).parallel(x)
    f2.compute_root().parallel(y).parallel(x)
    f3.compute_root().parallel(y).parallel(x)

    d0.compute_root().parallel(y).vectorize(x, 16)
    d1.compute_root().parallel(y).vectorize(x, 16)
    d2.compute_root().parallel(y).vectorize(x, 16)
    d3.compute_root().parallel(y).vectorize(x, 16)

    output.compute_root().parallel(y).vectorize(x, 16)

    output.update(0).parallel(r1.y)
    output.update(1).parallel(r1.y)
    output.update(2).parallel(r1.y)
    output.update(3).parallel(r1.y)
    output.update(4).parallel(r1.y)
    output.update(5).parallel(r1.y)
    output.update(6).parallel(r1.y)
    output.update(7).parallel(r1.y)

    return output


def srgb(input):
    srgb_matrix = hl.Func("srgb_matrix")
    output = hl.Func("srgb_output")

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    r = hl.RDom([(0, 3)])

    srgb_matrix[x, y] = hl.f32(0)

    srgb_matrix[0, 0] = hl.f32(0.964399)
    srgb_matrix[1, 0] = hl.f32(-1.119710)
    srgb_matrix[2, 0] = hl.f32(0.155311)
    srgb_matrix[0, 1] = hl.f32(-0.241156)
    srgb_matrix[1, 1] = hl.f32(1.673722)
    srgb_matrix[2, 1] = hl.f32(-0.432566)
    srgb_matrix[0, 2] = hl.f32(0.013887)
    srgb_matrix[1, 2] = hl.f32(-0.549820)
    srgb_matrix[2, 2] = hl.f32(1.535933)

    output[x, y, c] = hl.u16_sat(hl.sum(srgb_matrix[r, c] * input[x, y, r]))

    srgb_matrix.compute_root().parallel(y).parallel(x)

    return output

def gauss(input, k, r, name):
    blur_x = hl.Func(name + "_x")
    output = hl.Func(name)
    x, y, c, xi, yi = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("xi"), hl.Var("yi")

    if input.dimensions() == 2:
        blur_x[x, y] = hl.sum(input[x + r, y] * k[r])
        val = hl.sum(blur_x[x, y + r] * k[r])
        if input.output_types()[0] == hl.UInt(16):
            val = hl.cast(hl.UInt(16), val)
        output[x, y] = val
    else:
        blur_x[x, y, c] = hl.sum(input[x + r, y, c] * k[r])
        val = hl.sum(blur_x[x, y + r, c] * k[r])
        if input.output_types()[0] == hl.UInt(16):
            val = hl.cast(hl.UInt(16), val)
        output[x, y] = val

    blur_x.compute_at(output, x).vectorize(x, 16)

    output.compute_root().tile(x, y, xi, yi, 256, 128).vectorize(xi, 16).parallel(y)

    return output

def gauss_7x7(input, name):
    k = hl.Func("gauss_7x7_kernel")

    x = hl.Var("x")

    r = hl.RDom([(-3, 7)])

    k[x] = hl.cast(hl.Float(32), 0)

    k[-3] = 0.026267
    k[-2] = 0.100742
    k[-1] = 0.225511
    k[0] = 0.29496
    k[3] = 0.026267
    k[2] = 0.100742
    k[1] = 0.225511

    k.compute_root().parallel(x)

    return gauss(input, k, r, name)

def diff(im1, im2, name):
    output = hl.Func(name)

    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    if im1.dimensions() == 2:
        output[x, y] = hl.cast(hl.Int(32), im1[x, y]) - hl.cast(hl.Int(32), im2[x, y])
    else:
        output[x, y, c] = hl.cast(hl.Int(32), im1[x, y, c]) - hl.cast(hl.Int(32), im2[x, y, c])

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

    weight1 = hl.cast(hl.Float(32), dist[im1_mirror[x, y]])
    weight2 = hl.cast(hl.Float(32), dist[im2_mirror[x, y]])

    init_mask1[x, y] = weight1 / (weight1 + weight2)
    init_mask2[x, y] = 1 - init_mask1[x, y]

    mask1 = init_mask1
    mask2 = init_mask2

    num_layers = 2

    accumulator[x, y] = hl.i32(0)

    for i in range(num_layers):
        print('num_layers', i)

        prev_layer_str = str(i)
        layer_str = str(i + 1)

        laplace1 = diff(unblurred1, blurred1, "laplace1_layer_" + prev_layer_str)
        laplace2 = diff(unblurred2, blurred2, "laplace2_layer_" + layer_str)

        accumulator[x, y] += hl.cast(hl.Int(32), laplace1[x,y] * mask1[x,y]) + hl.cast(hl.Int(32), laplace2[x,y] * mask2[x,y])

        unblurred1 = blurred1
        unblurred2 = blurred2

        blurred1 = gauss_7x7(blurred1, "img1_layer_" + layer_str)
        blurred2 = gauss_7x7(blurred2, "img1_layer_" + layer_str)

        mask1 = gauss_7x7(mask1, "mask1_layer_" + layer_str)
        mask2 = gauss_7x7(mask2, "mask2_layer_" + layer_str)

    accumulator[x, y] += hl.cast(hl.Int(32), blurred1[x, y] * mask1[x, y]) + hl.cast(hl.Int(32), blurred2[x, y] * mask2[x, y])

    output[x, y] = hl.u16_sat(accumulator[x, y])

    init_mask1.compute_root().parallel(y).vectorize(x, 16)

    accumulator.compute_root().parallel(y).vectorize(x, 16)

    for i in range(num_layers):
        accumulator.update(i).parallel(y).vectorize(x, 16)

    return output


def brighten(input, gain):
    output = hl.Func("brighten_output")

    x, y = hl.Var("x"), hl.Var("y")

    output[x, y] = hl.u16_sat(gain * hl.cast(hl.UInt(32), input[x, y]))

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
        output[x, y] = hl.cast(hl.UInt(16), hl.select(input[x, y] < cutoff, gamma_toe * input[x, y],
                                                      gamma_fac * hl.pow(input[x, y], gamma_pow) + gamma_con))
    else:
        output[x, y, c] = hl.cast(hl.UInt(16), hl.select(input[x, y, c] < cutoff, gamma_toe * input[x, y, c],
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
        output[x, y] = hl.cast(hl.UInt(16), hl.select(input[x, y] < cutoff, gamma_toe * input[x, y],
                                                      hl.pow(hl.cast(hl.Float(32), input[x, y]) / 65535 + gamma_con,
                                                             gamma_pow) * gamma_fac))
    else:
        output[x, y, c] = hl.cast(hl.UInt(16), hl.select(input[x, y, c] < cutoff, gamma_toe * input[x, y, c],
                                                         hl.pow(
                                                             hl.cast(hl.Float(32), input[x, y, c]) / 65535 + gamma_con,
                                                             gamma_pow) * gamma_fac))

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def tone_map(input, width, height, compression, gain):
    normal_dist = hl.Func("luma_weight_distribution")
    grayscale = hl.Func("grayscale")
    output = hl.Func("tone_map_output")

    x, y, c, v = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("v")

    r = hl.RDom([(0, 3)])

    normal_dist[v] = hl.cast(hl.Float(32), hl.exp(-12.5 * hl.pow(hl.cast(hl.Float(32), v) / 65535 - 0.5, 2)))

    grayscale[x, y] = hl.cast(hl.UInt(16), hl.sum(hl.cast(hl.UInt(32), input[x, y, r])) / 3)

    dark = grayscale

    num_passes = 3

    comp_const = 1 + compression / num_passes
    gain_const = 1 + gain / num_passes

    comp_slope = (compression - comp_const) / (num_passes - 1)
    gain_slope = (gain - gain_const) / (num_passes - 1)

    for i in range(num_passes):
        print('num_pass', i)
        norm_comp = i * comp_slope + comp_const

        norm_gain = i * gain_slope + gain_const

        bright = brighten(dark, norm_comp)

        dark_gamma = gamma_correct(dark)
        bright_gamma = gamma_correct(bright)

        dark_gamma = combine(dark_gamma, bright_gamma, width, height, normal_dist)

        dark = brighten(gamma_inverse(dark_gamma), norm_gain)

    output[x, y, c] = hl.u16_sat(hl.cast(hl.UInt(32), input[x, y, c]) * hl.cast(hl.UInt(32), dark[x, y]) / hl.max(1, grayscale[x, y]))

    grayscale.compute_root().parallel(y).vectorize(x, 16)

    normal_dist.compute_root().vectorize(v, 16)

    return output

def u8bit_interleaved(input):
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
                 white_balance_b, compression, gain):

    print(f'\n{"=" * 30}\nFinishing image...\n{"=" * 30}')
    start = datetime.utcnow()

    print(black_point, white_point, white_balance_r, white_balance_g0, white_balance_g1,
                 white_balance_b, compression, gain)
    
    print("black_white_level")
    black_white_level_output = black_white_level(imgs, black_point, white_point)
    
    print("white_balance")
    white_balance_output = white_balance(black_white_level_output, width, height, white_balance_r, white_balance_g0,
                                         white_balance_g1, white_balance_b)
    
    print("demosaic")
    demosaic_output = demosaic(white_balance_output, width, height) # TODO

    # output = hl.Func("asf")
    # x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
    # output[x, y, c] = demosaic_output[x, y, c]
    

    # TODO
    # print('chroma_denoise')
    # chroma_denoised_output = chroma_denoise(demosaic_output, width, height, denoise_passes)

    # print("srgb")
    # srgb_output = srgb(demosaic_output)
    #
    # print("tone_map")
    # tone_map_output = tone_map(srgb_output, width, height, compression, gain)
    #
    # print("gamma_correct")
    # gamma_correct_output = gamma_correct(tone_map_output)

    # TODO
    # print('contrast')
    # contrast_output = contrast(gamma_correct_output, contrast_strength, black_level)
    #
    # print('sharpen')
    # sharpen_output = sharpen(contrast_output, sharpen_strength)

    u8bit_interleaved_output = u8bit_interleaved(demosaic_output)

    print(f'Finishing finished in {time_diff(start)} ms.\n')
    return u8bit_interleaved_output

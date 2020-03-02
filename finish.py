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

    x, y = hl.Var("x"), hl.Var("y")

    r = hl.RDom([(0, width / 2), (0, height / 2)])

    output[x, y] = hl.u16(0)

    output[r.x * 2, r.y * 2] = hl.u16_sat(white_balance_r * hl.cast(hl.Float(32), input[r.x * 2, r.y * 2]))
    output[r.x * 2 + 1, r.y * 2] = hl.u16_sat(white_balance_g0 * hl.cast(hl.Float(32), input[r.x * 2 + 1, r.y * 2]))
    output[r.x * 2, r.y * 2 + 1] = hl.u16_sat(white_balance_g1 * hl.cast(hl.Float(32), input[r.x * 2, r.y * 2 + 1]))
    output[r.x * 2 + 1, r.y * 2 + 1] = hl.u16_sat(
        white_balance_b * hl.cast(hl.Float(32), input[r.x * 2 + 1, r.y * 2 + 1]))

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



    # black_white_level_output = black_white_level(imgs, black_point, white_point)
    #
    # white_balance_output = white_balance(imgs, width, height, white_balance_r, white_balance_g0,
    #                                      white_balance_g1, white_balance_b)

    demosaic_output = demosaic(imgs, width, height)

    # TODO
    # chroma_denoised_output = chroma_denoise(demosaic_output, width, height, denoise_passes)
    #
    # srgb_output = srgb(demosaic_output)
    #
    # tone_map_output = tone_map(srgb_output, width, height, compression, gain)
    #
    # gamma_correct_output = gamma_correct(tone_map_output)
    #
    # contrast_output = contrast(gamma_correct_output, contrast_strength, black_level)
    #
    # sharpen_output = sharpen(contrast_output, sharpen_strength)

    u8bit_interleaved_output = u8bit_interleaved(demosaic_output)

    print(f'Finishing finished in {time_diff(start)} ms.\n')
    return u8bit_interleaved_output

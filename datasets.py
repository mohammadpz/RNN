# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 19:08:19 2015

@author: Mohammad Pezeshki
"""

import numpy
from images2gif import writeGif


def gaussian(grid, pos, s=0.75):
    return numpy.exp(-((grid[0] - pos[0]) ** 2 +
                       (grid[1] - pos[1]) ** 2) /
                     2 * s ** 2)


def save_as_gif(frames, path='results/last.gif'):
    writeGif(path, frames, duration=10.0 / frames.shape[0], dither=0)


def single_bouncing_ball(num_batches, batch_size,
                         num_frames, patch_size, ball_size):
    tempresolution = 50.
    num_frames_total = num_batches * batch_size * num_frames
    print 'Making data...'
    imgrid = numpy.array(numpy.meshgrid(numpy.arange(0, patch_size), numpy.arange(0, patch_size)))
    pos_x = numpy.tile(numpy.concatenate((
        numpy.linspace(ball_size - 1, patch_size - ball_size,
                       tempresolution - 1, endpoint=False),
        numpy.linspace(patch_size - ball_size, ball_size - 1,
                       tempresolution - 1, endpoint=False)),
        0), numpy.ceil(0.5 * num_frames_total / max((tempresolution - 1), 1)))

    pos_y = patch_size - numpy.abs(numpy.sin(numpy.linspace(
        0, 2 * numpy.pi * (num_frames_total / tempresolution),
        num_frames_total))) * (patch_size - 2 * ball_size)

    minshape = min(pos_x.shape[0], pos_y.shape[0])
    pos_x = pos_x[:minshape]
    pos_y = pos_y[:minshape]

    pos = numpy.concatenate((pos_x[:, None], pos_y[:, None]), 1)

    movie = numpy.zeros((num_frames_total, patch_size, patch_size),
                        dtype="float32")
    for i in range(num_frames_total):
        movie[i, :, :] = gaussian(imgrid, pos[i])

    movies = movie.reshape(-1, num_frames, patch_size ** 2)
    movies = movies[numpy.random.permutation(movies.shape[0])]
    numcases = movies.shape[0]

    train_features_numpy = movies.reshape(numcases, -1)
    train_features_numpy -= train_features_numpy.mean(1)[:, None]
    train_features_numpy /= train_features_numpy.std(1)[:, None]
    train_features_numpy -= train_features_numpy.min(1)[:, None]
    train_features_numpy /= train_features_numpy.max(1)[:, None]
    train_features_numpy = train_features_numpy.reshape(
        num_batches, batch_size, num_frames, patch_size ** 2)
    train_features_numpy = numpy.swapaxes(train_features_numpy, 1, 2)
    print "Data created:"
    print "(S, T, B, F) : " + str(train_features_numpy.shape)
    print
    return train_features_numpy


def random_noise(num_batches, batch_size, num_frames):
    # num_frames_total = num_batches * batch_size * num_frames

    random_walk = numpy.zeros((num_batches * batch_size, num_frames),
                              dtype=numpy.float32)

    for i in range(num_batches * batch_size):
        for j in range(num_frames):
            random_walk[i, j] = numpy.random.rand()

    random_walk[1:] = random_walk[numpy.random.permutation(random_walk.shape[0] - 1) + 1]

    random_walk = random_walk.reshape((num_batches, batch_size, num_frames, 1))

    random_walk = numpy.swapaxes(random_walk, 1, 2)

    return random_walk


def sine_wave(num_batches, batch_size, num_frames):
    num_frames_total = num_batches * batch_size * num_frames
    sine_wave = numpy.zeros((num_frames_total), dtype=numpy.float32)
    for i in range(0, num_frames_total):
        sine_wave[i] = 0.5 * numpy.sin(2 * numpy.pi * i / 30.0) + 0.5

    sine_wave = sine_wave.reshape((num_batches * batch_size, num_frames))
    sine_wave[1:] = sine_wave[numpy.random.permutation(sine_wave.shape[0] - 1) + 1]

    sine_wave = sine_wave.reshape((num_batches, batch_size, num_frames, 1))

    sine_wave = numpy.swapaxes(sine_wave, 1, 2)

    return sine_wave

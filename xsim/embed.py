#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import argparse
import logging
import os
import re
import sys
import time
from collections import namedtuple
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("embed")


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


class HuggingFaceEncoder:
    def __init__(self, encoder_name: str, verbose=False, **kwargs):
        encoder = f"sentence-transformers/{encoder_name}"
        if verbose:
            logger.info(f"loading HuggingFace encoder: {encoder}")
        self.encoder = SentenceTransformer(encoder, **kwargs)

    def encode_sentences(self, sentences):
        return self.encoder.encode(sentences)


def load_model(
        encoder: str,
        verbose=False,
        **encoder_kwargs,
) -> HuggingFaceEncoder:
    return HuggingFaceEncoder(encoder, verbose=verbose, **encoder_kwargs)


def encode_time(t):
    t = int(time.time() - t)
    if t < 1000:
        return "{:d}s".format(t)
    else:
        return "{:d}m{:d}s".format(t // 60, t % 60)


# Encode sentences (existing file pointers)
def encode_filep(
        encoder, inp_file, out_file, buffer_size=10000, fp16=False, verbose=False
):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        encoded = encoder.encode_sentences(sentences)
        if fp16:
            encoded = encoded.astype(np.float16)
        encoded.tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            logger.info("encoded {:d} sentences".format(n))
    if verbose:
        logger.info(f"encoded {n} sentences in {encode_time(t)}")


# Encode sentences (file names)
def encode_file(
        encoder,
        inp_fname,
        out_fname,
        buffer_size=10000,
        fp16=False,
        verbose=False,
        over_write=False,
        inp_encoding="utf-8",
):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            logger.info(
                "encoding {} to {}".format(
                    inp_fname if len(inp_fname) > 0 else "stdin",
                    out_fname,
                )
            )
        fin = (
            open(inp_fname, "r", encoding=inp_encoding, errors="surrogateescape")
            if len(inp_fname) > 0
            else sys.stdin
        )
        fout = open(out_fname, mode="wb")
        encode_filep(
            encoder, fin, fout, buffer_size=buffer_size, fp16=fp16, verbose=verbose
        )
        fin.close()
        fout.close()
    elif not over_write and verbose:
        logger.info("encoder: {} exists already".format(os.path.basename(out_fname)))


# Load existing embeddings
def EmbedLoad(fname, dim=1024, verbose=False, fp16=False):
    x = np.fromfile(fname, dtype=(np.float16 if fp16 else np.float32), count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(" - Embeddings: {:s}, {:d}x{:d}".format(fname, x.shape[0], dim))
    return x


# Get memory mapped embeddings
def EmbedMmap(fname, dim=1024, dtype=np.float32, verbose=False):
    nbex = int(os.path.getsize(fname) / dim / np.dtype(dtype).itemsize)
    E = np.memmap(fname, mode="r", dtype=dtype, shape=(nbex, dim))
    if verbose:
        print(" - embeddings on disk: {:s} {:d} x {:d}".format(fname, nbex, dim))
    return E


def embed_sentences(
        ifname: str,
        output: str,
        encoder: HuggingFaceEncoder = None,
        verbose: bool = False,
        buffer_size: int = 10000,
        max_sentences: Optional[int] = None,
        fp16: bool = False,
):
    assert encoder, "Provide initialised encoder"
    buffer_size = max(buffer_size, 1)
    assert (
            not max_sentences or max_sentences <= buffer_size
    ), "--max-sentences/--batch-size cannot be larger than --buffer-size"

    if not ifname:
        ifname = ""  # default to stdin

    encode_file(
        encoder,
        ifname,
        output,
        verbose=verbose,
        over_write=False,
        buffer_size=buffer_size,
        fp16=fp16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed sentences")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input text file",
    )
    parser.add_argument("--encoder", type=str, required=True, help="encoder to be used")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    parser.add_argument(
        "-o", "--output", required=True, help="Output sentence embeddings"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Buffer size (sentences)"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to process in a batch",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store embedding matrices in fp16 instead of fp32",
    )

    args = parser.parse_args()
    embed_sentences(
        ifname=args.input,
        encoder_path=args.encoder,
        verbose=args.verbose,
        output=args.output,
        buffer_size=args.buffer_size,
        max_sentences=args.max_sentences,
        fp16=args.fp16,
    )

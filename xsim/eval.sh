#!/bin/bash
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
#-------------------------------------------------------
#
# This bash script installs the flores200 dataset,
# performs xsim (multilingual similarity) evaluation with ratio margin

if [ -z ${SRCDIR} ] ; then
  echo "Please set the environment variable 'SRCDIR'"
  exit
fi

ddir="${SRCDIR}/data"
cd $ddir  # move to data directory

if [ ! -d $ddir/flores200 ] ; then
    echo " - Downloading flores200..."
    wget --trust-server-names -q https://tinyurl.com/flores200dataset
    tar -xf flores200_dataset.tar.gz
    /bin/mv flores200_dataset flores200
    /bin/rm flores200_dataset.tar.gz
else
    echo " - flores200 already downloaded"
fi

corpus_part="devtest"
corpus="flores200"

# note: example evaluation script expects format: basedir/corpus/corpus_part/lang.corpus_part

echo " - calculating xsim"
python3 $SRCDIR/xsim/eval.py                \
    --src-encoder meta-llama/Llama-3.2-3B    \
    --base-dir $ddir                         \
    --corpus $corpus                         \
    --corpus-part $corpus_part               \
    --margin ratio                           \
    --src-langs afr_Latn,fin_Latn,fra_Latn,hin_Deva,tha_Thai,eng_Latn      \
    --nway --verbose

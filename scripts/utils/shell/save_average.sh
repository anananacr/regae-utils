#!/bin/sh
label=20220722_Au_free
date=20220722

mkdir /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/safe/average/$label
mkdir /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/safe/merged/$label

cp -r /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/processed/average/$date/${label}* /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/safe/average/$label/
cp -r /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/processed/merged/$date/${label}* /asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/safe/merged/$label/

#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit

model_1_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2"
model_2_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2"

for model_1 in $model_1_list; do
    for model_2 in $model_2_list; do
        python affine_transformation.py --model "${model_1}_None_0_${model_2}_None_0" --setting A_forward
        python affine_transformation.py --model "${model_1}_None_0_${model_2}_None_0" --setting B_forward
    done
done

#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --output=transformer_small.out
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
###########################


mkdir -p /scatch/aishikc/checkpoints/multilingual_transformer_small
fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
  --max-epoch 100 \
  --ddp-backend=no_c10d \
  --task multilingual_translation --lang-pairs de-en,fr-en --encoder-langtok src\
  --arch synsem_transformer_iwslt_de_en_small \
  --share-encoders --share-decoders --share-decoder-embeddings --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
  --dropout 0.1 --weight-decay 0.0001 \
  --save-dir /scratch/aishikc/checkpoints/multilingual_transformer_small \
  --max-tokens 1000 --update-freq 8 --fp16 --batch-size 64


# python save_embeddings.py data-bin/iwslt17.de_fr.en.bpe16k/ --path /scratch/aishikc/checkpoints/multilingual_transformer_small/checkpoint_best.pt --task multilingual_translation --lang-pairs de-en,fr-en --encoder-langtok src --decoder-langtok --gen-subset valid

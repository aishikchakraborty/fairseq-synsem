mkdir -p checkpoints/multilingual_transformer
# fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
#   --max-epoch 50 \
#   --ddp-backend=no_c10d \
#   --task multilingual_translation --lang-pairs de-en,fr-en --encoder-langtok tgt\
#   --arch multilingual_transformer_iwslt_de_en \
#   --share-encoders --share-decoders --share-decoder-embeddings --share-decoder-input-output-embed \
#   --optimizer adam --adam-betas '(0.9, 0.98)' \
#   --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
#   --warmup-updates 4000 --warmup-init-lr '1e-07' \
#   --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
#   --dropout 0.3 --weight-decay 0.0001 \
#   --save-dir checkpoints/multilingual_transformer \
#   --max-tokens 1000 --update-freq 8 --fp16 --max-sentences 8 --batch-size 8
fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
  --max-epoch 50 \
  --ddp-backend=no_c10d \
  --task multilingual_translation --lang-pairs de-en,fr-en --encoder-langtok src --decoder-langtok\
  --arch synsem_transformer_iwslt_de_en_small \
  --share-encoders --share-decoders --share-decoder-embeddings --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
  --dropout 0.3 --weight-decay 0.0001 \
  --save-dir checkpoints/multilingual_transformer \
  --max-tokens 4000 --update-freq 8 --fp16 --max-sentences 8 --batch-size 16

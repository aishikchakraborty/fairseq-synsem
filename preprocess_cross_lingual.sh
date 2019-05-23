TEXT=synsem/iwslt17.de_fr.en.bpe16k

fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train.de-en --validpref $TEXT/valid.de-en \
    --joined-dictionary \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10 --fp16

fairseq-preprocess \
    --source-lang fr --target-lang en \
    --trainpref $TEXT/train.fr-en --validpref $TEXT/valid.fr-en \
    --joined-dictionary --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10 --fp16

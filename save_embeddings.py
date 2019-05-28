from fairseq import checkpoint_utils, data, options, tasks
import torch

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='multilingual_translation')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
for valid_sub_split in args.gen_subset.split(','):
    task.load_dataset(valid_sub_split, combine=True, epoch=0)
# Load model
print('| loading model from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
model = models[0]

M = model.models['de-en'].encoder.M
N = model.models['de-en'].encoder.N

no_langs = N.size(0)

lang2idx = task.lang2idx
idx2lang = {}
for keys in lang2idx.keys():
    idx2lang[lang2idx[keys]] = keys

lang2idx2idx = model.models['de-en'].encoder.lang2idx2idx
idx2idx2lang = {i.item():idx for idx, i in enumerate(lang2idx2idx)}
# import pdb; pdb.set_trace()

embed_matrix = model.models['de-en'].encoder.embed_tokens.weight
print(embed_matrix.size())
sem_matrix = torch.mm(embed_matrix, M)
torch.save(sem_matrix, 'embeddings/sem_matrix.pb')

for i in range(no_langs):
    N_t = N[i, :, :].squeeze(0)
    syn_matrix = torch.mm(embed_matrix, N_t)
    langid = idx2idx2lang[i]
    lang = idx2lang[langid]
    torch.save(sem_matrix, 'embeddings/' + lang +  '_syn_matrix.pb')

torch.save(task.dicts['en'].symbols, 'embeddings/vocab.pb')

# while True:
#     sentence = input('\nInput: ')
#
#     # Tokenize into characters
#     chars = ' '.join(list(sentence.strip()))
#     tokens = task.source_dictionary.encode_line(
#         chars, add_if_not_exist=False,
#     )
#
#     # Build mini-batch to feed to the model
#     batch = data.language_pair_dataset.collate(
#         samples=[{'id': -1, 'source': tokens}],  # bsz = 1
#         pad_idx=task.source_dictionary.pad(),
#         eos_idx=task.source_dictionary.eos(),
#         left_pad_source=False,
#         input_feeding=False,
#     )
#
#     # Feed batch to the model and get predictions
#     preds = model(**batch['net_input'])
#
#     # Print top 3 predictions and their log-probabilities
#     top_scores, top_labels = preds[0].topk(k=3)
#     for score, label_idx in zip(top_scores, top_labels):
#         label_name = task.target_dictionary.string([label_idx])
#         print('({:.2f})\t{}'.format(score, label_name))

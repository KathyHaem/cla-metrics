import numpy as np
import torch
from scipy.stats import pearsonr


def anc_score(a: np.ndarray, b: np.ndarray):
    """ My best understanding of Del & Fishel (AACL 2022), based on their code and paper.
    I will say some of it is confusing as hell because of mismatches between said code and paper.

    What exactly are a and b?
    src[mean_l], tgt[mean_l]  -> those are data frames. "l" is the layer number
    src is English, tgt some limited set of target langs
    At that point the datasets are already being loaded from disk

    From the preprocessing script---this is happening for each language + each layer. 'out_dict' is per-language.
         sent_reps_curr_layer_mean = masked_mean(
            enc_batch.hidden_states[layer_num], tok_batch.attention_mask.unsqueeze(2).bool(), 1
        )
        out_dict[f'mean_{layer_num}'] = sent_reps_curr_layer_mean.detach().cpu().numpy()
    (This should be saving one mean vector per input sequence.)

    Then they also have versions where they save the CLS token as the representation, or the first token.
    Neither of those seem to be actually used in the end.

    Their pandas-based data processing might be useful but...let's see.
    """

    # from their repo here:
    # https://github.com/TartuNLP/xsim/blob/c8a25a5096b9a183ddc6cfb0f36a118d6d7962f0/examples/util.py#L291
    # a, b = torch.Tensor(a), torch.Tensor(b)
    # a, b = a.cuda(), b.cuda()
    # center
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    # they have cosine in their repo but the paper says it should be pearson correlation
    # score = F.cosine_similarity(a, b, dim=0)
    score = pearsonr(a, b, axis=1).statistic
    # ugh it's confusing with the dims...but I want a result for each example
    # score = score.cpu().numpy()
    return np.absolute(score).mean().item()

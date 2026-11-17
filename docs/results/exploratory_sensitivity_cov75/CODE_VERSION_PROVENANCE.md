# Code-version provenance for these results

These results were produced by `src/music_recession/emotions.py` at

    sha256 614f862be7c480bc40864f79c4d2af63eef5b2270f17fac222bee1c743f777d1

The file committed in the repository is

    sha256 e5600454033eb04373c73a2238de1625d12bc80f0e0b8a9912e0ccad16a32aeb

The only difference between the two is a guard added to the tensor-batching
line so that non-tensor test doubles pass through unchanged:

    -  batched = {k: v.unsqueeze(0) if v.dim() == 1 else v ...}
    +  batched = {k: v.unsqueeze(0) if hasattr(v, 'dim') and v.dim() == 1 else v ...}

For real inference the operand is always a `torch.Tensor`, for which
`hasattr(v, 'dim')` is always True, so the two expressions are identical.

## Empirical verification, not assumption

Because the emotion cache key embeds the implementation hash, a partial re-run
scored part of the corpus under the committed version before being stopped.
Comparing the two sets of cached scores for songs scored under BOTH versions:

    songs scored under both versions : 7,490 of 16,290 (46.0%)
    bit-identical score dictionaries : 7,490 of 7,490 (100.00%)
    maximum absolute difference      : 0.000e+00
    token_count mismatches           : 0
    chunk_count mismatches           : 0
    model config identical           : yes
    software versions identical      : yes

Reproduce with `.agents/step9_verify_equivalence.py`.

## What remains unverified

The remaining 8,800 songs (54.0%) were NOT re-scored under the committed
version, so equivalence is demonstrated on 46.0% of the corpus rather than all
of it. No numerical difference was found anywhere it was checked, and the code
difference cannot affect a tensor operand, but a full byte-for-byte
reproduction of the monthly features and model metrics under the committed
hash has not been performed.

To obtain one, re-run the commands in `docs/exploratory_course_policy.md`
against a fresh cache directory; expect roughly 50 minutes of CPU inference.

from transformers import PreTrainedTokenizerBase


_PATCHED = False


def install_explicit_truncation_guard() -> None:
    """
    Match Transformers' implicit fallback for `max_length` by making it explicit.

    This prevents the noisy warning emitted when downstream code passes
    `max_length=...` without also setting `truncation=True`.
    """

    global _PATCHED
    if _PATCHED:
        return

    original = PreTrainedTokenizerBase._get_padding_truncation_strategies

    def _wrapped(self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs):
        if max_length is not None and padding is False and truncation is None:
            truncation = True
            verbose = False
        return original(
            self,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

    PreTrainedTokenizerBase._get_padding_truncation_strategies = _wrapped
    _PATCHED = True

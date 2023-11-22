import abc
import typing

import vod_configs
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Sectionizer(abc.ABC):
    """A function that sectionizes a chunk of text."""

    @abc.abstractmethod
    def __call__(
        self,
        text: str,
        prefix: str = "",
        add_prefix: bool = True,
    ) -> typing.Iterable[str]:
        """Sectionize a chunk of text."""
        ...


class SentenceSectionizer(Sectionizer):
    """A sectionizer that splits text into sentences."""

    def __init__(self, delimiter: str = ". ") -> None:
        self.delimiter = delimiter

    def __call__(
        self,
        text: str,
        prefix: str = "",
        add_prefix: bool = True,
    ) -> typing.Iterable[str]:
        """Sectionize a chunk of text."""
        for sentence in text.split(self.delimiter):
            if add_prefix:
                yield prefix + sentence
            else:
                yield sentence


class FixedLengthSectionizer(Sectionizer):
    """A tokenizer that splits text into fixed-length chunks.

    NOTE: `ellipsis_start` is added to the start of each chunk except the first.
    NOTE: `ellipsis_end` is added to the end of each chunk except the last.
    NOTE: `prefix` is added accounted in the token count for each chunk such as to ensure a fixed `max_length`.
    NOTE: `add_prefix` determines whether the `prefix` is added to each yielded chunk.
    """

    def __init__(  # noqa: PLR0913
        self,
        max_length: int,
        stride: int,
        tokenizer_name_or_path: str,
        ellipsis_start: str = "... ",
        ellipsis_end: str = " ...",
        add_special_tokens: bool = True,
    ) -> None:
        self.length = max_length
        self.stride = stride

        self.ellipsis_start = ellipsis_start
        self.ellipsis_end = ellipsis_end
        self.add_special_tokens = add_special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
        )

        # Find the start and end special tokens
        self.start_special_tokens, self.end_special_tokens = self._get_special_tokens(
            self.tokenizer,
            self.add_special_tokens,
        )

    @staticmethod
    def _get_prefix_tokens(
        tokenizer: PreTrainedTokenizerBase,
        prefix: str,
    ) -> list[int]:
        if len(prefix) == 0:
            return []
        return tokenizer(prefix, add_special_tokens=False).input_ids

    @staticmethod
    def _get_special_tokens(
        tokenizer: PreTrainedTokenizerBase,
        add_special_tokens: bool,
    ) -> tuple[list[int], list[int]]:
        with_special_tokens = tokenizer(". ", add_special_tokens=add_special_tokens).input_ids
        raw_encoded = tokenizer(". ", add_special_tokens=False).input_ids[0]
        start_special_tokens = with_special_tokens[: with_special_tokens.index(raw_encoded)]
        end_special_tokens = with_special_tokens[with_special_tokens.index(raw_encoded) + 1 :]
        return start_special_tokens, end_special_tokens

    def __call__(
        self,
        text: str,
        prefix: str = "",
        add_prefix: bool = True,
    ) -> typing.Iterable[str]:
        """Sectionize a chunk of text."""
        ellipsis_start_input_ids = self.tokenizer(self.ellipsis_start, add_special_tokens=False).input_ids
        ellipsis_end_input_ids = self.tokenizer(self.ellipsis_end, add_special_tokens=False).input_ids
        encoded_text = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

        # count the number of tokens in the prefix
        n_prefix_tokens = len(self._get_prefix_tokens(self.tokenizer, prefix))

        token_idx = 0
        tok_idx_end: None | int = None
        while tok_idx_end is None or tok_idx_end < (len(encoded_text.input_ids) - 1):
            is_start = token_idx == 0
            is_end = token_idx + self.length >= len(encoded_text.input_ids)
            start_ellipsis_tokens = [] if is_start else ellipsis_start_input_ids
            end_ellipsis_tokens = [] if is_end else ellipsis_end_input_ids

            # Calculate the tokens to be included in the chunk
            n_chunk_tokens = (
                self.length
                - n_prefix_tokens
                - len(start_ellipsis_tokens)
                - len(end_ellipsis_tokens)
                - len(self.start_special_tokens)
                - len(self.end_special_tokens)
            )
            if n_chunk_tokens <= 0:
                raise RuntimeError(
                    f"The resulting chunk length is non-positive: `{n_chunk_tokens}`. This is unexpected. "
                    f"Please report this bug."
                )

            # Get the start token and end token ids
            tok_idx_start = token_idx
            tok_idx_end = min(token_idx + n_chunk_tokens, len(encoded_text.input_ids) - 1)

            # Get the start and end indices for the chunk of text
            char_idx_start = encoded_text.offset_mapping[tok_idx_start][0]
            char_idx_end = encoded_text.offset_mapping[tok_idx_end][0]

            # Get the chunk of text and append the ellipsis if necessary
            chunk = text[char_idx_start:char_idx_end]
            if not is_start:
                chunk = self.ellipsis_start + chunk
            if not is_end:
                chunk += self.ellipsis_end

            # Add the prefix if necessary
            if add_prefix:
                chunk = prefix + chunk

            # Yield and increment
            yield chunk
            token_idx += min(self.stride, n_chunk_tokens)


def init_sectionizer(
    config: vod_configs.SectioningConfig,
) -> Sectionizer:
    """Initialize a sectionizer."""
    if isinstance(config, vod_configs.SentenceSectioningConfig):
        return SentenceSectionizer(config.delimiter)
    if isinstance(config, vod_configs.FixedLengthSectioningConfig):
        return FixedLengthSectionizer(
            config.max_length,
            config.stride,
            config.tokenizer_name_or_path,
            config.ellipsis_start,
            config.ellipsis_end,
            config.add_special_tokens,
        )
    raise TypeError(f"Unexpected type `{type(config)}`")

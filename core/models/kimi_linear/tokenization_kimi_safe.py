"""
安全版本的 Kimi Tokenizer - 使用延迟导入避免 Segmentation Fault
"""
import os
from logging import getLogger
from pathlib import Path
from typing import (
    cast,
    Tuple,
    Dict,
    Iterator,
    List,
    Union,
    Optional,
)
from shutil import copyfile
from tokenizers import AddedToken, pre_tokenizers, Regex
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from typing import Any


logger = getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tiktoken.model"}


class TikTokenTokenizer(PreTrainedTokenizer):
    """
    安全版本的 Tokenizer - 延迟导入 tiktoken 避免段错误
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256

    pat_str = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    def __init__(
        self,
        vocab_file,
        bos_token: Union[str, AddedToken]="[BOS]",
        eos_token: Union[str, AddedToken]="[EOS]",
        unk_token: Union[str, AddedToken, None]=None,
        pad_token: Union[str, AddedToken, None]=None,
        additional_special_tokens: List[str]=None,
        added_tokens_decoder: Optional[dict] = None,
        **kwargs,
    ):
        # 延迟导入 tiktoken - 只在实际使用时导入
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
            self._tiktoken_available = True
        except ImportError as e:
            logger.warning(f"tiktoken 导入失败: {e}")
            self._tiktoken_available = False
            raise ImportError(
                "tiktoken is required for KimiTokenizer. "
                "Install it with: pip install tiktoken==0.5.1"
            )
        
        assert os.path.isfile(vocab_file), f"Vocab file not found: {vocab_file}"

        if additional_special_tokens is None:
            additional_special_tokens = [
                "<|im_end|>",
                "<|im_user|>", 
                "<|im_assistant|>", 
                "<|start_header_id|>", 
                "<|end_header_id|>", 
                "[EOT]", 
                "<|im_system|>", 
                "<|im_middle|>",
            ]
        
        if added_tokens_decoder is not None:
            special_tokens_mapping = {
                i: added_tokens_decoder[i].content for i in added_tokens_decoder
            }
        else:
            special_tokens_mapping = {}

        # 使用 load_tiktoken_bpe 加载
        try:
            mergeable_ranks = load_tiktoken_bpe(vocab_file)
        except Exception as e:
            logger.error(f"加载 tiktoken model 失败: {e}")
            raise RuntimeError(
                f"Failed to load tiktoken model from {vocab_file}. "
                "This may be due to a corrupted file or incompatible tiktoken version."
            ) from e

        # 为特殊 token 分配 ID
        special_tokens_base_id = len(mergeable_ranks)
        special_tokens = [bos_token, eos_token]
        
        if unk_token is not None:
            special_tokens.append(unk_token)
        if pad_token is not None:
            special_tokens.append(pad_token)
            
        special_tokens.extend(additional_special_tokens)
        
        self.special_tokens = {
            token: special_tokens_base_id + i 
            for i, token in enumerate(special_tokens)
        }

        try:
            self.model = tiktoken.Encoding(
                name=Path(vocab_file).name,
                pat_str=self.pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )
        except Exception as e:
            logger.error(f"创建 tiktoken Encoding 失败: {e}")
            raise RuntimeError(
                "Failed to create tiktoken Encoding. "
                "Try: pip install --upgrade tiktoken"
            ) from e

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            added_tokens_decoder=added_tokens_decoder,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.model.n_vocab

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        tokens = self.model.encode(text, allowed_special="all")
        return [self.model.decode_single_token_bytes(token) for token in tokens]

    def _convert_token_to_id(self, token):
        return self.model.encode_single_token(token)

    def _convert_id_to_token(self, index):
        if index < self.vocab_size:
            return self.model.decode_single_token_bytes(index)
        raise ValueError("Index out of vocabulary")

    def convert_tokens_to_string(self, tokens):
        text = b"".join(tokens).decode("utf-8", errors="replace")
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return None
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(vocab_file):
            copyfile(self.vocab_file, vocab_file)
            
        return (vocab_file,)


# 创建别名
KimiTokenizer = TikTokenTokenizer


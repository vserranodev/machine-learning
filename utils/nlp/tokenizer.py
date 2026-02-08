from dataclasses import dataclass

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]     
    merges: dict[tuple[int, int], int] 

class BPETokenizer():
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8"))) 
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices)) 
        string = b"".join(bytes_list).decode("utf-8") 
        return string

# def get_compression_ratio(string: str, indices: list[int]) -> float:
#     """Given `string` that has been tokenized into `indices`, ."""
#     num_bytes = len(bytes(string, encoding="utf-8")) 
#     return num_bytes / num_tokens

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  
    i = 0  
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


if __name__ == "__main__":
    print("ON DEVELOPMENT....")


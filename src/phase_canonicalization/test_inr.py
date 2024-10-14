from functorch import make_functional
import sys
sys.path.append('../scalegmn')
from src.scalegmn.inr import INR, reconstruct_inr


def test_inr(w, b):
    wb_tuple = _wb_to_tuple(w,b)
    inr = INR()
    inr_func, _ = make_functional(inr)
    reconstruct_inr(wb_tuple, inr_func, img_name='canonicalized', save=True, last_batch=True)


def _wb_to_tuple(w, b):
    """Converts a list of weight tensors and a list of biases into a list of layers of weights and biases.
    Assumes the state_dict key order is [0.weight, 0.bias, 1.weight, 1.bias, ...]
    """
    keys = ['seq.0.weight', 'seq.0.bias', 'seq.1.weight', 'seq.1.bias', 'seq.2.weight', 'seq.2.bias']
    batch_size = w[0].shape[0]
    state_tuple = [list() for _ in range(batch_size)]
    layer_idx = 0

    while layer_idx < len(keys):
        for batch_idx in range(batch_size):
            state_tuple[batch_idx].append(w[layer_idx // 2][batch_idx, :, :].squeeze(-1))
            state_tuple[batch_idx].append(b[(layer_idx + 1) // 2][batch_idx, :].squeeze(-1))
        layer_idx += 2
    state_tuple = [tuple(_) for _ in state_tuple]
    return state_tuple

import numpy as np

PAD_IDX = 0

def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_IDX, dtype=np.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


def set_GPU_memory_growth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


import tensorflow as tf
from tensorflow import keras
import numpy as np

import FrNumCorpus
import utils

import time
import pickle
import os

class TransformerConfig():
    def __init__(self,  model_dim=32, max_len=30, num_layer=3, num_head=4, drop_rate=0.1, savedFolder="saved"):
        self.model_dim = model_dim
        self.max_len = max_len
        self.num_layer = num_layer
        self.num_head = num_head
        self.drop_rate = drop_rate
        self.saved_folder = savedFolder

class MultiHead(keras.layers.Layer):
    def __init__(self, num_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // num_head
        self.num_head = num_head
        self.model_dim = model_dim
        self.wq = keras.layers.Dense(num_head * self.head_dim)
        self.wk = keras.layers.Dense(num_head * self.head_dim)
        self.wv = keras.layers.Dense(num_head * self.head_dim)

        self.o_dense = keras.layers.Dense(model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def call(self, q, k, v, mask, training):
        _q = self.wq(q)
        _k, _v = self.wk(k), self.wv(v)
        _q = self.split_heads(_q)
        _k, _v = self.split_heads(_k), self.split_heads(_v)
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)
        o = self.o_dense(context)
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.num_head, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(self.attention, v)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o         # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, num_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)]
        self.mh = MultiHead(num_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, training, mask):
        attn = self.mh.call(xz, xz, xz, mask, training)
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.ffn.call(o1), training)
        o = self.ln[1](ffn + o1)
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, num_head, model_dim, drop_rate, num_layer):
        super().__init__()
        self.ls = [EncodeLayer(num_head, model_dim, drop_rate) for _ in range(num_layer)]

    def call(self, xz, training, mask):
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz


class DecoderLayer(keras.layers.Layer):
    def __init__(self, num_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(3)]
        self.drop = keras.layers.Dropout(drop_rate)
        self.mh = [MultiHead(num_head, model_dim, drop_rate) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        attn = self.mh[0].call(yz, yz, yz, yz_look_ahead_mask, training)
        o1 = self.ln[0](attn + yz)
        attn = self.mh[1].call(o1, xz, xz, xz_pad_mask, training)
        o2 = self.ln[1](attn + o1)
        ffn = self.drop(self.ffn.call(o2), training)
        o = self.ln[2](ffn + o2)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, num_head, model_dim, drop_rate, num_layer):
        super().__init__()
        self.ls = [DecoderLayer(num_head, model_dim, drop_rate) for _ in range(num_layer)]

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for l in self.ls:
            yz = l.call(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, num_vocab):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        pos_em = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim)
        pos_em[:, 0::2] = np.sin(pos_em[:, 0::2])
        pos_em[:, 1::2] = np.cos(pos_em[:, 1::2])
        pos_em = pos_em[None, :, :]
        self.pos_em = tf.constant(pos_em, dtype=tf.float32)
        self.embeddings = keras.layers.Embedding(
            input_dim=num_vocab, output_dim=model_dim,
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

    def call(self, x):
        x_embed = self.embeddings(x) + self.pos_em
        return x_embed


class Transformer(keras.Model):
    def __init__(self, config, num_vocab):
        super().__init__()
        self.max_len = config.max_len
        self.padding_idx = utils.PAD_IDX

        self.embed = PositionEmbedding(config.max_len, config.model_dim, num_vocab)
        self.encoder = Encoder(config.num_head, config.model_dim, config.drop_rate, config.num_layer)
        self.decoder = Decoder(config.num_head, config.model_dim, config.drop_rate, config.num_layer)
        self.o = keras.layers.Dense(num_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(0.002)
        self.saved_folder = config.saved_folder

    def call(self, x, y, training=None):
        x_embed, y_embed = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder.call(x_embed, training, mask=pad_mask)
        decoded_z = self.decoder.call(
            y_embed, encoded_z, training, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=pad_mask)
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.call(x, y[:, :-1], training=True)
            pad_mask = tf.math.not_equal(y[:, 1:], self.padding_idx)
            loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(y[:, 1:], logits), pad_mask))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, logits

    def _pad_bool(self, seqs):
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def _look_ahead_mask(self, seqs):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        mask = tf.where(self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask

    def translate(self, src, dict_word_idx, dict_idx_word):
        src_pad = utils.pad_zero(src, self.max_len)
        tgt = utils.pad_zero(np.array([[dict_word_idx["<BEGIN>"], ] for _ in range(len(src))]), self.max_len+1)
        tgti = 0
        x_embed = self.embed(src_pad)
        encoded_z = self.encoder.call(x_embed, False, mask=self._pad_mask(src_pad))
        while True:
            y = tgt[:, :-1]
            y_embed = self.embed(y)
            decoded_z = self.decoder.call(
                y_embed, encoded_z, False, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=self._pad_mask(src_pad))
            logits = self.o(decoded_z)[:, tgti, :].numpy()
            idx = np.argmax(logits, axis=1)
            tgti += 1
            tgt[:, tgti] = idx
            if tgti >= self.max_len:
                break
        result = []
        for j in range(len(src)):
            result.append( [dict_idx_word[i] for i in tgt[j, 1:tgti] if "<" not in dict_idx_word[i]])
        return [" ".join(res) for res in result] 

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.mh[0].attention.numpy() for l in self.decoder.ls],
                "mh2": [l.mh[1].attention.numpy() for l in self.decoder.ls],
        }}
        return attentions


def train(model, data, step):
    # training
    t0 = time.time()
    for t in range(step):
        bx, by = data.sample(64)
        bx, by = utils.pad_zero(bx, max_len=model.max_len), utils.pad_zero(by, max_len=model.max_len + 1)
        loss, logits = model.step(bx, by)
        if t % 50 == 0:
            logits = logits[0].numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.4f" % loss.numpy(),
                "| target: ", " ".join([data.dict_idx_word[i] for i in by[0, 1:10]]),
                "| inference: ", " ".join([data.dict_idx_word[i] for i in np.argmax(logits, axis=1)[:10]]),
            )
            t0 = t1

    os.makedirs("./" + model.saved_folder + "/models/transformer", exist_ok=True)
    model.save_weights("./" + model.saved_folder + "/models/transformer/model.ckpt")
    os.makedirs("./" + model.saved_folder + "/tmp", exist_ok=True)
    with open("./" + model.saved_folder + "/tmp/transformer_dicts_word_idx.pkl", "wb") as f:
        pickle.dump({"dict_word_idx": data.dict_word_idx, "dict_idx_word": data.dict_idx_word}, f)


def save_training_results(model, data):
    with open("./" + model.saved_folder + "/tmp/transformer_dicts_word_idx.pkl", "rb") as f:
        dic = pickle.load(f)
    model.load_weights("./" + model.saved_folder + "/models/transformer/model.ckpt")
    bx, by = data.sample(32)
    model.translate(bx, dic["dict_word_idx"], dic["dict_idx_word"])
    attnum_data = {
        "src": [[data.dict_idx_word[i] for i in bx[j]] for j in range(len(bx))],
        "tgt": [[data.dict_idx_word[i] for i in by[j]] for j in range(len(by))],
        "attentions": model.attentions}
    path = "./" + model.saved_folder + "/tmp/transformer_attentionum_matrix.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(attnum_data, f)


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', dest='training', default=False, action='store_true', help='choose to train the model')
    parser.add_argument('--cpu', dest='usingCPU', default=False, action='store_true', help='choose to use CPU')
    parser.add_argument('--translate', type=int, dest='translate', default=None, help='translate!')

    args = parser.parse_args()
    print(args)

    if not args.usingCPU:
        utils.set_GPU_memory_growth()

    ###  FrNum corpus training
    d = FrNumCorpus.frNumData("corpus/frNumData_6000.json")
    savePath = "savedModel_frCorpus"


    print("====================== Corpus example ============================")
    print(" - Digital format: ", d.num_digit[0], "\n - French format: ", d.num_fr[0])
    print("====================== Vocabulary set ============================")
    print(d.vocab)
    print("====================== Index examples ============================")
    print("x index sample: \n{}\n{}".format(d.idx2str(d.x[0]), d.x[0]),
          "\ny index sample: \n{}\n{}".format(d.idx2str(d.y[0]), d.y[0]))

    fr_config = TransformerConfig(32, 40, 3, 4, 0.1, savePath) # model_dim=32, max_len=40, num_layer=3, num_head=4, drop_rate=0.1

    # make sure that the config max length is not less than the max length of real data
    assert   fr_config.max_len >= d.x_maxLen
    assert   fr_config.max_len >= d.y_maxLen

    m = Transformer(fr_config, d.num_word)
    if args.training:
        train(m, d, step=2000)
        save_training_results(m, d)
    else:
        with open("./" + m.saved_folder + "/tmp/transformer_dicts_word_idx.pkl", "rb") as f:
            dic = pickle.load(f)
        m.load_weights("./" + m.saved_folder + "/models/transformer/model.ckpt")
        
        if args.translate is not None:
            
            print("\n====================== Translating user input ============================")
            print("Testing the model now...")
            
            bx,by = d.fresh_sample(True, args.translate)
            res = m.translate(bx, dic["dict_word_idx"], dic["dict_idx_word"])
            print(" - translation result: ", res)

        else:    
            print("\n====================== Testing the model ============================")
            print("Testing the model now...")

            bx,by = d.sample(1, True)

            res = m.translate(bx, dic["dict_word_idx"], dic["dict_idx_word"])
            print(" - tranlation result: ", [word for word in res if not "<" in word ])
            
            print("\nTesting a fresh new data...")
            bx,by = d.fresh_sample(True, 61250001)

            res = m.translate(bx, dic["dict_word_idx"], dic["dict_idx_word"])
            print(" - translation result: ", res)

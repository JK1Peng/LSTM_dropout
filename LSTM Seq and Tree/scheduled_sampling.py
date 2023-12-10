
# Constant Declaration
import sys
import os
import _pickle as pickle
import tensorflow as tf
import numpy as np
import nltk
from sklearn.metrics import f1_score

data_dir = 'data'
ckpt_dir = 'checkpoint'
word_embd_dir = 'checkpoint\word_embd'
model_dir = 'checkpoint\mainModel'

word_embd_dim = 100
pos_embd_dim = 25
dep_embd_dim = 25

word_vocab_size = 400001
pos_vocab_size = 10
dep_vocab_size = 21

relation_classes = 10

state_size = 100
batch_size = 10
channels = 3

max_len_seq = 90
max_len_path = 20
max_num_childs = 20

lambda_l2 = 0.0001
init_learning_rate = 0.001
decay_steps = 2000
decay_rate = 0.96
gradient_clipping = 10


# Graph Construction
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.name_scope("input"):

    fp_length = tf.compat.v1.placeholder(
        tf.int32, shape=[batch_size], name="fp_ength")
    fp = tf.compat.v1.placeholder(
        tf.int32, [batch_size, 2, max_len_seq], name="full_path")

    sp_length = tf.compat.v1.placeholder(
        tf.int32, shape=[batch_size, 2], name="sp_length")
    sp = tf.compat.v1.placeholder(
        tf.int32, [batch_size, 2, max_len_path], name="shortest_path")

    sp_pos = tf.compat.v1.placeholder(
        tf.int32, [batch_size, 2, max_len_path], name="sp_pos")

    sp_childs = tf.compat.v1.placeholder(
        tf.int32, [batch_size, 2, max_len_path, max_num_childs], name="sp_childs")
    sp_num_childs = tf.compat.v1.placeholder(
        tf.int32, [batch_size, 2, max_len_path], name="sp_num_childs")

    relation = tf.compat.v1.placeholder(
        tf.int32, [batch_size], name="relation")
    y_entity = tf.compat.v1.placeholder(
        tf.int32, [batch_size, max_len_seq], name="y_enity")

with tf.compat.v1.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(
        0.0, shape=[word_vocab_size, word_embd_dim]), name="W")
    embedding_placeholder = tf.compat.v1.placeholder(
        tf.float32, [word_vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)
    embd_fp_word = tf.nn.embedding_lookup(W, fp[:, 0])
    word_embedding_saver = tf.compat.v1.train.Saver({"word_embedding/W": W})

with tf.compat.v1.name_scope("pos_embedding"):
    W = tf.Variable(tf.random.uniform(
        [pos_vocab_size, pos_embd_dim]), name="W")
    embd_fp_pos = tf.nn.embedding_lookup(W, fp[:, 1])
    pos_embedding_saver = tf.compat.v1.train.Saver({"pos_embedding/W": W})

with tf.compat.v1.name_scope("dep_embedding"):
    W = tf.Variable(tf.random.uniform(
        [dep_vocab_size, dep_embd_dim]), name="W")
    embd_sp = tf.nn.embedding_lookup(W, sp)
    dep_embedding_saver = tf.compat.v1.train.Saver({"dep_embedding/W": W})

embd_fp = tf.concat([embd_fp_word, embd_fp_pos], axis=2)
embd_fp_rev = tf.reverse(embd_fp, [1])
fp_length_rev = tf.reverse(fp_length, [0])


# LSTM cell with backward and forward initiated separately.

def lstm_seq_init(channel, embedding_dim, state_size):
    init_const = tf.zeros([1, state_size])
    with tf.compat.v1.variable_scope(channel):
        W_i = tf.compat.v1.get_variable("W_i", shape=[embedding_dim, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_i = tf.compat.v1.get_variable("U_i", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_i = tf.compat.v1.get_variable("b_i", initializer=init_const)

        W_f = tf.compat.v1.get_variable("W_f", shape=[embedding_dim, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_f = tf.compat.v1.get_variable("U_f", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_f = tf.compat.v1.get_variable("b_f", initializer=init_const)

        W_o = tf.compat.v1.get_variable("W_o", shape=[embedding_dim, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_o = tf.compat.v1.get_variable("U_o", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_o = tf.compat.v1.get_variable("b_o", initializer=init_const)

        W_g = tf.compat.v1.get_variable("W_g", shape=[embedding_dim, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_g = tf.compat.v1.get_variable("U_g", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_g = tf.compat.v1.get_variable("b_g", initializer=init_const)


lstm_seq_init("lstm_fw", word_embd_dim + pos_embd_dim, state_size)
lstm_seq_init("lstm_bw", word_embd_dim + pos_embd_dim, state_size)


def cond1(i, const, steps, *agrs):
    return i < steps


def cond2(i, steps, *agrs):
    return i < steps


init_state_seq = tf.zeros([2, 1, state_size])

x = tf.constant(0)


def lstm_seq(input_embd, seq_len, scope):
    def body(j, const, steps, input_embd, states_seq, states_series):
        inputs = tf.expand_dims(input_embd[j], [0])

        hs = states_seq[0]
        cs = states_seq[1]

        hs_ = states_series[0]
        cs_ = states_series[1]
        with tf.compat.v1.variable_scope(scope, reuse=True):
            W_i = tf.compat.v1.get_variable("W_i")
            U_i = tf.compat.v1.get_variable("U_i")
            b_i = tf.compat.v1.get_variable("b_i")

            W_f = tf.compat.v1.get_variable("W_f")
            U_f = tf.compat.v1.get_variable("U_f")
            b_f = tf.compat.v1.get_variable("b_f")

            W_o = tf.compat.v1.get_variable("W_o")
            U_o = tf.compat.v1.get_variable("U_o")
            b_o = tf.compat.v1.get_variable("b_o")

            W_g = tf.compat.v1.get_variable("W_g")
            U_g = tf.compat.v1.get_variable("U_g")
            b_g = tf.compat.v1.get_variable("b_g")

            input_gate = tf.sigmoid(
                tf.matmul(inputs, W_i) + tf.matmul(hs, U_i) + b_i)
            forget_gate = tf.sigmoid(
                tf.matmul(inputs, W_f) + tf.matmul(hs, U_f) + b_f)
            output_gate = tf.sigmoid(
                tf.matmul(inputs, W_o) + tf.matmul(hs, U_o) + b_o)
            gt = tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(hs, U_g) + b_g)
            cs = input_gate * gt + forget_gate * cs
            hs = output_gate * tf.tanh(cs)

            hs_ = tf.cond(tf.equal(j, const), lambda: hs,
                          lambda: tf.concat([hs_, hs], 0))
            cs_ = tf.cond(tf.equal(j, const), lambda: cs,
                          lambda: tf.concat([cs_, cs], 0))

            states_seq = tf.stack([hs, cs], axis=0)
            states_series = tf.stack([hs_, cs_], axis=0)

            return j+1, const, steps, input_embd, states_seq, states_series

    _, _, _, _, _, state_series_seq = tf.while_loop(cond=cond1, body=body,
                                                    loop_vars=[
                                                        0, 0, seq_len, input_embd, init_state_seq, init_state_seq],
                                                    shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(), input_embd.get_shape(),
                                                                      init_state_seq.get_shape(),
                                                                      tf.TensorShape([2, None, state_size])])

    return state_series_seq


states_series_fw = []
states_series_bw = []
hidden_states_seq = []

for b in range(batch_size):
    seq_len = fp_length[b]
    input_embd = embd_fp[b]
    states_series_fw.append(lstm_seq(input_embd, seq_len, "lstm_fw"))

    input_embd = embd_fp_rev[b]
    states_series_bw.append(tf.reverse(
        lstm_seq(input_embd, seq_len, "lstm_bw"), [1]))
    hidden_states_seq.append(
        tf.concat([states_series_fw[b][0], states_series_bw[b][0]], 1))


def lstm_dep_init(channel, dep_input_size, state_size):
    init_const = tf.zeros([1, state_size])

    with tf.compat.v1.variable_scope(channel):
        W_i = tf.compat.v1.get_variable("W_i", shape=[dep_input_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_i = tf.compat.v1.get_variable("U_i", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_i = tf.compat.v1.get_variable("b_i", initializer=init_const)
        U_it = tf.compat.v1.get_variable("U_it", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))

        W_f = tf.compat.v1.get_variable("W_f", shape=[dep_input_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_f = tf.compat.v1.get_variable("U_f", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_f = tf.compat.v1.get_variable("b_f", initializer=init_const)
        U_fsp = tf.compat.v1.get_variable("U_fsp", shape=[2, state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_ffp = tf.compat.v1.get_variable("U_ffp", shape=[max_num_childs, state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))

        W_o = tf.compat.v1.get_variable("W_o", shape=[dep_input_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_o = tf.compat.v1.get_variable("U_o", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_o = tf.compat.v1.get_variable("b_o", initializer=init_const)
        U_ot = tf.compat.v1.get_variable("U_ot", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))

        W_u = tf.compat.v1.get_variable("W_u", shape=[dep_input_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        U_u = tf.compat.v1.get_variable("U_u", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))
        b_u = tf.compat.v1.get_variable("b_u", initializer=init_const)
        U_ut = tf.compat.v1.get_variable("U_ut", shape=[state_size, state_size], initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"))


dep_input_size = state_size * 2 + dep_embd_dim

lstm_dep_init("lstm_btup", dep_input_size, state_size)
lstm_dep_init("lstm_tpdn", dep_input_size, state_size)


init_state = tf.zeros([2, 1, 1, state_size])


def lstm_dep(b, p, start, seq_len, input_embd, input_pos, input_childs, input_num_child, states_seq, init_state_dep, scope):

    def loop_over_seq(index, const, steps, input_pos, input_embd, input_childs, input_num_child, states_seq, states_dep, states_series):

        inputs = tf.expand_dims(tf.concat(
            [hidden_states_seq[b][input_pos[p][index]], input_embd[p][index]], 0), 0)
        childs = input_childs[p][index]
        num_child = input_num_child[p][index]
        num_child_sp = tf.shape(states_dep[0])[0]

        with tf.compat.v1.variable_scope(scope, reuse=True):
            W_i = tf.compat.v1.get_variable("W_i")
            U_i = tf.compat.v1.get_variable("U_i")
            b_i = tf.compat.v1.get_variable("b_i")
            U_it = tf.compat.v1.get_variable("U_it")

            W_f = tf.compat.v1.get_variable("W_f")
            U_f = tf.compat.v1.get_variable("U_f")
            b_f = tf.compat.v1.get_variable("b_f")
            U_fsp = tf.compat.v1.get_variable("U_fsp")
            U_ffp = tf.compat.v1.get_variable("U_ffp")

            W_o = tf.compat.v1.get_variable("W_o")
            U_o = tf.compat.v1.get_variable("U_o")
            b_o = tf.compat.v1.get_variable("b_o")
            U_ot = tf.compat.v1.get_variable("U_ot")

            W_u = tf.compat.v1.get_variable("W_u")
            U_u = tf.compat.v1.get_variable("U_u")
            b_u = tf.compat.v1.get_variable("b_u")
            U_ut = tf.compat.v1.get_variable("U_ut")

            it = tf.matmul(inputs, W_i) + b_i + \
                tf.matmul(states_dep[0][0], U_i)
            ft = tf.matmul(inputs, W_f) + b_f + \
                tf.matmul(states_dep[0][0], U_f)
            ot = tf.matmul(inputs, W_o) + b_o + \
                tf.matmul(states_dep[0][0], U_o)
            ut = tf.matmul(inputs, W_u) + b_u + \
                tf.matmul(states_dep[0][0], U_u)

            def matmul(k, steps, it, ft, ot, ut):
                it += tf.matmul(states_dep[0][k], U_i)
                ft += tf.matmul(states_dep[0][k], U_f)
                ot += tf.matmul(states_dep[0][k], U_o)
                ut += tf.matmul(states_dep[0][k], U_u)
                return k+1, steps, it, ft, ot, ut

            _, _, it, ft, ot, ut = tf.while_loop(cond=cond2, body=matmul, loop_vars=[
                                                 1, num_child_sp, it, ft, ot, ut])

            def child_sum(k, steps, out, U):
                out += tf.matmul(states_seq[0][childs[k]], U)
                return k+1, steps, out, U

            _, _, ht_i, _ = tf.while_loop(cond=cond2, body=child_sum, loop_vars=[
                                          0, num_child, it, U_it])
            _, _, ht_o, _ = tf.while_loop(cond=cond2, body=child_sum, loop_vars=[
                                          0, num_child, ot, U_ot])
            _, _, ht_u, _ = tf.while_loop(cond=cond2, body=child_sum, loop_vars=[
                                          0, num_child, ut, U_ut])

            input_gate = tf.sigmoid(ht_i)
            output_gate = tf.sigmoid(ht_o)
            u_input = tf.tanh(ht_u)

            cell_state = input_gate * u_input

            def cell_state_sp(k, steps, cell_state):
                _, _, f_sp, _ = tf.while_loop(cond=cond2, body=child_sum, loop_vars=[
                                              0, num_child, ft, U_fsp[k]])
                cell_state += tf.sigmoid(f_sp) * states_dep[1][k]
                return k+1, steps, cell_state

            _, _, cell_state = tf.while_loop(cond=cond2, body=cell_state_sp, loop_vars=[
                                             0, num_child_sp, cell_state])

            def cell_states_fp(k, steps, ctl):
                _, _, f_fp, _ = tf.while_loop(cond=cond2, body=child_sum, loop_vars=[
                                              i, num_child, ft, U_ffp[k]])
                ctl += tf.sigmoid(f_fp) * states_seq[1][childs[k]]
                return k+1, steps, ctl

            _, _, cds = tf.while_loop(cond=cond2, body=cell_states_fp, loop_vars=[
                                      0, num_child, cell_state])

            hds = tf.expand_dims(output_gate * tf.tanh(cds), 0)

            cds = tf.expand_dims(cds, 0)

            states_dep = tf.stack([hds, cds], axis=0)

            hds_ = tf.cond(tf.equal(index, const), lambda: states_dep[0],
                           lambda: tf.concat([states_series[0], states_dep[0]], 0))

            cds_ = tf.cond(tf.equal(index, const), lambda: states_dep[1],
                           lambda: tf.concat([states_series[1], states_dep[1]], 0))

            states_series = tf.stack([hds_, cds_], axis=0)

        return index+1, const, steps, input_pos, input_embd, input_childs, input_num_child, states_seq, states_dep, states_series

    _, _, _, _, _, _, _, _, _, states_series_dep = tf.while_loop(cond=cond1, body=loop_over_seq, loop_vars=[start, start, seq_len,
                                                                                                            input_pos, input_embd, input_childs, input_num_child, states_seq, init_state_dep,
                                                                                                            init_state], shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                                                                                                                                           input_pos.get_shape(), input_embd.get_shape(), input_childs.get_shape(),
                                                                                                                                           input_num_child.get_shape(), states_seq.get_shape(),
                                                                                                                                           tf.TensorShape([2, None, 1, state_size]), tf.TensorShape([2, None, 1, state_size])])

    return states_series_dep


lca_series_btup = []
dp_series_tpdn = []

for i in range(batch_size):

    input_pos = sp_pos[i]
    input_embd = embd_sp[i]
    input_childs = sp_childs[i]
    input_num_child = sp_num_childs[i]

    input_pos_rev = tf.reverse(sp_pos[i], [1])
    input_embd_rev = tf.reverse(embd_sp[i], [1])
    input_childs_rev = tf.reverse(sp_childs[i], [1])
    input_num_child_rev = tf.reverse(sp_num_childs[i], [1])

    states_seq_fw = tf.expand_dims(states_series_fw[i], 2)
    states_seq_bw = tf.expand_dims(states_series_bw[i], 2)

    s1 = lstm_dep(i, 0, 0, sp_length[i][0]-1, input_embd, input_pos, input_childs, input_num_child,
                  states_seq_fw, init_state, "lstm_btup")

    lca_btup = tf.cond(
        sp_length[i][0] > 1, lambda: s1[:, sp_length[i][0]-2], lambda: init_state[:, 0])

    s2 = lstm_dep(i, 1, 0, sp_length[i][1], input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev,
                  states_seq_bw, init_state, "lstm_btup")

    lca_btup = tf.cond(sp_length[i][1] > 0, lambda: tf.stack(
        [lca_btup, s2[:, sp_length[i][1]-1]], axis=1), lambda: tf.expand_dims(lca_btup, 1))

    lca_series_btup.append(lstm_dep(i, 0, sp_length[i][0]-1, sp_length[i][0], input_embd,
                           input_pos, input_childs, input_num_child, states_seq_fw, lca_btup, "lstm_btup")[0, 0])

    lca_tpdn = lstm_dep(i, 0, 0, 1, input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev,
                        states_seq_bw, init_state, "lstm_tpdn")

    dp1 = lstm_dep(i, 0, 1, sp_length[i][0], input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev,
                   states_seq_bw, lca_tpdn, "lstm_tpdn")[0, -1]

    dp1 = tf.cond(sp_length[i][0] > 1, lambda: dp1, lambda: lca_tpdn[0][0])

    dp2 = lstm_dep(i, 1, 0, sp_length[i][1], input_embd, input_pos, input_childs, input_num_child,
                   states_seq_fw, lca_tpdn, "lstm_tpdn")[0, -1]

    dp2 = tf.cond(sp_length[i][1] > 0, lambda: dp2, lambda: lca_tpdn[0][0])

    dp_series_tpdn.append(tf.concat([dp1, dp2], 1))


for i in range(batch_size):

    temp = tf.concat([lca_series_btup[i], dp_series_tpdn[i]], 1)
    if (i == 0):
        dp_series = temp
    else:
        dp_series = tf.concat([dp_series, temp], axis=0)


with tf.compat.v1.name_scope("hidden_layer_seq"):
    W = tf.Variable(tf.random.truncated_normal(
        [200, 100], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([100]), name="b")

    y_hidden_layer = []
    y_hl = tf.zeros([1, 100])

    for batch in range(batch_size):
        s_seq = tf.expand_dims(hidden_states_seq[batch], 1)

        def matmul_hl(j, const, steps, input_seq, out_seq):
            temp = tf.tanh(tf.matmul(input_seq[j], W) + b)
            out_seq = tf.cond(tf.equal(j, const), lambda: temp,
                              lambda: tf.concat([out_seq, temp], 0))
            return j+1, const, steps, input_seq, out_seq

        _, _, _, _, output_seq = tf.while_loop(cond=cond1, body=matmul_hl,
                                               loop_vars=[
                                                   0, 0, fp_length[batch], s_seq, y_hl],
                                               shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                                                                 s_seq.get_shape(), tf.TensorShape([None, 100])])

        y_hidden_layer.append(output_seq)

with tf.name_scope("dropout_hidden_layer_seq"):
    y_hidden_layer_drop = tf.nn.dropout(y_hidden_layer, 0.3)

with tf.compat.v1.name_scope("softmax_layer_seq"):
    W = tf.Variable(tf.random.truncated_normal([100, 2], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([2]), name="b")

    logits_entity = []
    predictions_entity = []
    lg = tf.zeros([1, 2])

    for batch in range(batch_size):

        def matmul_softmax(j, const, steps, y_hl, lg):
            temp = tf.matmul(y_hl[j], W) + b
            lg = tf.cond(tf.equal(j, const), lambda: temp,
                         lambda: tf.concat([lg, temp], 0))
            return j+1, const, steps, y_hl, lg

        y_hl = tf.expand_dims(y_hidden_layer[batch], 1)

        _, _, _, _, logit = tf.while_loop(cond=cond1, body=matmul_softmax,
                                          loop_vars=[
                                              0, 0, fp_length[batch], y_hl, lg],
                                          shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                                                            y_hl.get_shape(), tf.TensorShape([None, 2])])

        logits_entity.append(logit)
        predictions_entity.append(tf.argmax(logit, 1))

Y_en = [y_entity[i][:fp_length[i]] for i in range(batch_size)]

Y_softmax = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_entity[i], labels=Y_en[i])) for i in range(batch_size)]

with tf.compat.v1.name_scope("loss_seq"):
    loss_seq = tf.reduce_mean(Y_softmax)

# HERE
with tf.compat.v1.name_scope("loss_seq_svm"):
    # Convert y_entity to -1 or 1 labels for SVM
    y_entity_svm = tf.cast(tf.multiply(2, y_entity) - 1, tf.float32)
    # Calculate hinge loss
    logits_entity_svm = tf.reduce_mean(
        logits_entity, axis=2)  # Adjust shape if necessary
    loss_seq_svm = tf.reduce_mean(tf.maximum(
        0., 1. - logits_entity_svm * y_entity_svm))

with tf.compat.v1.name_scope("hidden_layer_dep"):
    W = tf.Variable(tf.random.truncated_normal(
        [300, 100], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([100]), name="b")
    y_p = tf.tanh(tf.matmul(dp_series, W) + b)

with tf.compat.v1.name_scope("softmax_layer_dep"):
    W = tf.Variable(tf.random.truncated_normal(
        [100, relation_classes], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([relation_classes]), name="b")
    logits = tf.matmul(y_p, W) + b
    predictions_dep = tf.argmax(logits, 1)


with tf.compat.v1.name_scope("loss_dep"):
    loss_dep = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=relation))

# HERE
with tf.compat.v1.name_scope("loss_dep_svm"):
    # Convert relation to -1 or 1 labels for SVM
    relation_svm = tf.cast(tf.multiply(
        2.0, tf.cast(relation, tf.float32)) - 1.0, tf.float32)
    # Calculate hinge loss
    logits_dep_svm = tf.cast(logits, tf.float32)  # Cast logits to float32
    loss_dep_svm = tf.reduce_mean(tf.maximum(
        0., 1. - logits_dep_svm * tf.expand_dims(relation_svm, -1)))


tv_all = tf.compat.v1.trainable_variables()

tv_regu = []

non_reg = ["word_embedding/W:0", "pos_embedding/W:0", 'dep_embedding/W:0']

for t in tv_all:
    if t.name not in non_reg:
        if t.name.find('b_') == -1:
            if t.name.find('b:') == -1:
                tv_regu.append(t)

with tf.compat.v1.name_scope("total_loss"):
    l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
    total_loss = l2_loss + loss_seq + loss_dep

# HERE
with tf.compat.v1.name_scope("total_loss_svm"):
    l2_loss_svm = lambda_l2 * \
        tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
    total_loss_svm = l2_loss_svm + loss_seq_svm + loss_dep_svm

global_step_seq = tf.Variable(0, trainable=False, name="global_step_seq")
global_step_dep = tf.Variable(0, trainable=False, name="global_step_dep")

learning_rate_seq = tf.compat.v1.train.exponential_decay(
    init_learning_rate, global_step_seq, decay_steps, decay_rate, staircase=True)
learning_rate_dep = tf.compat.v1.train.exponential_decay(
    init_learning_rate, global_step_dep, decay_steps, decay_rate, staircase=True)

optimizer_seq = tf.compat.v1.train.AdamOptimizer(
    learning_rate_seq).minimize(loss_seq, global_step=global_step_seq)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_dep)
# HERE: New optimizer and training operation for SVM-based total loss
optimizer_svm = tf.compat.v1.train.AdamOptimizer(learning_rate_dep)

grads_vars = optimizer.compute_gradients(total_loss)

for g, v in grads_vars:
    if (g == None):
        print(g, v)

clipped_grads = [(tf.clip_by_norm(grad, gradient_clipping), var)
                 for grad, var in grads_vars]
train_op = optimizer.apply_gradients(
    clipped_grads, global_step=global_step_dep)

# HERE: New optimizer and training operation for SVM-based total loss
train_op_svm = optimizer_svm.minimize(
    total_loss_svm, global_step=global_step_dep)
#  In[13]:


f = open(data_dir + '/vocab_glove', 'rb')
vocab = pickle.load(f)
f.close()

word2id = dict((w, i) for i, w in enumerate(vocab))
id2word = dict((i, w) for i, w in enumerate(vocab))

unknown_token = "UNKNOWN_TOKEN"

pos_tags_vocab = []
for line in open(data_dir + '/pos_tags.txt'):
    pos_tags_vocab.append(line.strip())

dep_vocab = []
for line in open(data_dir + '/dependency_types.txt'):
    dep_vocab.append(line.strip())

relation_vocab = []
for line in open(data_dir + '/relation_typesv3.txt'):
    relation_vocab.append(line.strip())

rel2id = dict((w, i) for i, w in enumerate(relation_vocab))
id2rel = dict((i, w) for i, w in enumerate(relation_vocab))

pos_tag2id = dict((w, i) for i, w in enumerate(pos_tags_vocab))
id2pos_tag = dict((i, w) for i, w in enumerate(pos_tags_vocab))

dep2id = dict((w, i) for i, w in enumerate(dep_vocab))
id2dep = dict((i, w) for i, w in enumerate(dep_vocab))

pos_tag2id['OTH'] = 9
id2pos_tag[9] = 'OTH'

dep2id['OTH'] = 20
id2dep[20] = 'OTH'

JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']


def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 9


# In[14]:


def prepare_input(words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2, relations):

    length = len(words_seq)

    pos_path1 = [[i-1 for i in w] for w in pos_path1]
    pos_path2 = [[i-1 for i in w] for w in pos_path2]

    for i in range(length):
        words_seq[i] = [w for w in words_seq[i] if w != None]
        deps_seq[i] = [w for w in deps_seq[i] if w != None]
        pos_tags_seq[i] = [w for w in pos_tags_seq[i] if w != None]

    entity = np.zeros([length, max_len_seq])
    for i in range(length):
        entity[i][pos_path1[i][0]] = 1
        if (pos_path2[i] == []):
            entity[i][pos_path1[i][-1]] = 1
        else:
            entity[i][pos_path2[i][0]] = 1

    len_path1 = []
    len_path2 = []

    num_child_path1 = np.ones([length, max_len_path], dtype=int)
    num_child_path2 = np.ones([length, max_len_path], dtype=int)

    for w in word_path1:
        len_path1.append(len(w))

    for w in word_path2:
        len_path2.append(len(w))

    for i, w in enumerate(childs_path1):
        if (w != []):
            for j, c in enumerate(w):
                num_child_path1[i][j] = len(c)
        else:
            num_child_path1[i][0] = 0

    for i, w in enumerate(childs_path2):
        if (w != []):
            for j, c in enumerate(w):
                num_child_path2[i][j] = len(c)
        else:
            num_child_path2[i][0] = 0

    for i in range(length):
        if (childs_path2[i] != []):
            for j, c in enumerate(childs_path2[i]):
                if (c == []):
                    childs_path2[i][j] = [-1]
        else:
            childs_path2[i] = [[-1]]

        if (childs_path1[i] != []):
            for j, c in enumerate(childs_path1[i]):
                if (c == []):
                    childs_path1[i][j] = [-1]
        else:
            childs_path1[i] = [[-1]]

    for i in range(length):
        if (word_path2[i] == []):
            word_path2[i].append(unknown_token)
        if (dep_path2[i] == []):
            dep_path2[i].append('OTH')
        if (pos_tags_path2 == []):
            pos_tags_path2[i].append('OTH')

    for i in range(length):
        for j, word in enumerate(words_seq[i]):
            word = word.lower()
            words_seq[i][j] = word if word in word2id else unknown_token

        for l, d in enumerate(deps_seq[i]):
            deps_seq[i][l] = d if d in dep2id else 'OTH'

        for j, word in enumerate(word_path1[i]):
            word = word.lower()
            word_path1[i][j] = word if word in word2id else unknown_token

        for l, d in enumerate(dep_path1[i]):
            dep_path1[i][l] = d if d in dep2id else 'OTH'

        for j, word in enumerate(word_path2[i]):
            word = word.lower()
            word_path2[i][j] = word if word in word2id else unknown_token

        for l, d in enumerate(dep_path2[i]):
            dep_path2[i][l] = d if d in dep2id else 'OTH'

    words_seq_id = np.ones([length, max_len_seq], dtype=int)
    deps_seq_id = np.ones([length, max_len_seq], dtype=int)
    pos_tags_seq_id = np.ones([length, max_len_seq], dtype=int)

    word_path1_id = np.ones([length, max_len_path], dtype=int)
    word_path2_id = np.ones([length, max_len_path], dtype=int)

    dep_path1_id = np.ones([length, max_len_path], dtype=int)
    dep_path2_id = np.ones([length, max_len_path], dtype=int)

    pos_tags_path1_id = np.ones([length, max_len_path], dtype=int)
    pos_tags_path2_id = np.ones([length, max_len_path], dtype=int)

    pos_path1_ = np.ones([length, max_len_path], dtype=int)
    pos_path2_ = np.ones([length, max_len_path], dtype=int)

    childs_path1_ = np.ones([length, max_len_path, max_num_childs], dtype=int)
    childs_path2_ = np.ones([length, max_len_path, max_num_childs], dtype=int)

    seq_len = []

    for i in range(length):

        temp = []
        seq_len.append(len(words_seq[i]))

        for j, w in enumerate(pos_path1[i]):
            pos_path1_[i][j] = w

        for j, w in enumerate(pos_path2[i]):
            pos_path2_[i][j] = w

        for j, child in enumerate(childs_path1[i]):
            for k, c in enumerate(child):
                childs_path1_[i][j][k] = c - 1

        for j, child in enumerate(childs_path2[i]):
            for k, c in enumerate(child):
                childs_path2_[i][j][k] = c - 1

        for j, w in enumerate(words_seq[i]):
            words_seq_id[i][j] = word2id[w]

        for j, w in enumerate(pos_tags_seq[i]):
            pos_tags_seq_id[i][j] = pos_tag(w)

        for j, w in enumerate(deps_seq[i]):
            deps_seq_id[i][j] = dep2id[w]

        for j, w in enumerate(word_path1[i]):
            word_path1_id[i][j] = word2id[w]

        for j, w in enumerate(pos_tags_path1[i]):
            pos_tags_path1_id[i][j] = pos_tag(w)

        for j, w in enumerate(dep_path1[i]):
            dep_path1_id[i][j] = dep2id[w]

        for j, w in enumerate(word_path2[i]):
            word_path2_id[i][j] = word2id[w]

        for j, w in enumerate(pos_tags_path2[i]):
            pos_tags_path2_id[i][j] = pos_tag(w)

        for j, w in enumerate(dep_path2[i]):
            dep_path2_id[i][j] = dep2id[w]

    rel_ids = np.array([rel2id[rel] for rel in relations])

    return seq_len, words_seq_id, pos_tags_seq_id, deps_seq_id, len_path1, len_path2, pos_path1_, pos_path2_, dep_path1_id, dep_path2_id, childs_path1_, childs_path2_, num_child_path1, num_child_path2, rel_ids, entity


# # In[ ]:
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()


# # In[16]:


# f = open('data/word_embedding', 'rb')
# word_embedding = pickle.load(f)
# f.close()
# sess.run(embedding_init, feed_dict={embedding_placeholder:word_embedding})
# word_embedding_saver.save(sess, word_embd_dir + '/word_embd')


# # In[16]:


model = tf.train.latest_checkpoint(model_dir)
# saver.restore(sess, model)


# # In[19]:


latest_embd = tf.train.latest_checkpoint(word_embd_dir)
# word_embedding_saver.restore(sess, latest_embd)


# # In[17]:


f = open(data_dir + '/train_pathsv3', 'rb')
words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2 = pickle.load(
    f)
f.close()

relations = []
for line in open(data_dir + '/train_relationsv3.txt'):
    relations.append(line.strip().split()[1])

length = len(words_seq)
num_batches = int(length/batch_size)

seq_len, words_seq_id, pos_tags_seq_id, deps_seq_id, len_path1, len_path2, pos_path1, pos_path2, dep_path1_id, dep_path2_id, childs_path1, childs_path2, num_child_path1, num_child_path2, rel_ids, entity = prepare_input(
    words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2, relations)


print("839, I'm done")

# # In[ ]:


num_epochs = 15

for i in range(num_epochs):

    loss_per_epoch = 0

    for j in range(num_batches):

        s = j * batch_size
        end = (j+1) * batch_size

        feed_dict = {
            fp_length: seq_len[s:end],
            fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
            sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
            sp: [[dep_path1_id[k], dep_path2_id[k]] for k in range(s, end)],
            sp_pos: [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
            sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
            sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
            relation: rel_ids[s:end],
            y_entity: entity[s:end]}
        

        _, _loss, step = sess.run(
            [train_op, total_loss, global_step_dep], feed_dict)
#         _, _loss, step = sess.run([optimizer_seq, loss_seq, global_step_seq], feed_dict)
        loss_per_epoch += _loss

        if (j+1) % num_batches == 0:
            print("Epoch:", i+1, "Step:", step,
                  "loss:", loss_per_epoch/num_batches)

    saver.save(sess, model_dir + '/model')
    print("Saved Model")


# In[18]:


# training accuracy
all_predictions = []
for j in range(num_batches):
    s = j * batch_size
    end = (j+1) * batch_size

    feed_dict = {
        fp_length: seq_len[s:end],
        fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
        sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
        sp: [[dep_path1_id[k], dep_path2_id[k]] for k in range(s, end)],
        sp_pos: [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
        sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
        sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
        relation: rel_ids[s:end],
        y_entity: entity[s:end]}

    batch_predictions = sess.run(predictions_dep, feed_dict)
    all_predictions.append(batch_predictions)

y_pred = []
for i in range(num_batches):
    for pred in all_predictions[i]:
        y_pred.append(pred)

count = 0
for i in range(batch_size*num_batches):
    count += y_pred[i] == rel_ids[i]
accuracy = count/(batch_size*num_batches) * 100

f1 = f1_score(rel_ids[:batch_size*num_batches], y_pred, average='macro')*100
print("train accuracy", accuracy, " F1 Score", f1)


# In[19]:


f = open(data_dir + '/test_pathsv3', 'rb')
words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2 = pickle.load(
    f)
f.close()

relations = []
for line in open(data_dir + '/test_relationsv3.txt'):
    relations.append(line.strip().split()[0])

length = len(words_seq)
num_batches = int(length/batch_size)

seq_len, words_seq_id, pos_tags_seq_id, deps_seq_id, len_path1, len_path2, pos_path1, pos_path2, dep_path1_id, dep_path2_id, childs_path1, childs_path2, num_child_path1, num_child_path2, rel_ids, entity = prepare_input(
    words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2, relations)

# test predictions

all_predictions = []
for j in range(num_batches):
    s = j * batch_size
    end = (j+1) * batch_size

    feed_dict = {
        fp_length: seq_len[s:end],
        fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
        sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
        sp: [[dep_path1_id[k], dep_path2_id[k]] for k in range(s, end)],
        sp_pos: [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
        sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
        sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
        relation: rel_ids[s:end],
        y_entity: entity[s:end]}

    batch_predictions = sess.run(predictions_dep, feed_dict)
    all_predictions.append(batch_predictions)
y_pred = []
for i in range(num_batches):
    for pred in all_predictions[i]:
        y_pred.append(pred)

count = 0
for i in range(batch_size*num_batches):
    count += y_pred[i] == rel_ids[i]
accuracy = count/(batch_size*num_batches) * 100

f1 = f1_score(rel_ids[:batch_size*num_batches], y_pred, average='macro')*100
print("test accuracy", accuracy, " F1 Score", f1)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from mll_data import MllData, load_data

flags = tf.flags
logging = tf.logging
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "data/spoken_sentences_mll_wav",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp64", False,
                  "Train using 64-bit floats instead of 32-bit floats")
FLAGS = flags.FLAGS


def data_type():
    return tf.float64 if FLAGS.use_fp64 else tf.float32


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 104
    epoch_size = 30
    constant_lr_max_epoch = 4
    max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 10
    validation_batch_size = 3


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 416
    epoch_size = 40
    constant_lr_max_epoch = 6
    max_epoch = 16
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    validation_batch_size = 5


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    hidden_size = 1300
    epoch_size = 60
    constant_lr_max_epoch = 8
    max_epoch = 24
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    validation_batch_size = 5


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


class MLLModel(object):
    """The MLL model."""

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def inputs_ph(self):
        return self._inputs_ph

    @property
    def labels_ph(self):
        return self._labels_ph

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def loss(self):
        return self._loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    def __init__(self, is_training, config, input_):
        self._input = input_
        self._epoch_size = config.epoch_size if is_training else 1

        self._batch_size = batch_size = config.batch_size if is_training else config.validation_batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        num_classes = input_.num_classes

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        self._inputs_ph = tf.placeholder(data_type(), shape=[batch_size, num_steps, size])
        if is_training and config.keep_prob < 1:
            self._inputs_ph = tf.nn.dropout(self._inputs_ph, config.keep_prob)
        self._labels_ph = tf.placeholder(data_type(), shape=[batch_size, num_classes])

        with tf.variable_scope("RNN"):
            inputs = tf.unstack(self._inputs_ph, num=num_steps, axis=1)
            outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        self._final_state = state

        print("outputs len: ", len(outputs))
        output = outputs.pop()
        print("output shape: ", output.get_shape())

        softmax_w = tf.get_variable(
            "softmax_w", [size, num_classes], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=data_type())
        classes = tf.matmul(output, softmax_w) + softmax_b
        print("classes shape: ", classes.get_shape())

        classes_w = tf.get_variable(
            "classes_w", [num_classes, num_classes], dtype=data_type())
        classes_b = tf.get_variable("classes_b", [num_classes], dtype=data_type())
        logits = tf.matmul(classes, classes_w) + classes_b
        print("logits shape: ", logits.get_shape())

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self._labels_ph)
        self._loss = loss
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.epoch_size):
        _inputs, _labels = model.input.get_batch(model.batch_size)

        vals = session.run(fetches, feed_dict={
            model.inputs_ph: _inputs,
            model.labels_ph: _labels
        })
        # state = vals["final_state"]
        costs += vals["cost"]
        iters += 1

        if verbose and step % (model.epoch_size // 10) == 10:
            print("%.3f Accuracy: %.3f speed: %.0f sentences/sec" %
                  (step * 1.0 / model.epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to MLL data directory")

    config = get_config()
    train_data, valid_data = load_data(FLAGS.data_path)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = MllData(train_data['mfcc'], train_data['labels'], config.hidden_size)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = MLLModel(is_training=True, config=config, input_=train_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = MllData(valid_data['mfcc'], valid_data['labels'], config.hidden_size)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = MLLModel(is_training=False, config=config, input_=valid_input)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.constant_lr_max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_accuracy = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Accuracy: %.3f" % (i + 1, train_accuracy))
                valid_accuracy = run_epoch(session, mvalid)
                print("Epoch: %d Valid Accuracy: %.3f" % (i + 1, valid_accuracy))

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()

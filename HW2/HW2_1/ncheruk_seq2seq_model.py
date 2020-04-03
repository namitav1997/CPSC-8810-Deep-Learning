#Namita Vagdevi Cherukuru
#HW 2: Deep Learning (CPSC 8810)
#ncheruk@clemson.edu

import tensorflow as tenf
import numpy as nump
import random


class Ncheruk_Seq2Seq_Model():
    def __init__(self, rnn_sz, numof_layers, dim_vft, embed_size,
                 lr_rate, word_2_i, mode, max_gradnorm,
                 atten, bsearch, bsize,
                 maxer_step, maxdecstep):
        tenf.set_random_seed(5677)
        nump.random.seed(5677)
        random.seed(5677)


        self.word_2_idx = word_2_i
        self.mode = mode
        self.max_gradient_norm = max_gradnorm

        self.max_decoder_steps = maxdecstep
        self.rnn_size = rnn_sz
        self.num_layers = numof_layers
        self.dim_video_feat = dim_vft
        self.embedding_size = embed_size
        self.learning_rate = lr_rate

        self.use_attention = atten
        self.beam_search = bsearch
        self.beam_size = bsize
        self.max_encoder_steps = maxer_step

        self.vocab_size = len(self.word_2_idx)

        self.building_model()

    def crte_rnn_cell(self):
        def sgle_rnn_cell():
            single_cell = tenf.contrib.rnn.GRUCell(self.rnn_size)
            cell = tenf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.prob_placeholder, seed=9487)
            return cell

        cell = tenf.contrib.rnn.MultiRNNCell([sgle_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def building_model(self):
        tenf.set_random_seed(5677)
        nump.random.seed(5677)
        random.seed(5677)

        self.encdr_inputs = tenf.placeholder(tenf.float32, [None, None, None], name='encoder_inputs')
        self.encdr_inp_length = tenf.placeholder(tenf.int32, [None], name='encoder_inputs_length')

        self.batch_sz = tenf.placeholder(tenf.int32, [], name='batch_size')
        self.prob_placeholder = tenf.placeholder(tenf.float32, name='keep_prob_placeholder')

        self.dec_targets = tenf.placeholder(tenf.int32, [None, None], name='decoder_targets')
        self.der_target_len = tenf.placeholder(tenf.int32, [None], name='decoder_targets_length')

        self.max_sequence_length = tenf.reduce_max(self.der_target_len, name='max_target_len')
        self.mask = tenf.sequence_mask(self.der_target_len, self.max_sequence_length, dtype=tenf.float32,
                                       name='masks')



        with tenf.variable_scope('decoder', reuse=tenf.AUTO_REUSE):
            encoder_inputs_length = self.encdr_inp_length

            if self.beam_search:
                print("Using beamsearch decoding...")
                encoder_outputs = tenf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = tenf.contrib.framework.nest.map_structure(
                    lambda s: tenf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tenf.contrib.seq2seq.tile_batch(self.encdr_inp_length,
                                                                        multiplier=self.beam_size)

            batch_size = self.batch_sz if not self.beam_search else self.batch_sz * self.beam_size

            projection_layer = tenf.layers.Dense(units=self.vocab_size,
                                                 kernel_initializer=tenf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                                                                      seed=9487))

            embedding_decoder = tenf.Variable(tenf.random_uniform([self.vocab_size, self.rnn_size], -0.1, 0.1, seed=9487),
                                              name='embedding_decoder')

            decoder_cell = self.crte_rnn_cell()

            if self.use_attention:
                attention_mechanism = tenf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.rnn_size,
                    memory=encoder_outputs,
                    normalize=True,
                    memory_sequence_length=encoder_inputs_length)

                decoder_cell = tenf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.rnn_size,
                    name='Attention_Wrapper')

                decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tenf.float32).clone(
                    cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            output_layer = tenf.layers.Dense(self.vocab_size,
                                             kernel_initializer=tenf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                                                                  seed=9487))


            ending = tenf.strided_slice(self.dec_targets, [0, 0], [self.batch_sz, -1], [1, 1])
            decoder_inputs = tenf.concat([tenf.fill([self.batch_sz, 1], self.word_2_idx['<bos>']), ending], 1)

            start_tokens = tenf.ones([self.batch_sz, ], tenf.int32) * self.word_2_idx['<bos>']
            end_token = self.word_2_idx['<eos>']

            if self.beam_search:
                inference_decoder = tenf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_size,
                    output_layer=output_layer)
            else:
                inference_decoding_helper = tenf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token)
                inference_decoder = tenf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=inference_decoding_helper,
                    initial_state=decoder_initial_state,
                    output_layer=output_layer)

            inference_decoder_outputs, _, _ = tenf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder,
                maximum_iterations=self.max_decoder_steps)

            if self.beam_search:
                self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
                self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output
            else:
                self.decoder_predict_decode = tenf.expand_dims(inference_decoder_outputs.sample_id, -1)
                self.decoder_predict_logits = inference_decoder_outputs.rnn_output

        with tenf.variable_scope('encoder', reuse=tenf.AUTO_REUSE):
            enr_inputs_flatten = tenf.reshape(self.encdr_inputs, [-1, self.dim_video_feat])
            enr_inputs_embed = tenf.layers.dense(enr_inputs_flatten, self.embedding_size, use_bias=True)
            enr_inputs_embed = tenf.reshape(enr_inputs_embed,
                                            [self.batch_sz, self.max_encoder_steps, self.rnn_size])

            encoder_cell = self.crte_rnn_cell()

            encoder_outputs, encoder_state = tenf.nn.dynamic_rnn(
                encoder_cell, enr_inputs_embed,
                sequence_length=self.encdr_inp_length,
                dtype=tenf.float32)

        self.saver = tenf.train.Saver(tenf.global_variables(), max_to_keep=50)


            decoder_inputs_embedded = tenf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

            training_helper = tenf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_inputs_embedded,
                sequence_length=self.der_target_len,
                time_major=False, name='training_helper')
            training_decoder = tenf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper,
                initial_state=decoder_initial_state,
                output_layer=output_layer)


            decoder_outputs, _, _ = tenf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=self.max_sequence_length)

            self.decoder_logits_train = tenf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tenf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')

            self.loss = tenf.contrib.seq2seq.sequence_loss(
                logits=self.decoder_logits_train,
                targets=self.dec_targets,
                weights=self.mask)

            tenf.summary.scalar('loss', self.loss)
            self.summary_op = tenf.summary.merge_all()

            optimizer = tenf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tenf.trainable_variables()
            gradients = tenf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tenf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))




    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        feed_dict = {self.encdr_inputs: encoder_inputs,
                     self.encdr_inp_length: encoder_inputs_length,
                     self.dec_targets: decoder_targets,
                     self.der_target_len: decoder_targets_length,
                     self.prob_placeholder: 0.8,
                     self.batch_sz: len(encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, encoder_inputs, encoder_inputs_length):
        feed_dict = {self.encdr_inputs: encoder_inputs,
                     self.encdr_inp_length: encoder_inputs_length,
                     self.prob_placeholder: 1.0,
                     self.batch_sz: len(encoder_inputs)}
        predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
        return predict, logits

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        feed_dict = {self.encdr_inputs: encoder_inputs,
                     self.encdr_inp_length: encoder_inputs_length,
                     self.dec_targets: decoder_targets,
                     self.der_target_len: decoder_targets_length,
                     self.prob_placeholder: 1.0,
                     self.batch_sz: len(encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary


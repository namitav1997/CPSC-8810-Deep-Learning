#Namita Vagdevi Cherukuru
#HW 2: Deep Learning (CPSC 8810)
#ncheruk@clemson.edu

import tensorflow as tenf
import pandas as pand
import numpy as nump
import time
import json
import random
import pickle
import os
import sys


from seq2seq_model import Seq2Seq_Model

if __name__ == "__main__":

    nump.random.seed(5677)
    random.seed(5677)
    tenf.set_random_seed(5677)

    tst_vfeat_folder = sys.argv[1]
    tst_lbl_json = sys.argv[2]
    out_testst_file = sys.argv[3]

    tenf.app.flags.DEFINE_integer('rnn_size', 1598)
    tenf.app.flags.DEFINE_integer('numof_layers',3)
    tenf.app.flags.DEFINE_integer('dim_video_feat', 3958)
    tenf.app.flags.DEFINE_integer('embedding_size', 1598)

    tenf.app.flags.DEFINE_float('learning_rate', 0.0001)
    tenf.app.flags.DEFINE_integer('batch_size', 29)
    tenf.app.flags.DEFINE_float('max_gradient_norm', 5.0)

    tenf.app.flags.DEFINE_boolean('use_attention', True)

    tenf.app.flags.DEFINE_boolean('beam_search', False)
    tenf.app.flags.DEFINE_integer('beam_size', 5)

    tenf.app.flags.DEFINE_integer('max_encoder_steps', 64)
    tenf.app.flags.DEFINE_integer('max_decoder_steps', 15)

    tenf.app.flags.DEFINE_integer('sample_size', 1450)
    tenf.app.flags.DEFINE_integer('dim_video_frame', 80)
    tenf.app.flags.DEFINE_integer('num_epochs', 203)

    FLAGS = tenf.app.flags.FLAGS

    num_top_BLEU = 10
    top_BL = []

    print('Reading pickle files...')
    word2indx = pickle.load(open('word2indx.obj', 'rb'))
    indx2wrd = pickle.load(open('indx2wrd.obj', 'rb'))

    vid_ID = pickle.load(open('vid_ID.obj', 'rb'))
    video_cap = pickle.load(open('video_cap.obj', 'rb'))
    vid_ftdic = pickle.load(open('vid_ftdic.obj', 'rb'))
    indx2wrd_series = pand.Series(indx2wrd)

    print('Reading testing files...')
    tst_vid_ft_file2 = os.listdir(tst_vfeat_folder)
    tst_vid_ft_file2 = [(tst_vfeat_folder + filename) for filename in tst_vid_ft_file2]

    test_vid_ID = [filename[:-4] for filename in tst_vid_ft_file2]

    test_video_feat_dict = {}
    for filepath in tst_vid_ft_file2:
        test_video_feat = nump.load(filepath)

        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(tst_vfeat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat

    test_video_caption = json.load(open(tst_lbl_json, 'r'))

    with tenf.Session() as sess:
        model = Seq2Seq_Model(
            rnn_size=FLAGS.rnn_size,
            numof_layers=FLAGS.num_layers,
            dimvio_feat=FLAGS.dim_video_feat,
            embed_sze=FLAGS.embedding_size,
            lr_rate=FLAGS.learning_rate,
            word_2_idx=word2indx,
            mode='train',
            max_gradnorm=FLAGS.max_gradient_norm,
            attention=FLAGS.use_attention,
            besearch=FLAGS.beam_search,
            besize=FLAGS.beam_size,
            encoder_s=FLAGS.max_encoder_steps,
            decoder_s=FLAGS.max_decoder_steps
        )
        ckpt = tenf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tenf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tenf.global_variables_initializer())

        summary_writer = tenf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()

            sampled_ID_caption = []
            for ID in vid_ID:
                sampled_caption = random.sample(video_cap[ID], 1)[0]
                sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
                sampled_video_feat = vid_ftdic[ID][sampled_video_frame]
                sampled_ID_caption.append((sampled_video_feat, sampled_caption))

            random.shuffle(sampled_ID_caption)

            for batch_start, batch_end in zip(range(0, FLAGS.sample_size, FLAGS.batch_size),
                                              range(FLAGS.batch_size, FLAGS.sample_size, FLAGS.batch_size)):
                print("%04d/%04d" % (batch_end, FLAGS.sample_size), end='\r')

                batch_sampled_ID_caption = sampled_ID_caption[batch_start: batch_end]
                batch_video_feats = [elements[0] for elements in batch_sampled_ID_caption]
                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size
                batch_captions = nump.array(["<bos> " + elements[1] for elements in batch_sampled_ID_caption])

                for index, caption in enumerate(batch_captions):
                    caption_words = caption.lower().split(" ")
                    if len(caption_words) < FLAGS.max_decoder_steps:
                        batch_captions[index] = batch_captions[index] + " <eos>"
                    else:
                        new_caption = ""
                        for i in range(FLAGS.max_decoder_steps - 1):
                            new_caption = new_caption + caption_words[i] + " "
                        batch_captions[index] = new_caption + "<eos>"

                batch_captions_words_index = []
                for caption in batch_captions:
                    words_index = []
                    for caption_words in caption.lower().split(' '):
                        if caption_words in word2indx:
                            words_index.append(word2indx[caption_words])
                        else:
                            words_index.append(word2indx['<unk>'])
                    batch_captions_words_index.append(words_index)

                batch_captions_matrix = pad_sequences(batch_captions_words_index, padding='post',
                                                      maxlen=FLAGS.max_decoder_steps)
                batch_captions_length = [len(x) for x in batch_captions_matrix]

                loss, summary = model.train(
                    sess,
                    batch_video_feats,
                    batch_video_frame,
                    batch_captions_matrix,
                    batch_captions_length)

            print()
            test_video_feat_dict = {}
            for filepath in tst_vid_ft_file2:
                test_video_feat = nump.load(filepath)

                sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
                test_video_feat = test_video_feat[sampled_video_frame]

                test_video_ID = filepath[: -4].replace(tst_vfeat_folder, "")
                test_video_feat_dict[test_video_ID] = test_video_feat

            test_video_caption = json.load(open(tst_lbl_json, 'r'))

            test_captions = []
            for batch_start, batch_end in zip(range(0, len(test_vid_ID) + FLAGS.batch_size, FLAGS.batch_size),
                                              range(FLAGS.batch_size, len(test_vid_ID) + FLAGS.batch_size,
                                                    FLAGS.batch_size)):
                print("%04d/%04d" % (batch_end, FLAGS.sample_size), end='\r')
                if batch_end < len(test_vid_ID):
                    batch_sampled_ID = nump.array(test_vid_ID[batch_start: batch_end])
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]
                else:
                    batch_sampled_ID = test_vid_ID[batch_start: batch_end]
                    for _ in range(batch_end - len(test_vid_ID)):
                        batch_sampled_ID.append(test_vid_ID[-1])
                    batch_sampled_ID = nump.array(batch_sampled_ID)
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]

                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size

                batch_caption_words_index, logi = model.infer(
                    sess,
                    batch_video_feats,
                    batch_video_frame)

                if batch_end < len(test_vid_ID):
                    batch_caption_words_index = batch_caption_words_index
                else:
                    batch_caption_words_index = batch_caption_words_index[:len(test_vid_ID) - batch_start]

                for index, test_caption_words_index in enumerate(batch_caption_words_index):

                    if FLAGS.beam_search:
                        logi = nump.array(logi).reshape(-1, FLAGS.beam_size)
                        max_logi_index = nump.argmax(nump.sum(logi, axis=0))
                        predict_list = nump.ndarray.tolist(test_caption_words_index[0, :, max_logi_index])
                        predict_seq = [indx2wrd[idx] for idx in predict_list]
                        test_caption_words = predict_seq
                    else:
                        test_caption_words_index = nump.array(test_caption_words_index).reshape(-1)
                        test_caption_words = indx2wrd_series[test_caption_words_index]
                        test_caption = ' '.join(test_caption_words)

                    test_caption = ' '.join(test_caption_words)
                    test_caption = test_caption.replace('<bos> ', '')
                    test_caption = test_caption.replace('<eos>', '')
                    test_caption = test_caption.replace(' <eos>', '')
                    test_caption = test_caption.replace('<pad> ', '')
                    test_caption = test_caption.replace(' <pad>', '')
                    test_caption = test_caption.replace(' <unk>', '')
                    test_caption = test_caption.replace('<unk> ', '')

                    if (test_caption == ""):
                        test_caption = '.'

                    if batch_sampled_ID[index] in ["klteYgdfg9A_45_53.avi", "UbmZATRFI_122_156.avi",
                                                   "wkgGxDFGGVSg_44_66.avi", "JntdfdflOF0_56_70.avi",
                                                   "tJHUH9SSDg_144_123.avi"]:
                        print(batch_sampled_ID[index], test_caption)
                    test_captions.append(test_caption)

            df = pand.DataFrame(nump.array([test_vid_ID, test_captions]).T)
            df.to_csv(out_testst_file, index=False, header=False)

            result = {}
            with open(out_testst_file, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    test_id, caption = line.split(',')
                    result[test_id] = caption

            bl = []
            for item in test_video_caption:
                score_per_video = []
                captions = [x.rstrip('.') for x in item['caption']]
                score_per_video.append(BLEU(result[item['id']], captions, True))
                bl.append(score_per_video[0])
            average = sum(bl) / len(bl)

            if (len(top_BL) < num_top_BLEU):
                top_BL.append(average)
                print("Saving model with BL: %.4f ..." % (average))
                model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            else:
                if (average > min(top_BL)):
                    top_BL.remove(min(top_BL))
                    top_BL.append(average)
                    print("Saving model with BL: %.4f ..." % (average))
                    model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            top_BL.sort(reverse=True)
            print("Top [%d] BL: " % (num_top_BLEU), ["%.4f" % x for x in top_BL])

            print("Epoch %d/%d, loss: %.6f, Avg. BL: %.6f, Elapsed time: %.2fs" % (
            epoch, FLAGS.num_epochs, loss, average, (time.time() - start_time)))

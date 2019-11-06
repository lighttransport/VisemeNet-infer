import tensorflow as tf
import numpy as np

from src.model import model
from src.utl.load_param import *
from src.utl.utl import *
import math
import time
from src.create_dataset_csv import create_dataset_csv
from src.eval_viseme import eval_viseme


def load_graph(frozen_graph_filename):
    # Load the protobuf file to unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph_def into a new Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    # Print all operators
    for op in graph.get_operations():
        if op.type == 'Placeholder':
            print(op.name)

    return graph


def test_frozen(graph, test_audio_name):

    # Output nodes
    v_cls = graph.get_tensor_by_name('net2_output/add_1:0')
    v_reg = graph.get_tensor_by_name('net2_output/add_4:0')
    jali = graph.get_tensor_by_name('net2_output/add_6:0')

    # Input nodes
    x = graph.get_tensor_by_name('input/Placeholder_1:0')
    x_face_id = graph.get_tensor_by_name('input/Placeholder_2:0')
    phase = graph.get_tensor_by_name('input/phase:0')
    dropout = graph.get_tensor_by_name('net1_shared_rnn/Placeholder:0')

    # ------------ Modification of original train_visemenet.py ----------------
    csv_test_audio = csv_dir + test_audio_name + '/'
    print(csv_test_audio)

    try_mkdir(pred_dir)

    data_dir = {'train': {}, 'test': {}}
    data_dir['test']['wav'] = open(csv_test_audio + "test/wav.csv", 'r')
    data_dir['test']['clip_len'] = open(csv_test_audio + "test/clip_len.csv", 'r')
    cv_file_len = simple_read_clip_len(data_dir['test']['clip_len'])
    print('Loading wav_raw.txt file in {:}'.format(csv_test_audio))

    train_wav_raw = np.loadtxt(csv_test_audio + 'wav_raw.csv')
    test_wav_raw = train_wav_raw

    # ============================== TRAIN SET CHUNK ITERATION ============================== #

    for key in ['train', 'test']:
        for lpw_key in data_dir[key].keys():
            data_dir[key][lpw_key].seek(0)

    print("===================== TEST/CV CHUNK - {:} ======================".format(csv_test_audio))
    eof = False
    chunk_num = 0
    chunk_size_sum = 0

    batch_size = test_wav_raw.shape[0]
    chunk_size = batch_size * batch_per_chunk_size

    with tf.compat.v1.Session(graph=graph) as sess:

        while (not eof):
            cv_data, eof = read_chunk_data(data_dir, 'test', chunk_size)
            chunk_num += 1
            chunk_size_sum += len(cv_data['wav'])

            print('Load Chunk {:d}, size {:d}, total_size {:d} ({:2.2f})'
                  .format(chunk_num, len(cv_data['wav']), chunk_size_sum, chunk_size_sum / cv_file_len))

            full_idx_array = np.arange(len(cv_data['wav']))
            # np.random.shuffle(full_idx_array)
            for next_idx in range(0, int(np.floor(len(cv_data['wav']) / batch_size))):
                batch_idx_array = full_idx_array[next_idx * batch_size: (next_idx + 1) * batch_size]
                batch_x, batch_x_face_id, batch_x_pose, batch_y_landmark, batch_y_phoneme, batch_y_lipS, batch_y_maya_param = \
                    read_next_batch_easy_from_raw(test_wav_raw, cv_data, 'face_close', batch_idx_array, batch_size, n_steps, n_input, n_landmark,
                                         n_phoneme, n_face_id)
                npClose = np.loadtxt(lpw_dir + 'saved_param/maya_close_face.txt')
                batch_x_face_id = np.tile(npClose, (batch_x_face_id.shape[0], 1))

                # Forward session
                pred_v_cls, pred_v_reg, pred_jali = sess.run(
                        [v_cls, v_reg, jali],
                        feed_dict={x: batch_x, x_face_id: batch_x_face_id,
                                   dropout: 0, phase: 0 })

                def save_output(filename, npTxt, fmt):
                    f = open(filename, 'wb')
                    np.savetxt(f, npTxt, fmt=fmt)
                    f.close()

                try_mkdir(pred_dir + test_audio_name)

                def sigmoid(x):
                    return 1/(1+np.exp(-x))
                save_output(pred_dir + test_audio_name + "/mayaparam_pred_cls.txt",
                            np.concatenate([pred_jali, sigmoid(pred_v_cls)], axis=1), '%.4f')
                save_output(pred_dir + test_audio_name + "/mayaparam_pred_reg.txt",
                            np.concatenate([pred_jali, pred_v_reg], axis=1), '%.4f')


if __name__ == '__main__':
    # Audio filename to process.
    test_audio_name = 'visemenet_intro.wav'

    # convert audio wav to network input format
    create_dataset_csv(csv_dir, test_audio_name=test_audio_name)

    # feedforward testing
    # Edit path to `visemenet_frozen.pb` as you wish
    graph = load_graph('visemenet_frozen.pb')
    test_frozen(graph=graph, test_audio_name=test_audio_name[:-4])
    print('Finish forward testing.')

    # output viseme parameter
    eval_viseme(test_audio_name[:-4])
    print('Done.')

# src/eval/seld_validator.py
import os
import codecs
import numpy as np
import pandas as pd
import tqdm

from src.eval.wav_convertor import WavConvertor
from src.eval.seld_predictor import SELDClassifier, SELDDetector
from src.eval.seld_eval_dcase2022 import all_seld_eval


class SELDValidator(object):
    def __init__(self, args, monitor_path):
        self._args = args
        self._monitor_path = monitor_path

        if self._args.val:
            self._tag = 'TMP4VAL'
        elif self._args.eval:
            self._tag = '{}_{}'.format(
                os.path.splitext(os.path.basename(self._args.eval_wav_txt))[0],
                os.path.splitext(os.path.basename(self._args.eval_model))[0][-7:])
        self._pred_directory = os.path.join(monitor_path, 'pred_{}'.format(self._tag))
        os.makedirs(self._pred_directory, exist_ok=True)

        self._wav_convertor = WavConvertor(self._args)

        txt = self._args.val_wav_txt if self._args.val else self._args.eval_wav_txt
        self._wav_path_list = pd.read_table(txt, header=None).values.tolist()

        self._wav_dict = {}
        self._duration_dict = {}
        self._label_dict = {}
        for row in tqdm.tqdm(self._wav_path_list, desc='[Val initial setup]'):
            wav_path = row[0]
            wav_pad, duration = self._wav_convertor.wav_path2wav(wav_path)
            label_pad = self._wav_convertor.wav_path2label(wav_path, duration)
            self._wav_dict[wav_path] = wav_pad
            self._duration_dict[wav_path] = duration
            self._label_dict[wav_path] = label_pad

    def validation(self, model_path):
        self._args.eval_model = model_path
        self._seld_classifier = SELDClassifier(self._args)
        self._seld_detector = SELDDetector(self._args)

        val_loss = 0
        for row in tqdm.tqdm(self._wav_path_list, desc='[Val]'):
            val_loss += self._pred_wav(row[0])
        val_loss = val_loss / len(self._wav_path_list)

        if self._args.val:
            all_test_metric = all_seld_eval(self._args, pred_directory=self._pred_directory)
        else:
            result_path = os.path.join(self._monitor_path, 'result_{}.txt'.format(self._tag))
            all_test_metric = all_seld_eval(self._args, pred_directory=self._pred_directory,
                                            result_path=result_path)
        return all_test_metric, val_loss

    def _pred_wav(self, wav_path):
        spec_pad = self._wav_convertor.wav2spec(self._wav_dict[wav_path])
        duration = self._duration_dict[wav_path]
        label_pad = self._label_dict[wav_path]

        # Dauer auf die verfuegbaren Videoframes begrenzen (aus der .npy, keine Videodatei noetig)
        npy_path = wav_path.replace('foa_dev', 'visual_features').replace('.wav', '.npy')
        n_frames = len(np.load(npy_path, mmap_mode='r'))
        duration = min(duration, n_frames / self._args.video_fps)

        self._seld_classifier.set_input(spec_pad, label_pad, wav_path)
        self._seld_detector.set_duration(duration)
        time_array = self._seld_detector.get_time_array()

        wav_loss = 0
        for index, _ in enumerate(time_array[::self._args.batch_size]):
            self._seld_classifier.receive_input(
                time_array[index * self._args.batch_size: (index + 1) * self._args.batch_size])
            self._seld_classifier.calc_output()
            wav_loss += self._seld_classifier.get_loss()
            self._seld_detector.set_minibatch_result(
                index=index, result=self._seld_classifier.get_output())
        self._seld_detector.minibatch_result2raw_output_array()
        wav_loss = wav_loss / len(time_array[::self._args.batch_size])

        for index, time in enumerate(time_array):
            self._seld_detector.detect(index=index, time=time)

        pred_path = os.path.join(
            self._pred_directory,
            '{}.csv'.format(os.path.splitext(os.path.basename(wav_path))[0]))
        self._seld_detector.save_df(pred_path)
        return wav_loss

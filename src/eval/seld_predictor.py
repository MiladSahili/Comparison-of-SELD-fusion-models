# src/eval/seld_predictor.py
import torch
import numpy as np
import cv2
import json
import pandas as pd

from src.models.midlevel.net_seld import create_net_seld
from src.losses.adpit import MSELoss_ADPIT
from dcase2022_task3_seld_metrics.SELD_evaluation_metrics import distance_between_spherical_coordinates_rad


class SELDClassifier(object):
    def __init__(self, args):                      # (1) kein object_detection
        self._args = args
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._net = create_net_seld(self._args)
        self._net.to(self._device)
        self._net.eval()
        checkpoint = torch.load(self._args.eval_model, map_location=lambda s, l: s)
        self._net.load_state_dict(checkpoint['model_state_dict'])

        self._criterion = MSELoss_ADPIT()

    def set_input(self, spec_pad, label_pad, wav_path):
        self._spec_pad = spec_pad
        self._label_pad = label_pad
        self._wav_path = wav_path
        # .npy einmal pro Clip laden (identisch zum dataloader)
        npy_path = wav_path.replace('foa_dev', 'visual_features').replace('.wav', '.npy')
        self._visual_frames = np.load(npy_path)

    def receive_input(self, time_array):
        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._frame_length = round(self._args.train_wav_length * fs / self._args.stft_hop_size) + 1

        features = np.zeros((self._args.batch_size,) + self._spec_pad[:, :, :self._frame_length].shape)
        labels = np.zeros((self._args.batch_size,) + self._label_pad[:, :, :, :self._frame_length].shape)
        videos = np.zeros((self._args.batch_size, 2, 6, 37))       # (3) feste Form statt get_dist_shape()

        for index, time in enumerate(time_array):
            frame_idx = int(time * self._frame_per_sec)
            features[index] = self._spec_pad[:, :, frame_idx: frame_idx + self._frame_length]
            labels[index] = self._label_pad[:, :, :, frame_idx: frame_idx + self._frame_length]

            # ===== (4) HIER: vorberechnete .npy laden, exakt wie im dataloader =====
            videos[index] = self._load_visual(time)

        self._input_a = torch.tensor(features, dtype=torch.float).to(self._device)
        self._input_v = torch.tensor(videos, dtype=torch.float).to(self._device)
        self._label = torch.tensor(labels, dtype=torch.float).to(self._device)

    def _load_visual(self, start_sec):
        frame_idx = int(start_sec * self._args.video_fps)
        frame_idx = min(frame_idx, len(self._visual_frames) - 1)
        return self._visual_frames[frame_idx].astype(np.float32)

    def calc_output(self):
        self._output = self._net(self._input_a, self._input_v)

    def get_output(self):
        hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)
        cut_frame = int(np.floor((self._frame_length - hop_frame) / 2))
        output = self._output.cpu().detach().numpy()
        self._output = 0
        return output[:, :, :, :, cut_frame: cut_frame + hop_frame]

    def get_loss(self):
        loss = self._criterion(self._output, self._label).cpu().detach().numpy()
        return loss

class SELDDetector(object):
    def __init__(self, args):
        self._args = args
        with open(args.threshold_config, 'r') as f:
            threshold_config = json.load(f)
        self._thresh_bin = threshold_config['threshold_presence']
        self._thresh_dist = threshold_config['threshold_unification']

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)

    def set_duration(self, duration):
        hop = self._args.eval_wav_hop_length
        if (duration % hop == 0) or (np.abs((duration % hop) - hop) < 1e-10):
            self._time_array = np.arange(0, duration + hop, hop)
        else:
            self._time_array = np.arange(0, duration, hop)

        self._rows = []                                    # statt pd.DataFrame()
        self._minibatch_result = np.zeros((
            len(self._time_array) + self._args.batch_size,
            3, 3, self._args.class_num, self._hop_frame))
        self._raw_output_array = np.zeros((
            3, 3, self._args.class_num,
            len(self._time_array) * self._hop_frame))

    def get_time_array(self):
        return self._time_array

    def set_minibatch_result(self, index, result):
        self._minibatch_result[
            index * self._args.batch_size: (index + 1) * self._args.batch_size] = result

    def minibatch_result2raw_output_array(self):
        array_len = self._minibatch_result.shape[0] * self._hop_frame
        result_array = np.zeros((3, 3, self._args.class_num, array_len))
        for index, each_result in enumerate(self._minibatch_result):
            result_array[:, :, :, index * self._hop_frame: (index + 1) * self._hop_frame] = each_result
        self._raw_output_array = result_array[:, :, :, : len(self._time_array) * self._hop_frame]

    def detect(self, index, time):
        s = index * self._hop_frame
        e = (index + 1) * self._hop_frame
        for event_class in range(self._args.class_num):
            r = self._raw_output_array
            self._each_detect(
                time, event_class,
                r[0, 0, event_class, s:e], r[0, 1, event_class, s:e], r[0, 2, event_class, s:e],
                r[1, 0, event_class, s:e], r[1, 1, event_class, s:e], r[1, 2, event_class, s:e],
                r[2, 0, event_class, s:e], r[2, 1, event_class, s:e], r[2, 2, event_class, s:e])

    def _each_detect(self, time, event_class, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        azi0, ele0, bin0 = self._xyz2azi_ele_bin(x0, y0, z0)
        azi1, ele1, bin1 = self._xyz2azi_ele_bin(x1, y1, z1)
        azi2, ele2, bin2 = self._xyz2azi_ele_bin(x2, y2, z2)

        frame_per_sec4csv = 10
        hop_frame4csv = int(self._hop_frame / (self._frame_per_sec / frame_per_sec4csv))
        rad2deg = 180 / np.pi

        for csv_idx, frame in enumerate(
                range(int(time * frame_per_sec4csv),
                      int(time * frame_per_sec4csv) + hop_frame4csv)):
            csv2net = int(self._frame_per_sec / frame_per_sec4csv)
            a, b = csv_idx * csv2net, (csv_idx + 1) * csv2net
            azi_m0, ele_m0, bin_m0 = self._azi_ele_bin2mean(azi0, ele0, bin0, a, b, event_class)
            azi_m1, ele_m1, bin_m1 = self._azi_ele_bin2mean(azi1, ele1, bin1, a, b, event_class)
            azi_m2, ele_m2, bin_m2 = self._azi_ele_bin2mean(azi2, ele2, bin2, a, b, event_class)

            f01 = self._similar_location(azi_m0, ele_m0, azi_m1, ele_m1)
            f12 = self._similar_location(azi_m1, ele_m1, azi_m2, ele_m2)
            f20 = self._similar_location(azi_m2, ele_m2, azi_m0, ele_m0)

            if f01 + f12 + f20 == 0:
                if bin_m0 > self._thresh_bin[event_class]:
                    self._rows.append((frame, event_class, azi_m0 * rad2deg, ele_m0 * rad2deg))
                if bin_m1 > self._thresh_bin[event_class]:
                    self._rows.append((frame, event_class, azi_m1 * rad2deg, ele_m1 * rad2deg))
                if bin_m2 > self._thresh_bin[event_class]:
                    self._rows.append((frame, event_class, azi_m2 * rad2deg, ele_m2 * rad2deg))
            elif f01 + f12 + f20 == 1:
                if f01:
                    if bin_m2 > self._thresh_bin[event_class]:
                        self._rows.append((frame, event_class, azi_m2 * rad2deg, ele_m2 * rad2deg))
                    azi_u = (azi_m0 * bin_m0 + azi_m1 * bin_m1) / (bin_m0 + bin_m1)
                    ele_u = (ele_m0 * bin_m0 + ele_m1 * bin_m1) / (bin_m0 + bin_m1)
                    self._rows.append((frame, event_class, azi_u * rad2deg, ele_u * rad2deg))
                elif f12:
                    if bin_m0 > self._thresh_bin[event_class]:
                        self._rows.append((frame, event_class, azi_m0 * rad2deg, ele_m0 * rad2deg))
                    azi_u = (azi_m1 * bin_m1 + azi_m2 * bin_m2) / (bin_m1 + bin_m2)
                    ele_u = (ele_m1 * bin_m1 + ele_m2 * bin_m2) / (bin_m1 + bin_m2)
                    self._rows.append((frame, event_class, azi_u * rad2deg, ele_u * rad2deg))
                elif f20:
                    if bin_m1 > self._thresh_bin[event_class]:
                        self._rows.append((frame, event_class, azi_m1 * rad2deg, ele_m1 * rad2deg))
                    azi_u = (azi_m2 * bin_m2 + azi_m0 * bin_m0) / (bin_m2 + bin_m0)
                    ele_u = (ele_m2 * bin_m2 + ele_m0 * bin_m0) / (bin_m2 + bin_m0)
                    self._rows.append((frame, event_class, azi_u * rad2deg, ele_u * rad2deg))
            else:
                denom = bin_m0 + bin_m1 + bin_m2
                azi_u = (azi_m0 * bin_m0 + azi_m1 * bin_m1 + azi_m2 * bin_m2) / denom
                ele_u = (ele_m0 * bin_m0 + ele_m1 * bin_m1 + ele_m2 * bin_m2) / denom
                self._rows.append((frame, event_class, azi_u * rad2deg, ele_u * rad2deg))

    def _xyz2azi_ele_bin(self, x, y, z):
        azi = np.arctan2(y, x)
        ele = np.arctan2(z, np.sqrt(x**2 + y**2))
        b = np.sqrt(x**2 + y**2 + z**2)
        b[b > 1] = 1
        return azi, ele, b

    def _azi_ele_bin2mean(self, azi, ele, b, idx_start, idx_end, event_class):
        bin_mean = np.mean(b[idx_start: idx_end])
        azi_mean, ele_mean = None, None
        if bin_mean > self._thresh_bin[event_class]:
            w = b[idx_start: idx_end]
            azi_mean = np.sum(w * azi[idx_start: idx_end]) / np.sum(w)
            ele_mean = np.sum(w * ele[idx_start: idx_end]) / np.sum(w)
        return azi_mean, ele_mean, bin_mean

    def _similar_location(self, azi0, ele0, azi1, ele1):
        if (azi0 is not None) and (azi1 is not None):
            if distance_between_spherical_coordinates_rad(azi0, ele0, azi1, ele1) < self._thresh_dist:
                return 1
        return 0

    def save_df(self, pred_path):
        self._df = pd.DataFrame(self._rows)
        if not self._df.empty:
            self._df = self._df.sort_values(0)
        self._df.to_csv(pred_path, sep=',', index=False, header=False)
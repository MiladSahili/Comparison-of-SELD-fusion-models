# src/eval/seld_eval_dcase2022.py
# Basiert auf Sony Group Corporation, audio-visual-seld-dcase2023
# Angepasst: explizite _dev-Pfadersetzung, Debug-Ausgabe des Referenzordners

import os
import codecs

from dcase2022_task3_seld_metrics import parameters, cls_compute_seld_results


def all_seld_eval(args, pred_directory, result_path=None):
    if args.eval:
        with open(args.eval_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
    elif args.val:
        with open(args.val_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
    wav_dir = os.path.dirname(wav_file_list[0])

    # explizit mit _dev, damit nicht versehentlich Teile des Projektpfads ersetzt werden
    ref_desc_files = wav_dir.replace("foa_dev", "metadata_dev").replace("mic_dev", "metadata_dev")
    ref_files_folder = os.path.dirname(ref_desc_files)
    print("[eval] ref_files_folder:", ref_files_folder)

    pred_output_format_files = pred_directory

    params = parameters.get_params()
    score_obj = cls_compute_seld_results.ComputeSELDResults(
        params, ref_files_folder=ref_files_folder)
    er20, f20, le, lr, seld_err, classwise_test_scr = score_obj.get_SELD_Results(
        pred_output_format_files)

    print('SELD scores')
    print('All\tER\tF\tLE\tLR\tSELD_error')
    print('All\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
        er20, f20, le, lr, seld_err))

    if params['average'] == 'macro':
        print('Class-wise results')
        print('Class\tER\tF\tLE\tLR\tSELD_error')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                cls_cnt,
                classwise_test_scr[0][cls_cnt],
                classwise_test_scr[1][cls_cnt],
                classwise_test_scr[2][cls_cnt],
                classwise_test_scr[3][cls_cnt],
                classwise_test_scr[4][cls_cnt]))

    if args.eval and result_path is not None:
        with codecs.open(result_path, 'w', 'utf-8') as f:
            print('SELD scores', file=f)
            print('All\tER\tF\tLE\tLR\tSELD_error', file=f)
            print('All\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                er20, f20, le, lr, seld_err), file=f)
            if params['average'] == 'macro':
                print('Class-wise results', file=f)
                print('Class\tER\tF\tLE\tLR\tSELD_error', file=f)
                for cls_cnt in range(params['unique_classes']):
                    print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                        cls_cnt,
                        classwise_test_scr[0][cls_cnt],
                        classwise_test_scr[1][cls_cnt],
                        classwise_test_scr[2][cls_cnt],
                        classwise_test_scr[3][cls_cnt],
                        classwise_test_scr[4][cls_cnt]), file=f)

    return er20, f20, le, lr, seld_err
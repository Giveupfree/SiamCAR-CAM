from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory, OTBDataset, UAVDataset, LaSOTDataset, GOT10kDataset,\
    VOTDataset, NFSDataset, VOTLTDataset
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from pysot.tracker.siamcar_tracker import SiamCARTracker

import optuna
import logging


def rename(tune_dir,dataset,video,model_name,result):
    oldname = os.path.join(tune_dir, dataset, video, model_name)
    newname = os.path.join(tune_dir, dataset, video, "{:.3f}".format(result) + model_name)
    os.rename(oldname, newname)

def eval(dataset, tracker_name):
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    # root = os.path.join(root, dataset)
    tracker_dir = "./"
    trackers = [tracker_name]
    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'GOT' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'NFS' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao
    elif 'VOT2018-LT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=False)

    return 0

# fitness function
def objective(trial):
    # different params
    WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.000, 1.000)
    PENALTY_K = trial.suggest_uniform('penalty_k', 0.000, 1.000)
    LR = trial.suggest_uniform('scale_lr', 0.000, 1.000)
    # cfg.TRACK.WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.45, 0.48)
    # cfg.TRACK.PENALTY_K = trial.suggest_uniform('penalty_k', 0.07, 0.11)
    # cfg.TRACK.LR = trial.suggest_uniform('scale_lr', 0.43, 0.49)
    hp = {'lr': LR, 'penalty_k': PENALTY_K, 'window_lr': WINDOW_INFLUENCE}
    # rebuild tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    tracker_name = os.path.join('tune_results', args.dataset, model_name, model_name + \
                    '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
                    '_pk-{:.3f}'.format(PENALTY_K) + \
                    '_lr-{:.3f}'.format(LR))
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img, hp)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                # if idx == 0:
                #     cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(tracker_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        eao = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, EAO: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, PENALTY_K, LR, eao)
        logging.getLogger().info(info)
        print(info)
        rename('tune_results', args.dataset, model_name, model_name + \
                    '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
                    '_pk-{:.3f}'.format(PENALTY_K) + \
                    '_lr-{:.3f}'.format(LR), eao)
        return eao
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    # scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img,hp)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    # scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                # if idx == 0:
                #     cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                # result_path = os.path.join(video_path,
                #                            '{}_001_confidence.value'.format(video.name))
                # with open(result_path, 'w') as f:
                #     for x in scores:
                #         f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif "GOT" in args.dataset:
                video_path = os.path.join(model_path, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))
        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
            model_name, WINDOW_INFLUENCE, PENALTY_K, LR, auc)
        logging.getLogger().info(info)
        print(info)
        rename('tune_results', args.dataset, model_name, model_name + \
               '_wi-{:.3f}'.format(WINDOW_INFLUENCE) + \
               '_pk-{:.3f}'.format(PENALTY_K) + \
               '_lr-{:.3f}'.format(LR), auc)
        return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tuning for SiamPD')
    parser.add_argument('--dataset_root', default='/path/to/dataset', type=str, help='datasetdir')
    parser.add_argument('--dataset', default='UAV123', type=str, help='dataset')
    parser.add_argument('--config', default='./experiments/SiamPD_r50/config.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='./snapshot/model_general.pth', type=str,
                        help='snapshot of models to eval')
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")
    parser.add_argument('--num', '-n', default=1, type=int,
                        help='number of thread to eval')
    args = parser.parse_args()

    torch.set_num_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # load config
    cfg.merge_from_file(args.config)
    dataset_root = os.path.join(args.dataset_root, args.dataset)
    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    # Eval dataset
    if args.dataset == "GOT-10k":
        root = os.path.join(args.dataset_dir, "GOT10K", "test")
    else:
        root = os.path.join(args.dataset_dir, args.dataset)
    if 'OTB' in args.dataset:
        dataset_eval = OTBDataset(args.dataset, root)
    elif 'LaSOT' in args.dataset:
        dataset_eval = LaSOTDataset(args.dataset, root)
    elif 'GOT-10k' == args.dataset:
        dataset_eval = GOT10kDataset(args.dataset, root)
    elif 'UAV' in args.dataset:
        dataset_eval = UAVDataset(args.dataset, root)
    elif 'NFS' in args.dataset:
        dataset_eval = NFSDataset(args.dataset, root)
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset_eval = VOTDataset(args.dataset, root)
    elif 'VOT2018-LT' == args.dataset:
        dataset_eval = VOTLTDataset(args.dataset, root)
    
    tune_result = os.path.join('tune_results', args.dataset)
    if not os.path.isdir(tune_result):
                os.makedirs(tune_result)
    log_path = os.path.join(tune_result, (args.snapshot).split('/')[-1].split('.')[0] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(study_name=args.dataset,
                                direction='maximize',
                                storage='sqlite:///{}.db'.format(args.dataset),
                                load_if_exists=True)
    study.optimize(objective, n_trials=500)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))



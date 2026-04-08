import os
import cv2
import py_sod_metrics
import argparse
import metrics as M
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True,
                    help="path to the prediction results")
parser.add_argument("--pred_path", type=str, required=True,
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str, required=True,
                    help="path to the ground truth masks")
args = parser.parse_args()

# 使用 py_sod_metrics 初始化指标
FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()
MSIOU = py_sod_metrics.MSIoU()

# 初始化 FmeasureV2 及其相关指标
sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
        "rec": py_sod_metrics.RecallHandler(**sample_gray),
        "fpr": py_sod_metrics.FPRHandler(**sample_gray),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
        "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
        "ber": py_sod_metrics.BERHandler(**sample_gray),
        "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
        "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
        "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
        "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
        "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
        "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
        "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
        "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
    }
)

pred_root = args.pred_path
mask_root = args.gt_path
mask_name_list = sorted(os.listdir(mask_root))
for i, mask_name in enumerate(mask_name_list):
    print(f"[{i}] Processing {mask_name}...")
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    FMv2.step(pred=pred, gt=mask)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
fmv2 = FMv2.get_results()

# 计算新添加的指标
maxFm = fm["curve"].max()
maxEm = em["curve"].max() if em["curve"] is not None else None
fnr = None  # 由于不清楚 FNR 具体计算方式，这里先设为 None，你可以根据实际情况修改

curr_results = {
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "meaniou": fmv2["iou"]["dynamic"].mean(),
    'Smeasure': sm,
    "wFmeasure": wfm,
    "adpFm": fm["adp"],
    "meanEm": em["curve"].mean(),
    "MAE": mae,
    "maxFm": maxFm,
    "maxEm": maxEm,
    "fnr": fnr
}

print(args.dataset_name)
print("mDice:       ", format(curr_results['meandice'], '.4f'))
print("mIoU:        ", format(curr_results['meaniou'], '.4f'))
print("S_{alpha}:   ", format(curr_results['Smeasure'], '.4f'))
print("F^{w}_{beta}:", format(curr_results['wFmeasure'], '.4f'))
print("F_{beta}:    ", format(curr_results['adpFm'], '.4f'))
print("E_{phi}:     ", format(curr_results['meanEm'], '.4f'))
print("MAE:         ", format(curr_results['MAE'], '.4f'))
print("maxFm:       ", format(curr_results['maxFm'], '.4f')) if curr_results['maxFm'] is not None else print("maxFm:       -")
print("maxEm:       ", format(curr_results['maxEm'], '.4f')) if curr_results['maxEm'] is not None else print("maxEm:       -")
print("fnr:         ", format(curr_results['fnr'], '.4f')) if curr_results['fnr'] is not None else print("fnr:         -")
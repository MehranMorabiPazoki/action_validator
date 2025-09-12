from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
import logging
import os

logger = logging.getLogger("mmaction_inference")

config_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
label_file = 'tools/data/kinetics/label_map_k400.txt'

# Initialize once
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

# Load labels once
with open(label_file) as f:
    labels = [x.strip() for x in f.readlines()]


def mmaction_inference(video_dict:dict,delete_videos:bool = True):
    """
    Run inference on multiple camera videos and aggregate results.

    Args:
        video_dict (dict): {camera_id: video_path}

    Returns:
        results (list): top-5 [(label, score)]
    """
    if not video_dict:
        raise ValueError("No video input provided to mmaction_inference")

    logger.info("Running MMAction2 inference on %d cameras", len(video_dict))

    all_scores = []

    for cam_id, video_path in video_dict.items():
        logger.info("🔍 Inference on camera %s → %s", cam_id, video_path)

        pred_result = inference_recognizer(model, video_path)
        pred_scores = np.array(pred_result.pred_score.tolist())
        all_scores.append(pred_scores)

    # --- Aggregate across cameras ---
    mean_scores = np.mean(all_scores, axis=0)

    # --- Top-5 labels ---
    score_tuples = tuple(zip(range(len(mean_scores)), mean_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    results = [(labels[k[0]], float(k[1])) for k in top5_label]

    logger.info('The top-5 labels with aggregated scores are:')
    for lbl, score in results:
        logger.info(f"{lbl}: {score:.4f}")

    # --- Delete video file after inference ---
    if delete_videos:
        for cam_id, video_path in video_dict.items():
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Deleted temporary file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {video_path}: {e}")
                    
    return results

from operator import itemgetter
from mmaction.apis import inference_recognizer, init_recognizer
from mmcv.video import VideoReader
from mmcv.transforms import Compose
from mmpose.utils import register_all_modules
from mmpose.apis import init_model
from mmengine import Config
import numpy as np
from tqdm import tqdm
import torch
import logging
import os
import json

logger = logging.getLogger("mmaction_inference")

register_all_modules()


mmpose_config_path = './models/pose/config.py'
mmpose_checkpoint_path = './models/pose/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth'

device = "cuda:0"
mmpose_model = init_model(mmpose_config_path, mmpose_checkpoint_path, device=device)

# Define your val pipeline
val_pipeline = [
    # dict(type='LoadImageFromNDArray'),
    dict(type='BottomupResize', input_size=(640, 640), pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('ori_shape', 'img_shape', 'input_size', 'input_center', 'input_scale')
    )
]
pipeline = Compose(val_pipeline)


mmaction_config_path = './models/mmaction/20250924_133942/vis_data/config.py'
mmaction_checkpoint_path = './models/mmaction/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb_20220815-af2f119a.pth' # can be a local path

# Initialize once
mmaction_model = init_recognizer(mmaction_config_path, mmaction_checkpoint_path, device=device)  # device can be 'cuda:0'


with open(file='action.json',mode='r+') as file:
    actions = json.load(file)

def pad_pose_sequence(pose_results, max_person=5, num_joints=17):
    """
    Convert a variable-person pose sequence from MMPose into a fixed-size tensor.
    Args:
        pose_results: list of pose outputs (each frame = pred_instances)
        max_person: maximum number of people per frame (others truncated)
        num_joints: number of keypoints (default = 17 for COCO)
    Returns:
        keypoints: np.ndarray (max_person, num_frames, num_joints, 2)
        keypoints_visible: np.ndarray (max_person, num_frames, num_joints)
    """
    num_frames = len(pose_results)
    keypoints = np.zeros((max_person, num_frames, num_joints, 2), dtype=np.float32)
    keypoints_visible = np.zeros((max_person, num_frames, num_joints), dtype=np.float32)

    for t, ps in enumerate(pose_results):
        # Extract per-frame keypoints and visibility
        kpts = ps.pred_instances.keypoints
        vis = ps.pred_instances.keypoints_visible

        num_person = kpts.shape[0]
        n = min(num_person, max_person)

        # Optionally, sort persons by mean visibility score to keep best ones
        mean_scores = vis.mean(axis=1)
        order = np.argsort(mean_scores)[::-1][:n]

        keypoints[:n, t, :, :] = kpts[order, :, :]
        keypoints_visible[:n, t, :] = vis[order, :]

    return keypoints, keypoints_visible

def run_pose_inference(video_path):
    video_reader = VideoReader()
    batch_size = 32
    batch_input = []
    batch_data_sample = []
    pose_results = []
    height = video_reader.height
    width = video_reader.width

    for i, frame in tqdm(enumerate(video_reader)):

        data = dict(
            img=frame,
            ori_shape=(height, width)   # <-- required by BottomupResize
        )
        data = pipeline(data)
        batch_input.append(data['inputs'])
        batch_data_sample.append(data['data_samples'])
        # Run batched inference
        if len(batch_input) == batch_size or i==len(video_reader)-1 :
            
            batch_input_tensor = torch.stack(batch_input).float().to(device)
            
            with torch.no_grad():
                outputs = mmpose_model.predict(batch_input_tensor,batch_data_sample)
            # for j, out in enumerate(outputs):
            #     print(f"Frame {i - batch_size + j + 1}: {out.pred_instances.keypoints.shape}")
            pose_results.extend(outputs)
            batch_input = []
            batch_data_sample = []


    keypoints, keypoints_visible = pad_pose_sequence(pose_results)

    data = {
            'keypoint': keypoints,          # (1, T, K, 2)
            'keypoint_score': keypoints_visible,
            'total_frames' : len(pose_results),
            'img_shape' : (height, width)
        }
    return data


def mmaction_inference(video_dict:dict,delete_videos:bool = False):
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
        pose_data = run_pose_inference(video_path=video_path)
        
        mmaction_result = inference_recognizer(mmaction_model, pose_data)

        pred_label = mmaction_result.pred_label.detach().cpu().numpy()
        pred_scores = mmaction_result.pred_score.detach().cpu().numpy()
        all_scores.append(pred_scores)

    # --- Aggregate across cameras ---
    mean_scores = np.mean(all_scores, axis=0)

    # --- Top-5 labels ---
    score_tuples = tuple(zip(range(len(mean_scores)), mean_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    results = [(actions[f'A{k[0]+1}'], float(k[1])) for k in top5_label]

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

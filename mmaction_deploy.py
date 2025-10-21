from mmaction.apis import inference_recognizer, init_recognizer
from mmcv.video import VideoReader
from mmcv.transforms import Compose
from mmpose.utils import register_all_modules
from mmpose.apis import init_model
from mmengine import Config
import numpy as np
from tqdm import tqdm
import torch


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



register_all_modules()
cfg_path = './work_dirs/slowonly_r50_8xb16_u48_240e_ntu60_xsub_limb_custom/yoloxpose_s_8xb32-300e_coco-640.py'
ckpt_path = './work_dirs/slowonly_r50_8xb16_u48_240e_ntu60_xsub_limb_custom/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth'

device = "cuda"
mmpose_model = init_model(cfg_path, ckpt_path, device=device)

video_path = "./demo/demo_skeleton.mp4"

video_reader = VideoReader(video_path)


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


# Read video
video_reader = VideoReader(video_path)
batch_size = 32
batch_input = []
batch_data_sample = []
pose_results = []
height = video_reader.height
width = video_reader.width

max_person = 3
min_person = 10
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

pose_sequence = []
pose_scores = []


keypoints, keypoints_visible = pad_pose_sequence(pose_results)

data = {
    'keypoint': keypoints,          # (1, T, K, 2)
    'keypoint_score': keypoints_visible,
    'total_frames' : len(pose_results),
    'img_shape' : (height, width)
}



config_path = './work_dirs/slowonly_r50_8xb16_u48_240e_ntu60_xsub_limb_custom/20250924_133942/vis_data/config.py'
checkpoint_path = './work_dirs/slowonly_r50_8xb16_u48_240e_ntu60_xsub_limb_custom/best_acc_top1_epoch_1.pth' # can be a local path
actions = {
    "A1": "drink water",
    "A2": "eat meal/snack",
    "A3": "brushing teeth",
    "A4": "brushing hair",
    "A5": "drop",
    "A6": "pickup",
    "A7": "throw",
    "A8": "sitting down",
    "A9": "standing up (from sitting position)",
    "A10": "clapping",
    "A11": "reading",
    "A12": "writing",
    "A13": "tear up paper",
    "A14": "wear jacket",
    "A15": "take off jacket",
    "A16": "wear a shoe",
    "A17": "take off a shoe",
    "A18": "wear on glasses",
    "A19": "take off glasses",
    "A20": "put on a hat/cap",
    "A21": "take off a hat/cap",
    "A22": "cheer up",
    "A23": "hand waving",
    "A24": "kicking something",
    "A25": "reach into pocket",
    "A26": "hopping (one foot jumping)",
    "A27": "jump up",
    "A28": "make a phone call/answer phone",
    "A29": "playing with phone/tablet",
    "A30": "typing on a keyboard",
    "A31": "pointing to something with finger",
    "A32": "taking a selfie",
    "A33": "check time (from watch)",
    "A34": "rub two hands together",
    "A35": "nod head/bow",
    "A36": "shake head",
    "A37": "wipe face",
    "A38": "salute",
    "A39": "put the palms together",
    "A40": "cross hands in front (say stop)",
    "A41": "sneeze/cough",
    "A42": "staggering",
    "A43": "falling",
    "A44": "touch head (headache)",
    "A45": "touch chest (stomachache/heart pain)",
    "A46": "touch back (backache)",
    "A47": "touch neck (neckache)",
    "A48": "nausea or vomiting condition",
    "A49": "use a fan (with hand or paper)/feeling warm",
    "A50": "punching/slapping other person",
    "A51": "kicking other person",
    "A52": "pushing other person",
    "A53": "pat on back of other person",
    "A54": "point finger at the other person",
    "A55": "hugging other person",
    "A56": "giving something to other person",
    "A57": "touch other person's pocket",
    "A58": "handshaking",
    "A59": "walking towards each other",
    "A60": "walking apart from each other",
    "A61": "put on headphone",
    "A62": "take off headphone",
    "A63": "shoot at the basket",
    "A64": "bounce ball",
    "A65": "tennis bat swing",
    "A66": "juggling table tennis balls",
    "A67": "hush (quiet)",
    "A68": "flick hair",
    "A69": "thumb up",
    "A70": "thumb down",
    "A71": "make ok sign",
    "A72": "make victory sign",
    "A73": "staple book",
    "A74": "counting money",
    "A75": "cutting nails",
    "A76": "cutting paper (using scissors)",
    "A77": "snapping fingers",
    "A78": "open bottle",
    "A79": "sniff (smell)",
    "A80": "squat down",
    "A81": "toss a coin",
    "A82": "fold paper",
    "A83": "ball up paper",
    "A84": "play magic cube",
    "A85": "apply cream on face",
    "A86": "apply cream on hand back",
    "A87": "put on bag",
    "A88": "take off bag",
    "A89": "put something into a bag",
    "A90": "take something out of a bag",
    "A91": "open a box",
    "A92": "move heavy objects",
    "A93": "shake fist",
    "A94": "throw up cap/hat",
    "A95": "hands up (both hands)",
    "A96": "cross arms",
    "A97": "arm circles",
    "A98": "arm swings",
    "A99": "running on the spot",
    "A100": "butt kicks (kick backward)",
    "A101": "cross toe touch",
    "A102": "side kick",
    "A103": "yawn",
    "A104": "stretch oneself",
    "A105": "blow nose",
    "A106": "hit other person with something",
    "A107": "wield knife towards other person",
    "A108": "knock over other person (hit with body)",
    "A109": "grab other person’s stuff",
    "A110": "shoot at other person with a gun",
    "A111": "step on foot",
    "A112": "high-five",
    "A113": "cheers and drink",
    "A114": "carry something with other person",
    "A115": "take a photo of other person",
    "A116": "follow other person",
    "A117": "whisper in other person’s ear",
    "A118": "exchange things with other person",
    "A119": "support somebody with hand",
    "A120": "finger-guessing game (playing rock-paper-scissors)"
}

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cuda:0'
# test a single image
result = inference_recognizer(model, data)
pred_label = result.pred_label.detach().cpu().numpy()
pred_scores = result.pred_score.detach().cpu().numpy()
description = actions[f'A{pred_label[0]+1}']
print(pred_label,pred_scores,description)
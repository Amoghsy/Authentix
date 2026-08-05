"""
face_detector.py

This module contains the FaceDetector and supporting components for face detection and
tracking. It implements the VGG-backed S3FD (Single Shot Scale-invariant Face Detector)
architecture, box decoding, non-maximum suppression (NMS), and contiguous track linkage
with linear interpolation for missing frames.

If sfd_face.pth weights are not found locally, they are automatically downloaded
from Hugging Face to the checkpoints folder.
"""

import logging
from pathlib import Path
import sys
import urllib.request
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config


class FaceDetectionError(Exception):
    """Raised when face detection or model loading fails."""
    pass


class L2Norm(nn.Module):
    """
    L2 Normalization layer used in S3FD to normalize channel activations
    across specific VGG feature map layers.
    """
    def __init__(self, n_channels: int, scale: float = 1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(self.n_channels).fill_(self.scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class s3fd(nn.Module):
    """
    PyTorch S3FD network architecture modularized to match standard pre-trained checkpoint keys.
    """
    def __init__(self):
        super(s3fd, self).__init__()
        
        # VGG backbone structure matching state_dict keys (e.g. vgg.0.weight, etc.)
        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # 0
            nn.ReLU(inplace=True),                                   # 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),   # 2
            nn.ReLU(inplace=True),                                   # 3
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 4
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 5
            nn.ReLU(inplace=True),                                   # 6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 7
            nn.ReLU(inplace=True),                                   # 8
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 9
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 10
            nn.ReLU(inplace=True),                                   # 11
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 12
            nn.ReLU(inplace=True),                                   # 13
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 14
            nn.ReLU(inplace=True),                                   # 15
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 16
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 17
            nn.ReLU(inplace=True),                                   # 18
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 19
            nn.ReLU(inplace=True),                                   # 20
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 21
            nn.ReLU(inplace=True),                                   # 22
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 23
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 24
            nn.ReLU(inplace=True),                                   # 25
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 26
            nn.ReLU(inplace=True),                                   # 27
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 28
            nn.ReLU(inplace=True),                                   # 29
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 30
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3),# 31
            nn.ReLU(inplace=True),                                   # 32
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),# 33
            nn.ReLU(inplace=True),                                   # 34
        ])
        
        # Norm layers matching state_dict keys
        self.L2Norm3_3 = L2Norm(256, scale=10)
        self.L2Norm4_3 = L2Norm(512, scale=8)
        self.L2Norm5_3 = L2Norm(512, scale=5)
        
        # Extras layers matching state_dict keys
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0), # 0
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 1
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),  # 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 3
        ])
        
        # Loc heads matching state_dict keys
        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1),
        ])
        
        # Conf heads matching state_dict keys
        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1),  # cls1 has 4 channels for maxout
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Block 1
        h = self.vgg[1](self.vgg[0](x))
        h = self.vgg[3](self.vgg[2](h))
        h = self.vgg[4](h)

        # Block 2
        h = self.vgg[6](self.vgg[5](h))
        h = self.vgg[8](self.vgg[7](h))
        h = self.vgg[9](h)

        # Block 3
        h = self.vgg[11](self.vgg[10](h))
        h = self.vgg[13](self.vgg[12](h))
        h = self.vgg[15](self.vgg[14](h))
        f3_3 = h
        h = self.vgg[16](h)

        # Block 4
        h = self.vgg[18](self.vgg[17](h))
        h = self.vgg[20](self.vgg[19](h))
        h = self.vgg[22](self.vgg[21](h))
        f4_3 = h
        h = self.vgg[23](h)

        # Block 5
        h = self.vgg[25](self.vgg[24](h))
        h = self.vgg[27](self.vgg[26](h))
        h = self.vgg[29](self.vgg[28](h))
        f5_3 = h
        h = self.vgg[30](h)

        # FC layers (treated as conv layers)
        h = self.vgg[32](self.vgg[31](h))
        h = self.vgg[34](self.vgg[33](h))
        ffc7 = h

        # Extra blocks
        h = F.relu(self.extras[0](ffc7), inplace=True)
        f6_2 = F.relu(self.extras[1](h), inplace=True)
        
        h = F.relu(self.extras[2](f6_2), inplace=True)
        f7_2 = F.relu(self.extras[3](h), inplace=True)

        # Normalization
        f3_3 = self.L2Norm3_3(f3_3)
        f4_3 = self.L2Norm4_3(f4_3)
        f5_3 = self.L2Norm5_3(f5_3)

        # Predictions
        cls1 = self.conf[0](f3_3)
        reg1 = self.loc[0](f3_3)
        cls2 = self.conf[1](f4_3)
        reg2 = self.loc[1](f4_3)
        cls3 = self.conf[2](f5_3)
        reg3 = self.loc[2](f5_3)
        cls4 = self.conf[3](ffc7)
        reg4 = self.loc[3](ffc7)
        cls5 = self.conf[4](f6_2)
        reg5 = self.loc[4](f6_2)
        cls6 = self.conf[5](f7_2)
        reg6 = self.loc[5](f7_2)

        # Maxout background label for cls1
        chunk = torch.chunk(cls1, 4, dim=1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


def decode_boxes(loc: np.ndarray, priors: np.ndarray, variances: List[float]) -> np.ndarray:
    """
    Decodes predicted coordinate offsets using prior anchors.
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    """
    Performs standard Non-Maximum Suppression (NMS) on bounding boxes.
    """
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class FaceDetector:
    """
    S3FD-based face detector. Handles automatic weight retrieval, batched image inference,
    NMS box decoding, and multiple speaker trajectory tracking.
    """
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.device = torch.device(self.config.models.device)
        self.weights_path = self.config.paths.checkpoints_dir / self.config.models.sfd_weights_name
        
        self._ensure_weights()
        self._init_model()

    def _ensure_weights(self) -> None:
        """Downloads pretrained SFD weights from Hugging Face if not present."""
        if self.weights_path.exists():
            self.logger.info(f"Loaded existing face detector weights from: {self.weights_path}")
            return
            
        self.config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        url = self.config.models.sfd_weights_url
        self.logger.warning(f"SFD face detector weights not found. Downloading from Hugging Face: {url}")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="sfd_face.pth") as t:
                urllib.request.urlretrieve(
                    url,
                    filename=str(self.weights_path),
                    reporthook=t.update_to
                )
            self.logger.info("Weights downloaded successfully.")
        except Exception as e:
            if self.weights_path.exists():
                self.weights_path.unlink()
            raise FaceDetectionError(f"Failed to download SFD weights from {url}. Error: {e}")

    def _init_model(self) -> None:
        """Initializes the S3FD network and loads weights."""
        try:
            self.net = s3fd()
            state_dict = torch.load(self.weights_path, map_location="cpu")
            self.net.load_state_dict(state_dict)
            self.net.to(self.device)
            self.net.eval()
            self.logger.info(f"Initialized S3FD detector model on {self.device}")
        except Exception as e:
            raise FaceDetectionError(f"Failed to initialize S3FD model: {e}")

    def _get_predictions(self, olist: List[np.ndarray], batch_size: int, conf_threshold: float) -> List[np.ndarray]:
        """
        Decodes prior boxes and scores from network output maps.
        """
        bboxlists = [[] for _ in range(batch_size)]
        variances = [0.1, 0.2]
        
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # strides: 4, 8, 16, 32, 64, 128
            
            # Find elements exceeding minimum threshold
            b_idx, h_idx, w_idx = np.where(ocls[:, 1, :, :] > conf_threshold)
            
            for b, h, w in zip(b_idx, h_idx, w_idx):
                axc = stride / 2.0 + w * stride
                ayc = stride / 2.0 + h * stride
                priors = np.array([[axc, ayc, stride * 4.0, stride * 4.0]])
                
                score = ocls[b, 1, h, w]
                loc = oreg[b, :, h, w][None, :]  # shape [1, 4]
                
                box = decode_boxes(loc, priors, variances)  # [1, 4]
                bboxlists[b].append([box[0, 0], box[0, 1], box[0, 2], box[0, 3], score])
                
        final_bboxlists = []
        for b in range(batch_size):
            if len(bboxlists[b]) == 0:
                final_bboxlists.append(np.empty((0, 5), dtype=np.float32))
            else:
                final_bboxlists.append(np.stack(bboxlists[b], axis=0))
                
        return final_bboxlists

    def detect_faces(
        self,
        frames: List[np.ndarray],
        batch_size: int = 4,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> List[List[Tuple[Tuple[int, int, int, int], float]]]:
        """
        Runs face detection over a sequence of video frames.
        
        Args:
            frames (List[np.ndarray]): List of BGR frames.
            batch_size (int): Number of frames processed simultaneously.
            conf_threshold (float): Bounding box confidence filter threshold.
            nms_threshold (float): Non-maximum suppression IoU threshold.
            
        Returns:
            List[List[Tuple[Tuple[int, int, int, int], float]]]: List for each frame,
            containing tuples of ((x1, y1, x2, y2), score).
        """
        all_detections = []
        
        for idx in range(0, len(frames), batch_size):
            batch = frames[idx:idx + batch_size]
            b_size = len(batch)
            
            processed_frames = []
            for img in batch:
                img_t = img.transpose(2, 0, 1)  # C, H, W
                processed_frames.append(img_t)
                
            img_batch = np.stack(processed_frames, axis=0).astype(np.float32)
            # Subtract color channel mean values (BGR order)
            img_batch[:, 0, :, :] -= 104.0
            img_batch[:, 1, :, :] -= 117.0
            img_batch[:, 2, :, :] -= 123.0
            
            img_tensor = torch.from_numpy(img_batch).to(self.device)
            
            with torch.no_grad():
                olist = self.net(img_tensor)
                
            # Softmax confidence maps
            for i in range(len(olist) // 2):
                olist[i * 2] = F.softmax(olist[i * 2], dim=1)
                
            olist_numpy = [oelem.cpu().numpy() for oelem in olist]
            
            # Extract candidates
            batch_preds = self._get_predictions(olist_numpy, b_size, conf_threshold=0.05)
            
            # Apply NMS and filter by confidence
            for b in range(b_size):
                preds = batch_preds[b]
                keep = nms(preds, nms_threshold)
                filtered_preds = preds[keep]
                
                frame_dets = []
                for det in filtered_preds:
                    x1, y1, x2, y2, score = det
                    if score >= conf_threshold:
                        frame_dets.append(((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(score)))
                        
                all_detections.append(frame_dets)
                
        return all_detections

    def track_face(
        self,
        detections: List[List[Tuple[Tuple[int, int, int, int], float]]],
        iou_threshold: float = 0.3
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Links bounding boxes across frames to track a single speaker.
        Selects the most stable/active track and uses linear interpolation
        to fill any missing bounding boxes.
        
        Args:
            detections (List[List[Tuple[Tuple[int, int, int, int], float]]]): Detected boxes per frame.
            iou_threshold (float): Minimum IoU threshold to link detections.
            
        Returns:
            List[Optional[Tuple[int, int, int, int]]]: Single bounding box (x1, y1, x2, y2) per frame.
        """
        num_frames = len(detections)
        if num_frames == 0:
            return []
            
        tracks = []
        
        for frame_idx, frame_dets in enumerate(detections):
            for bbox, score in frame_dets:
                matched_track_idx = -1
                best_iou = iou_threshold
                
                for t_idx, track in enumerate(tracks):
                    if track[-1]["frame_idx"] == frame_idx - 1:
                        iou = self._calculate_iou(bbox, track[-1]["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            matched_track_idx = t_idx
                            
                new_node = {"frame_idx": frame_idx, "bbox": bbox, "score": score}
                if matched_track_idx != -1:
                    tracks[matched_track_idx].append(new_node)
                else:
                    tracks.append([new_node])
                    
        if len(tracks) == 0:
            self.logger.warning("No face trajectories identified in the video.")
            return [None] * num_frames
            
        # Select track with largest spatial-temporal footprint
        best_track = None
        max_metric = -1.0
        
        for track in tracks:
            length = len(track)
            avg_area = sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in track) / length
            metric = avg_area * length
            if metric > max_metric:
                max_metric = metric
                best_track = track
                
        linked_boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * num_frames
        for node in best_track:
            linked_boxes[node["frame_idx"]] = node["bbox"]
            
        interpolated_boxes = self._interpolate_gaps(linked_boxes)
        return interpolated_boxes

    def _calculate_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def _interpolate_gaps(self, boxes: List[Optional[Tuple[int, int, int, int]]]) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Interpolates missing frames linearly in a bounding box sequence.
        """
        n = len(boxes)
        valid_indices = [i for i, b in enumerate(boxes) if b is not None]
        
        if len(valid_indices) == 0:
            return boxes
            
        new_boxes = list(boxes)
        
        # 1. Fill leading None values using the first valid box
        first_valid_idx = valid_indices[0]
        for i in range(first_valid_idx):
            new_boxes[i] = boxes[first_valid_idx]
            
        # 2. Fill trailing None values using the last valid box
        last_valid_idx = valid_indices[-1]
        for i in range(last_valid_idx + 1, n):
            new_boxes[i] = boxes[last_valid_idx]
            
        # 3. Linearly interpolate gaps between valid indices
        for k in range(len(valid_indices) - 1):
            start_idx = valid_indices[k]
            end_idx = valid_indices[k + 1]
            
            if end_idx - start_idx > 1:
                start_box = np.array(boxes[start_idx])
                end_box = np.array(boxes[end_idx])
                
                for i in range(start_idx + 1, end_idx):
                    ratio = (i - start_idx) / (end_idx - start_idx)
                    interp_box = start_box + ratio * (end_box - start_box)
                    new_boxes[i] = (
                        int(round(interp_box[0])),
                        int(round(interp_box[1])),
                        int(round(interp_box[2])),
                        int(round(interp_box[3]))
                    )
                    
        return new_boxes


if __name__ == "__main__":
    print("Executing self-test for face_detector.py...")
    
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"Failed to initialize FaceDetector: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Running model execution test on {detector.device}...")
    dummy_input = torch.randn(1, 3, 256, 256, device=detector.device)
    
    try:
        with torch.no_grad():
            outputs = detector.net(dummy_input)
        print(f"Model Forward pass successful. Output heads count: {len(outputs)}")
        assert len(outputs) == 12  # 6 scales * (cls + loc)
        
        # Test detection logic on synthetic blank image
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        dets = detector.detect_faces([dummy_frame], batch_size=1)
        print(f"Synthetic image detection finished. Bounding boxes: {dets}")
        assert len(dets) == 1
        
        # Test tracking interpolation logic
        mock_dets = [
            [((10, 10, 50, 50), 0.9)],
            [],  # gap
            [((12, 12, 52, 52), 0.9)]
        ]
        tracked = detector.track_face(mock_dets)
        print(f"Tracked sequence length: {len(tracked)}. Box at gap index 1: {tracked[1]}")
        assert len(tracked) == 3
        assert tracked[1] is not None
        assert abs(tracked[1][0] - 11) <= 1
        
        print("All face_detector.py self-tests: PASSED")
        
    except Exception as e:
        print(f"Verification tests failed with error: {e}", file=sys.stderr)
        sys.exit(1)

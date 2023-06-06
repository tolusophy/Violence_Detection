import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SalientSuperImage(Dataset):
    def __init__(self, root_dir, num_secs=5, k=60, sampler='uniform', aspect_ratio='144p_B', grid_shape=(10, 6), transform=None):
        """
        Args:
            root_dir (str): The path to the video file.
            num_secs (int): The number of seconds to extract frames from.
            k (int): The number of frames to be sampled from the input array.
            sampler (str): The sampling method to use. Can be one of 'uniform', 'random', 'continuous',
                        'mean_abs', 'LK', 'consecutive' or 'centered'. Defaults to 'uniform'.
            aspect_ratio: A tuple of integers specifying the desired size (width, height) of the cropped and resized frames.
                        options include: '144p_A', '240p_A', '360p_A', '480p_A', '144p_B', '240p_B', '360p_B', '480p_B'
                        'square', 'vertical'
            grid_shape (tuple): A tuple of integers specifying the desired grid shape for the concatenated frames (rows, cols). 
                        The product of rows and cols must be equal to the number of frames in the input array.

        Returns:
            tuple: Concantenated frame, idx, tokenized text
        """

        shapes = {
                    '144p_A': (192, 144), '240p_A': (320, 240), '360p_A': (480, 360), '480p_A': (640, 480),
                    '144p_B': (256, 144), '240p_B': (426, 240), '360p_B': (640, 360), '480p_B': (852, 480),
                    'square': (360, 360), 'vertical': (270, 450)
                }

        self.samples = []
        self.labels = []
        self.aspect_ratio = shapes[aspect_ratio]
        self.num_secs = num_secs
        self.k = k
        self.sampler = sampler
        self.grid_shape = grid_shape
        self.class_to_idx = {'Normal': 0, 'Violence': 1, 'Weaponized': 2} # mapping of class names to integer indices
        self.transform = transform
        
        videos = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if not file_name.endswith(".mp4") and not file_name.endswith(".avi") and not file_name.endswith(".mpg") and not file_name.endswith(".mkv") and not file_name.endswith(".mov"):
                    continue
                video_path = os.path.join(class_dir, file_name)
                videos.append((video_path, class_name))
        self.videos = videos

    def __len__(self):
        return len(self.videos)


    def extract_frames(self, video_path, num_secs):
        """
        Extracts all frames in the last "num_secs" seconds of a video file.

        Args:
            video_path (str): The path to the video file.
            num_secs (int): The number of seconds to extract frames from.

        Returns:
            numpy array: A numpy array of frames extracted from the last "num_secs" seconds of the video file.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate the start and end frame indices for the last "num_secs" seconds of the video.
        end_frame = total_frames
        start_frame = end_frame - (num_secs * fps)
        if start_frame < 0:
            start_frame = 0

        # Extract frames from the video.
        frames = []
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Convert the frames list to a numpy array.
        frames = np.array(frames)

        cap.release()

        return frames


    def sample_frames(self, frames, k, sampler='uniform'):
        """
        Returns a numpy array of sampled frames from the input numpy array of frames.

        Args:
            frames (numpy array): An array of frames (numpy arrays) extracted from the video.
            k (int): The number of frames to be sampled from the input array.
            sampler (str): The sampling method to use. Can be one of 'uniform', 'random', 'continuous',
                        'mean_abs', 'LK', 'consecutive' or 'centered'. Defaults to 'uniform'.

        Returns:
            numpy array: An array containing sampled frames from the input array.
        """
        num_frames = len(frames)
        if k >= num_frames:
            return frames

        k = min(k, num_frames)
        sampled_frames = []

        if sampler == 'uniform':
            stride = num_frames // k
            indices = np.linspace(0, num_frames - 1, k).astype(int)
            sampled_frames = [frames[i] for i in indices]
        elif sampler == 'random':
            indices = np.random.choice(num_frames, k, replace=False)
            sampled_frames = [frames[i] for i in indices]
        elif sampler == 'continuous':
            stride = (num_frames - 1) / (k - 1)
            indices = [int(i * stride) for i in range(k)]
            sampled_frames = [frames[i] for i in indices]
        elif sampler == 'mean_abs':
            # Calculate the average absolute difference between adjacent frames
            diffs = np.abs(np.diff(frames.astype(np.float32)))
            avg_diffs = np.mean(diffs, axis=(1, 2, 3))
            
            # Select frames with the smallest average absolute difference
            indices = np.argsort(avg_diffs)[:k]
            sampled_frames = [frames[i] for i in indices]
        elif sampler == 'LK':
            # Compute optical flow using Lucas-Kanade algorithm
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            sampled_frames = [frames[0]]
            prev_frame = gray_frames[0]
            for i in range(1, k):
                next_frame = gray_frames[int((i / k) * (num_frames - 1))]
                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                mag = mag.astype(np.uint8)
                _, mag = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)
                sampled_frames.append(frames[int((i / k) * (num_frames - 1))])
                prev_frame = next_frame
        elif sampler == 'centered':
            mid = num_frames // 2
            half_k = k // 2
            stride = mid // half_k
            indices = np.linspace(0, mid - 1, half_k).astype(int)
            sampled_frames += [frames[i] for i in indices]
            indices = np.linspace(mid, num_frames - 1, half_k).astype(int)
            sampled_frames += [frames[i] for i in indices]
        elif sampler == 'consecutive':
            indices = [i for i in range(k)]
            sampled_frames = [frames[i] for i in indices]
        else:
            raise ValueError("Unsupported sampler type: {}".format(sampler))
        
        return np.array(sampled_frames)

    def remove_black_border(self, frames, target_size):
        """
        Removes black borders from each frame in the input array, resizes the cropped frames to the target size,
        and returns a numpy array of the processed frames.

        Args:
            frames (numpy array): An array of frames (numpy arrays) extracted from the video.
            target_size (tuple): A tuple of integers specifying the desired size (width, height) of the cropped and resized frames.

        Returns:
            numpy array: An array containing processed frames (numpy arrays) with black borders removed and resized to the target size.
        """
        processed_frames = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

            # Apply morphological transformations to fill in small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            thresh = cv2.bitwise_not(opened)

            # Find contours and get bounding rectangle of largest contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                max_area = 0
                best_cnt = None

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        best_cnt = cnt

                x, y, w, h = cv2.boundingRect(best_cnt)
                cropped_frame = frame[y:y+h, x:x+w]

                # Resize cropped frame to target size
                resized_frame = cv2.resize(cropped_frame, target_size)

                processed_frames.append(resized_frame)
            else:
                resized_frame = cv2.resize(frame, target_size)
                processed_frames.append(resized_frame)

        return np.array(processed_frames)

    def concatenate_frames(self, frames, grid_shape):
        """
        Concatenates the input frames into a grid shape specified by the `grid_shape` argument.

        Args:
            frames (numpy array): An array of frames (numpy arrays) to be concatenated.
            grid_shape (tuple): A tuple of integers specifying the desired grid shape for the concatenated frames (rows, cols). The product of rows and cols must be equal to the number of frames in the input array.

        Returns:
            PIL Image: A concatenated image (PIL Image) containing the input frames in the specified grid shape.
        """

        n_frames = len(frames)
        rows, cols = grid_shape
        assert rows * cols == n_frames, "The product of rows and cols must be equal to the number of frames in the input array."

        frame_h, frame_w = frames[0].shape[:2]
        canvas_h = frame_h * rows
        canvas_w = frame_w * cols

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i in range(n_frames):
            r = i // cols
            c = i % cols
            frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            canvas[r*frame_h:(r+1)*frame_h, c*frame_w:(c+1)*frame_w, :] = frame

        return Image.fromarray(canvas)

    def __getitem__(self, idx):
        video_path, class_name = self.videos[idx]
        class_idx = self.class_to_idx[class_name]
        
        frames = self.extract_frames(video_path, self.num_secs)
        if len(frames) == 0:
            return None
        
        sampled_frames = self.sample_frames(frames, self.k, self.sampler)
        cropped_frames = self.remove_black_border(sampled_frames, self.aspect_ratio)
        canvas = self.concatenate_frames(cropped_frames, self.grid_shape)
        
        if self.transform is not None:
            canvas = self.transform(canvas)
            
        return canvas, class_idx
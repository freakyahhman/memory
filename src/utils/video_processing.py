from decord import VideoReader, cpu, gpu
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from pathlib import Path
import cv2
import torch
from transformers import VideoMAEVideoProcessor, VideoMAEModel, WhisperProcessor, WhisperModel
import torchaudio




def video_processing(video_path: str, base_dir: str):
    video_name = Path(video_path).stem
    output_dir = Path(base_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #scene detect and collect frames
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    print(f"Số cảnh: {len(scene_list)}")
    frames_to_sample = [(start.get_frames() + end.get_frames()) // 2 for start, end in scene_list]
    
    try:
        gpu = gpu(0)
        vr = VideoReader(video_path, ctx=gpu)
    except Exception:
        vr = VideoReader(video_path, ctx=cpu())
        
    fps = vr.get_avg_fps()
    
    #write frames to data/{video_name}/{time}.jpg
    for frame in frames_to_sample:
        image = vr[frame].asnumpy()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        time = frame / fps
        time = round(time, 2)
        out_path = output_dir / f"{time}.jpg"
        cv2.imwrite(str(out_path), image_bgr)

def make_embedding(video_path: str):
    video_processor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-base")
    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    
    audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    audio_model = WhisperModel.from_pretrained("openai/whisper-base")
    
    wav, sr = torchaudio.load(video_path)
    
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    video_inputs = video_processor(video_path, return_tensors="pt")
    audio_inputs = audio_processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        video_out = video_model(**video_inputs).last_hidden_state
        video_features = video_out.mean(dim=1)
        audio_out = audio_model.encoder(**audio_inputs).last_hidden_state
        audio_features = audio_out.mean(dim=1)
        combined_features = torch.cat([video_features, audio_features], dim=-1)
        return combined_features
    


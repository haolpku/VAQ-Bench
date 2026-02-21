# Usage

## Prepare

Download VATEX-EVAL dataset in the following link

```
https://drive.google.com/drive/folders/1jAfZZKEgkMEYFF2x1mhYo39nH-TNeGm6?usp=sharing
```

Download YOLO model checkpoint yolo11x-seg in the following link

```
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt
```

Download PAC-S++ clip model checkpoint PAC++_clip_ViT-L-14 in the following link

```
https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth
```

Download corresponding clip model code following instructions under

```
https://github.com/aimagelab/pacscore
```

## Compute EVQAScore

If your data is a jsonl where each line contains `video_path` and `caption`, you can prepare EVQAScore inputs first, then run extraction/scoring.

### 1) Convert jsonl to EVQAScore files

```
python preprocess.py \
  --jsonl-file /your_path.jsonl \
  --video-root /your_video_root \
  --info-out ./vid_cans_score_dict.json \
  --cand-pkl-out ./candidates_list.pkl
```

### 2) Extract keywords with sglang server

Start your server:

```
python -m sglang.launch_server \
  --model-path /your_model_path \
  --host 127.0.0.1 \
  --port 30000 \
  --tp 8 \
  --served-model-name your_model_name \
  --disable-cuda-graph
```

```
python keywords_extraction.py \
  --api-base http://127.0.0.1:30000/v1 \
  --served-model-name your_model_name \
  --cand-pkl ./candidates_list.pkl \
  --info-file ./vid_cans_score_dict.json \
  --output-json ./cand_keywords.json \
  --num-workers 32
```

### 3) Preprocess video chunks

```
mkdir -p cache results
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python evqascore.py \
    --preprocess \
    --interval 30 \
    --num-chunks 8 \
    --chunk-idx $i \
    --run-name myrun \
    --info-file ./vid_cans_score_dict.json \
    --key-file ./cand_keywords.json \
    --cache-folder ./cache \
    --video-folder /your_video_folder \
    --yolo-path ./yolo11x-seg.pt \
    --clip-model-name ViT-L/14 \
    --clip-weights ./PAC++_clip_ViT-L-14.pth &
done
wait
```

### 4) Compute final EVQAScore

```
python evqascore.py \
  --interval 30 \
  --num-chunks 8 \
  --run-name myrun \
  --info-file ./vid_cans_score_dict.json \
  --key-file ./cand_keywords.json \
  --cache-folder ./cache \
  --result-folder ./results \
  --video-folder /your_video_folder \
  --yolo-path ./yolo11x-seg.pt \
  --clip-model-name ViT-L/14 \
  --clip-weights ./PAC++_clip_ViT-L-14.pth
```

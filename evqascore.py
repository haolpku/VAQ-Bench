import os, sys, torch, cv2, json, argparse, math
from tqdm import tqdm
from PIL import Image
from functools import partial
from ultralytics import YOLO

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAC_DIR = os.path.join(THIS_DIR, "pacscore")
if PAC_DIR not in sys.path:
    sys.path.insert(0, PAC_DIR)

from models.clip_lora import clip_lora


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_models(args):
    clip_model, clip_preprocess = clip_lora.load(
        args.clip_model_name, device=device, lora=args.clip_lora_r
    )
    clip_model = clip_model.to(device).float()
    clip_tokenizer = partial(clip_lora.tokenize, truncate=True)

    if args.clip_weights and os.path.exists(args.clip_weights):
        state = torch.load(args.clip_weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        clip_model.load_state_dict(state, strict=True)
    elif args.clip_weights:
        raise FileNotFoundError(f"clip weights not found: {args.clip_weights}")

    clip_model.eval()
    yolo = YOLO(args.yolo_path)
    return clip_model, clip_preprocess, clip_tokenizer, yolo


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def hmean(a): # harmonic mean
    return len(a) / sum([1 / i for i in a])

def mean(a): # mean
    return sum(a) / len(a)

def metric(similarity):
    return hmean([
        torch.mean(torch.max(similarity, dim=0)[0]),
        torch.mean(torch.max(similarity, dim=1)[0]),
    ])

def normalize(A):
    A_norm = torch.linalg.norm(A, dim=-1, keepdim=True)
    return A / A_norm

def read_video(vid, interval, video_folder):
    addr = os.path.join(video_folder, f"{vid}.mp4")
    frames = []
    cap = cv2.VideoCapture(addr)
    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_count += 1
    return frames

def encode_video(images, clip_model):
    BATCH = 64
    res = []
    with torch.no_grad():
       for i in range(0, len(images), BATCH):
            feats = clip_model.encode_image(images[i : i+BATCH].to(device)).float()
            res.append(feats)
    res = torch.cat(res, dim=0)
    res /= res.norm(dim=-1, keepdim=True)
    return res

def segment_video(frames, yolo):
    BATCH = 64
    images = []
    for i in range(0, len(frames), BATCH):
        results = yolo(frames[i : i+BATCH], stream=True, verbose=False)
        for j, r in enumerate(results):
            for xyxy in r.boxes.xyxy:
                x1, y1, x2, y2 = xyxy.cpu().numpy().astype(int)
                images.append(frames[i + j][y1:y2, x1:x2])
    return images

def preprocess(frames, clip_preprocess):
    rgb = [Image.fromarray(i).convert("RGB") for i in frames]
    pixels = [clip_preprocess(i) for i in rgb]
    return torch.stack(pixels)

def get_video_feats(
    vpaths,
    args,
    clip_model,
    clip_preprocess,
    yolo,
    cache_file=None,
    save_step=None,
    use_video_cache=None,
):
    if use_video_cache:
        res = {}
        for file in use_video_cache:
            r = torch.load(file)
            for k, v in r.items():
                res[k] = v
        return res
    
    vpaths = list(set(vpaths))
    video_feats = {}
    
    if cache_file and os.path.exists(cache_file):
        video_feats = torch.load(cache_file)
        vpaths = [i for i in vpaths if i not in video_feats]
    
    step = 0
    for vpath in tqdm(vpaths, desc=f'Processing Chunk {args.chunk_idx}' if args.chunk_idx is not None else 'Processing Videos', position=0, leave=True):
        #### read video
        frames = read_video(vpath, args.interval, args.video_folder)
        if frames is None or len(frames) == 0:
            continue
        video_feats[vpath] = {}
        #### get global feat
        images = preprocess(frames, clip_preprocess)
        image_feats = encode_video(images, clip_model)
        global_feat = normalize(torch.mean(image_feats, dim=0, keepdim=True))
        video_feats[vpath]['g'] = global_feat
        #### get local feat
        images = segment_video(frames, yolo)
        if len(images) == 0:
            images = frames
        images = preprocess(images, clip_preprocess)
        local_feat = encode_video(images, clip_model)
        video_feats[vpath]['l'] = local_feat
        #### update count and save
        step += 1
        if save_step and step == save_step:
            torch.save(video_feats, cache_file)
            step = 0
    if save_step and step > 0:
        torch.save(video_feats, cache_file)
    return video_feats

def get_text_feats(texts, args, is_key, clip_model, clip_tokenizer):
    texts = [i.split(',') if is_key else [i] for i in texts]
    texts = [clip_tokenizer(i).to(device) for i in tqdm(texts, desc='tokenizing')]
    with torch.no_grad():
        text_feats = []
        for i in tqdm(texts, desc='encoding'):
            feats = clip_model.encode_text(i)
            text_feats.append(feats)
    text_feats = [i / i.norm(dim=-1, keepdim=True) for i in tqdm(text_feats, desc='normalizing')]
    return text_feats

def get_score(vpaths, cands, keys, args, clip_model, clip_preprocess, clip_tokenizer, yolo, use_video_cache = None):
    video_feats = get_video_feats(
        vpaths,
        args,
        clip_model,
        clip_preprocess,
        yolo,
        use_video_cache=use_video_cache,
    )
    cand_feats = get_text_feats(cands, args, 0, clip_model, clip_tokenizer)
    key_feats = get_text_feats(keys, args, 1, clip_model, clip_tokenizer)
    
    pbar = tqdm(range(len(vpaths)), desc='get score')
    result = []
    for i in pbar:
        vpath, cand, key = vpaths[i], cands[i], keys[i]
        if vpath not in video_feats:
            result.append(-1)
            continue
        video_feat = video_feats[vpath]
        cand_feat = cand_feats[i]
        key_feat = key_feats[i]
        l_score = metric(video_feat['l'] @ key_feat.T)
        g_score = metric(video_feat['g'] @ cand_feat.T)
        score = (l_score + g_score) / 2
        n = len(key.split(','))
        result.append(float(score))
    return result

def main(args):
    clip_model, clip_preprocess, clip_tokenizer, yolo = build_models(args)
    run_name = f'{args.run_name}_{args.interval}'
    
    with open(args.info_file, 'r', encoding='utf-8') as f:
        info_ = json.load(f)
    with open(args.key_file, 'r', encoding='utf-8') as f:
        key_ = json.load(f)
    
    if args.preprocess:
        vpaths = []
        for vid in info_.keys():
            vpaths.append(vid)
        vpaths = get_chunk(vpaths, args.num_chunks, args.chunk_idx)
        
        cache_file = os.path.join(
            args.cache_folder,
            f'{run_name}_{args.num_chunks}_{args.chunk_idx}.pkl'
        )
        
        get_video_feats(
            vpaths,
            args,
            clip_model,
            clip_preprocess,
            yolo,
            cache_file=cache_file,
            save_step=10,
        )
    else:
        cands, keys = [], []
        vpaths = []
        for vid in info_.keys():
            for cand in info_[vid]['cands']:
                vpaths.append(vid)
                cands.append(cand)
            for key in key_[vid]['cands']:
                keys.append(key)
        
        use_video_cache = [
            os.path.join(
                args.cache_folder,
                f'{run_name}_{args.num_chunks}_{i}.pkl'
            ) for i in range(args.num_chunks)
        ]
        
        result = get_score(
            vpaths,
            cands,
            keys,
            args,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
            yolo,
            use_video_cache=use_video_cache,
        )
        
        res = {}
        ptr = 0
        for vid in info_.keys():
            cand_count = len(info_[vid]['cands'])
            one_result = result[ptr : ptr + cand_count]
            ptr += cand_count

            if len(one_result) == 0 or one_result[0] == -1:
                continue

            res[vid] = {
                '0_ref_cands_score': one_result,
            }
            if 'scores' in info_[vid]:
                try:
                    res[vid]['human_cands_score'] = eval(info_[vid]['scores'])
                except Exception:
                    pass
        
        store_file = os.path.join(args.result_folder, f'{run_name}.json')
        with open(store_file, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--info-file", default='vid_cans_score_dict.json')
    parser.add_argument("--key-file", default='cand_keywords.json')
    parser.add_argument("--result-folder", default='results')
    parser.add_argument("--cache-folder", default='cache')
    parser.add_argument("--video-folder", default='videos')
    parser.add_argument("--yolo-path", default='yolo11x-seg.pt')
    parser.add_argument("--clip-model-name", default='ViT-L/14')
    parser.add_argument("--clip-weights", default=None)
    parser.add_argument("--clip-lora-r", type=int, default=4)
    parser.add_argument("--interval", type=int, help='video sample interval')
    parser.add_argument("--run-name")
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int, required=False)
    args = parser.parse_args()
    
    main(args)

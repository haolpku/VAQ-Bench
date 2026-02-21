import argparse
import json
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert caption jsonl into EVQAScore input files."
    )
    parser.add_argument("--jsonl-file", required=True, help="Path to source jsonl file.")
    parser.add_argument(
        "--video-root",
        required=True,
        help="Root directory of videos. video_path will be converted to relative vid ids.",
    )
    parser.add_argument(
        "--info-out",
        default="vid_cans_score_dict.json",
        help="Output json path for EVQAScore info file.",
    )
    parser.add_argument(
        "--cand-pkl-out",
        default="candidates_list.pkl",
        help="Output pkl path for keywords_extraction.py input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    info = {}
    all_cands = []

    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "video_path" not in item or "caption" not in item:
                raise KeyError(
                    f"Missing required keys in line {line_no}. "
                    "Each json object must contain video_path and caption."
                )

            video_path = item["video_path"]
            caption = item["caption"]

            rel = os.path.relpath(video_path, args.video_root)
            if rel.endswith(".mp4"):
                vid = rel[:-4]
            else:
                vid = rel

            if vid not in info:
                info[vid] = {"cands": [], "scores": "[]"}
            info[vid]["cands"].append(caption)
            all_cands.append(caption)

    with open(args.info_out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    # keywords_extraction.py uses .tolist() on loaded object, so keep numpy array.
    with open(args.cand_pkl_out, "wb") as f:
        pickle.dump(np.array(all_cands, dtype=object), f)

    print(f"Wrote {args.info_out} with {len(info)} videos.")
    print(f"Wrote {args.cand_pkl_out} with {len(all_cands)} captions.")


if __name__ == "__main__":
    main()

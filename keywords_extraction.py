import argparse
import concurrent.futures
import json
import pickle
import time
import urllib.error
import urllib.request

from tqdm import tqdm

prompt_template = (
    "Extract the main words or phrases that best summarize the"
    "following sentence. Return only the words from the sentence in"
    "a space-separated format, without extra explanations or examples.\n\n"
    "Example 1:\nSentence: \"The teacher explains a complex math problem.\"\nKeywords: teacher, explains, math problem\n"
    "Example 2:\nSentence: \"A boy is playing soccer in the park.\"\nKeywords: boy, playing, soccer, park\n\n"
    "Now, extract the keywords from the following sentence:\n\"{}\""
)

def prepare_input_batch(sentences):
    return [[{"role": "user", "content": prompt_template.format(sentence)}] for sentence in sentences]


def clean_keywords(generated_text):
    cleaned_text = generated_text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    items = [k.strip() for k in cleaned_text.replace("\n", ",").split(",") if k.strip()]
    return ", ".join(items)


def sglang_chat_completion(messages, args):
    url = args.api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": messages,
        "temperature": 0.9,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )

    last_err = None
    for attempt in range(1, args.max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError) as e:
            last_err = e
            if attempt == args.max_retries:
                break
            time.sleep(args.retry_sleep)
    raise RuntimeError(f"sglang request failed after {args.max_retries} retries: {last_err}")


def extract_keywords(sentences, args, pbar=None):
    myinput = prepare_input_batch(sentences)
    keywords_list = [""] * len(myinput)

    def worker(item):
        idx, messages = item
        generated_text = sglang_chat_completion(messages, args).strip()
        return idx, clean_keywords(generated_text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(worker, (idx, messages)) for idx, messages in enumerate(myinput)]
        for future in concurrent.futures.as_completed(futures):
            idx, keyword = future.result()
            keywords_list[idx] = keyword
            if pbar is not None:
                pbar.update(1)
    return keywords_list


def process_cand_list(cands, args):
    with tqdm(total=len(cands), desc="Processing cands", unit="cand") as pbar:
        return extract_keywords(cands, args, pbar=pbar)


def build_keyword_dict(info_file, keyword_list):
    with open(info_file, "r", encoding="utf-8") as f:
        info_ = json.load(f)

    out = {}
    idx = 0
    for vid in info_.keys():
        cand_count = len(info_[vid]["cands"])
        out[vid] = {"cands": keyword_list[idx : idx + cand_count]}
        idx += cand_count

    if idx != len(keyword_list):
        raise ValueError(
            f"Keyword count mismatch: used {idx}, total {len(keyword_list)}. "
            "Please verify candidate order and source file."
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--cand-pkl", required=True, help="Path to candidates_list.pkl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--info-file", default=None, help="Optional: vid_cans_score_dict.json")
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    with open(args.cand_pkl, "rb") as f:
        samples_list = pickle.load(f)
    cands = samples_list.tolist()
    cand_keywords = process_cand_list(cands, args)

    output_data = cand_keywords
    if args.info_file:
        output_data = build_keyword_dict(args.info_file, cand_keywords)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Keywords extraction completed: {args.output_json}")

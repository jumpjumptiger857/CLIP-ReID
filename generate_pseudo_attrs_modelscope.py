import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from PIL import Image, ImageOps

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


MODEL_ID = "iic/cv_resnet50_pedestrian-attribute-recognition_image"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def default_record() -> Dict[str, Dict[str, Any]]:
    return {
        "upper_color": {"value": "unknown", "conf": 0.0},
        "upper_wear": {"value": "unknown", "conf": 0.0},
        "lower_wear": {"value": "unknown", "conf": 0.0},
        "hat": {"value": "unknown", "conf": 0.0},
        "backpack": {"value": "unknown", "conf": 0.0},
    }


def area_of_box(box: List[float]) -> float:
    if len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def choose_main_person(output: Dict[str, Any]) -> Tuple[List[float], Any]:
    """
    选择面积最大的检测框对应的人。
    """
    boxes = output.get("boxes", []) or output.get("bboxes", []) or []
    labels = output.get("labels", []) or []

    if not boxes:
        return [], None

    best_idx = max(range(len(boxes)), key=lambda i: area_of_box(boxes[i]))
    chosen_box = boxes[best_idx]

    if isinstance(labels, list) and len(labels) == len(boxes):
        chosen_labels = labels[best_idx]
    else:
        chosen_labels = labels

    return chosen_box, chosen_labels


def normalize_token(x: Any) -> str:
    return str(x).strip().lower().replace(" ", "").replace("_", "")


def parse_modelscope_labels(raw_labels: Any) -> Dict[str, Dict[str, Any]]:
    """
    解析 ModelScope 当前返回的固定顺序属性列表。

    例子:
    [
      "Male", "Age18-60", "Side",
      "No", "No", "No", "No", "No",
      "ShortSleeve", "Shorts", "red", "blue"
    ]

    顺序解释为:
    [gender, age, view, hat, glasses, handbag, shoulderbag, backpack,
     upper_wear, lower_wear, upper_color, lower_color]
    """
    rec = default_record()

    if raw_labels is None:
        return rec

    # 如果是 [[...]]，拍平一层
    if isinstance(raw_labels, list) and len(raw_labels) == 1 and isinstance(raw_labels[0], list):
        raw_labels = raw_labels[0]

    if not isinstance(raw_labels, list):
        return rec

    labels = [normalize_token(x) for x in raw_labels]

    if len(labels) < 12:
        return rec

    # 固定位置解析
    hat_raw = labels[3]
    backpack_raw = labels[7]
    upper_wear_raw = labels[8]
    lower_wear_raw = labels[9]
    upper_color_raw = labels[10]
    # lower_color_raw = labels[11]  # 当前最小版先不用

    # hat
    if hat_raw == "no":
        rec["hat"] = {"value": "no", "conf": 1.0}
    else:
        rec["hat"] = {"value": "yes", "conf": 1.0}

    # backpack
    if backpack_raw == "no":
        rec["backpack"] = {"value": "no", "conf": 1.0}
    else:
        rec["backpack"] = {"value": "yes", "conf": 1.0}

    # upper_wear
    upper_map = {
        "shortsleeve": "short_sleeve",
        "longsleeve": "long_sleeve",
    }
    if upper_wear_raw in upper_map:
        rec["upper_wear"] = {"value": upper_map[upper_wear_raw], "conf": 1.0}

    # lower_wear
    lower_map = {
        "trousers": "trousers",
        "shorts": "shorts",
        "skirt&dress": "skirt",
        "skirtdress": "skirt",
    }
    if lower_wear_raw in lower_map:
        rec["lower_wear"] = {"value": lower_map[lower_wear_raw], "conf": 1.0}

    # upper_color
    color_map = {
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "brown": "brown",
        "purple": "purple",
        "pink": "pink",
        "orange": "orange",
    }
    if upper_color_raw in color_map:
        rec["upper_color"] = {"value": color_map[upper_color_raw], "conf": 1.0}

    return rec


def run_par_once(par, pil_img: Image.Image) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        pil_img.save(tmp_path)
        output = par(tmp_path)
        return output
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_par_robust(par, img_path: str) -> Dict[str, Any]:
    """
    对 crop 图做多次尝试:
    1) 原图
    2) 25% padding
    3) 45% padding
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)

    trials = [
        img,
        ImageOps.expand(img, border=int(min_side * 0.25), fill=(114, 114, 114)),
        ImageOps.expand(img, border=int(min_side * 0.45), fill=(114, 114, 114)),
    ]

    last_err = None
    for trial_img in trials:
        try:
            output = run_par_once(par, trial_img)
            boxes = output.get("boxes", []) or output.get("bboxes", []) or []
            if len(boxes) > 0:
                return output
        except Exception as e:
            last_err = e

    if last_err is not None:
        raise last_err
    raise RuntimeError("No human detected after all retries.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="图片目录")
    parser.add_argument("--output_json", type=str, required=True, help="输出 json 文件")
    parser.add_argument("--save_rel_to", type=str, default=None, help="json key 相对哪个目录保存")
    parser.add_argument("--start", type=int, default=0, help="从第几张开始")
    parser.add_argument("--limit", type=int, default=-1, help="最多处理多少张，-1 表示全部")
    args = parser.parse_args()

    img_paths = list_images(args.input_dir)
    if not img_paths:
        raise ValueError(f"No images found under: {args.input_dir}")

    if args.start > 0:
        img_paths = img_paths[args.start:]
    if args.limit > 0:
        img_paths = img_paths[:args.limit]

    print(f"Found {len(img_paths)} images to process.")

    par = pipeline(
        task=Tasks.pedestrian_attribute_recognition,
        model=MODEL_ID,
        trust_remote_code=True
    )

    results = {}
    success = 0
    failed = 0

    for img_path in tqdm(img_paths, desc="Generating pseudo attributes"):
        try:
            output = run_par_robust(par, img_path)
            _, chosen_labels = choose_main_person(output)
            rec = parse_modelscope_labels(chosen_labels)
            success += 1
        except Exception:
            rec = default_record()
            failed += 1

        if args.save_rel_to is not None:
            key = os.path.relpath(img_path, args.save_rel_to)
        else:
            key = os.path.relpath(img_path, args.input_dir)

        key = key.replace("\\", "/")
        results[key] = rec

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} records to: {args.output_json}")
    print(f"Success: {success}, Failed: {failed}, Total: {len(results)}")
    if len(results) > 0:
        print(f"Success rate: {success / len(results):.4f}")


if __name__ == "__main__":
    main()
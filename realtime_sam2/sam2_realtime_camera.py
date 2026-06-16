import argparse
import json
import os
import random
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time SAM2 camera background replacement with multi-object prompts."
    )
    parser.add_argument("--camera", default="0", help="Camera index or video path.")
    parser.add_argument("--background", required=True, help="Replacement background image.")
    parser.add_argument("--prompt-image", default=None, help="Existing image used with --prompt-json to initialize masks.")
    parser.add_argument("--prompt-json", default=None, help="Existing multi-object prompt JSON.")
    parser.add_argument(
        "--prompt-image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Original prompt image size if --prompt-image is unavailable.",
    )
    parser.add_argument(
        "--no-scale-prompts",
        action="store_true",
        help="Do not scale prompt coordinates from prompt image size to camera frame size.",
    )
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default=None, help="cuda, cpu, cuda:0. Defaults to cuda if available.")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height.")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--process-every", type=int, default=1, help="Run SAM2 every N frames; reuse mask between runs.")
    parser.add_argument("--bg-fit", choices=["cover", "stretch"], default="cover")
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--edge-blur", type=float, default=1.5)
    parser.add_argument("--save-video", default=None, help="Optional path for composited output video.")
    parser.add_argument("--save-frames-dir", default=None, help="Optional directory to save composited frames.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_qt():
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/freefont")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_FONT_DPI", "96")


def open_capture(source, width, height):
    import cv2

    try:
        source_value = int(source)
    except ValueError:
        source_value = source
    cap = cv2.VideoCapture(source_value)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")
    return cap


def resize_background(bg, size, fit, rng):
    from PIL import Image

    target_w, target_h = size
    bg = bg.convert("RGB")
    if fit == "stretch":
        return bg.resize(size, Image.Resampling.BILINEAR)

    scale = max(target_w / bg.width, target_h / bg.height)
    new_size = (max(1, int(bg.width * scale)), max(1, int(bg.height * scale)))
    bg = bg.resize(new_size, Image.Resampling.BILINEAR)
    max_left = max(0, bg.width - target_w)
    max_top = max(0, bg.height - target_h)
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    return bg.crop((left, top, left + target_w, top + target_h))


def refine_mask(mask, dilate, edge_blur):
    import numpy as np
    from PIL import Image, ImageFilter

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if dilate > 0:
        kernel = max(3, dilate * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel))
    hard = np.asarray(mask_img, dtype=np.uint8) > 127
    if edge_blur > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(edge_blur))
    alpha = np.asarray(mask_img, dtype="float32")[..., None] / 255.0
    return hard, alpha


def composite_rgb(frame_rgb, bg_rgb, alpha):
    import numpy as np

    out = frame_rgb.astype(np.float32) * alpha + bg_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_rgb(frame_rgb, mask):
    import numpy as np

    red = np.zeros_like(frame_rgb, dtype=np.float32)
    red[..., 0] = 255
    alpha = mask.astype(np.float32)[..., None] * 0.45
    out = frame_rgb.astype(np.float32) * (1.0 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_prompt(prompt, default_obj_id):
    obj_id = int(prompt.get("obj_id", default_obj_id))
    box = prompt.get("box")
    if box is not None:
        if len(box) != 4:
            raise ValueError(f"Object {obj_id} box must have 4 numbers.")
        box = [float(v) for v in box]

    points = []
    for point in prompt.get("points", []):
        if len(point) != 3:
            raise ValueError(f"Object {obj_id} point must be [x, y, label].")
        x, y, label = float(point[0]), float(point[1]), int(point[2])
        if label not in (0, 1):
            raise ValueError(f"Object {obj_id} point label must be 0 or 1.")
        points.append((x, y, label))
    if box is None and not points:
        raise ValueError(f"Object {obj_id} has neither box nor points.")
    return {"obj_id": obj_id, "box": box, "points": points}


def load_prompt_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_objects = data if isinstance(data, list) else data.get("objects")
    if not raw_objects:
        raise ValueError(f"No objects found in prompt JSON: {path}")
    return [normalize_prompt(prompt, idx + 1) for idx, prompt in enumerate(raw_objects)]


def scale_prompts(prompts, src_size, dst_size):
    if src_size == dst_size:
        return prompts
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / src_w
    sy = dst_h / src_h
    scaled = []
    for prompt in prompts:
        box = prompt["box"]
        if box is not None:
            box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        points = [(x * sx, y * sy, label) for x, y, label in prompt["points"]]
        scaled.append({"obj_id": prompt["obj_id"], "box": box, "points": points})
    return scaled


def format_prompt_summary(prompts):
    lines = []
    for prompt in prompts:
        point_text = " ".join(
            f"--point {int(round(x))},{int(round(y))},{label}" for x, y, label in prompt["points"]
        )
        box_text = ""
        if prompt["box"] is not None:
            box_text = "--box " + " ".join(str(int(round(v))) for v in prompt["box"])
        lines.append(f"# object {prompt['obj_id']}: {box_text} {point_text}".strip())
    return "\n".join(lines)


def preview_initial_masks(frame_rgb, merged_mask):
    setup_qt()
    import cv2

    overlay = overlay_rgb(frame_rgb, merged_mask)
    canvas = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    lines = [
        "Initial prompt-image mask preview",
        "red area = kept foreground",
        "c/s/enter: start camera stream",
        "q/esc: quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30
    window = "Initial mask preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, canvas)
    accepted = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, ord("c"), ord("s")):
            accepted = True
            break
        if key in (27, ord("q")):
            break
    cv2.destroyWindow(window)
    if not accepted:
        raise RuntimeError("Initial mask preview cancelled.")


def select_prompt_interactively(frame_rgb, title):
    setup_qt()
    import cv2

    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    window = title
    mode = "point"
    box = None
    points = []
    dragging = False
    drag_start = None
    drag_end = None

    def clamp_xy(x, y):
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def draw():
        canvas = image_bgr.copy()
        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(canvas, "BOX", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        if dragging and drag_start and drag_end:
            cv2.rectangle(canvas, drag_start, drag_end, (0, 220, 255), 1)
        for x, y, label in points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            text = "FG" if label == 1 else "BG"
            cv2.circle(canvas, (int(round(x)), int(round(y))), 7, color, -1)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 9, (255, 255, 255), 2)
            cv2.putText(canvas, text, (int(round(x)) + 10, int(round(y)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        lines = [
            title,
            f"mode: {mode}",
            "left click: foreground | right click: background",
            "b: box drag | p: point mode | u: undo | x: clear box | r: reset",
            "enter/s: accept | q/esc: cancel",
        ]
        y = 24
        for line in lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal box, dragging, drag_start, drag_end
        x, y = clamp_xy(x, y)
        if mode == "box":
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                drag_start = (x, y)
                drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and dragging:
                dragging = False
                drag_end = (x, y)
                x1, y1 = drag_start
                x2, y2 = drag_end
                if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                    box = [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((float(x), float(y), 1))
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append((float(x), float(y), 0))
        draw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(width, 1280), min(height, 800))
    cv2.setMouseCallback(window, on_mouse)
    draw()
    accepted = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, ord("s")):
            accepted = True
            break
        if key in (27, ord("q")):
            break
        if key == ord("b"):
            mode = "box"
            draw()
        elif key == ord("p"):
            mode = "point"
            draw()
        elif key == ord("u"):
            if points:
                points.pop()
            draw()
        elif key == ord("x"):
            box = None
            draw()
        elif key == ord("r"):
            box = None
            points.clear()
            draw()
    cv2.destroyWindow(window)
    if not accepted or (box is None and not points):
        raise RuntimeError("Prompt selection cancelled or empty.")
    return {"box": box, "points": points}


def select_multi_prompts(frame_rgb):
    setup_qt()
    import cv2

    prompts = []
    while True:
        prompt = select_prompt_interactively(frame_rgb, f"Realtime object {len(prompts) + 1}")
        prompts.append(prompt)

        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        lines = [
            f"Selected {len(prompts)} object(s).",
            "a: add another object",
            "c/s/enter: start realtime",
            "q/esc: cancel",
        ]
        y = 34
        for line in lines:
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y += 36
        window = "Add object?"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, canvas)
        decision = "cancel"
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("a"):
                decision = "add"
                break
            if key in (13, 10, ord("c"), ord("s")):
                decision = "start"
                break
            if key in (27, ord("q")):
                decision = "cancel"
                break
        cv2.destroyWindow(window)
        if decision == "add":
            continue
        if decision == "start":
            return prompts
        raise RuntimeError("Multi-object selection cancelled.")


def predict_object(predictor, prompt=None, low_res_mask=None):
    import numpy as np

    point_coords = None
    point_labels = None
    box = None
    if prompt is not None:
        if prompt["points"]:
            point_coords = np.asarray([[x, y] for x, y, _ in prompt["points"]], dtype=np.float32)
            point_labels = np.asarray([label for _, _, label in prompt["points"]], dtype=np.int32)
        if prompt["box"] is not None:
            box = np.asarray(prompt["box"], dtype=np.float32)
    masks, scores, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=low_res_mask,
        multimask_output=False,
    )
    best_idx = int(np.argmax(scores)) if len(scores) else 0
    return masks[best_idx].astype(bool), low_res_masks[best_idx]


def main():
    args = parse_args()
    setup_qt()

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    rng = random.Random(args.seed)
    cap = open_capture(args.camera, args.width, args.height)
    ok, frame_bgr = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first camera frame.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    bg_source = Image.open(args.background).convert("RGB")
    bg = resize_background(bg_source, (width, height), args.bg_fit, rng)
    bg_rgb = np.asarray(bg, dtype=np.uint8)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SAM2ImagePredictor.from_pretrained(args.model_id, device=device)

    init_frame_rgb = frame_rgb
    if args.prompt_json:
        prompts = load_prompt_json(args.prompt_json)
        if args.prompt_image:
            prompt_image = Image.open(args.prompt_image).convert("RGB")
            init_frame_rgb = np.asarray(prompt_image, dtype=np.uint8)
            src_size = prompt_image.size
        elif args.prompt_image_size:
            src_size = tuple(args.prompt_image_size)
        else:
            src_size = (width, height)
        if not args.no_scale_prompts:
            prompts = scale_prompts(prompts, src_size, (init_frame_rgb.shape[1], init_frame_rgb.shape[0]))
        print(f"Loaded {len(prompts)} object prompt(s) from: {args.prompt_json}")
        print(format_prompt_summary(prompts))
    else:
        print("Select objects to keep. Suggested objects: conveyor/table, robot hands, bins, target object.")
        prompts = select_multi_prompts(frame_rgb)
        print(f"Selected {len(prompts)} object(s).")

    print("Initializing SAM2 masks...")
    low_res_by_obj = [None for _ in prompts]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=str(device).startswith("cuda"),
    ):
        predictor.set_image(init_frame_rgb)
        init_masks = []
        for obj_i, prompt in enumerate(prompts):
            mask, low_res = predict_object(predictor, prompt=prompt, low_res_mask=None)
            low_res_by_obj[obj_i] = low_res
            init_masks.append(mask)
    merged_mask = np.any(np.stack(init_masks, axis=0), axis=0)

    if args.prompt_image:
        preview_initial_masks(init_frame_rgb, merged_mask)

    writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps != fps:
            fps = 30.0
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    frames_dir = Path(args.save_frames_dir) if args.save_frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    if merged_mask.shape[:2] != (height, width):
        merged_mask = None
    frame_idx = 0
    last_time = time.time()
    fps_smooth = 0.0

    window = "SAM2 realtime background replacement"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if frame_idx == 0:
                current_bgr = frame_bgr
            else:
                ok, current_bgr = cap.read()
                if not ok:
                    break
            current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)

            if frame_idx % max(1, args.process_every) == 0 or merged_mask is None:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(device).startswith("cuda"),
                ):
                    predictor.set_image(current_rgb)
                    masks = []
                    for obj_i, prompt in enumerate(prompts):
                        use_prompt = prompt if low_res_by_obj[obj_i] is None else None
                        mask, low_res = predict_object(
                            predictor,
                            prompt=use_prompt,
                            low_res_mask=low_res_by_obj[obj_i],
                        )
                        low_res_by_obj[obj_i] = low_res
                        masks.append(mask)
                merged_mask = np.any(np.stack(masks, axis=0), axis=0)

            hard_mask, alpha = refine_mask(merged_mask, args.dilate, args.edge_blur)
            comp_rgb = composite_rgb(current_rgb, bg_rgb, alpha)
            overlay = overlay_rgb(current_rgb, hard_mask)

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_smooth = inst_fps if fps_smooth == 0.0 else fps_smooth * 0.9 + inst_fps * 0.1

            display_rgb = comp_rgb
            display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                display_bgr,
                f"fps {fps_smooth:.1f} | q quit | r reselect | o overlay",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_bgr,
                f"fps {fps_smooth:.1f} | q quit | r reselect | o overlay",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if args.display_scale != 1.0:
                display_bgr = cv2.resize(
                    display_bgr,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, display_bgr)

            if writer:
                writer.write(cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))
            if frames_dir:
                Image.fromarray(comp_rgb).save(frames_dir / f"{frame_idx:06d}.jpg", quality=95)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("o"):
                cv2.imshow("SAM2 realtime mask overlay", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if key == ord("r"):
                prompts = select_multi_prompts(current_rgb)
                low_res_by_obj = [None for _ in prompts]
                merged_mask = None
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

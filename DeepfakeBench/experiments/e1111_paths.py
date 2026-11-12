"""E1111 paths after the user-specified /home/user1 -> /mnt/data2 move."""

from pathlib import Path

OLD_ROOT = "/home/user1"
NEW_ROOT = "/mnt/data2"


def relocated(value):
    text = str(value).replace("\\", "/")
    if text == OLD_ROOT or text.startswith(OLD_ROOT + "/"):
        return NEW_ROOT + text[len(OLD_ROOT):]
    return str(value)


def normalize_arguments(args, bench, config=None):
    config = {} if config is None else config
    defaults = {
        "dataset_json_folder": config.get("dataset_json_folder") or bench / "preprocessing/dataset_json",
        "rgb_root": config.get("rgb_root_override") or OLD_ROOT + "/effort/data",
        "clip_pretrained_path": config.get("clip_pretrained_path") or bench / "training/models--openai--clip-vit-large-patch14",
    }
    for name, default in defaults.items():
        value = getattr(args, name, None)
        value = relocated(default if value is None else value)
        # Keep the original child-runner config types safe for YAML dumping.
        setattr(args, name, Path(value) if name == "dataset_json_folder" else str(value))
    for name in ("base_run", "base_checkpoint", "base_config", "raw_root",
                 "predictor_path", "pair_map", "output_dir"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(relocated(value)))
    return args


def input_file(path, rgb_root):
    text = str(path).replace("\\", "/")
    if text == OLD_ROOT or text.startswith(OLD_ROOT + "/"):
        raise ValueError(
            f"Dataset index still contains an obsolete absolute path: {text}. "
            "Update that index prefix; regenerating images is not required if the new file exists.")
    path = Path(text)
    return path if path.is_absolute() else Path(rgb_root) / path

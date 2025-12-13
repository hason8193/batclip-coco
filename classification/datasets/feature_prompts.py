import json
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional


FeatureVariant = Literal["has", "may_or_may_not_have"]


def load_feature_dict(path: str | Path) -> Dict[str, List[str]]:
    """Load a class->features mapping.

    Expected format:
        {
          "acoustic_guitar": ["hollow body", ...],
          ...
        }

    Keys may use underscores; values are feature strings.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(data).__name__}")

    normalized: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if v is None:
            feats: List[str] = []
        elif isinstance(v, list):
            feats = [str(x).strip() for x in v if str(x).strip()]
        else:
            feats = [str(v).strip()] if str(v).strip() else []

        normalized[k] = feats

    return normalized


def _format_feature_prompt(
    class_name: str,
    features: List[str],
    *,
    variant: FeatureVariant,
) -> str:
    class_name = str(class_name).strip()
    features = [str(x).strip() for x in (features or []) if str(x).strip()]

    if not features:
        return f"{class_name}."

    feature_blob = ", ".join(features)
    if variant == "has":
        return f"{class_name} which has {feature_blob}."
    if variant == "may_or_may_not_have":
        return f"{class_name} which may or may not have {feature_blob}."

    raise ValueError(f"Unknown feature prompt variant: {variant}")


def build_imagenet_prompt_json(
    imagenet_class_names: Iterable[str],
    feature_dict: Dict[str, List[str]],
    *,
    variant: FeatureVariant = "has",
    fallback_template: str = "a photo of a {}.",
) -> Dict[str, List[str]]:
    """Build a CuPL-style prompt JSON for all ImageNet classes.

    Returns a dict mapping *space-separated* class names (as used by get_class_names)
    to a list of prompt strings.

    For classes included in feature_dict (underscored keys), use the feature prompt.
    For classes not included, use fallback_template.
    """

    # Normalize keys to both underscored and space-separated forms.
    features_by_space: Dict[str, List[str]] = {}
    for k, feats in (feature_dict or {}).items():
        key_space = str(k).replace("_", " ").strip()
        features_by_space[key_space] = [str(x).strip() for x in (feats or []) if str(x).strip()]

    prompt_json: Dict[str, List[str]] = {}
    for cname in imagenet_class_names:
        cname = str(cname).strip()
        if cname in features_by_space:
            prompt = _format_feature_prompt(cname, features_by_space[cname], variant=variant)
            prompt_json[cname] = [prompt]
        else:
            prompt_json[cname] = [fallback_template.format(cname)]

    return prompt_json


def write_prompt_json(prompt_json: Dict[str, List[str]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(prompt_json, f, ensure_ascii=False, indent=2)
    return out_path

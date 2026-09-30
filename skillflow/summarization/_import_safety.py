from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import sys
import types


OPTIONAL_AUDIO_PACKAGES = {
    "av",
    "librosa",
    "soundfile",
    "soxr",
    "speech_recognition",
    "torchaudio",
    "torchcodec",
}

OPTIONAL_VISION_PACKAGES = {
    "torchvision",
    "torchvision.io",
    "torchvision.transforms",
}

OPTIONAL_MEDIA_PACKAGES = OPTIONAL_AUDIO_PACKAGES | OPTIONAL_VISION_PACKAGES


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    is_package = name == "torchvision"
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=is_package)
    module.__version__ = "0.0"
    if is_package:
        module.__path__ = []
    if name == "av":
        logging = types.SimpleNamespace(
            ERROR=40,
            set_level=lambda *_args, **_kwargs: None,
        )
        module.logging = logging
    elif name == "torchvision.io":
        class ImageReadMode:
            UNCHANGED = 0
            GRAY = 1
            GRAY_ALPHA = 2
            RGB = 3
            RGB_ALPHA = 4

        def decode_image(*_args, **_kwargs):
            raise ImportError("torchvision is intentionally hidden for text-only model loading")

        module.ImageReadMode = ImageReadMode
        module.decode_image = decode_image
    elif name == "torchvision.transforms":
        class InterpolationMode:
            NEAREST = 0
            NEAREST_EXACT = 0
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
            LANCZOS = 1

        module.InterpolationMode = InterpolationMode
    return module


def disable_transformers_audio_detection() -> None:
    module = sys.modules.get("transformers.utils.import_utils")
    if module is None:
        return
    for attr in (
        "_av_available",
        "_librosa_available",
        "_soundfile_available",
        "_soxr_available",
        "_speech_available",
        "_torchaudio_available",
        "_torchcodec_available",
        "_torchvision_available",
    ):
        if hasattr(module, attr):
            setattr(module, attr, False)


@contextmanager
def hide_optional_audio_packages():
    """Keep text model imports from failing on broken optional media deps."""
    real_find_spec = importlib.util.find_spec
    previous_modules = {name: sys.modules.get(name) for name in OPTIONAL_MEDIA_PACKAGES}

    def patched_find_spec(name, package=None):
        root = name.split(".", 1)[0]
        if root in OPTIONAL_MEDIA_PACKAGES:
            return None
        return real_find_spec(name, package)

    for name in OPTIONAL_MEDIA_PACKAGES:
        sys.modules[name] = _stub_module(name)
    importlib.util.find_spec = patched_find_spec
    try:
        disable_transformers_audio_detection()
        yield
        disable_transformers_audio_detection()
    finally:
        importlib.util.find_spec = real_find_spec
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

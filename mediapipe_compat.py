"""Compatibility helpers for the installed MediaPipe build."""

from __future__ import annotations

import ctypes
import os
import platform
from importlib import resources


def patch_mediapipe_loader() -> None:
    """Patch MediaPipe's ctypes loader for Python 3.14 on Windows.

    The MediaPipe wheel expects ``ctypes.CDLL(None).free`` to exist, but that
    lookup fails on this Python/Windows combination. MediaPipe's native library
    can still use ``free`` from ``ucrtbase.dll``, so we inject that symbol before
    the first task model is created.
    """

    from mediapipe.tasks.python.core import mediapipe_c_bindings

    if getattr(mediapipe_c_bindings.load_raw_library, "_patched_for_python314", False):
        return

    def load_raw_library(signatures=()):
        if not hasattr(load_raw_library, "_shared_lib"):
            load_raw_library._shared_lib = None  # type: ignore[attr-defined]

        shared_lib = load_raw_library._shared_lib  # type: ignore[attr-defined]
        if shared_lib is None:
            if os.name == "posix":
                if platform.system() == "Darwin":
                    lib_filename = "libmediapipe.dylib"
                else:
                    lib_filename = "libmediapipe.so"
            else:
                lib_filename = "libmediapipe.dll"

            lib_path_context = resources.files("mediapipe.tasks.c")
            absolute_lib_path = str(lib_path_context / lib_filename)
            shared_lib = ctypes.CDLL(absolute_lib_path)

            if not hasattr(shared_lib, "free"):
                shared_lib.free = ctypes.CDLL("ucrtbase.dll").free

            load_raw_library._shared_lib = shared_lib  # type: ignore[attr-defined]

        for signature in signatures:
            c_func = getattr(shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        shared_lib.free.argtypes = [ctypes.c_void_p]
        shared_lib.free.restype = None

        return shared_lib

    load_raw_library._patched_for_python314 = True  # type: ignore[attr-defined]
    mediapipe_c_bindings.load_raw_library = load_raw_library
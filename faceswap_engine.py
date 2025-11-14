# faceswap_engine.py

def swap_face(source_bytes: bytes, target_bytes: bytes) -> dict:
    """
    Placeholder face swap engine.

    This function currently does not perform real face swapping.
    It only returns metadata of the received images.

    Parameters:
        source_bytes: The bytes of the source image.
        target_bytes: The bytes of the target image.

    Returns:
        A dictionary containing debug info.
    """

    return {
        "status": "ok",
        "note": "Placeholder engine. Real face swapping will be added later.",
        "source_size_bytes": len(source_bytes),
        "target_size_bytes": len(target_bytes)
    }

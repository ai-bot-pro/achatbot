import base64


def image_bytes_to_base64_data_uri(img_bytes: bytes,
                                   format: str = "jpeg", encoding="utf-8"):
    base64_data = base64.b64encode(img_bytes).decode(encoding)
    return f"data:image/{format};base64,{base64_data}"


def image_to_base64_data_uri(file_path: str,
                             format: str = "jpeg", encoding="utf-8"):
    with open(file_path, "rb") as img_file:
        return image_bytes_to_base64_data_uri(
            img_file.read(), format=format, encoding=encoding)

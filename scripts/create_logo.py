import os
import base64

# A tiny 1x1 transparent PNG used as a safe placeholder logo.
B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8A"
    "AnsB9wE7k3MAAAAASUVORK5CYII="
)


def ensure_logo(path=None):
    """Ensure an `arabiers.png` file exists at `path` (or ./assets/arabiers.png).
    If missing, decode the embedded base64 and write a valid PNG file.
    """
    if path is None:
        path = os.path.join(os.getcwd(), "assets", "arabiers.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        data = base64.b64decode(B64_PNG)
        with open(path, "wb") as f:
            f.write(data)
        # Return True when file was created
        return True
    return False

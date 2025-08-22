from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
from io import BytesIO
import warnings
import logging

# === Suppress noisy warnings ===
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message="Error fetching version info", module="albumentations")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", module="pydantic")

# Reduce logging noise
logging.getLogger("albumentations").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# === Load model safely ===
print("Loading LaTeX OCR model...")
try:
    from pix2tex.cli import LatexOCR
    model = LatexOCR()  # Now works! (after GitHub install)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

def b64_to_image(data_url: str):
    try:
        if ',' in data_url:
            _, b64 = data_url.split(',', 1)
        else:
            b64 = data_url
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img = b64_to_image(image_data)
        latex = model(img)
        return jsonify({"latex": latex})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)


if False:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from pix2tex.cli import LatexOCR
    from PIL import Image
    import base64
    from io import BytesIO
    import warnings
    import torch

    # Suppress torch.load warning
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
        module="torch"
    )


    app = Flask(__name__)
    CORS(app)   # <-- this is critical

    model = LatexOCR()

    def b64_to_image(data_url: str):
        _, b64 = data_url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    @app.route("/ocr", methods=["POST"])
    def ocr():
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "no image provided"}), 400
        img = b64_to_image(image_data)
        latex = model(img)   # real OCR here
        return jsonify({"latex": latex})


    import base64
    from io import BytesIO

    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from PIL import Image

    USE_PIX2TEX = False  # <-- flip to True after you verify the end-to-end path

    # If/when you enable pix2tex:
    # pip install pix2tex torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
    # from pix2tex.cli import LatexOCR
    # model = LatexOCR()

    app = Flask(__name__)
    CORS(app)  # allow requests from the extension (localhost)

    def b64_to_image(data_url: str) -> Image.Image:
        """
        Accepts a 'data:image/png;base64,....' string and returns PIL Image.
        """
        if "," in data_url:
            _, b64 = data_url.split(",", 1)
        else:
            b64 = data_url
        raw = base64.b64decode(b64)
        return Image.open(BytesIO(raw)).convert("RGB")

    @app.route("/ocr", methods=["POST"])
    def ocr():
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "no image provided"}), 400

        try:
            img = b64_to_image(image_data)
        except Exception as e:
            return jsonify({"error": f"invalid image: {e}"}), 400

        if USE_PIX2TEX:
            # Real OCR
            # latex = model(img)  # returns a string
            # return jsonify({"latex": latex})
            return jsonify({"error": "pix2tex not enabled in this snippet"}), 501

        # Echo OCR (MVP sanity check)
        # Here we just return a dummy formula so you can verify the extension flow.
        return jsonify({"latex": r"\int_a^b f(x)\,dx = F(b)-F(a)"}), 200

    if __name__ == "__main__":
        # Run on localhost:5000
        app.run(host="127.0.0.1", port=5000, debug=True)

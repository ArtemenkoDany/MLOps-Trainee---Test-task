import argparse
import io
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        else:
            file = request.files["file"]

            img_bytes = file.read()
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            img = Image.open(io.BytesIO(img_bytes))
            results = model(img, size=640)

            results.render()  # imgs with boxes and labels
            for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save("static/image0.jpg", format="JPEG")
            return redirect("static/image0.jpg")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
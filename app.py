from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import webbrowser
import threading

app = Flask(__name__)

model = load_model("fabric_model.keras")
class_names = ['floral', 'geometric', 'plain', 'polka_dot', 'striped']

# Set upload folder path
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        # Predict
        preds = model.predict(img)
        pred_class = class_names[np.argmax(preds)]
        prediction = f"Predicted Pattern: {pred_class}"

    return render_template("index.html", prediction=prediction, filename=filename)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    import sys
    from werkzeug.serving import run_simple
    # Open browser after starting Flask app
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)


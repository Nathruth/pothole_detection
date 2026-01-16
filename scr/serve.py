# src/serve.py
from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error":"No file provided"}), 400
    file = request.files["file"]
    file.save("temp.jpg")
    result = predict("temp.jpg")
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

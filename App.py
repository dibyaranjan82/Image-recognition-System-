from flask import Flask, render_template, request
import os
from utils import preprocess_image, load_model_and_predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(path)
            prediction = load_model_and_predict(path)
            return render_template('index.html', image_path=path, prediction=prediction)
    return render_template('index.html', image_path=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

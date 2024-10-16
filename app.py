import os
import subprocess
import uuid
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file

app = Flask(__name__)
app.secret_key = "b7c2f3d8a5e4f6b3c1a7d9e8b5a6c9e4"
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
object_detection_output_dir = './model1_result'
defect_detection_output_dir = './model2_result'
# object_detection_trained_weight = r"D:\GBM\trained weights\best.pt"
object_detection_trained_weight = "./Models/object_detection.pt"
defect_detection_trained_weight = "./Models/defect_detection.pt"
result_image_path = None
greyscale_image_folder = "./greyscale_images"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(filename)


@app.route('/imageResult')
def image_result():
    global result_image_path
    if result_image_path and os.path.exists(result_image_path):
        return send_file(result_image_path, mimetype='image/jpeg')
    else:
        return "No image available or image not found", 404


@app.route('/uploadImage', methods=['POST'])
def upload_image():
    global result_image_path
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    files = request.files.getlist('image')
    filename = None

    # Save the uploaded image to a specific directory
    for file in files:
        print(file.filename)
        if file.filename:
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

    if filename:
        folder_name = str(uuid.uuid4())
        print(folder_name)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyscale_image_path = os.path.join(greyscale_image_folder, file.filename)
        cv2.imwrite(greyscale_image_path, grayscale_image)

        # First YOLOv9 object detection model
        yolov9_command_object_detection = [
            'python', './yolov9/detect.py',
            '--weights', object_detection_trained_weight,
            '--conf', '0.4',
            '--source', greyscale_image_path,
            '--device', 'cpu',
            '--save-txt',
            '--save-conf',
            '--project', object_detection_output_dir,
            '--name', folder_name
        ]

        try:
            result = subprocess.run(yolov9_command_object_detection, check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print('Model1 prediction done')
            output = result.stdout.decode('utf-8')
            print(output)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8')
            print(error_message)

        # Defect detection model
        model1_result_image = f"{object_detection_output_dir}/{folder_name}/{filename}"
        yolov9_command_defect_detection = [
            'python', './yolov9/detect.py',
            '--weights', defect_detection_trained_weight,
            '--conf', '0.4',
            '--source', model1_result_image,
            '--device', 'cpu',
            '--save-txt',
            '--save-conf',
            '--project', defect_detection_output_dir,
            '--name', folder_name
        ]

        try:
            result_defect_detection = subprocess.run(yolov9_command_defect_detection, check=True,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('Model2 prediction done')
            output = result_defect_detection.stdout.decode('utf-8')
            print(output)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8')
            print(error_message)

        # Set the path for the result image to be accessed in the image_result endpoint
        result_image_path = f"{defect_detection_output_dir}/{folder_name}/{filename}"

        return redirect(url_for('image_result'))


if __name__ == "__main__":
    app.run()

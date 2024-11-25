import argparse
import os
import time
import cv2
from flask import Flask, render_template, request, Response, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from werkzeug.utils import safe_join, send_file

app = Flask(__name__)

# Add configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output\detections'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_unique_filename(filename):
    """Generate a unique filename using timestamp"""
    name, ext = os.path.splitext(filename)
    timestamp = int(time.time() * 1000)
    return f"{name}_{timestamp}{ext}"

def create_required_directories():
    """Ensure required directories exist"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs('results', exist_ok=True)  # Ensure the output directory exists

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return 'No file uploaded', 400

        f = request.files['file']
        if f.filename == '':
            return 'No file selected', 400

        if f and allowed_file(f.filename):
            # Secure the filename and make it unique
            filename = secure_filename(f.filename)
            unique_filename = get_unique_filename(filename)
            
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            f.save(filepath)

            file_extension = unique_filename.rsplit('.', 1)[1].lower()
            
            # Initialize the model
            model = YOLO('best.pt')

            if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
                img = cv2.imread(filepath)
                
                # Generate unique output filename
                output_filename = f"detection_{unique_filename}"
                
                # Perform the detection and save results
                results = model(img, save=True, project=app.config['OUTPUT_FOLDER'], 
                              name='', save_txt=False)

                output_dir = results[0].save_dir if hasattr(results[0], 'save_dir') else None

                if output_dir:
                    print(f"OK: {output_dir}")
                
                # Save additional copy to output/outputImage.jpg
                res_plotted = results[0].plot()
                cv2.imwrite('results/outputImage.jpg', res_plotted)
                print("\033[92mAdditional image saved at: results/outputImage.jpg\033[0m")

                time.sleep(1)
                image_path = os.path.join(output_dir, 'image0.jpg')
                if os.path.exists(image_path):
                    x = image_path
                    print(f"{image_path} exists.")
                else:
                    print(f"{image_path} does not exist.")

                # Rename the output file to our unique filename
                output_files = os.listdir(app.config['OUTPUT_FOLDER'])
                if output_files:
                    latest_file = max(output_files, key=lambda x: os.path.getctime(
                        os.path.join(app.config['OUTPUT_FOLDER'], x)))
                    final_path = os.path.join(output_dir, output_filename)
                    os.rename(
                        os.path.join(app.config['OUTPUT_FOLDER'], latest_file),
                        os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    )
                print(f"\033[92mImage saved at: {image_path}\033[0m")
                relative_image_path = os.path.relpath(image_path, app.config['OUTPUT_FOLDER'])
                print(relative_image_path)
                #return render_template('myImage.html')
                return send_from_directory('results', 'outputImage.jpg')

            elif file_extension == 'mp4':
                # Video processing code remains the same
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output/output.mp4', fourcc, 30.0, 
                                    (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True, project='output', name='detections')
                    res_plotted = results[0].plot()

                    cv2.imshow("result", res_plotted)
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                cap.release()
                out.release()
                #return send_from_directory('output', 'output.mp4')
                return video_feed()

    return render_template('index.html')

@app.route('/output/detections/<path:filename>')
def display(filename):
    try:
        file_path = safe_join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return f"File not found: {filename}", 404
            
        print(f"Serving file from: {file_path}")
        
        return send_file(
            file_path,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=None
        )
        
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return "Error serving file", 500

def get_frame():
    video = cv2.VideoCapture('output/output.mp4')
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_feed/<path:filename>')
def image_feed(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    create_required_directories()
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
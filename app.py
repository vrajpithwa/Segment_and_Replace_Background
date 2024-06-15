
from cosine_sim import cosine_similarity, make_pickle_files
from flask import Flask, render_template, request, jsonify
import os
import cosine_sim
import pickle
from flask_cors import CORS, cross_origin
from flask import Flask, request, render_template, send_from_directory  # Importing necessary modules from Flask
from ultralytics import YOLO  # Importing YOLO object detection model from Ultralytics
from segment_anything import sam_model_registry, SamPredictor  # Importing custom segmentation model
import cv2  # Importing OpenCV for image processing
import numpy as np  # Importing NumPy for numerical operations
import os  # Importing OS module for file operations
from PIL import Image  # Importing PIL for image handling
import subprocess  # Importing subprocess module for running external commands
import shutil
from flask_cors import CORS, cross_origin
from promt_image_gen import query
import io 
from PIL import Image 

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# segment anything VRAJ's CODE
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = './uploads'  # Setting the upload folder path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configuring the upload folder in Flask app

@app.route('/segment')  # Defining a route for the homepage
@cross_origin()

def index():
    return render_template('index.html')  # Rendering an HTML template

# Function to handle uploaded files and run
@app.route('/select_person', methods=['POST'])  # Defining a route for file upload
@cross_origin()
def upload():
    uploaded_file = request.files['file']
    genBg = request.form.get('genBg')
    print(f'genbg {genBg}')
    if(genBg == 'true'):
        background_file = request.form.get('background')

        image_bytes = query(background_file)
        image = Image.open(io.BytesIO(image_bytes))
        background_file = 'visible_image1.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_file)
        print(background_path)
        image.save(background_path)  # Save as PNG
        uploaded_file.save(file_path)
        # background_file.save(background_path)
    else:
        background_file = request.files['background']
        print('zzzzzzzz')
        print(background_file)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_file.filename)
        print(background_path)
        uploaded_file.save(file_path)
        background_file.save(background_path)
    det_img = detect_image(file_path)[0]
    print('select-perseon end reached')
    return {'det_img': det_img, 'file_path': file_path, 'background_path': background_path}

@app.route('/upload', methods=['POST'])  # selecting the index of the person to be shown
@cross_origin()
def select_person():
    selected_indices_str = request.form['selected_indices']  # Getting the selected indices from the form data
    file_path = request.form['file_path']
    background_path = request.form['background_path']

    print('selected indices string')
    print(selected_indices_str)
    selected_indices = [int(idx) for idx in selected_indices_str.replace(",", " ").split()]  # Converting the selected indices to a list of integers
    print(selected_indices)
    print('file_path from select_pers')
    print(file_path)
    processed_image_path = process_image(file_path, background_path,selected_indices_str)
    if processed_image_path is None:
        return "No person detected in the image."
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed_image_path)
 
 
def detect_image(file_path):
    subprocess.run(["yolo", "predict", "model=yolov8n.pt", f"source='{file_path}'"])  # Running YOLO model prediction
    model = YOLO('./yolov8n.pt')  # Initializing YOLO model
    results = model.predict(source=file_path, conf=0.25)  # Running YOLO prediction on the image 
    print(results)
    detected_image_folder = 'runs/detect/predict/'
    detected_image_name = os.listdir(detected_image_folder)[0]
    print("mmmmmmmmmm")
    print(detected_image_name)
    full_detected_image_path = os.path.join(detected_image_folder, detected_image_name)
    # moving the detected image to the upload folder reanming it to detected_image , if already exist then overwrite it
    destination_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_image.jpg')
    if os.path.exists(destination_path):
    # Remove the existing file before moving the new one
        os.remove(destination_path)
    shutil.move(full_detected_image_path, os.path.join(app.config['UPLOAD_FOLDER'], 'detected_image.jpg'))
    #deleting the predict folder
    os.rmdir('runs/detect/predict')
    detected_image= os.path.join(app.config['UPLOAD_FOLDER'], 'detected_image.jpg')
    return [detected_image, results]

def apply_inpainting(image_path, mask_path):
    person_image = cv2.imread(image_path)
    if person_image is None:
        print(f"Failed to load image: {image_path}")
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return None

    # Create a mask for the inpainting region
    inpaint_mask = cv2.bitwise_not(mask)

    # Inpaint the image
    inpainted_image = cv2.inpaint(person_image, inpaint_mask, 3, cv2.INPAINT_TELEA)

    return inpainted_image


def process_image(file_path, background_path,selected_indices):
    results = detect_image(file_path)[1]
   
    image = cv2.imread(file_path)  # Reading the uploaded image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting the image to grayscale
    person_bboxes = []  # Initializing a list for person bounding boxes
    all_person_boxes = []
    for result in results:  # Looping through YOLO results
        boxes = result.boxes  # Getting bounding boxes from YOLO result
        for box in boxes:  # Looping through each bounding box
            cls = box.cls  # Getting the class of the bounding box
            if cls == 0:  # Checking if the class is 'person'
                bbox = box.xyxy.tolist()  # Converting bounding box to list
                person_bboxes.append(bbox)  # Adding the bounding box to person_bboxes list
                all_person_boxes.append(bbox)
    if not person_bboxes:
        return None  # Returning None if no person bounding boxes are found

    if len(person_bboxes) > 1:
        # selected_indices_str = input("Enter the index(es) of the person(s) to segment (separated by commas or spaces): ")  # Getting user input for selected person indices
        selected_indices = [int(idx) for idx in selected_indices.replace(",", " ").split()]  # Converting user input to list of integers
        person_bboxes = [person_bboxes[idx] for idx in selected_indices]  # Filtering person bounding boxes based on selected indices
    
    sam_checkpoint = "./sam_vit_h_4b8939.pth"  # Setting the path to the segmentation model checkpoint
    model_type = "vit_h"  # Setting the segmentation model type
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # Initializing the segmentation model
    predictor = SamPredictor(sam)  # Initializing the segmentation predictor

    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Converting grayscale image to RGB
    predictor.set_image(rgb_image)  # Setting the image for segmentation
    segmentation_masks = []  # Initializing a list for segmentation masks
    for bbox in person_bboxes:  # Looping through person bounding boxes
        input_box = np.array(bbox)  # Converting bounding box to NumPy array
        masks, _, _ = predictor.predict(  # Running segmentation prediction
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        segmentation_masks.append(masks[0])  # Adding segmentation mask to the list

    all_segmentation_masks = []  # Initializing a list for segmentation masks
    for bbox in all_person_boxes:  # Looping through person bounding boxes
        input_box = np.array(bbox)  # Converting bounding box to NumPy array
        masks, _, _ = predictor.predict(  # Running segmentation prediction
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        all_segmentation_masks.append(masks[0])  # Adding segmentation mask to the list

  

    #get mask of individual person and creaate final segmented image also save it in the folder names each person as per the index
    for i,bbox in enumerate(all_person_boxes):
        # combined_mask = np.max(np.stack(all_segmentation_masks), axis=0)  # Combining segmentation masks
        combined_mask = all_segmentation_masks[i]   # Combining segmentation masks
        binary_mask = np.where(combined_mask > 0.5, 1, 0)  # Creating binary mask based on combined mask
        person_image = image * binary_mask[..., np.newaxis]  # Creating final segmented image
        person_image_name = f'person_{i}.jpg'  # Defining name of the output file in each_persons folder
        os.makedirs('each_persons', exist_ok=True)  # Creating the folder to save the final segmented image
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], person_image_name), person_image.astype(np.uint8))  # Saving the final segmented image
        # save the segmentation mask of seach person in diff image
        mask = binary_mask * 255
        
        cv2.imwrite("each_persons/mask_"+person_image_name, mask.astype(np.uint8))
        print(f'person_image_path: {person_image_name}')


    
    # Combining all masks into one mask
    
    combined_mask = np.max(np.stack(segmentation_masks), axis=0)  # Combining segmentation masks
    binary_mask = np.where(combined_mask > 0.5, 1, 0)  # Creating binary mask based on combined mask

    background_image = cv2.imread(background_path)  # Reading the background image
    background_image = cv2.resize(background_image, (rgb_image.shape[1], rgb_image.shape[0]))  # Resizing background image
    final_image = background_image * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]  # Creating final segmented image
    
    processed_image_name = 'processed_image.jpg'  # Setting the processed image name
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_image_name) 
     # Creating the processed image path
    print(f'processed_image_path: {processed_image_path}')
    # there will be always one image inside the folder getting that image
    cv2.imwrite(processed_image_path, final_image.astype(np.uint8))  # Saving the final segmented image

    # for i, bbox in enumerate(all_person_boxes):
    #     # persons image is in uploads folder
    #     person_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'person_{i}.jpg')
    #     # mask is in each_persons folder oustide the uploads folder
    #     mask_path = os.path.join('each_persons', f'mask_person_{i}.jpg')
    #     inpainted_person_image = apply_inpainting(person_image_path, mask_path, image)
    #     if inpainted_person_image is not None:
    #         inpainted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'each_persons', f'inpainted_person_{i}.jpg')
    #         cv2.imwrite(inpainted_image_path, inpainted_person_image)
    #         print(f"Inpainted image saved: {inpainted_image_path}")    
    return processed_image_name   # Returning the processed image name 

@app.route('/uploads/<filename>')  # Defining a route for uploaded files
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  # Sending the uploaded file to the client


# get matching images DARSHAN's CODE


# #getting a new image and finding the closest match
# import cosine_sim
# import pickle
# import os
from cosine_sim import cosine_similarity, make_pickle_files

@cross_origin
@app.route('/', methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        # Get the uploaded images
        person1_image = request.files.get('person1')
        person2_image = request.files.get('person2')

        # Save the images to the server
        person1_path = os.path.join('uploads', person1_image.filename)
        person1_image.save(person1_path)

        # Process the person1 image
        person1_embedding = cosine_sim.make_pickle_files(person1_path)
        print("Person1 embedding")
        print(len(person1_embedding))
        if len(person1_embedding) > 1:
            return jsonify({'images': [],'response': 'Please Enter a single face image'})
        if len(person1_embedding) == 0:
            return jsonify({'images': [],'response': 'No face detected in the image'})
        person1_embedding = cosine_sim.make_pickle_files(person1_path)[0]
        
        output_folder = 'data_new/new_dbscann_jitter_1new_eps0.34_large_final_new'
        person1_folder_path = cosine_sim.find_closest_match_majority(person1_path, output_folder, 0.95)
        print(f'Person1 folder path: {person1_folder_path}')
        if(not person1_folder_path):
            return jsonify({'images': []})
        image_paths_person_1 = []
        # vraj_darshan.1.jpg , name will be like this we have to extract vraj_darshan
        for image in os.listdir(person1_folder_path):
            image_name = image.split('.')[0]
            image_name += '.jpg'
            image_paths_person_1.append(os.path.join('data_new/train_dataset_new', image_name))
        images_with_person1 = list(set(image_paths_person_1))
        # images_with_person1 = [ os.path.join('data_new/train_dataset_new', image) for image in os.listdir(person1_folder_path) if image.endswith('.jpeg') or image.endswith('.jpg') or image.endswith('.png')]

        images_with_both_persons = []

        # Check if person2 image is provided
        if person2_image:
            person2_path = os.path.join('uploads', person2_image.filename)
            person2_image.save(person2_path)

            # Process the person2 image
            person2_embedding = cosine_sim.make_pickle_files(person2_path)[0]
            person2_folder_path = cosine_sim.find_closest_match_majority(person2_path, output_folder, 0.9)

            cropped_face_folder = 'data_new/cropped_faces_new'
            original_images_folder = 'data_new/train_dataset_new'

            for image in os.listdir(person1_folder_path):
                postfix = image.split('.')[0]
                person1_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
                for embedding in os.listdir(person1_cropped_folder_path):
                    if embedding.endswith('.pkl'):
                        with open(os.path.join(person1_cropped_folder_path, embedding), 'rb') as f:
                            embeddings = pickle.load(f)
                            embedding_data = embeddings[0]
                            if cosine_similarity(embedding_data, person2_embedding) > 0.95:
                                images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpg'))

            for image in os.listdir(person2_folder_path):
                postfix = image.split('.')[0]
                person2_cropped_folder_path = os.path.join(cropped_face_folder, postfix)
                for embedding in os.listdir(person2_cropped_folder_path):
                    if embedding.endswith('.pkl'):
                        with open(os.path.join(person2_cropped_folder_path, embedding), 'rb') as f:
                            embeddings = pickle.load(f)
                            embedding_data = embeddings[0]
                            if cosine_similarity(embedding_data, person1_embedding) > 0.95:
                                images_with_both_persons.append(os.path.join(original_images_folder, embedding.split('.')[0] + '.jpg'))

            print(images_with_both_persons)
            images_with_both_persons = list(set(images_with_both_persons))
            return jsonify({'images': images_with_both_persons})

        else:
            
            print(images_with_person1)
            return jsonify({'images': images_with_person1})

    # return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
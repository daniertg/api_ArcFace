# face_recognition_model.py
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceRecognitionModel:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        self.registered_faces = {}

        # Ensure the upload folder exists
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

        # Initialize the FaceAnalysis model with RetinaFace and ArcFace
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use GPUExecutionProvider if CUDA is available
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def save_image(self, image_file):
        try:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(self.upload_folder, filename)
            image_file.save(image_path)
            return image_path
        except Exception as e:
            print(f"Error saving image: {e}")
            raise

    def encode_face(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            faces = self.app.get(image)
            if len(faces) == 0:
                return []
            
            # Extract embeddings from detected faces
            face_encodings = [face.embedding for face in faces]
            return face_encodings
        except Exception as e:
            print(f"Error encoding face: {e}")
            raise

    def register_face(self, image_file, user_name):  # Add user_name parameter
        try:
            image_path = self.save_image(image_file)
            face_encodings = self.encode_face(image_path)
            
            if len(face_encodings) > 0:
                # Register the face with the first encoding and associate with user_name
                self.registered_faces[user_name] = face_encodings[0]
                return {"status": "success", "message": f"Face registered successfully for {user_name}"}, 200
            else:
                return {"status": "error", "message": "No face detected"}, 400
        except Exception as e:
            print(f"Error registering face: {e}")
            return {"status": "error", "message": str(e)}, 500

    def verify_face(self, image_file):
        try:
            image_path = self.save_image(image_file)
            face_encodings = self.encode_face(image_path)
            
            if len(face_encodings) > 0:
                for name, registered_face in self.registered_faces.items():
                    similarity = np.dot(registered_face, face_encodings[0]) / (np.linalg.norm(registered_face) * np.linalg.norm(face_encodings[0]))
                    if similarity > 0.95:  # Adjust the threshold based on your needs
                        return {"status": "success", "name": name, "message": "Face verified successfully"}, 200

            return {"status": "error", "message": "Face verification failed"}, 400
        except Exception as e:
            print(f"Error verifying face: {e}")
            return {"status": "error", "message": str(e)}, 500

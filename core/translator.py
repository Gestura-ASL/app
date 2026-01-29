import pandas as pd
import tensorflow as tf
import numpy as np
import os

class SignModel:
    def __init__(self, model_path="models/model.tflite", train_csv_path="assets/train.csv"):
        self.ROWS_PER_FRAME = 543
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        
        train = pd.read_csv(train_csv_path)
        train['sign_ord'] = train['sign'].astype('category').cat.codes
        self.ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, landmark_buffer):
        if not landmark_buffer:
            return None, 0.0

        # 1. Combine buffer into one DataFrame
        data = pd.concat(landmark_buffer)
        
        # 2. Extract X, Y, Z values (Preserve NaNs!)
        xyz_data = data[['x', 'y', 'z']].values
        
        # 3. Dynamic Reshape (Exactly like original model_inference.py)
        # We do NOT pad. We use whatever number of frames we captured.
        current_frames = len(landmark_buffer)
        xyz_np = xyz_data.reshape(current_frames, self.ROWS_PER_FRAME, 3)
        xyz_np = xyz_np.astype(np.float32)

        try:
            # 4. Run Inference
            prediction = self.prediction_fn(inputs=xyz_np)
            sign_logits = prediction['outputs']
            
            # 5. Convert Logits to Probability
            # Your logs showed values like 2.8, 4.0. These are logits.
            # We must Softmax them to get a 0-1 confidence score.
            sign_probs = self._softmax(sign_logits)
            
            top_index = np.argmax(sign_probs)
            confidence = sign_probs[top_index]
            sign_label = self.ORD2SIGN.get(top_index, "Unknown")
            
            return sign_label, float(confidence)
        except Exception as e:
            print(f"Inference Error: {e}")
            return None, 0.0
        
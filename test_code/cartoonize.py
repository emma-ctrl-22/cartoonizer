import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io

# Set up Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
log_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'cartoonization_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Global variables for TensorFlow session and model
sess = None
input_photo = None
network_out = None
final_out = None
saver = None

def initialize_model(model_path):
    """Initialize the TensorFlow model and session"""
    global sess, input_photo, network_out, final_out, saver
    
    try:
        logger.info(f"Initializing model from: {model_path}")
        
        # Verify model files exist
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found: {model_path}")
            return False
        if not tf.train.latest_checkpoint(model_path):
            logger.error(f"No model checkpoint found in: {model_path}")
            return False

        # Define the network
        input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
        network_out = network.unet_generator(input_photo)
        final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

        # Initialize variables
        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)
        
        # Configure and start session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        # Load the model
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        logger.info(f"Loading model from: {latest_checkpoint}")
        saver.restore(sess, latest_checkpoint)
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        return False

def resize_crop(image):
    """Resize and crop the input image"""
    try:
        h, w, c = np.shape(image)
        logger.info(f"Original image dimensions: {h}x{w}x{c}")
        
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
            logger.info(f"Resized to: {h}x{w}")
            
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        logger.info(f"Final cropped dimensions: {h}x{w}")
        
        return image
    except Exception as e:
        logger.error(f"Error in resize_crop: {str(e)}")
        raise

def process_image(image):
    """Process a single image through the model"""
    try:
        # Resize and crop the image
        image = resize_crop(image)
        
        # Prepare the image for the model
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        
        # Run the network
        logger.info("Running network...")
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise

@app.route('/cartoonize', methods=['POST'])
def cartoonize_endpoint():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Process the image
        cartoon = process_image(image)
        
        # Convert the cartoonized image to base64
        _, buffer = cv2.imencode('.jpg', cartoon)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'message': 'Image cartoonized successfully'
        })
    
    except Exception as e:
        logger.error(f"Error in cartoonize endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_initialized': sess is not None
    })

if __name__ == '__main__':
    # Initialize the model
    model_path = 'saved_models'
    if not initialize_model(model_path):
        logger.error("Failed to initialize model. Exiting...")
        exit(1)
    
    # Start the Flask server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
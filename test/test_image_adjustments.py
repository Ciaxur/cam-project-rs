#!/usr/bin/env python3
import logging
import requests
import base64
import math
import numpy
from io import BytesIO
from PIL import Image, UnidentifiedImageError

API_HOST = 'localhost'
API_PORT = 3000
CLIENT_CERT_PATH = 'certs/cam-client1.crt'
CLIENT_KEY_PATH = 'certs/cam-client1.key'
TRUSTED_CA_PATH = 'certs/4bitCA.crt'

snap_endpoint = f'https://{API_HOST}:{API_PORT}/camera/snap'
list_endpoint = f'https://{API_HOST}:{API_PORT}/camera/list'

def apply_image_adjustments(image: Image, cam_ip: str, adjustment_mp: dict) -> numpy.ndarray:
    """
        Adjusts a given image with respect to the stored camera adjustments.

        Args:
            image: The image instance to adjust.
            cam_ip: The camera's IP used as a key to lookup corresponding adjustments for.

        Return:
            Adjusted image as a numpy array.
    """
    # Convert the image to a numpy array.
    image_np = numpy.asarray(image)

    if cam_ip in adjustment_mp:
        cam_adjustments = adjustment_mp[cam_ip]
        image_frame_crop_height = cam_adjustments['CropFrameHeight']
        image_frame_crop_width = cam_adjustments['CropFrameWidth']
        image_frame_crop_x = cam_adjustments['CropFrameX']
        image_frame_crop_y = cam_adjustments['CropFrameY']

        # Crop the image
        h, w = image_np.shape[:2]
        h_ajustment = math.floor(h * image_frame_crop_height)
        w_ajustment = math.floor(w * image_frame_crop_width)
        x, y = image_frame_crop_x, image_frame_crop_y

        logging.debug(f'Applying frame crop H={h_ajustment} H={w_ajustment} on x={x}, y={y}')
        new_h = h - h_ajustment
        new_w = w - w_ajustment
        image_np = image_np[y:y+new_h, x:x+new_w]

    return image_np

def main():
    logging.basicConfig(level=logging.INFO)

    # Camera metadata.
    API_CAMERA_ADJUSTMENTS_MAP = {}
    API_CAMERA_MAP = {}
    captured_images = []


    # Query camera states.
    logging.info(f'Querying the state of connected cameras')
    with requests.get(
        list_endpoint,
        cert=(CLIENT_CERT_PATH, CLIENT_KEY_PATH),
        verify=TRUSTED_CA_PATH,
        json={},
    ) as response:
        if response.status_code == 200:
            obj = response.json()
            # Store the current state.
            for cam in obj['Cameras']:
                cam_ip = cam['IP']
                cam_adjustments = cam['Adjustment']
                API_CAMERA_ADJUSTMENTS_MAP[cam_ip] = cam_adjustments
            logging.debug(f'Camera list query result: {API_CAMERA_ADJUSTMENTS_MAP}')

        else:
            raise Exception(f"API Endpoint connection failed with status code {response.status_code}")

    # Query a snapshot.
    with requests.get(
        snap_endpoint,
        cert=(CLIENT_CERT_PATH, CLIENT_KEY_PATH),
        verify=TRUSTED_CA_PATH,
        json={},
    ) as response:
        if response.status_code == 200:
            obj = response.json()
            # Store the current state.
            API_CAMERA_MAP = obj['cameras']
        else:
            raise Exception(f"API Endpoint connection failed with status code {response.status_code}")

    # Extract and apply adjustments to images.
    for camera_ip in API_CAMERA_MAP:
        cam = API_CAMERA_MAP[camera_ip]
        cam_name = cam['name']
        cam_data = cam['data']
        device_name = f"{cam_name}:{camera_ip}"

        # Extract current camera's image.
        try:
            img_b64 = cam_data
            if not img_b64:
                continue

            img = base64.b64decode(img_b64)
            _img = Image.open(BytesIO(img))
            nd_array = apply_image_adjustments(_img, camera_ip, API_CAMERA_ADJUSTMENTS_MAP)
            logging.debug(f'[loop] ESP Device {device_name} captured image -> {nd_array.shape}')
            captured_images.append((device_name, nd_array))
        except UnidentifiedImageError:
                logging.error(f'[loop] Failed to interpret image from ESP Device {device_name}')
        except Exception as e:
            logging.warning(f'[loop] Unknown exception while interpreting image from ESP Device {device_name}: {e}')

    # DEBUG:
    logging.info(API_CAMERA_ADJUSTMENTS_MAP)
    logging.info(API_CAMERA_MAP.keys())

    for img in captured_images:
        device_name, img_array = img
        print(device_name)

        img = Image.fromarray(img_array)
        print(img)
        img.show()



if __name__ == '__main__':
    main()
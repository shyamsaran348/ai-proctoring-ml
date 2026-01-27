import cv2
import os
from django.conf import settings


def validate_image(image_path):
    """Validate that an image file is readable and contains valid image data"""
    try:
        if not os.path.exists(image_path):
            return False, "File does not exist"
        
        if os.path.getsize(image_path) == 0:
            return False, "File is empty"
        
        # Try to read the image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, "Cannot read image with OpenCV"
        
        # Check if image has valid dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False, "Image has invalid dimensions"
        
        return True, "Image is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def capture_reference_image(student_id: str) -> str:
	"""Capture a single reference image from webcam and save to media/students.
	Returns absolute file path of the saved image.
	"""
	# Ensure media directory exists
	# Save captured references under static/uploads/students to match registration storage
	students_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', 'students')
	os.makedirs(students_dir, exist_ok=True)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError('Could not open webcam')

	print('[INFO] Press SPACE to capture image or ESC to exit')
	path = ''
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				continue
			cv2.imshow('Capture Reference - Press SPACE', frame)
			key = cv2.waitKey(1)
			if key % 256 == 27:  # ESC
				raise RuntimeError('Capture cancelled by user')
			elif key % 256 == 32:  # SPACE
				filename = f"{student_id}_reference.jpg"
				path = os.path.join(students_dir, filename)
				cv2.imwrite(path, frame)
				print(f"[INFO] Image saved to {path}")
				
				# Validate the captured image
				is_valid, message = validate_image(path)
				if is_valid:
					print(f"[INFO] Image validation successful: {message}")
					break
				else:
					print(f"[WARNING] Image validation failed: {message}")
					print("[INFO] Please try capturing again...")
	finally:
		cap.release()
		cv2.destroyAllWindows()

	return path


__all__ = ['capture_reference_image']

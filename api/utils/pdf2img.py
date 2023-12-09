from pdf2image import convert_from_path
from datetime import datetime
import pathlib
import shutil
import os

def get_poppler_path():
    pdftoppm_path = shutil.which("pdftoppm")
    if pdftoppm_path:
        return pathlib.Path(pdftoppm_path).parent
    else:
        return None

def pdf2img(pdf_path: str, parent_path: str, pdf_name: str):
	current_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
	result_folder_name = f"{pdf_name}_{current_time}"
	result_path = os.path.join(parent_path, "pdf2img_results", result_folder_name)
	os.makedirs(result_path, exist_ok=True)  # This line will create the directory if it doesn't exist

	poppler_path = get_poppler_path()
	print(poppler_path)

	try:
		images = convert_from_path(str(pdf_path), poppler_path=poppler_path)
		for i, img in enumerate(images):
			img.save(f'{result_path}/output_{i}.jpg', 'JPEG')
        
		result = "success"
		print("Result:", result)
		return result

	except Exception as e:
		result = f"An error occurred: {e}"
		print("Result:", result)
		return result

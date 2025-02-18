import os
import json

def create_json_entry(target_path, prompt):
    json_entries = []
    for case_folder in os.listdir(target_path):
        target_case_path = os.path.join(target_path, case_folder)
        

        if os.path.exists(target_case_path):
            for image_file in os.listdir(target_case_path):
                if image_file.endswith('.jpg'):
                    target_image_path = os.path.join('data',target_case_path, image_file)
                    json_entry = {"target": target_image_path, "prompt": prompt}
                    json_entries.append(json_entry)

    return json_entries

#target_folder = "./SUN/Train/Negative/"  # Replace with your target_folder (folder containing training images)
target_folder = "./isit_umr/Adenomatous/NBI/" # Replace with your target_folder (folder containing training images)

#prompt_value = "colonoscopy image without polyp"
prompt_value = "colonoscopy image with adenomatous polyp, narrow band imaging"  # Change prompt according to the target file data

json_entries = create_json_entry(target_folder, prompt_value)


# Save each JSON entry as a separate object in the file
with open("train.json", "a") as json_file:
    for entry in json_entries:
        json.dump(entry, json_file)
        json_file.write("\n")


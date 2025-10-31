import os
import xml.etree.ElementTree as ET

# Automatically locate XML folder
base_dir = os.path.dirname(__file__)
xml_folder = os.path.join(base_dir, "fire_face_detector", "datasets", "labels", "train")
output_folder = os.path.join(base_dir, "fire_face_detector", "datasets", "labels", "train_yolo")

# Define your classes
class_map = {
    "fire": 0,
    "face": 1
}

os.makedirs(output_folder, exist_ok=True)

print("Looking for XML files in:", xml_folder)

if not os.path.exists(xml_folder):
    print("❌ Folder not found! Check your path.")
else:
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} XML files")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        txt_file = os.path.join(output_folder, xml_file.replace(".xml", ".txt"))
        with open(txt_file, "w") as f:
            for obj in root.findall("object"):
                label = obj.find("name").text.lower().strip()

                # Check if label exists in your map
                if label not in class_map:
                    print(f"⚠️ Warning: Unknown class '{label}' in {xml_file}, skipping...")
                    continue

                class_id = class_map[label]

                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # Convert to YOLO format (normalized)
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

    print(f"✅ Conversion complete! Saved YOLO labels to:\n{output_folder}")

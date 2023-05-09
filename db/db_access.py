import os
from gridfs import GridFS


def download_image(image_id, output_file_path, db):
    gridfs = GridFS(db)
    
    if gridfs.exists(image_id):
        with open(output_file_path, "wb") as output_file:
            output_file.write(gridfs.get(image_id).read())
        print(f"Image '{image_id}' downloaded to '{output_file_path}'.")
    else:
        print(f"Image with ID '{image_id}' not found in GridFS.")

def upload_image(image_path, gridfs):
    with open(image_path, "rb") as image_file:
        image_name = os.path.basename(image_path)
        image_id = gridfs.put(image_file, filename=image_name)
    return image_id

def create_tree(path, db):
    gridfs = GridFS(db)

    tree = {"name": os.path.basename(path), "type": "directory", "children": []}
    for entry in os.scandir(path):
        if entry.is_file():
            image_id = upload_image(entry.path, gridfs)
            tree["children"].append({"name": entry.name, "type": "file", "image_id": image_id})
            print(f"Inserted {entry.name} in db")
        elif entry.is_dir():
            tree["children"].append(create_tree(entry.path, db))
            print(f"Creating directory {entry.path}")
    return tree

def recreate_tree(tree, parent_path, db):
    if tree["type"] == "directory":
        new_dir_path = os.path.join(parent_path, tree["name"])
        os.makedirs(new_dir_path, exist_ok=True)
        
        for child in tree["children"]:
            recreate_tree(child, new_dir_path, db)
    elif tree["type"] == "file":
        output_file_path = os.path.join(parent_path, tree["name"])
        download_image(tree["image_id"], output_file_path, db)

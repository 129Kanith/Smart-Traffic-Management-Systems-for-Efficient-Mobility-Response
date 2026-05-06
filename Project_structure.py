import os

# Define the project structure
project_structure = {
    "smart_traffic_system": [
        "app.py",
        "requirements.txt",
        {"videos": []},
        {"outputs": []},
        {"models": []},
        {"modules": []},
        {"assets": []}
    ]
}

def create_structure(base_path, structure):
    for item in structure:
        if isinstance(item, str):
            # Handle files
            file_path = os.path.join(base_path, item)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("")  # create empty file
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)

# Run the script
base_dir = os.getcwd()  # current working directory
create_structure(base_dir, project_structure["smart_traffic_system"])

print("Project structure created successfully!")

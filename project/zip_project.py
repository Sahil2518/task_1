import os
import zipfile

def make_zip(source_dir, output_filename):
    print(f"Creating {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Exclude unnecessary directories
            dirs[:] = [d for d in dirs if d not in ('__pycache__', '.ipynb_checkpoints', '.gradio', '.git', 'node_modules')]
            for file in files:
                # Exclude existing zip files or the zip script itself
                if file.endswith('.zip') or file == 'zip_project.py':
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print("Done!")

if __name__ == "__main__":
    make_zip('.', 'task14_submission.zip')

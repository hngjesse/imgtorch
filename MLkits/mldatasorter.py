from pathlib import Path
import shutil

class MLDataSorter:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        
    def flatten(self, target_dir, rename_if_conflict=True):
        """
        Flatten all files from source_dir (including subfolders) into target_dir.

        Parameters:
        -----------
        source_dir : str or Path
            Root directory to flatten.
        target_dir : str or Path
            Destination directory for flattened files.
        rename_if_conflict : bool
            Whether to rename files to avoid overwriting (default True).
        """
        source = Path(self.source_dir)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        counter = 0
        num_copied = 0

        for file in source.rglob('*.*'):
            if file.is_file():
                dest_path = target / file.name

                if dest_path.exists() and rename_if_conflict:
                    # Only rename if there's a conflict
                    while True:
                        new_name = f"{file.stem}_{counter}{file.suffix}"
                        dest_path = target / new_name
                        counter += 1
                        if not dest_path.exists():
                            break

                shutil.copy2(file, dest_path)
                num_copied += 1

        print(f"Flattened {num_copied} files into '{target}'.")

    def separate(self, classes: list[str], output_folder, move_files=False, rename=False):
        """
        Separate files into subfolders based on keyword matches in filenames (case-insensitive).

        Parameters:
        -----------
        classes : list[str]
            List of keywords to match in filenames.
        output_folder : str or Path
            Folder where the class subfolders will be created.
        move_files : bool
            If True, files will be moved. If False, files will be copied.
        rename : bool
            If True, rename the files to <classname><index>.<ext> (e.g., cat1.jpg, cat2.png).
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        matches = 0
        counters = {cls: 1 for cls in classes}
        classes_lower = [cls.lower() for cls in classes]

        for file in self.source_dir.rglob('*.*'):
            if file.is_file():
                filename_lower = file.name.lower()
                for keyword_lower, original_case in zip(classes_lower, classes):
                    if keyword_lower in filename_lower:
                        target_dir = output_folder / original_case
                        target_dir.mkdir(parents=True, exist_ok=True)

                        if rename:
                            idx = counters[original_case]
                            new_filename = f"{original_case.lower()}_{idx}{file.suffix.lower()}"
                            counters[original_case] += 1
                        else:
                            new_filename = file.name

                        target_path = target_dir / new_filename

                        if move_files:
                            shutil.move(file, target_path)
                        else:
                            shutil.copy2(file, target_path)

                        matches += 1
                        break  # Only assign to first matching class

        print(f"Separated {matches} files into {len(classes)} folders at '{output_folder}'.")
    def rename_all(self, keyword, output_folder):
        """
        Copy and rename all files in source_dir into a new folder using keyword-based names.

        Files will be named as: keyword_1.ext, keyword_2.ext, ...

        Parameters:
        -----------
        keyword : str
            Prefix for the new filenames.
        output_folder : str or Path
            Target directory where renamed files will be saved.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in self.source_dir.iterdir() if f.is_file()])
        count = 1
        copied = 0

        for file in files:
            new_name = f"{keyword.lower()}_{count}{file.suffix.lower()}"
            target_path = output_folder / new_name
            shutil.copy2(file, target_path)
            count += 1
            copied += 1

        print(f"Copied and renamed {copied} files into '{output_folder}' as '{keyword}_<n>.<ext>'.")




source = 'flatten_data22'
target = 'grouped_data22'
classes = ['23243543', 'cat', 'Airplane', 'dog', 'human', 'solar']
Dir1 = MLDataSorter(source)


Dir1.separate(classes, target, rename=True)


Dir2 = MLDataSorter('picture/cat')
Dir2.rename_all('cat','picture/cat_renamed')
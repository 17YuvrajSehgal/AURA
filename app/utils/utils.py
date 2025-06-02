import os


def read_files_content(file_paths):
    """
    Reads the content of a list of files and returns a list of their contents as strings.

    Args:
        file_paths (list): A list of file paths to read.

    Returns:
        list: A list of strings, each containing the content of a corresponding file.

    Notes:
        Logs any files that cannot be read and continues with others.
    """
    contents = []

    for file_path in file_paths:
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue

            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()  # Remove leading/trailing whitespaces

            contents.append(content)

        except Exception as e:
            print(f"[ERROR] Error reading file {file_path}: {str(e)}")

    return contents


# Helper function to peek into a file
def read_file_peek(file_path: str, num_lines: int = 5) -> str:
    try:
        with open(file_path, 'r') as file:
            lines = [file.readline().strip() for _ in range(num_lines)]
        return f"File: {file_path}\nSample:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"
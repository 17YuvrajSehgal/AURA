import bz2
import gzip
import logging
import lzma
import os
import shutil
import tarfile
import zipfile
from concurrent.futures import as_completed

import rarfile
import zstandard
from py7zr import SevenZipFile

# Configure logging
log_path = os.path.abspath('../../algo_outputs/logs/md_file_extraction.log')
log_dir = os.path.dirname(log_path)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

# Configure rarfile backend (ensure you have unrar or rar installed and available in PATH)
rarfile.UNRAR_TOOL = "unrar"  # or "rar" if available


def extract_md_files_from_archive(archive_path, extraction_base_dir):
    """
    Extract .md files from an archive to a specific directory.
    """
    md_files = []

    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    for ext in ('.tar.gz', '.tar.bz2', '.tar.xz', '.tgz'):
        if archive_path.endswith(ext):
            archive_name = os.path.splitext(os.path.splitext(os.path.basename(archive_path))[0])[0]
            break

    archive_extraction_dir = os.path.join(extraction_base_dir, archive_name)
    os.makedirs(archive_extraction_dir, exist_ok=True)

    try:
        # ZIP files
        if archive_path.endswith(".zip"):
            logging.info(f"Processing {archive_name} using .zip...")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith(".md"):
                        destination_path = os.path.join(archive_extraction_dir, file)
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        zip_ref.extract(file, archive_extraction_dir)
                        md_files.append(destination_path)

        # TAR files
        elif archive_path.endswith((".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")):
            logging.info(f"Processing {archive_name} using TAR-based archive...")
            try:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    for member in tar_ref.getmembers():
                        if member.isfile() and member.name.endswith(".md"):
                            destination_path = os.path.join(archive_extraction_dir, member.name)
                            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                            with tar_ref.extractfile(member) as source, open(destination_path, "wb") as out_f:
                                shutil.copyfileobj(source, out_f)
                            md_files.append(destination_path)
            except tarfile.TarError as e:
                logging.error(f"Error extracting TAR archive {archive_path}: {e}")

        # 7z files
        elif archive_path.endswith(".7z"):
            logging.info(f"Processing {archive_name} using .7z...")
            with SevenZipFile(archive_path, mode="r") as seven_z_ref:
                for file in seven_z_ref.getnames():
                    if file.endswith(".md"):
                        destination_path = os.path.join(archive_extraction_dir, file)
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        seven_z_ref.extract(targets=[file], path=archive_extraction_dir)
                        md_files.append(destination_path)

        # GZ files
        elif archive_path.endswith(".gz"):
            logging.info(f"Processing {archive_name} using .gz...")
            try:
                base_name = os.path.basename(archive_path)[:-3]  # remove .gz
                if base_name.endswith(".md"):
                    os.makedirs(archive_extraction_dir, exist_ok=True)
                    output_path = os.path.join(archive_extraction_dir, base_name)
                    with gzip.open(archive_path, "rb") as gz_ref, open(output_path, "wb") as out_f:
                        out_f.write(gz_ref.read())
                    logging.info(f"Extracted .md file: {output_path}")
                    md_files.append(output_path)
                else:
                    logging.info(f"Skipped non-.md file inside .gz: {base_name}")
            except Exception as e:
                logging.error(f"Error processing .gz file {archive_path}: {e}")

        # BZ2 files
        elif archive_path.endswith((".bz2", ".bz")):
            logging.info(f"Processing {archive_name} using .bz2 or .bz...")
            base_name = os.path.basename(archive_path)
            if archive_path.endswith(".bz2"):
                base_name = base_name[:-4]
            elif archive_path.endswith(".bz"):
                base_name = base_name[:-3]

            output_path = os.path.join(archive_extraction_dir, base_name)
            with bz2.BZ2File(archive_path, "rb") as bz2_ref, open(output_path, "wb") as out_f:
                out_f.write(bz2_ref.read())
            if output_path.endswith(".md"):
                md_files.append(output_path)

        # XZ files
        elif archive_path.endswith(".xz"):
            logging.info(f"Processing {archive_name} using .xz...")
            base_name = os.path.basename(archive_path)[:-3]
            output_path = os.path.join(archive_extraction_dir, base_name)
            with lzma.open(archive_path, "rb") as xz_ref, open(output_path, "wb") as out_f:
                out_f.write(xz_ref.read())
            if output_path.endswith(".md"):
                md_files.append(output_path)

        # RAR files
        elif archive_path.endswith(".rar"):
            logging.info(f"Processing {archive_name} using .rar...")
            with rarfile.RarFile(archive_path, "r") as rar_ref:
                for file in rar_ref.namelist():
                    if file.endswith(".md"):
                        destination_path = os.path.join(archive_extraction_dir, file)
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        rar_ref.extract(file, archive_extraction_dir)
                        md_files.append(destination_path)

        # ZST files
        elif archive_path.endswith(".zst"):
            logging.info(f"Processing {archive_name} using .zst...")
            try:
                base_name = os.path.splitext(os.path.basename(archive_path))[0]
                decompressed_path = os.path.join(archive_extraction_dir, base_name)
                dctx = zstandard.ZstdDecompressor()

                # Decompress the .zst file
                with open(archive_path, "rb") as fh_in, open(decompressed_path, "wb") as fh_out:
                    dctx.copy_stream(fh_in, fh_out)

                # Check if the decompressed file is a .tar archive
                if tarfile.is_tarfile(decompressed_path):
                    logging.info(f"Detected .tar archive inside .zst: {decompressed_path}")
                    with tarfile.open(decompressed_path, "r:*") as tar_ref:
                        for member in tar_ref.getmembers():
                            if member.isfile() and member.name.endswith(".md"):
                                destination_path = os.path.join(archive_extraction_dir, member.name)
                                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                                with tar_ref.extractfile(member) as source, open(destination_path, "wb") as out_f:
                                    shutil.copyfileobj(source, out_f)
                                md_files.append(destination_path)
                    # Remove the intermediate .tar file after processing
                    os.remove(decompressed_path)
                    logging.info(f"Removed intermediate .tar file: {decompressed_path}")
                elif decompressed_path.endswith(".md"):
                    # If the decompressed file itself is an .md file
                    md_files.append(decompressed_path)
                else:
                    logging.info(f"Skipped non-.md file inside .zst: {decompressed_path}")
                    os.remove(decompressed_path)  # Clean up non-.md decompressed file
            except Exception as e:
                logging.error(f"Error extracting .zst archive {archive_path}: {e}")


    except Exception as e:
        logging.error(f"Error processing archive {archive_path}: {e}")

    return md_files


def find_md_files_in_artifact(artifact_dir, artifact_name, output_base_dir):
    """
    Find all .md files in the given artifact directory and its archives.

    Args:
        artifact_dir (str): Path to the artifact directory.
        artifact_name (str): The name of the artifact (used to organize output).
        output_base_dir (str): Base directory for extracted files.

    Returns:
        list: List of paths to .md files.
    """
    md_files = []
    artifact_output_dir = os.path.join(output_base_dir, artifact_name)
    os.makedirs(artifact_output_dir, exist_ok=True)

    for root, _, files in os.walk(artifact_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, artifact_dir)
            destination_dir = os.path.join(artifact_output_dir, relative_path)
            os.makedirs(destination_dir, exist_ok=True)

            # If it's a direct .md file, just copy it
            if file.endswith(".md"):
                destination_path = os.path.join(destination_dir, file)
                try:
                    shutil.copy(file_path, destination_path)
                    md_files.append(destination_path)
                except Exception as e:
                    logging.error(f"Failed to copy .md file {file_path}: {e}")

            # If it's a known archive, extract .md files
            elif file.lower().endswith((
                    ".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz",
                    ".7z", ".gz", ".bz2", ".bz", ".xz", ".rar",
                    ".ttgz", ".zst",
                    ".001", ".002", ".003", ".004", ".005", ".006", ".007"
            )):
                try:
                    md_files.extend(
                        extract_md_files_from_archive(file_path, destination_dir)
                    )
                except Exception as e:
                    logging.error(f"Failed to extract archive {file_path}: {e}")

    return md_files


def process_artifact_directory(artifact_dir, output_base_dir):
    """
    Processes an artifact directory by iterating through its contents,
    calling find_md_files_in_artifact for each subdirectory.
    """
    if os.path.exists(artifact_dir):
        for artifact_name in os.listdir(artifact_dir):
            artifact_path = os.path.join(artifact_dir, artifact_name)
            if os.path.isdir(artifact_path):
                logging.info(f"Processing artifact directory: {artifact_name}")
                try:
                    find_md_files_in_artifact(
                        artifact_path, artifact_name, output_base_dir
                    )
                except Exception as e:
                    logging.error(f"Failed to process artifact {artifact_name}: {e}")
            else:
                logging.warning(f"Skipping non-directory: {artifact_path}")
    else:
        logging.warning(f"Artifact directory does not exist: {artifact_dir}")


def main():
    # List of artifact directories
    artifact_directories = [
        "D:\\misc_artifact_downloads",
        "D:\\acm_artifact_downloads",
        "D:\\acm_zenodo_downloads"
    ]

    # Base directory to store extracted .md files
    output_base_dir = '../../algo_outputs/md_file_extraction'
    os.makedirs(output_base_dir, exist_ok=True)

    # Use ThreadPoolExecutor to process each artifact directory in parallel
    # Increase or decrease max_workers depending on system resources
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for artifact_dir in artifact_directories:
            # Submit each artifact directory for parallel processing
            futures.append(executor.submit(process_artifact_directory, artifact_dir, output_base_dir))

        # Wait for all submitted tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # if an exception occurred in a worker, this will raise it
            except Exception as e:
                logging.error(f"Error while processing a directory in a thread: {e}")

    logging.info(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()

import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def validate_file_existence(file_path):
    try:
        with open(file_path):
            pass
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
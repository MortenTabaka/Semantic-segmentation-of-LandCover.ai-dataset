from os import path

import requests
from tqdm import tqdm

abs_path = path.abspath("")
output = abs_path + "/data/raw/"

if not path.isfile(output):
    url = "https://huggingface.co/datasets/MortenTabaka/LandCover-Aerial-Imagery-for-semantic-segmentation/resolve/main/landcover.zip"
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(path.join(output, "landcover.zip"), "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Download incomplete")

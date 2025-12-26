# NOTE:
# Client ID and Secret are intentionally omitted for security reasons.
# Users should insert their own Sentinel Hub credentials to run this script.


pip install sentinelhub geopandas shapely pillow

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox
)
import time
import pandas as pd
import shutil
import os


# CONFIGURATION

CLIENT_ID = "<SENTINELHUB_CLIENT_ID>"
CLIENT_SECRET = "<SENTINELHUB_CLIENT_SECRET>"

BASE_PATH = r"D:\AWANI DOCUMENTS\CDC_Project\Data"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")

TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, "Images", "Train")
TEST_IMAGE_DIR = os.path.join(BASE_PATH, "Images", "Test")

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)


# SENTINEL HUB AUTH

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET

# EVALSCRIPT

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  const gain = 3.0;
  const decode = (v) => {
    let val = v * gain;
    return val <= 0.0031308 ? 12.92 * val : 1.055 * Math.pow(val, 1/2.4) - 0.055;
  };
  return [decode(sample.B04), decode(sample.B03), decode(sample.B02)];
}
"""

# DOWNLOAD FUNCTION

def download_images(csv_path, image_dir, split_name):
    df = pd.read_csv(csv_path)

    print(f"\n Starting {split_name} image download: {len(df)} samples")

    for idx, row in df.iterrows():
        house_id = row["id"]
        lat, lon = row["lat"], row["long"]

        final_path = os.path.join(image_dir, f"{house_id}.png")
        if os.path.exists(final_path):
            continue

        print(f" [{split_name}] {idx+1}/{len(df)} → ID {house_id}")

        bbox = BBox(
            bbox=[lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005],
            crs=CRS.WGS84
        )

        try:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=("2022-01-01", "2024-01-01"),
                        mosaicking_order="leastCC",
                        maxcc=0.1
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.PNG)
                ],
                bbox=bbox,
                size=[256, 256],
                config=config,
                data_folder=image_dir
            )

            request.get_data(save_data=True)

            saved_file = request.get_filename_list()[0]
            full_saved_path = os.path.join(image_dir, saved_file)

            shutil.move(full_saved_path, final_path)

            temp_dir = os.path.dirname(full_saved_path)
            if temp_dir != image_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

            time.sleep(0.2)  # rate limiting

        except Exception as e:
            print(f" Error for ID {house_id}: {e}")

    print(f"\n {split_name} download complete!")

# FULL PIPELINE

download_images(TRAIN_CSV, TRAIN_IMAGE_DIR, "TRAIN")
download_images(TEST_CSV, TEST_IMAGE_DIR, "TEST")

print("\n ALL IMAGES DOWNLOADED SUCCESSFULLY!")
import cv2
import numpy as np
import pandas as pd

# Membaca citra
image_path = "static/dataset/Indra-Mahasiswa/2.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (160, 160))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mendapatkan dimensi citra
height, width, _ = image.shape

# Membuat array koordinat piksel
coordinates = np.array(list(np.ndindex((height, width))), dtype=int)

# Memisahkan kanal warna RGB
blue_channel, green_channel, red_channel = cv2.split(image)

# Menggabungkan data RGB dan koordinat piksel ke dalam satu DataFrame
data = pd.DataFrame({
    'Red': red_channel.flatten(),
    'Green': green_channel.flatten(),
    'Blue': blue_channel.flatten(),
    'X': coordinates[:, 0],
    'Y': coordinates[:, 1]
})

# Menyimpan data ke dalam file Excel
excel_path = "data_citra_rgb_1.xlsx"
data.to_excel(excel_path, index=False)
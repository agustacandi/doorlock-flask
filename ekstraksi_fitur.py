# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from keras.models import Model, load_model
# import pandas as pd

# classifier = load_model('static/models/model_4_rgb.h5')

# input_image = cv2.imread("static/dataset/Fara-Mahasiswa/1.jpg")
# input_image = cv2.resize(input_image, (224, 224))
# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# input_image = input_image.reshape(1, 224, 224, 3)

# hidden_layer_name = 'conv2d'

# # Dapatkan output dari lapisan tersembunyi
# hidden_layer_output = classifier.get_layer(hidden_layer_name).output

# # Buat model yang mengeluarkan output dari lapisan tersembunyi
# hidden_layer_model = Model(inputs=classifier.input, outputs=hidden_layer_output)

# # Lakukan prediksi menggunakan model lapisan tersembunyi
# hidden_layer_predictions = hidden_layer_model.predict(input_image)
# flattened_features = hidden_layer_predictions.reshape(hidden_layer_predictions.shape[0], -1)

# # Tampilkan hasil ekstraksi fitur
# # print("Hasil Ekstraksi Fitur dari Lapisan Tersembunyi (FLATTEN):")
# print("Hasil Ekstraksi Fitur Konvolusi Pertama:")
# # print(hidden_layer_predictions.shape)
# # print(hidden_layer_predictions)

# # import pandas as pd
# # dataExcel = pd.DataFrame(flattened_features)
# # dataExcel.to_hdf('ekstraksi_fitur.h5', key='data', mode='w')
# # dataExcelTranspose = dataExcel.T
# # dataExcelTranspose.to_excel('hasil_ekstraksi_input.xlsx', index=False)
# # dataExcel.to_excel('hasil_ekstraksi_input.xlsx', index=False)
import cv2
import numpy as np
import pandas as pd

# Membaca citra
img = cv2.imread('static/dataset/Fara-Mahasiswa/1.jpg')

# Mengubah citra menjadi array NumPy
data_angka_rgb = np.array(img)

# Reshape array agar menjadi satu dimensi
flattened_data_rgb = data_angka_rgb.reshape(-1, 3)  # 3 karena citra RGB memiliki tiga saluran warna

# Membuat DataFrame menggunakan Pandas
df_rgb = pd.DataFrame(flattened_data_rgb, columns=['R', 'G', 'B'])

# Menyimpan DataFrame ke dalam file Excel
df_rgb.to_excel('data_citra_rgb.xlsx', index=False)

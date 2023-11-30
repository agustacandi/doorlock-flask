# import io
# import json
# label = []

# with io.open('static/labels/labels.txt', 'r', encoding='utf-8') as f:
    # labels = f.readlines()
    # labels = [line.rstrip('\n') for line in labels]
    # print(labels)
    # for i in labels:
      # label.append(i)
# for i in label:
  # split = i.split('-')
  # print(split[0])
  

# with open('static/labels/labels.json', 'r') as json_file:
#   data = json.load(json_file)
  
# for i in data:
#   label.append(i)
#   print(i)

# Menggunakan perulangan foreach
def cek_semua_sama(arr):
    # Memeriksa apakah semua nilai dalam array sama
    if all(elem == arr[0] for elem in arr):
        return "Berhasil"
    else:
        return "Gagal"

array_contoh_1 = [3, 3, 3, 3]
array_contoh_2 = [2, 2, 2, 2]

hasil_1 = cek_semua_sama(array_contoh_1)
hasil_2 = cek_semua_sama(array_contoh_2)

print("Hasil 1:", hasil_1)
print("Hasil 2:", hasil_2)
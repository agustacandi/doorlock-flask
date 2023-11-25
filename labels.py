import io

label = []

with io.open('static/labels/labels.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = [line.rstrip('\n') for line in labels]
    print(labels)
    for i in labels:
      label.append(i)
for i in label:
  split = i.split('-')
  print(split[0])
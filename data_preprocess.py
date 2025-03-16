import json
from sklearn.model_selection import train_test_split

og_train_data = json.load(open("training.json"))
og_val_data = json.load(open("validation.json"))
og_test_data = json.load(open("test.json"))


train_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in og_train_data:
    train_dist[str(int(elt["stars"]))] += 1
val_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in og_val_data:
    val_dist[str(int(elt["stars"]))] += 1
test_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in og_test_data:
    test_dist[str(int(elt["stars"]))] += 1

print("Data Before Preprocessing")
print("Train Data Distribution: ", train_dist)
print("Validation Data Distribution: ", val_dist)
print("Test Data Distribution: ", test_dist)

data_combined = og_train_data + og_val_data + og_test_data

train_data, temp_data = train_test_split(data_combined, test_size=0.2, random_state=69)

val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=69)

with open("train_data.json", "w") as f, open("val_data.json", "w") as g, open("test_data.json", "w") as h:
    json.dump(train_data, f)
    json.dump(val_data, g)
    json.dump(test_data, h)

train_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in train_data:
    train_dist[str(int(elt["stars"]))] += 1
val_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in val_data:
    val_dist[str(int(elt["stars"]))] += 1
test_dist = {'1':0, '2':0, '3':0, '4':0, '5':0}
for elt in test_data:
    test_dist[str(int(elt["stars"]))] += 1

print("\n")
print("Data After Preprocessing")
print("Train Data Distribution: ", train_dist)
print("Validation Data Distribution: ", val_dist)
print("Test Data Distribution: ", test_dist)
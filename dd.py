import pickle

with open("model_paths.pkl", "rb") as f:
    model_paths = pickle.load(f)

print(model_paths)
print(type(model_paths[0]))
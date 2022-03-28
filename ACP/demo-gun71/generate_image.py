import pickle
from PIL import Image

home = "/home/mohit/EPIC-KITCHENS"
data = pickle.load(open("../scripts/generate_metadata/nosup/train_contact_videos.pkl", "rb"))
data_f = [f for f in data if f[0][:2] != (None, None)]

index = 200000
hand, vid, pid, fname = data_f[index]
l_hand, r_hand, objects, scores = hand

path = f"{home}/{pid}/rgb_frames/{vid}/{fname}"
img = Image.open(path)
img.save(f"data/{vid}_{fname}")

pickle.dump(data_f[index], open(f"data/{vid}_{fname[:-4]}.pkl", "wb"))






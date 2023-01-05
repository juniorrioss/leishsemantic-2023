import json
import cv2
import numpy as np

# função para pegar os pontos dos poligonos da label do arquivo json
def get_json(ann_path):

    with open(ann_path, "rb") as handle:
        data = json.load(handle)

    shape_dicts = data["shapes"]
    img_height = data["imageHeight"]
    img_width = data["imageWidth"]

    return shape_dicts, (img_height, img_width)


def label2poly_multiclass(shape, shape_dicts):

    label2id = {
        "leishmania": 1,
        "macrofago contavel": 2,
        "macrofago nao contavel": 3,
    }

    # coletando todas as labels
    labels = [x["label"] for x in shape_dicts]

    # coletando todos os poligonos
    poly = [np.array(x["points"], dtype=np.int32) for x in shape_dicts]

    # juntando as labels e seus poligonos
    # EX: ('macrofago nao contavel', array([[2919, 2320],
    #    [2826, 2306],
    #    [2789, 2223],
    #    [2817, 2127],
    #    [2893, 2081],
    #    [2956, 2109]]))
    blank_channel = np.full(shape[:2], dtype=np.uint8, fill_value=0)

    for i in range(len(labels)):
        ## Preenchendo os poligonos com suas respectivas classes
        if labels[i] != "leishmania":
            cv2.fillPoly(blank_channel, [poly[i]], label2id[labels[i]])

    for i in range(len(labels)):
        if labels[i] == "leishmania":
            cv2.fillPoly(blank_channel, [poly[i]], label2id[labels[i]])

    return blank_channel


if __name__ == "__main__":

    import os
    from tqdm.auto import tqdm

    json_path = "data/raw_data"
    save_dir = "data/label"
    os.makedirs(save_dir, exist_ok=True)
    json_list = [i for i in os.listdir(json_path) if i.endswith(".json")]
    print(len(json_list), "json files founded!")
    for json_name in tqdm(json_list):
        shape_dicts, shape_img = get_json(os.path.join(json_path, json_name))
        mask = label2poly_multiclass(shape_img, shape_dicts)

        img_name = json_name.split(".")[0] + ".png"
        cv2.imwrite(os.path.join(save_dir, img_name), mask)

import json, numpy as np, os, torch, cv2

from torch.utils.data import Dataset


class HulaDataset(Dataset):

    def __init__(self, metadata_file, root_dir, categories, mode="train"):
        """
        Args:
            metadata_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            categories (array): list of category to load
                on a sample.
        """
        f = open(root_dir + "meta/" + metadata_file, "r")
        if len(categories) > 0:
            self.metadata = [
                x
                for x in json.load(f)
                if x.get("category", "nocategory") in categories]
        else:
            self.metadata = json.load(f)
        self.root_dir = root_dir
        self.categories = categories
        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata[idx].get("img"))
        image = cv2.imread(img_name)
        image = cv2.resize(image, (1920 // 120 * 10, 1080 // 120 * 10))

        # valori di test
        edges = cv2.Canny(image, 55, 230)
        image = image.astype(np.float32)
        edges = edges.astype(np.float32)
        # change interval from [0, 255] to [0.0, 1.0]
        image /= 255.0
        edges /= 255.0
        # change the planes from HxVxRGB into RGBxHxV
        a = [image[:, :, x] for x in range(0, 3)]
        a.append(edges)
        image = torch.tensor(a)
        bbox = self.metadata[idx].get("rects")
        category = self.metadata[idx].get("category", "nocategory")
        # sample = {'image': image, 'bbox': bbox, "category": category, 'label': self.categories.index(category)}

        # print(image.shape, bbox, type(bbox), category)
        # return image, self.categories.index(category), {'bbox': bbox, 'category': category}
        if self.mode == "train":
            return image, self.categories.index(category)
        elif self.mode == "inference":
            return image, self.categories.index(category), (self.metadata[idx].get("img")), category

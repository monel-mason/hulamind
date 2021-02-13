from datetime import datetime
import glob
import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from hulaload import HulaDataset
from nn import HulaNet

desired_width = 320
# pd.set_option('display.width', desired_width) for Panda
np.set_printoptions(linewidth=desired_width)


def imgrid(shape, ndarray, labels=None):
    # TODO: function a se sante
    rgb, net_kern_rgb = plt.subplots(shape[0], shape[1], sharex="col", sharey="row",
                                     gridspec_kw=dict(height_ratios=np.ones(shape[0], dtype=int),
                                                      width_ratios=np.ones(shape[1], dtype=int)))
    canny, net_kern_canny = plt.subplots(shape[0], shape[1], sharex="col", sharey="row",
                                         gridspec_kw=dict(height_ratios=np.ones(shape[0], dtype=int),
                                                          width_ratios=np.ones(shape[1], dtype=int)))
    for i in range(shape[0] * shape[1]):
        tmp = ndarray[i]
        # tmp = net.conv1.weight[i]
        tmp = tmp.cpu().detach().numpy()
        tmp = np.swapaxes(np.swapaxes(tmp, 0, 2), 0, 1)
        # a = [tmp[x, :, :] for x in range(0, 4)]
        # tmp = np.array(a)
        rgb_image = np.clip(tmp[:, :, 0:3], 0, 1)
        canny_image = np.clip(tmp[:, :, 3:], 0, 1)
        net_kern_rgb[i // shape[0], i % shape[1]].imshow(rgb_image)
        net_kern_canny[i // shape[0], i % shape[1]].imshow(canny_image)
        if labels is not None:
            net_kern_rgb[i // shape[0], i % shape[1]].set_title(labels[i])
    # TODO:END
    return rgb, canny


def train_mosse():
    hula_path = '/home/mason/Pictures/hula/'
    categories = ["mosse", "nonmosse"]
    writer = SummaryWriter()

    json_list = glob.glob(hula_path + "meta/*json")
    json_list = [os.path.basename(x) for x in json_list]
    json_list.sort()
    print("using metadata: %s" % (json_list[-1:][0]))
    load = HulaDataset(os.path.basename(hula_path + json_list[-1:][0]), hula_path, categories)

    trainer = torch.utils.data.DataLoader(load, batch_size=12, shuffle=True)
    net = HulaNet(0, 0, len(categories))
    net.cuda()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.5)

    for epoch in range(120):  # loop over the dataset multiple times
        for i, data in enumerate(trainer, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            actual_loss = loss(outputs, labels)
            actual_loss.backward()
            optimizer.step()

            if i % 5 == 0:
                writer.add_scalar('loss', actual_loss, epoch * len(trainer) + i)
                writer.flush()
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, actual_loss.item()))

                rgb, canny = imgrid((2, 2), net.conv1.weight)
                writer.add_figure('conv1 rgb masks', rgb, global_step=epoch * len(trainer) + i)
                writer.flush()
                writer.add_figure('conv1 canny mask', canny, global_step=epoch * len(trainer) + i)
                writer.flush()
            if i == 10 and epoch % 20 == 0:
                values, indexes = torch.max(outputs[:4], 1)
                values = values.cpu().detach().numpy()
                indexes = indexes.cpu().detach().numpy()
                lookup = np.array(categories)
                labels = ["lab: " + lookup[indexes[x]] + " p: " + str(values[x]) for x in range(4)]
                result, canny = imgrid((2, 2), inputs[:4], labels)
                writer.add_figure('result example', result, global_step=epoch * len(trainer) + i)
                writer.flush()

    if actual_loss < 0.1:
        print('[%d, %5d] loss: %.3f saved' % (epoch + 1, i + 1, actual_loss.item()))
        torch.save(net.state_dict(), "nets/nn_" + datetime.now().strftime('%Y%m%d-%H%M%S:%f') + '-' + str(
            round(actual_loss.item(), 3)) + ".pt")


def rebase():
    root = "/home/mason/Pictures/hula/meta/"

    # base = "mm_20210102-192545:355500.json"
    # auto = "mm_20210102-204329:039089-auto.json"
    # retrain = "mm_20210103-111641:695242.json"
    base, auto, retrain = None, None, None

    json_list = glob.glob(root + "*json")
    json_list = [os.path.basename(x) for x in json_list]
    json_list.sort(reverse=True)

    retrain = json_list[0]
    i = 1
    while base is None:
        if "-auto.json" in json_list[i]:
            auto = json_list[i]
            base = json_list[i + 1]
        i = i + 1
    result = []

    base_set = json.load(open(root + base, "r"))
    auto_set = json.load(open(root + auto, "r"))
    retrain_set = json.load(open(root + retrain, "r"))

    assert (len(base_set) == len(auto_set) == len(retrain_set))

    for i, data in enumerate(base_set):
        if base_set[i].get("img") == auto_set[i].get("img"):
            if base_set[i].get("category", "nocategory") == "nocategory" and auto_set[i].get("category") != \
                    retrain_set[i].get("category"):
                result.append(retrain_set[i])
            else:
                result.append(base_set[i])
    f = open("/home/mason/Pictures/hula/meta/mm_" + datetime.now().strftime('%Y%m%d-%H%M%S:%f') + '-retrain.json', "w")
    json.dump(result, f)


def inference():
    net_path = '/home/mason/PycharmProjects/hulamind/nets/'
    hula_path = '/home/mason/Pictures/hula/'
    categories = ["mosse", "nonmosse"]

    net_list = glob.glob(net_path + "*pt")
    net_list = [os.path.basename(x) for x in net_list]
    net_list.sort(reverse=True)

    print("using net %s" % net_list[0])
    net = HulaNet(0, 0, 2)  # fisso a 2, non posso prenderle dalle categories
    net.load_state_dict(torch.load(net_path + net_list[0]))
    net.eval()

    net.cuda()
    json_list = glob.glob(hula_path + "meta/*json")
    json_list = [os.path.basename(x) for x in json_list]
    json_list.sort()
    print("using meta %s" % json_list[-1])

    merge = []
    load = HulaDataset(os.path.basename(hula_path + json_list[-1:][0]), hula_path, "nocategory", "inference")

    inferencer = torch.utils.data.DataLoader(load, batch_size=1)

    for i, data in enumerate(inferencer, 0):
        # get the inputs; data is a list of [inputs, labels]
        if (i + 1) % 30 == 0:
            print("%d - %d" % (i, len(inferencer)))
            # break
        inputs, labels, image_name, category = data
        inputs = inputs.cuda()

        outputs = net(inputs)
        values, indexes = torch.max(outputs[:4], 1)
        indexes = indexes.cpu().detach().numpy()
        lookup = np.array(categories)
        elem = {"img": image_name[0], "category": lookup[indexes[0]], "rects": []}
        merge.append(elem)
    merge = sorted(merge, key=lambda x: x.get("img"))
    f = open(hula_path + "meta/" + json_list[-1:][0], "r")
    base = json.load(f)
    counter = 0
    for i, data in enumerate(base):
        if data.get("img") == merge[counter].get("img"):
            if data.get("category") is None:
                data["category"] = merge[counter].get("category")
            else:
                print("%d: Warning found category %s on %s" % (i, data.get("category"), data.get("img")))
            counter = counter + 1

    f = open("/home/mason/Pictures/hula/meta/mm_" + datetime.now().strftime('%Y%m%d-%H%M%S:%f') + '-auto.json', "w")
    json.dump(base, f)


if __name__ == "__main__":
    print(sys.argv[1])
    if sys.argv[1] == "train":
        train_mosse()
    if sys.argv[1] == "inference":
        inference()
    if sys.argv[1] == "rebase":
        rebase()

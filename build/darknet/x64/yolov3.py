from ctypes import *
import math
import os
import time

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def load_yolo_dll(dll_name):
    cwd = os.path.dirname(__file__)
    dll_path = os.path.join(cwd, dll_name)
    lib = CDLL(dll_path, RTLD_GLOBAL)

    ### setup dll interfaces
    # int network_width(network *net);
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int

    # int network_height(network *net);
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int
    
    # int network_channel(network *net);
    lib.network_channel.argtypes = [c_void_p]
    lib.network_channel.restype = c_int

    # int network_batch(network *net);
    lib.network_batch.argtypes = [c_void_p]
    lib.network_batch.restype = c_int

    # void cuda_set_device(int n);
    lib.cuda_set_device.argtypes = [c_int]

    # float *network_predict(network net, float *input);
    lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
    lib.network_predict.restype = POINTER(c_float)

    # float *network_predict(network net, float *input);
    lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
    lib.network_predict.restype = POINTER(c_float)

    # image make_image(int w, int h, int c);
    lib.make_image.argtypes = [c_int, c_int, c_int]
    lib.make_image.restype = IMAGE

    # detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
    lib.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
    lib.get_network_boxes.restype = POINTER(DETECTION)

    # detection *get_network_boxes_batch(network *net, int b, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
    lib.get_network_boxes_batch.argtypes = [c_void_p, c_int, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
    lib.get_network_boxes_batch.restype = POINTER(DETECTION)

    # detection *make_network_boxes(network *net, float thresh, int *num);
    lib.make_network_boxes.argtypes = [c_void_p, c_float, POINTER(c_int)]
    lib.make_network_boxes.restype = POINTER(DETECTION)

    # detection *make_network_boxes_batch(network *net, int b, float thresh, int *num);
    lib.make_network_boxes_batch.argtypes = [c_void_p, c_int, c_float, POINTER(c_int)]
    lib.make_network_boxes_batch.restype = POINTER(DETECTION)

    # void free_detections(detection *dets, int n);
    lib.free_detections.argtypes = [POINTER(DETECTION), c_int]

    # void free_ptrs(void **ptrs, int n);
    lib.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    # float *network_predict(network net, float *input);
    lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
    lib.network_predict.restype = POINTER(c_float)

    # float *network_predict_image(network *net, image im);
    lib.network_predict_image.argtypes = [c_void_p, IMAGE]
    lib.network_predict_image.restype = POINTER(c_float)

    # network *load_network_custom(char *cfg, char *weights, int clear, int batch);
    lib.load_network_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
    lib.load_network_custom.restype = c_void_p

    # void do_nms_sort(detection *dets, int total, int classes, float thresh);
    lib.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    # void do_nms_obj(detection *dets, int total, int classes, float thresh);
    lib.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    # void free_image(image m);
    lib.free_image.argtypes = [IMAGE]

    # image letterbox_image(image im, int w, int h); 
    lib.letterbox_image.argtypes = [IMAGE, c_int, c_int]
    lib.letterbox_image.restype = IMAGE

    # metadata get_metadata(char *file);
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    # image load_image_color(char *filename, int w, int h);
    lib.load_image_color.argtypes = [c_char_p, c_int, c_int]
    lib.load_image_color.restype = IMAGE

    # void rgbgr_image(image im);
    lib.rgbgr_image.argtypes = [IMAGE]

    return lib

libdarknet = load_yolo_dll("yolo_cpp_dll.dll")

def load_network(gpu, config_file, model_file, batch, lib=libdarknet):
    lib.cuda_set_device(gpu)
    net = lib.load_network_custom(config_file.encode("utf-8"), model_file.encode("utf-8"), 0, batch)
    return net

def detect(net, data, nclasses, thresh=.5, hier_thresh=.5, nms=.45, lib=libdarknet):
    st = time.perf_counter()
    lib.network_predict(net, data["X"])
    print("network_predict: {:.3f} sec".format(time.perf_counter() - st))
    
    num = c_int(0)
    pnum = pointer(num)
    ret = []
    for n, w, h in zip(range(data["n"]), data["w"], data["h"]):
        dets = lib.get_network_boxes_batch(net, n, w, h, thresh, hier_thresh, None, 0, pnum, 1)
        num = pnum[0]
        lib.do_nms_sort(dets, num, nclasses, nms)
        res = []
        for j in range(num):
            for i in range(nclasses):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((i, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        lib.free_detections(dets, num)
        ret.append(res)
    return ret

def show_dets(data, dets):
    from skimage import io, draw
    import numpy as np
    paths = data["paths"]
    for det, path in zip(dets, paths):
        image = io.imread(path)
        print("*** "+str(len(det))+" Results, color coded by confidence ***")
        imcaption = []
        for detection in det:
            label = str(detection[0])
            confidence = detection[1]
            pstring = label+": "+str(np.rint(100 * confidence))+"%"
            imcaption.append(pstring)
            print(pstring)
            bounds = detection[2]
            shape = image.shape

            yExtent = int(bounds[3])
            xEntent = int(bounds[2])
            # Coordinates are around the center
            xCoord = int(bounds[0] - bounds[2]/2)
            yCoord = int(bounds[1] - bounds[3]/2)
            boundingBox = [
                [xCoord, yCoord],
                [xCoord, yCoord + yExtent],
                [xCoord + xEntent, yCoord + yExtent],
                [xCoord + xEntent, yCoord]
            ]
            # Wiggle it around to make a 3px border
            rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
            rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
            rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
            rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
            rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
            boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
            draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
            draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
            draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
            draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
            draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
        io.imshow(image)
        io.show()

def load_batch_images(paths, net, lib=libdarknet):
    w = lib.network_width(net)
    h = lib.network_height(net)
    c = lib.network_channel(net)
    b = lib.network_batch(net)
    s = w*h*c
    X = (c_float*(s*b))()
    paths = paths[:b]
    iw = []
    ih = []
    for i, p in enumerate(paths):
        im = lib.load_image_color(p.encode("utf-8"), 0, 0)
        iw.append(im.w)
        ih.append(im.h)
        sized = lib.letterbox_image(im, w, h)
        X[i*s:(i+1)*s] = sized.data[0:s]
        lib.free_image(sized)
        lib.free_image(im)
    return {"w":iw, "h":ih, "X": X, "paths": paths, "n": len(paths)}

def load_batch_images_ex(paths, net, lib=libdarknet):
    import cv2
    import numpy as np
    

if __name__ == "__main__":
    import time
    st = time.perf_counter()
    net = load_network(3, "cfg/my-yolov3.cfg", "my-yolov3_final.weights", 4)
    print("load_network: {:.3f} sec".format(time.perf_counter() - st))
    st = time.perf_counter()
    data = load_batch_images(["1.jpg", "2.jpg", "3.jpg", "4.jpg"], net)
    print("load_batch_images: {:.3f} sec".format(time.perf_counter() - st))
    st = time.perf_counter()
    res = detect(net, data, 2)
    print("detect: {:.3f} sec".format(time.perf_counter() - st))
    show_dets(data, res)

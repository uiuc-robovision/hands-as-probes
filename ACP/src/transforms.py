from box_utils import BBox, BBox_lite
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal

def generate_bbox(bb1=None, bb2=None, srangex=(80, 300), srangey=(80, 300), rng=None, avoid_black=False):
    if bb1 is None and bb2 is None:
        return None
    elif bb1 is not None and bb2 is None:
        return generate_bbox_onehand(bb1, srangex, srangey, rng, avoid_black=avoid_black)
    elif bb2 is not None and bb1 is None:
        return generate_bbox_onehand(bb2, srangex, srangey, rng, avoid_black=avoid_black)
        
    bb1 = bb1.expand_to_square()
    bb2 = bb2.expand_to_square()
    # region_center = BBox.avg_tuples(bb1.get_center(), bb2.get_center())
    # width = max(srangex[1], max(bb1.right, bb2.right) - min(bb1.left, bb2.left))
    # height = max(srangey[1], max(bb1.bottom, bb2.bottom) - min(bb1.top, bb2.top))
    # region_bbox = BBox.from_center_dims(*region_center, width, height, bb1.img).get_in_frame()
    region_bbox = BBox(0, 0, 1.0, 1.0, bb1.img, True).get_in_frame()
    point = None
    count = 0
    while point is None:
        count += 1
        if count > 10:
            return None
        point = region_bbox.sample_point(rng=rng)
        if bb1.ispointin(point) or bb2.ispointin(point):
            point = None
            continue
    
        constraints1 = bb1.get_boundaries(point, avoid_black=avoid_black)
        constraints2 = bb2.get_boundaries(point, avoid_black=avoid_black)
        constraints = (max(constraints1[0], constraints2[0]),
                      max(constraints1[1], constraints2[1]),
                      min(constraints1[2], constraints2[2]),
                      min(constraints1[3], constraints2[3])
                     )
        constraint_bbox = BBox(*constraints, bb1.img)
        # constraint_bbox = BBox.intersect(region_bbox, constraint_bbox)
        if min(constraint_bbox.width, constraint_bbox.height) < srangex[0] or\
            min(constraint_bbox.width, constraint_bbox.height) < srangey[0]:
            point = None
    final_bbox = constraint_bbox.sample_square_bbox(srangex[0], srangey[0], srangex[1], srangey[1], rng=rng)
    
    return region_bbox, point, constraint_bbox, final_bbox

def generate_bbox_onehand(bb, srangex=(80, 800), srangey=(80, 800), rng=None, avoid_black=False):
    bb = bb.expand_to_square()
    # region_center = bb.get_center()
    
    # region_bbox = BBox.from_center_dims(*region_center, srangex[1], srangey[1], bb.img).get_in_frame()
    region_bbox = BBox(0, 0, 1.0, 1.0, bb.img, True).get_in_frame()
    point = None
    count = 0
    while point is None:
        count += 1
        if count > 10:
            return None
        point = region_bbox.sample_point(rng=rng)
        if bb.ispointin(point):
            point = None
            continue
    
        constraints = bb.get_boundaries(point, avoid_black=avoid_black)
        constraint_bbox = BBox(*constraints, bb.img)
        # constraint_bbox = BBox.intersect(region_bbox, constraint_bbox)
        if min(constraint_bbox.width, constraint_bbox.height) < srangex[0] or\
            min(constraint_bbox.width, constraint_bbox.height) < srangey[0]:
            point = None
    final_bbox = constraint_bbox.sample_square_bbox(srangex[0], srangey[0], srangex[1], srangey[1], rng=rng)
    
    return region_bbox, point, constraint_bbox, final_bbox

def generate_bbox_nohands(img, srangex=(80, 800), srangey=(80, 800), rng=None):
    
    region_bbox = BBox(0, 0, 1.0, 1.0, img, True).get_in_frame()
    point = None
    count = 0
    while point is None:
        count += 1
        if count > 10:
            return None
        point = region_bbox.sample_point(rng=rng)    
        constraint_bbox = region_bbox
        # constraint_bbox = BBox.intersect(region_bbox, constraint_bbox)
        if min(constraint_bbox.width, constraint_bbox.height) < srangex[0] or\
            min(constraint_bbox.width, constraint_bbox.height) < srangey[0]:
            point = None
    final_bbox = constraint_bbox.sample_square_bbox(srangex[0], srangey[0], srangex[1], srangey[1], rng=rng)
    return region_bbox, point, constraint_bbox, final_bbox


def generate_bbox_inference(img, bb1=None, bb2=None, srangex=(50, 800), srangey=(50, 800), rng=None):
    if bb1 is None and bb2 is None:
        region_bbox = BBox(0, 0, 1.0, 1.0, img, True).get_in_frame()
        
        point = None
        count = 0
        while point is None:
            count += 1
            if count > 10:
                return None
            point = region_bbox.sample_point(rng=rng)    
            constraint_bbox = region_bbox
            # constraint_bbox = BBox.intersect(region_bbox, constraint_bbox)
            if min(constraint_bbox.width, constraint_bbox.height) < srangex[0] or\
                min(constraint_bbox.width, constraint_bbox.height) < srangey[0]:
                point = None
        final_bbox = constraint_bbox.sample_square_bbox(srangex[0], srangey[0], srangex[1], srangey[1], rng=rng, rand=False)
        return region_bbox, point, constraint_bbox, final_bbox
    
    elif bb1 is not None and bb2 is None:
        return generate_bbox_onehand(bb1, srangex, srangey, rng)
    elif bb2 is not None and bb1 is None:
        return generate_bbox_onehand(bb2, srangex, srangey, rng)
        
    bb1 = bb1.expand_to_square()
    bb2 = bb2.expand_to_square()
    # region_center = BBox.avg_tuples(bb1.get_center(), bb2.get_center())
    # width = max(srangex[1], max(bb1.right, bb2.right) - min(bb1.left, bb2.left))
    # height = max(srangey[1], max(bb1.bottom, bb2.bottom) - min(bb1.top, bb2.top))
    # region_bbox = BBox.from_center_dims(*region_center, width, height, bb1.img).get_in_frame()
    region_bbox = BBox(0, 0, 1.0, 1.0, bb1.img, True).get_in_frame()
    point = None
    count = 0
    while point is None:
        count += 1
        if count > 10:
            return None
        point = region_bbox.sample_point(rng=rng)
        if bb1.ispointin(point) or bb2.ispointin(point):
            point = None
            continue
    
        constraints1 = bb1.get_boundaries(point)
        constraints2 = bb2.get_boundaries(point)
        constraints = (max(constraints1[0], constraints2[0]),
                      max(constraints1[1], constraints2[1]),
                      min(constraints1[2], constraints2[2]),
                      min(constraints1[3], constraints2[3])
                     )
        constraint_bbox = BBox(*constraints, bb1.img)
        # constraint_bbox = BBox.intersect(region_bbox, constraint_bbox)
        if min(constraint_bbox.width, constraint_bbox.height) < srangex[0] or\
            min(constraint_bbox.width, constraint_bbox.height) < srangey[0]:
            point = None
    final_bbox = constraint_bbox.sample_square_bbox(srangex[0], srangey[0], srangex[1], srangey[1], rng=rng)
    
    return region_bbox, point, constraint_bbox, final_bbox


def generate_bbox_around_forimg(img, minsize, maxsize, rng=None):
    region_bbox = BBox(0, 0, 1.0, 1.0, img, True).get_in_frame()
    bbsq = region_bbox.sample_square_bbox(minsize[0], minsize[1], maxsize[0], maxsize[1], rng=rng)

    bbaround = bbsq.scale(2.0)
    bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    
    return bbsq, bbaround

class VImage:
    def __init__(self, img):
        self.size = img.size

def create_bbox_around(i, j, width, height, img):

    bbsq = BBox_lite(i, j, i + width, j + height, img)
    bbaround = bbsq.scale(2.0)
    bbaround = bbaround.shift(shiftv=bbsq.height/2.)

    return (bbsq, bbaround)



def generate_bboxes_around_forimg(img, width, height, nh=20, nv=10, rng=None, margin=0.25):
    
    # bbsqs = []
    # for i in np.linspace(0, img.size[0], nh):
    #     for j in np.linspace(0, img.size[1], nh):
    #         bbsqs.append()
    vimg = VImage(img)

    bbsqs = Parallel(n_jobs=5, prefer="threads")(delayed(create_bbox_around)(int(i), int(j), width, height, vimg) \
                                                 for i in np.linspace(int(width * margin), img.size[0] - (1 + margin) * width, nh) \
                                                 for j in np.linspace(int(height * margin), img.size[1] - (1 + margin) * height, nv))
    return bbsqs


def generate_validbboxes_around_forimg(img, width, height, nh=20, nv=10, rng=None):
    
    # bbsqs = []
    # for i in np.linspace(0, img.size[0], nh):
    #     for j in np.linspace(0, img.size[1], nh):
    #         bbsqs.append()

    rb = BBox(0, 0, 1.0, 1.0, img, True)
    rb.set_valdims()
    vimg = VImage(img)

    bbsqs = Parallel(n_jobs=20, prefer="threads")(delayed(create_bbox_around)(int(i), int(j), width, height, vimg) \
                                                 for i in np.linspace(width * 0.25 + rb.valleft, rb.valright - 1.25 * width, nh) \
                                                 for j in np.linspace(height * 0.25 + rb.valtop, rb.valbottom - 1.25 * height, nv))
    return bbsqs

def generate_bbox_around(bb, low_scale=1.0, scale=1.5, shift=0.2, negres_fixed=False, rng=None, return_pos=False):
    bbsq = bb.expand_to_square().scale(1.1)
    if rng is not None:
        val = rng.standard_normal()
        if shift > 0:
            bbsq = bbsq.shift(shifth=rng.integers(-shift * bb.width, shift * bb.width + 1), shiftv=rng.integers(-shift * bb.height, shift * bb.height + 1))
        bbsq = bbsq.scale(rng.uniform(low_scale, scale))
    else:
        val = np.random.normal()
        if shift > 0:
            bbsq = bbsq.shift(shifth=np.random.randint(-shift * bb.width, shift * bb.width + 1), shiftv=np.random.randint(-shift * bb.height, shift * bb.height + 1))
        bbsq = bbsq.scale(np.random.uniform(low_scale, scale))
    if val > 0.:
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    else:
        bb.set_valdims()
        region_bbox = BBox(bb.valleft, bb.valtop, bb.valright, bb.valbottom, bb.img)
        if negres_fixed:
            bbsq = region_bbox.sample_square_bbox(30, 30, max(70, bb.width * scale), max(70, bb.height * scale), rng=rng)
        else:
            bbsq = region_bbox.sample_square_bbox(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)     
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    
    if return_pos:
        return (bbsq, bbaround), val > 0
    return bbsq, bbaround


def generate_bbox_aroundobj(bb, low_scale=0.5, scale=0.75, fshift=0.5, negres_fixed=False, rng=None):
    bbsq = bb.scale(1.2)
    if rng is not None:
        val = rng.standard_normal()
        if fshift > 0:
            bbsq = bbsq.shift(shifth=rng.integers(-fshift * bb.width, fshift * bb.width + 1), shiftv=rng.integers(-fshift * bb.height, fshift * bb.height + 1))
        # bbsq = bbsq.scale(rng.uniform(low_scale, scale))
    else:
        val = np.random.normal()
        if fshift > 0:
            bbsq = bbsq.shift(shifth=np.random.randint(-fshift * bb.width, fshift * bb.width + 1), shiftv=np.random.randint(-fshift * bb.height, fshift * bb.height + 1))
        # bbsq = bbsq.scale(np.random.uniform(low_scale, scale))
    if True:
        bbsq = bbsq.sample_square_bbox(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    else:
        bb.set_valdims()
        region_bbox = BBox(bb.valleft, bb.valtop, bb.valright, bb.valbottom, bb.img)
        # if region_bbox.height == -1:
        #     print(bb.valleft, bb.valtop, bb.valright, bb.valbottom)
        #     bb.img.save("temp.png")
        #     sys.exit()
        if negres_fixed:
            bbsq = region_bbox.sample_square_bbox(30, 30, max(70, bb.width * scale), max(70, bb.height * scale), rng=rng)
        else:
            bbsq = region_bbox.sample_square_bbox(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)     
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    
    return bbsq, bbaround


def generate_bbox_aroundobj_wnegs(bb, low_scale=0.5, scale=0.75, fshift=0.5, negres_fixed=False, rng=None):
    bbsq = bb.scale(1.2)
    # if int(fshift * bb.width) > 0 and int(fshift * bb.height) > 0:
    if rng is not None:
        val = rng.standard_normal()
        if fshift > 0:
            bbsq.shift(shifth=rng.integers(-fshift * bb.width, fshift * bb.width + 1), shiftv=rng.integers(-fshift * bb.height, fshift * bb.height + 1))
        # bbsq = bbsq.scale(rng.uniform(low_scale, scale))
    else:
        val = np.random.normal()
        if fshift > 0:
            bbsq.shift(shifth=np.random.randint(-fshift * bb.width, fshift * bb.width + 1), shiftv=np.random.randint(-fshift * bb.height, fshift * bb.height + 1))
        # bbsq = bbsq.scale(np.random.uniform(low_scale, scale))
    if val > 0:
        bbsq = bbsq.sample_square_bbox(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    else:
        bb.set_valdims()
        region_bbox = BBox(bb.valleft, bb.valtop, bb.valright, bb.valbottom, bb.img)
        # if region_bbox.height == -1:
        #     print(bb.valleft, bb.valtop, bb.valright, bb.valbottom)
        #     bb.img.save("temp.png")
        #     sys.exit()
        if negres_fixed:
            bbsq = region_bbox.sample_square_bbox(30, 30, max(70, bb.width * scale), max(70, bb.height * scale), rng=rng)
        else:
            bbsq = region_bbox.sample_square_bbox(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)     
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    
    return bbsq, bbaround


def generate_bbox_aroundv2(bb, low_scale=1.0, scale=1.5, shift=5, negres_fixed=False, rng=None):
    bbsq = bb.expand_to_square().scale(1.1)
    if rng is not None:
        val = rng.standard_normal()
        bbsq.shift(shifth=rng.integers(-shift, shift), shiftv=rng.integers(-shift, shift))
        bbsq = bbsq.scale(rng.uniform(low_scale, scale))
    else:
        val = np.random.normal()
        bbsq.shift(shifth=np.random.randint(-shift, shift), shiftv=np.random.randint(-shift, shift))
        bbsq = bbsq.scale(np.random.uniform(low_scale, scale))
    if val > 0.:
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    else:
        bb.set_valdims()
        region_bbox = BBox(bb.valleft, bb.valtop, bb.valright, bb.valbottom, bb.img)
        # if region_bbox.height == -1:
        #     print(bb.valleft, bb.valtop, bb.valright, bb.valbottom)
        #     bb.img.save("temp.png")
        #     sys.exit()
        if negres_fixed:
            bbsq = region_bbox.sample_square_bboxv2(30, 30, max(70, bb.width * scale), max(70, bb.height * scale), rng=rng)
        else:
            bbsq = region_bbox.sample_square_bboxv2(bb.width*low_scale, bb.height*low_scale, bb.width*scale, bb.height*scale, rng=rng)     
        bbaround = bbsq.scale(2.0)
        bbaround = bbaround.shift(shiftv=bbsq.height/2.)
    
    return bbsq, bbaround

def crop_image(img, bbox):
    img_crop = img.crop(bbox.get_bbox())
    return img_crop

def resize_image(img, width, height, type='best'):
    if type == "best":
        img_resized = img.resize((int(width), int(height)), Image.ANTIALIAS)
    elif type=='nn':
        img_resized = img.resize((int(width), int(height)), Image.NEAREST)
    return img_resized

def create_mask(img, boxes):
    npimg = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    for bbox in boxes:
        if bbox is not None:
            l, t, r, b = bbox.get_bbox_int()
            npimg[t:b, l:r] = 255
    return npimg

def create_gaussian_mask(img, boxes, scale=4.0):
    npimg = np.zeros((img.size[1], img.size[0]))
    pos = np.dstack(np.mgrid[0:img.size[1]:1, 0:img.size[0]:1])
    for bbox in boxes:
        if bbox is not None:
            rv = multivariate_normal(mean=[bbox.centery, bbox.centerx], cov=np.array([[max(bbox.height, 1), 0], [0, max(bbox.width, 1)]])*scale)
            npimg += rv.pdf(pos) / np.max(rv.pdf(pos))

    npimg = npimg / (np.max(npimg) + 1e-8)
    npimg = (npimg * 255).astype(np.uint8)

    return npimg


def save_nparray_asimg(arr, path, scaled_to_one=True):
    if not scaled_to_one:
        arr = arr.copy() / np.max(arr)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)

def nparray2img(arr, scaled_to_one=True):
    if not scaled_to_one:
        arr = arr.copy() / np.max(arr)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img

def create_validity_mask(imsize, masking=True, mask_location="bc"):
    '''
    imsize: size of the mask
    masking: True/False
    mask_location: bc -> bottom center, center -> center
    '''
    validity_mask = np.ones((imsize, imsize))
    if masking:
        if mask_location == 'bc':
            validity_mask[imsize * 2 // 4: imsize * 4 // 4 , imsize // 4: imsize * 3 // 4] = 0
        elif mask_location == 'center':
            validity_mask[imsize * 1 // 4: imsize * 3 // 4 , imsize // 4: imsize * 3 // 4] = 0
        else:
            print("Invalid argument")
            exit()
    
    return validity_mask

def get_rotation_matrix(theta1, theta2, theta3):
    R1 = np.array([[1, 0, 0],
              [0, np.cos(theta1), -np.sin(theta1)],
              [0, np.sin(theta1), np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), 0, np.sin(theta2)],
              [0, 1, 0],
              [-np.sin(theta2), 0, np.cos(theta2)]])
    R3 = np.array([[np.cos(theta3), -np.sin(theta3), 0],
                  [np.sin(theta3), np.cos(theta3), 0],
                  [0, 0, 1]])
    return R1 @ R2 @ R3

def merge_mask(inp_img, hand_img, hand_mask, imsize, rng=None):
    new_img = Image.new('RGB', (imsize, imsize))
    new_mask = Image.new('RGB', (imsize, imsize))
    if rng is None:
        sz = np.random.randint(imsize * 2 //3, imsize)
        i = np.random.randint(0, imsize)
        j = np.random.randint(0, imsize)
    else:
        sz = rng.integers(imsize * 2 //3, imsize)
        i = rng.integers(-sz, imsize)
        j = rng.integers(-sz, imsize)
    h_img = resize_image(hand_img, sz, sz)
    h_mask = resize_image(hand_mask, sz, sz)
    new_img.paste(h_img, (i, j))
    new_mask.paste(h_mask, (i, j))
    thresh = 70
    fn = lambda x : 255 if x > thresh else 0
    new_mask = new_mask.convert('L').point(fn, mode='1')
    return Image.composite(new_img, inp_img, new_mask)

def pad_image(img, pad, fill=(0, 0, 0)):
    if isinstance(pad, int):
        pad_tuple = (pad, pad, pad, pad)
    elif isinstance(pad, tuple):
        if len(pad) == 2:
            pad_tuple = (pad[0], pad[0], pad[1], pad[1])
        elif len(pad) == 4:
            pad_tuple = (pad[0], pad[1], pad[2], pad[3])
        else:
            raise ValueError('Invalid padding, tuple should be of length 2 or 4')
    else:
        raise ValueError('Invalid padding, should be of type int or a tuple')
    
    left, right, top, bottom = pad_tuple
    w, h = img.size
    nw = w + right + left
    nh = h + top + bottom
    result = Image.new(img.mode, (nw, nh), fill)
    result.paste(img, (left, top))
    return result

def unpad_image(img, pad):
    if isinstance(pad, int):
        pad_tuple = (pad, pad, pad, pad)
    elif isinstance(pad, tuple):
        if len(pad) == 2:
            pad_tuple = (pad[0], pad[0], pad[1], pad[1])
        elif len(pad) == 4:
            pad_tuple = (pad[0], pad[1], pad[2], pad[3])
        else:
            raise ValueError('Invalid padding, tuple should be of length 2 or 4')
    else:
        raise ValueError('Invalid padding, should be of type int or a tuple')
    
    left, right, top, bottom = pad_tuple
    w, h = img.size
    l, r, t, b = left, w - right, top, h - bottom
    return img.crop((l, t, r, b))
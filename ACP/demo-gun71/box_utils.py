import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

class BBox_lite:
    def __init__(self, left, top, right, bottom, img, inp_fraction=False):
        if inp_fraction:
            self.left = left * img.size[0]
            self.right = right * img.size[0]
            self.top = top * img.size[1]
            self.bottom = bottom * img.size[1]
        else:
            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom
        self.img = img
        self.set_dims()
           
    
    @classmethod
    def from_center_dims(cls, cx, cy, width, height, img):
        return cls(cx - width/2., cy - height/2., cx + width/2., cy + height/2., img)
    
    @classmethod
    def avg_tuples(cls, *args):
        return tuple(map(lambda y: sum(y) / float(len(y)), zip(*args)))
    
    @classmethod
    def intersect(cls, bb1, bb2):
        return BBox_lite(
                    max(bb1.left, bb2.left),
                    max(bb1.top, bb2.top),
                    min(bb1.right, bb2.right),
                    min(bb1.bottom, bb2.bottom),
                    bb1.img
                    )
    def flip_horizontal(self):
        left = self.img.size[0] - self.right
        right = self.img.size[0] - self.left
        
        return BBox_lite(left, self.top, right, self.bottom, self.img)
    
    def flip_vertical(self):
        top = self.img.size[1] - self.bottom
        bottom = self.img.size[1] - self.top
        
        return BBox_lite(self.left, top, self.right, bottom, self.img)
    
    def get_bbox(self):
        return (self.left, self.top, self.right, self.bottom)
    
    def get_bbox_int(self):
        return (int(self.left), int(self.top), int(self.right), int(self.bottom))
    
    def set_dims(self):
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.centerx = self.left / 2.0 + self.right / 2.0
        self.centery = self.bottom / 2.0 + self.top / 2.0
        self.max_width = float(self.img.size[0])
        self.max_height = float(self.img.size[1])
    
    def shift(self, shifth=0, shiftv=0):
        return BBox_lite(self.left - shifth, self.top - shiftv, self.right - shifth, self.bottom - shiftv, self.img)

    def get_center(self):
        return (self.centerx, self.centery)
            
    def scale(self, scale):
        return BBox_lite(self.centerx - self.width * scale / 2.0,
                    self.centery - self.height * scale / 2.0,
                    self.centerx + self.width * scale / 2.0,
                    self.centery + self.height * scale / 2.0,
                    self.img)

    def ispointin(self, point):
        x, y = point
        if (self.left <= x <= self.right) and (self.top <= y <= self.bottom):
            return True
        
        return False
    
    def sample_point(self, rng=None):
        if rng is not None:
            x = rng.uniform(self.left, self.right)
            y = rng.uniform(self.top, self.bottom)
            
            return x, y
        
        x = np.random.uniform(self.left, self.right)
        y = np.random.uniform(self.top, self.bottom)

        return x, y
        
    
    def get_in_frame(self):
        left = min(max(self.left, 0), self.max_width)
        right = min(max(self.right, 0), self.max_width)
        top = min(max(self.top, 0), self.max_height)
        bottom = min(max(self.bottom, 0), self.max_height)
        
        return BBox_lite(left, top, right, bottom, self.img)
    
    def make_square(self):
        best_dim = min(self.width, self.height)
        return BBox_lite.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
    
    def point_to_slope(self, point):
        x, y = point
        cx, cy = self.centerx, self.centery
        
        return 180 + np.arctan2(-cy + y, cx - x) * 180 / np.pi
    
    def expand_to_square(self):
        best_dim = max(self.width, self.height)
        left, top, right, bottom = self.get_bbox()
        if best_dim == self.width == self.height:
            return self
        if best_dim > self.width:
            margin = (best_dim - self.width) / 2.
            if left - 0 < margin:
                right = right + 2 * margin
            elif self.max_width - right < margin:
                left = left - 2 * margin
            else:
                return BBox_lite.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
            
            return BBox_lite(left, top, right, bottom, self.img)
        elif best_dim > self.height:
            margin = (best_dim - self.height) / 2.
            if top - 0 < margin:
                bottom = bottom + 2 * margin
            elif self.max_height - bottom < margin:
                top = top - 2 * margin
            else:
                return BBox_lite.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
            
            return BBox_lite(left, top, right, bottom, self.img)
    
    def sample_square_bbox(self, min_width=50, min_height=50, max_width=400, max_height=400, rng=None, rand=False):
        max_height = min(self.height, max_height)
        max_width = min(self.width, max_width)
        if rng is not None:
            w = rng.uniform(min_width, min(max_height, max_width))
            h = rng.uniform(min_height, min(max_height, max_width))
            if rand:
                w = h = rng.choice([w, h])
            else:
                w = h = max(w, h)
            l = rng.uniform(self.left, self.right - w)
            t = rng.uniform(self.top, self.bottom - h)
            if t + h > (self.bottom + 1):
                print("Error", t, h, self.top, self.bottom - h, self.bottom, max_height, self.height)
            if l + w > (self.right + 1):
                print("Error")
            return BBox_lite.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
            
        w = np.random.uniform(min_width, min(max_height, max_width))
        h = np.random.uniform(min_height, min(max_height, max_width))
        if rand:
                w = h = rng.choice([w, h])
        else:
            w = h = max(w, h)
        l = np.random.uniform(self.left, self.right - w)
        t = np.random.uniform(self.top, self.bottom - h)
        
        if t + h > self.bottom:
            print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
            print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
            print(f"t {t} boxt {self.top}")
            print("Vertical Error", t + h, self.bottom)
        if l + w > self.right:
            print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
            print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
            print(f"l {l} boxl {self.left}")
            print("Horizontal Error", l + w, self.right)

        return BBox_lite.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
    
    def sample_bbox(self, min_width=20, max_width=50, min_height=20, max_height=50, rng=None):
        if rng is not None:
            w = rng.uniform(min_width, min(self.width, max_width))
            h = rng.uniform(min_height, min(self.height, max_height))
            l = rng.uniform(self.left, self.right - w)
            t = rng.uniform(self.top, self.bottom - h)
            
            return BBox_lite.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
            
        w = np.random.uniform(min_width, min(self.width, max_width))
        h = np.random.uniform(min_height, min(self.height, max_height))
        l = np.random.uniform(self.left, self.right - w)
        t = np.random.uniform(self.top, self.bottom - h)

        return BBox_lite.from_center_dims(l + w / 2., t + h /2., w, h, self.img)

class BBox:
    def __init__(self, left, top, right, bottom, img, inp_fraction=False):
        if inp_fraction:
            self.left = left * img.size[0]
            self.right = right * img.size[0]
            self.top = top * img.size[1]
            self.bottom = bottom * img.size[1]
        else:
            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom
        self.img = img
        self.set_dims()
           
    
    @classmethod
    def from_center_dims(cls, cx, cy, width, height, img):
        return cls(cx - width/2., cy - height/2., cx + width/2., cy + height/2., img)
    
    @classmethod
    def avg_tuples(cls, *args):
        return tuple(map(lambda y: sum(y) / float(len(y)), zip(*args)))
    
    @classmethod
    def intersect(cls, bb1, bb2):
        return BBox(
                    max(bb1.left, bb2.left),
                    max(bb1.top, bb2.top),
                    min(bb1.right, bb2.right),
                    min(bb1.bottom, bb2.bottom),
                    bb1.img
                    )

    def set_valdims(self):
        self.npimg = np.array(self.img)[:, :, 0].astype(np.int32)
        rowsums = np.sum(self.npimg, axis=1)
        rowsums[rowsums > 0] = 1
        rowcumsums = np.cumsum(rowsums)
        rowcumsumsb = np.cumsum(rowsums[::-1])[::-1]
        # diffs = rowsums[1:] - rowsums[:-1]

        # self.valtop = np.argmax(diffs == 1) + 1 if diffs[np.argmax(diffs == 1)] == 1 else 0
        args = np.argwhere(rowcumsums == np.min(rowcumsums))
        self.valtop = np.max(args) if len(args) > 0 else 0
        # self.valbottom = np.argmax(diffs == -1) + 1 if diffs[np.argmax(diffs == -1)] == -1 else self.max_height
        args = np.argwhere(rowcumsumsb == np.min(rowcumsumsb))
        self.valbottom = np.min(args) if len(args) > 0 else self.max_height - 1.
        
        
        colsums = np.sum(self.npimg, axis=0)
        colsums[colsums > 0] = 1
        colcumsums = np.cumsum(colsums)
        colcumsumsb = np.cumsum(colsums[::-1])[::-1]
        # diffs = colsums[1:] - colsums[:-1]
        
        # self.valleft = np.argmax(diffs == 1) + 1 if diffs[np.argmax(diffs == 1)] == 1 else 0
        args = np.argwhere(colcumsums == np.min(colcumsums))
        self.valleft = np.max(args) if len(args) > 0 else 0
        # self.valright = np.argmax(diffs == -1) + 1 if diffs[np.argmax(diffs == -1)] == -1 else self.max_width
        args = np.argwhere(colcumsumsb == np.min(colcumsumsb))
        self.valright = np.min(args) if len(args) > 0 else self.max_width - 1.
        

        # If incorrect estimation, go to boundaries
        if self.valtop > self.bottom - 20:
            self.valtop = 0
            self.valbottom = self.max_height

        # If incorrect estimation, go to boundaries
        if self.valleft > self.valright - 20:
            self.valleft = 0
            self.valright = self.max_width

    def flip_horizontal(self):
        left = self.img.size[0] - self.right
        right = self.img.size[0] - self.left
        
        return BBox(left, self.top, right, self.bottom, self.img)
    
    def flip_vertical(self):
        top = self.img.size[1] - self.bottom
        bottom = self.img.size[1] - self.top
        
        return BBox(self.left, top, self.right, bottom, self.img)
    
    def get_bbox(self):
        return (self.left, self.top, self.right, self.bottom)
    
    def get_bbox_int(self):
        return (int(self.left), int(self.top), int(self.right), int(self.bottom))
    
    def set_dims(self):
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.centerx = self.left / 2.0 + self.right / 2.0
        self.centery = self.bottom / 2.0 + self.top / 2.0
        self.max_width = float(self.img.size[0])
        self.max_height = float(self.img.size[1])
    
    def shift(self, shifth=0, shiftv=0):
        return BBox(self.left - shifth, self.top - shiftv, self.right - shifth, self.bottom - shiftv, self.img)

    def get_center(self):
        return (self.centerx, self.centery)
            
    def scale(self, scale):
        return BBox(self.centerx - self.width * scale / 2.0,
                    self.centery - self.height * scale / 2.0,
                    self.centerx + self.width * scale / 2.0,
                    self.centery + self.height * scale / 2.0,
                    self.img)
    
    def scalexy(self, wscale, hscale):
        return BBox(self.centerx - self.width * wscale / 2.0,
                    self.centery - self.height * hscale / 2.0,
                    self.centerx + self.width * wscale / 2.0,
                    self.centery + self.height * hscale / 2.0,
                    self.img)

    def ispointin(self, point):
        x, y = point
        if (self.left <= x <= self.right) and (self.top <= y <= self.bottom):
            return True
        
        return False
    
    def sample_point(self, rng=None):
        if rng is not None:
            x = rng.uniform(self.left, self.right)
            y = rng.uniform(self.top, self.bottom)
            
            return x, y
        
        x = np.random.uniform(self.left, self.right)
        y = np.random.uniform(self.top, self.bottom)

        return x, y
        
    
    def get_in_frame(self):
        left = min(max(self.left, 0), self.max_width)
        right = min(max(self.right, 0), self.max_width)
        top = min(max(self.top, 0), self.max_height)
        bottom = min(max(self.bottom, 0), self.max_height)
        
        return BBox(left, top, right, bottom, self.img)
    
    def make_square(self):
        best_dim = min(self.width, self.height)
        return BBox.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
    
    def point_to_slope(self, point):
        x, y = point
        cx, cy = self.centerx, self.centery
        
        return 180 + np.arctan2(-cy + y, cx - x) * 180 / np.pi
    
    def get_boundaries(self, point, avoid_black=False):
        slope = self.point_to_slope(point)
        
        if avoid_black:
            if (0 <= slope <= 45) or (315 <= slope <= 360):
                return self.right, self.valtop, self.valright, self.valbottom
            elif (45 <= slope <= 135):
                return self.valleft, self.valtop, self.valright, self.top
            elif (135 <= slope <= 225):
                return self.valleft, self.valtop, self.left, self.valbottom
            elif (225 <= slope <= 315):
                return self.valleft, self.bottom, self.valright, self.valbottom
        
        else:
            if (0 <= slope <= 45) or (315 <= slope <= 360):
                return self.right, 0, self.max_width, self.max_height
            elif (45 <= slope <= 135):
                return 0, 0, self.max_width, self.top
            elif (135 <= slope <= 225):
                return 0, 0, self.left, self.max_height
            elif (225 <= slope <= 315):
                return 0, self.bottom, self.max_width, self.max_height
    
    def expand_to_square(self):
        best_dim = max(self.width, self.height)
        left, top, right, bottom = self.get_bbox()
        if best_dim == self.width == self.height:
            return self
        if best_dim > self.width:
            margin = (best_dim - self.width) / 2.
            if left - 0 < margin:
                right = right + 2 * margin
            elif self.max_width - right < margin:
                left = left - 2 * margin
            else:
                return BBox.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
            
            return BBox(left, top, right, bottom, self.img)
        elif best_dim > self.height:
            margin = (best_dim - self.height) / 2.
            if top - 0 < margin:
                bottom = bottom + 2 * margin
            elif self.max_height - bottom < margin:
                top = top - 2 * margin
            else:
                return BBox.from_center_dims(self.centerx, self.centery, best_dim, best_dim, self.img)
            
            return BBox(left, top, right, bottom, self.img)
    
    def sample_square_bbox(self, min_width=50, min_height=50, max_width=400, max_height=400, rng=None, rand=False):
        max_height = min(self.height, max_height)
        max_width = min(self.width, max_width)
        if rng is not None:
            w = rng.uniform(min_width, min(max_height, max_width))
            h = rng.uniform(min_height, min(max_height, max_width))
            if rand:
                w = h = rng.choice([w, h])
            else:
                w = h = max(w, h)
            l = rng.uniform(self.left, self.right - w)
            t = rng.uniform(self.top, self.bottom - h)
            # if t + h > (self.bottom + 1):
                # print("Error", t, h, self.top, self.bottom - h, self.bottom, max_height, self.height)
                # print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
                # print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
                # print(f"t {t} boxt {self.top}")
                # print("Vertical Error", t + h, self.bottom)
            # if l + w > (self.right + 1):
                # print("Error")
            return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
            
        w = np.random.uniform(min_width, min(max_height, max_width))
        h = np.random.uniform(min_height, min(max_height, max_width))
        if rand:
                w = h = rng.choice([w, h])
        else:
            w = h = max(w, h)
        l = np.random.uniform(self.left, self.right - w)
        t = np.random.uniform(self.top, self.bottom - h)
        
        # if t + h > self.bottom:
        #     print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
        #     print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
        #     print(f"t {t} boxt {self.top}")
        #     print("Vertical Error", t + h, self.bottom)
        # if l + w > self.right:
        #     print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
        #     print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
        #     print(f"l {l} boxl {self.left}")
        #     print("Horizontal Error", l + w, self.right)

        return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)

    def sample_square_bboxv2(self, min_width=50, min_height=50, max_width=400, max_height=400, rng=None, rand=False):
        max_height = min(self.height, max_height)
        max_width = min(self.width, max_width)
        mindim = min(min_width, min_height)
        if rng is not None:
            w = rng.uniform(mindim, min(max_height, max_width))
            h = rng.uniform(mindim, min(max_height, max_width))
            if rand:
                w = h = rng.choice([w, h])
            else:
                w = h = max(w, h)
            l = rng.uniform(self.left, self.right - w)
            t = rng.uniform(self.top, self.bottom - h)
            # if t + h > (self.bottom + 1):
            #     print("Error", t, h, self.top, self.bottom - h, self.bottom, max_height, self.height)
            #     # print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
            #     # print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
            #     # print(f"t {t} boxt {self.top}")
            #     # print("Vertical Error", t + h, self.bottom)
            # if l + w > (self.right + 1):
            #     print("Error")
            return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
            
        w = np.random.uniform(mindim, min(max_height, max_width))
        h = np.random.uniform(mindim, min(max_height, max_width))
        if rand:
                w = h = rng.choice([w, h])
        else:
            w = h = max(w, h)
        l = np.random.uniform(self.left, self.right - w)
        t = np.random.uniform(self.top, self.bottom - h)
        
        # if t + h > self.bottom:
        #     print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
        #     print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
        #     print(f"t {t} boxt {self.top}")
        #     print("Vertical Error", t + h, self.bottom)
        # if l + w > self.right:
        #     print(f"h {h} w {w} boxh {self.height} minheight {min_height} maxposs {min(self.width, self.height)}")
        #     print(f"w {w} boxw w {w} {self.width} minwidth {min_width} maxposs {min(self.width, self.height)}")
        #     print(f"l {l} boxl {self.left}")
        #     print("Horizontal Error", l + w, self.right)

        return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
    
    def sample_bbox(self, min_width=20, max_width=50, min_height=20, max_height=50, rng=None):
        if rng is not None:
            w = rng.uniform(min_width, min(self.width, max_width))
            h = rng.uniform(min_height, min(self.height, max_height))
            l = rng.uniform(self.left, self.right - w)
            t = rng.uniform(self.top, self.bottom - h)
            
            return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)
            
        w = np.random.uniform(min_width, min(self.width, max_width))
        h = np.random.uniform(min_height, min(self.height, max_height))
        l = np.random.uniform(self.left, self.right - w)
        t = np.random.uniform(self.top, self.bottom - h)

        return BBox.from_center_dims(l + w / 2., t + h /2., w, h, self.img)

def draw_bbox(img, bbox, color="white", text=None):
    if bbox is None:
        return img
    source_img = img.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    draw.rectangle(bbox.get_bbox(), outline=color, width=4)
    if text is not None:
        draw.text(bbox.get_center(), text)
#     source_img.show()
    return source_img

def draw_point(img, point, color="white", r=3):
    source_img = img.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    x, y = point
    leftUpPoint = (x-r, y-r)
    rightDownPoint = (x+r, y+r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)
    return source_img
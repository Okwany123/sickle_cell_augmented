"""Augmented inference from the project's deployed detector and segmentation checkpoints."""

from io import BytesIO
from pathlib import Path
from threading import Lock
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d


ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / 'model_config.json').read_text())
CFG = SETTINGS['detector']
SEG = SETTINGS['segmenter']
DEVICE = torch.device('cpu')
INFERENCE_LOCK = Lock()

torch.set_num_threads(min(2, os.cpu_count() or 1))
cv2.setNumThreads(1)


def read_rgb(source):
    with Image.open(source) as image:
        w, h = image.size
        if min(w,h) < 32 or w*h > 36_000_000:
            raise ValueError(f'Unsupported image dimensions {w} × {h}')
        return np.asarray(image.convert('RGB')).copy()


def letterbox(rgb, size=None):
    size = CFG['size'] if size is None else size
    h, w = rgb.shape[:2]
    scale = size/max(w,h)
    nw, nh = max(1,round(w*scale)), max(1,round(h*scale))
    x,y = (size-nw)//2, (size-nh)//2
    canvas = np.full((size,size,3),127,np.uint8)
    canvas[y:y+nh,x:x+nw] = cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
    return canvas,(x,y,nw,nh)


def build_detector():
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone = resnet_fpn_backbone(backbone_name='resnet50',weights=None,
                                   norm_layer=FrozenBatchNorm2d,trainable_layers=0).to(DEVICE)
    for module in backbone.modules():
        if isinstance(module,FrozenBatchNorm2d):
            module.eps = 0.
    backbone.requires_grad_(False)
    anchors = AnchorGenerator(tuple((s,) for s in CFG['anchor_sizes']),
                              tuple(tuple(CFG['anchor_ratios']) for _ in CFG['anchor_sizes']))
    roi_pool = MultiScaleRoIAlign(featmap_names=['0','1','2'],output_size=3,sampling_ratio=2,
                                 canonical_scale=56,canonical_level=3)
    head = TwoMLPHead(256*3*3,256)
    detector = FasterRCNN(backbone,num_classes=None,min_size=CFG['size'],max_size=CFG['size'],
        rpn_anchor_generator=anchors,rpn_head=RPNHead(256,3),
        rpn_pre_nms_top_n_train=1000,rpn_pre_nms_top_n_test=650,
        rpn_post_nms_top_n_train=420,rpn_post_nms_top_n_test=300,
        rpn_batch_size_per_image=192,box_batch_size_per_image=128,
        box_roi_pool=roi_pool,box_head=head,box_predictor=FastRCNNPredictor(256,2),
        box_score_thresh=CFG['score_floor'],box_nms_thresh=CFG['nms'],
        box_detections_per_img=CFG['max_detections'])
    return detector.to(DEVICE)


def tensor_image(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255


def feature_levels(detector, batch):
    with torch.no_grad():
        levels = detector.backbone(batch)
    return {key:levels[key] for key in ('0','1','2')}


def transform_inputs(detector, images, targets=None):
    return detector.transform([x.to(DEVICE) for x in images],targets)


def detector_heads(detector,features,targets=None):
    from torchvision.models.detection.image_list import ImageList
    side = CFG['size']
    n = features['0'].shape[0]
    # Region proposal code uses only the batch shape and image dimensions here.
    image_list = ImageList(features['0'].new_empty((n,0,side,side)),[(side,side)]*n)
    proposals, rpn_losses = detector.rpn(image_list,features,targets)
    predictions, roi_losses = detector.roi_heads(features,proposals,image_list.image_sizes,targets)
    return predictions, {**rpn_losses,**roi_losses}


def unpack_boxes(out, geom):
    x,y,w,h = geom
    boxes = out['boxes'].detach().cpu().numpy()
    scores = out['scores'].detach().cpu().numpy()
    boxes = np.clip(boxes,[x,y,x,y],[x+w,y+h,x+w,y+h])
    visible = (boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    return dict(boxes=boxes[visible], scores=scores[visible])


@torch.inference_mode()
def predict_canvas(detector,rgb,geom):
    detector.eval()
    batch,_=transform_inputs(detector,[tensor_image(rgb)])
    features=feature_levels(detector,batch.tensors)
    features={k:v.half().float() for k,v in features.items()}
    outputs,_=detector_heads(detector,features)
    return unpack_boxes(outputs[0],geom)


def load_heads(detector,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    mismatch=detector.load_state_dict(state,strict=False)
    if mismatch.unexpected_keys or any(not k.startswith('backbone.') for k in mismatch.missing_keys):
        raise ValueError(f'Incompatible detector checkpoint {path}.')


def roi_geometry(box):
    left,top,right,bottom = map(float,box)
    side = max(12, int(np.ceil(SEG['context']*max(right-left,bottom-top))))
    return int(np.floor((left+right-side)/2)),int(np.floor((top+bottom-side)/2)),side


def roi_image(rgb, box):
    left,top,side = geom = roi_geometry(box)
    h,w = rgb.shape[:2]
    x1,y1,x2,y2 = max(left,0),max(top,0),min(left+side,w),min(top+side,h)
    if x2<=x1 or y2<=y1:
        raise ValueError('ROI does not intersect the image.')
    crop=rgb[y1:y2,x1:x2]
    crop=cv2.copyMakeBorder(crop,y1-top,top+side-y2,x1-left,left+side-x2,
                           borderType=cv2.BORDER_CONSTANT,value=(127,127,127))
    resize=cv2.INTER_AREA if side>=SEG['side'] else cv2.INTER_LINEAR
    patch=cv2.resize(crop,(SEG['side'],)*2,interpolation=resize)
    coordinates=(np.arange(SEG['side'])+.5)*side/SEG['side']
    xx,yy=np.meshgrid(coordinates+left,coordinates+top)
    b=np.asarray(box)
    support=(xx>=max(0,b[0]))&(xx<min(w,b[2]))&(yy>=max(0,b[1]))&(yy<min(h,b[3]))
    return patch,support,geom


def largest_roi_component(mask, support, min_pixels=8, fill=True):
    binary=(mask&support).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    if n<2:
        return np.zeros_like(mask,dtype=bool)
    center=np.argwhere(support).mean(0)
    yy,xx=np.ogrid[:mask.shape[0],:mask.shape[1]]
    weight=np.exp(-((yy-center[0])**2+(xx-center[1])**2)/(2*(SEG['side']*.24)**2))
    scores=np.bincount(labels.ravel(),weights=weight.ravel(),minlength=n)
    scores[0]=0
    scores[stats[:,cv2.CC_STAT_AREA]<min_pixels]=0
    if not scores.max():
        return np.zeros_like(mask,dtype=bool)
    selected=(labels==scores.argmax()).astype(np.uint8)
    if fill:
        contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(selected,contours,-1,1,cv2.FILLED)
    return selected.astype(bool)&support


class SickleUNet(nn.Module):
    """A compact three-level binary U-Net for detector crops."""
    def __init__(self):
        super().__init__()
        self.down=nn.ModuleList([self.block(a,b) for a,b in [(3,12),(12,24),(24,48)]])
        self.middle=self.block(48,72)
        self.up=nn.ModuleList([self.block(a,b) for a,b in [(120,48),(72,24),(36,12)]])
        self.output=nn.Conv2d(12,1,1)

    @staticmethod
    def block(a,b):
        return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True),
                             nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True))

    def forward(self,x):
        skips=[]
        for layer in self.down:
            x=layer(x)
            skips.append(x)
            x=F.max_pool2d(x,2)
        x=self.middle(x)
        for layer,skip in zip(self.up,reversed(skips)):
            x=layer(torch.cat((F.interpolate(x,scale_factor=2,mode='nearest'),skip),dim=1))
        return self.output(x)[:,0]


def build_segmenter():
    return SickleUNet().to(DEVICE,memory_format=torch.channels_last)


def seg_input(images):
    x=torch.from_numpy(np.ascontiguousarray(images.transpose(0,3,1,2))).to(DEVICE)
    return x.to(dtype=torch.float32,memory_format=torch.channels_last).div_(127.5).sub_(1.)


@torch.inference_mode()
def seg_scores(model,images):
    model.eval()
    if not len(images):
        return np.empty((0,SEG['side'],SEG['side']),np.float32)
    return np.concatenate([model(seg_input(images[i:i+SEG['batch']])).sigmoid().cpu().numpy()
                           for i in range(0,len(images),SEG['batch'])])


def mask_threshold(p,s,t):
    return largest_roi_component((p>=t)&s,s,SEG['minimum_component'])


def field_mask_shape(rgb):
    h,w=rgb.shape[:2]
    scale=min(1.,768/max(h,w))
    return max(1,round(h*scale)),max(1,round(w*scale))


def paste_roi(mask, geom, original_hw, target):
    """Place a square ROI in a field-aligned evaluation canvas or display mask."""
    left,top,side=geom
    h,w=original_hw
    th,tw=target.shape
    x1,y1=int(np.floor(left*tw/w)),int(np.floor(top*th/h))
    x2,y2=int(np.ceil((left+side)*tw/w)),int(np.ceil((top+side)*th/h))
    xa,ya,xb,yb=max(x1,0),max(y1,0),min(x2,tw),min(y2,th)
    if xb<=xa or yb<=ya:
        return
    enlarged=cv2.resize(mask.astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
    target[ya:yb,xa:xb] |= enlarged[ya-y1:yb-y1,xa-x1:xb-x1].astype(bool)


def predict_field_mask(model,rgb,boxes,threshold):
    out=np.zeros(field_mask_shape(rgb),bool)
    if not len(boxes):
        return out,[]
    patches,supports,geometries=zip(*(roi_image(rgb,b) for b in boxes))
    probabilities=seg_scores(model,np.stack(patches))
    masks=[mask_threshold(p,s,threshold) for p,s in zip(probabilities,supports)]
    for mask,geom in zip(masks,geometries):
        paste_roi(mask,geom,rgb.shape[:2],out)
    return out,masks


def colour_mask(rgb,mask,colour=(24,226,200)):
    picture=cv2.resize(rgb,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_AREA)
    picture[mask]=(picture[mask].astype(np.float32)*.59+np.asarray(colour)*.41).astype(np.uint8)
    return picture


def combine_upload_mask(mask,shape):
    return cv2.resize(mask.astype(np.uint8)*255,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)


def load_augmented_model():
    detector = build_detector().eval()
    # This checkpoint is the public COCO source of the project's frozen backbone.
    official = torch.hub.load_state_dict_from_url(
        SETTINGS['backbone_url'], map_location='cpu', check_hash=True, weights_only=True, progress=False
    )
    backbone = {name.removeprefix('backbone.'): tensor for name, tensor in official.items()
                if name.startswith('backbone.')}
    detector.backbone.load_state_dict(backbone)
    del official, backbone
    load_heads(detector, ROOT / 'augmented_detector.pt')

    segmenter = build_segmenter().eval()
    segmenter.load_state_dict(torch.load(ROOT / 'augmented_segmenter.pt', map_location='cpu', weights_only=True))
    return detector, segmenter


def analyze_image(data, models):
    rgb = read_rgb(BytesIO(data))
    detector, segmenter = models
    canvas, geometry = letterbox(rgb)
    x, y, scaled_width, scaled_height = geometry
    height, width = rgb.shape[:2]

    with INFERENCE_LOCK:
        predictions = predict_canvas(detector, canvas, geometry)
        scores = predictions['scores']
        cutoff = SETTINGS['detector_threshold']
        score = float(max(scores, default=0.0))
        selected = scores >= cutoff
        scale_x, scale_y = width / scaled_width, height / scaled_height
        boxes = (predictions['boxes'][selected] - [x, y, x, y]) * [scale_x, scale_y, scale_x, scale_y]
        mask, masks = predict_field_mask(segmenter, rgb, boxes, SETTINGS['mask_threshold'])

    overlay = Image.fromarray(colour_mask(rgb, mask, (18, 227, 124)))
    preview_height, preview_width = mask.shape
    sx, sy = preview_width / width, preview_height / height
    draw = ImageDraw.Draw(overlay)
    for box in boxes * [sx, sy, sx, sy]:
        draw.rectangle(tuple(box.tolist()), outline='#ffbd39', width=max(2, min(4, preview_width // 200)))
    original = Image.fromarray(cv2.resize(rgb, (preview_width, preview_height), interpolation=cv2.INTER_AREA))
    mask_png = BytesIO()
    Image.fromarray(combine_upload_mask(mask, (height, width))).save(mask_png, format='PNG')
    return {
        'prediction': 'Positive' if score >= cutoff else 'Negative',
        'score': score,
        'boxes': len(boxes),
        'masks': sum(int(roi.any()) for roi in masks),
        'original': original,
        'overlay': overlay,
        'mask_png': mask_png.getvalue(),
    }

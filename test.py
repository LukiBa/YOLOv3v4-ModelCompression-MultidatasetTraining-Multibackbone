import argparse
import json
from pathlib import Path
import shutil
from torch.utils.data import DataLoader
from models import *
from utils.datasets import *
from utils.utils import *
import torch
import torch.utils
import utils.quantized.quantized_intuitus_new as q_new

torch.set_default_dtype(torch.float32)
DEVICE = 'cpu'

def _create_parser():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--model', type=str, default=None, help='model path') #'pt_models/dorefa.pt'
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3tiny/yolov3-tiny-quant.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017_val_split.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/rt.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--img_outpath', type=str, default=None)#'./detect_imgs', help='Path to output images. If set to None images are not saved.') #'./detect_imgs'
    parser.add_argument('--param_outpath', type=str, default='./parameters/int8_6')#'./detect_imgs', help='Path to output images. If set to None images are not saved.') #'./detect_imgs'
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--quantized', type=int, default=6,help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--a-bit', type=int, default=8, help='a-bit')
    parser.add_argument('--w-bit', type=int, default=6, help='w-bit')
    parser.add_argument('--FPGA', type=bool, default=False)#action='store_true', help='FPGA')
    parser.add_argument('--load_model', type=str, default=None, help='weights saved as model')
    return parser.parse_args()    

def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=True,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True,
         quantized=-1,
         a_bit=8,
         w_bit=8,
         FPGA=False,
         rank=None,
         img_outpath=None,
         param_outpath=None,
         load_model=None):
    
    torch.cuda.empty_cache()
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz, quantized=quantized, a_bit=a_bit, w_bit=w_bit,
                        FPGA=FPGA)

        if FPGA:
            dev = 'cpu'
        else:
            dev = DEVICE 

        # Load weights
        #attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            if opt.load_model != None:
                model =  torch.load(load_model)
            else:
                chkpt = torch.load(weights, map_location=device)
                if True:
                    weight_dict = {}
                    for name in chkpt['model'].keys():
                        if '.0.0.' in name:
                            new_name = name.replace('.0.0.','.0.Conv2d.')
                            
                        elif '.0.' in name and not 'Conv2d' in name and not 'BatchNorm2d' in name:
                            new_name = name.replace('.0.','.Conv2d.')
                        else:
                            new_name = name  
                        weight_dict[new_name] = chkpt['model'][name]
                else:
                    weight_dict = chkpt['model']
                    
                chkpt['model'] = {k: v for k, v in weight_dict.items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(weight_dict, strict=False)                
            #layer = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            #layer[3].fold_bn = True
            
        else:  # darknet format
            load_darknet_weights(model, weights, FPGA=FPGA)

        # Fuse
        model.fuse(quantized=quantized, FPGA=opt.FPGA)
        #print(model)
        model.to(dev)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        # summary(model, input_size=(3, imgsz, imgsz))
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    
    if img_outpath != None:
        img_outpath = Path(img_outpath).absolute()
        img_predpath = img_outpath / 'pred'
        img_gndpath = img_outpath / 'gnd'
        if img_predpath.exists():
            shutil.rmtree(img_predpath, ignore_errors=False, onerror=None)
        if img_gndpath.exists():
            shutil.rmtree(img_gndpath, ignore_errors=False, onerror=None)    
        img_predpath.mkdir(parents=True, exist_ok=True)
        img_gndpath.mkdir(parents=True, exist_ok=True)
        
    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels2(path, imgsz, batch_size, single_cls=single_cls)
        test = dataset[0]
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=1,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    
    # _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    #pbar = tqdm(dataloader, desc=s) if rank in [-1, 0] else dataloader
    pbar = tqdm(dataloader, desc=s)
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
        
    model.to(DEVICE)
    model.eval()
    
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        
        imgs = imgs.to(device).float()
        #if opt.quantized != 5:
        imgs = imgs/255.0

        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()

            inf_out, train_out, _ = model(imgs, augment=augment)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                #tbox = xywh2xyxy(labels[:, 1:5])
                #tbox = labels[:, 1:5]
                
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if img_outpath != None:#if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=str(img_gndpath/f))  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=str(img_predpath/f))  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    if rank in [-1, 0]:
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # cocovision = opt.data.split('\\')[-1].split('.')[0]
            # print(cocovision)
            # cocoGt = COCO(glob.glob('data/'+cocovision+'/instances_val*.json')[0])  # initialize COCO ground truth api
            coco_instances_path = str(Path(data['instances']).absolute())
            cocoGt = COCO(glob.glob(coco_instances_path)[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Write Parameter to txt 
    if param_outpath != None:
        model.write_parameter_to_txt(param_outpath)
    
    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
   
    opt = _create_parser()     
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.data = list(glob.iglob('./**/' + opt.data, recursive=True))[0]  # find file
    model = None
    if opt.model != None:
        model = torch.load(opt.model)

    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        results, maps = test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             model = model,
             quantized=opt.quantized,
             a_bit=opt.a_bit,
             w_bit=opt.w_bit,
             FPGA=opt.FPGA,
             param_outpath = opt.param_outpath,
             img_outpath = opt.img_outpath,
             load_model = opt.load_model)
        
        s = ('P','R','mAP@0.5','F1')
        print('%10s' * 4 % s + '\n')
        print('%10.3g' * 4 % results[:4] + '\n')

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

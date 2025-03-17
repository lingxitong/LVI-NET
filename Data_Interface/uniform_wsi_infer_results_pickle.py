
import pickle,os
from PIL import Image
import concurrent.futures
import time
import functools
import numpy as np

class Uniform_wsi_infer_results_pickle:
    def __init__(self,pickle_path):
        self.pickle_path = pickle_path
        assert self._check_is_uniform_wsi_infer_results_pickle(self.pickle_path)
        with open(self.pickle_path,'rb') as f:
            self.uniform_wsi_infer_results_pickle_data = pickle.load(self.pickle_path)

    def cal_time(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{func.__name__} took {elapsed_time:.4f} seconds")
            return result
        return wrapper
    
    def save_pickle(self,save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.uniform_wsi_infer_results_pickle_data, f)
        print(f"Filtered results saved to {save_path}")


    def _check_masks(masks:list):
        assert isinstance(masks, list)
        for mask in masks:
            assert all(isinstance(mask_point, tuple) and len(mask_point) == 2 and all(isinstance(x, int) for x in mask_point) for mask_point in mask)
            assert mask[0] == mask[-1]

    def _check_bboxes(bboxes:list):
        assert isinstance(bboxes, list)
        for box in bboxes:
            assert isinstance(box, list)
            assert len(box) == 4
            assert all(isinstance(x, int) for x in box)
            x_min,y_min,x_max,y_max = tuple(box)
            assert x_min < x_max - 1    
            assert y_min < y_max - 1
    def _check_is_uniform_wsi_infer_results_pickle(self):
        try:
            with open(self.pickle_path,'rb') as f:
                pickle_data = pickle.load(f)
            masks = pickle_data['masks']
            bboxes = pickle_data['bboxes']
            scores = pickle_data['scores']
            assert len(masks) == len(bboxes) == len(scores)
            num_instances = len(masks)
            if num_instances != 0:
                self._check_bboxes(bboxes)
                self._check_masks(masks)
            return True
        except:
            return False
        
    def _convert_box_to_qupath_box(self,box:list[int]):
        x_min,y_min,x_max,y_max = tuple(box)
        qupath_box = [[[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max],[x_min,y_min]]]
        return qupath_box
    
    def _convert_mask_to_qupath_mask(self,mask:list[tuple[int,int]]):
        qupath_mask = [[list(mask_point) for mask_point in mask]]
        return qupath_mask
        
    def convert_to_qupath_json(self, qupath_json_save_path, remove_bboxes = False, remove_masks = False, box_color = [185,203,90], mask_color = [100,203,70]):
        assert qupath_json_save_path.endswith('.json')
        assert not (remove_bboxes and remove_masks)
        qupath_json_dict = {"type": "FeatureCollection", "features": []}
        if not remove_bboxes:
            bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes']
            box_features = [{"type":"feature",
                             "id":f"box_{index+1}",
                             "geometry":{"type":"Polygon","coordinates":self._convert_box_to_qupath_box(box)},
                             "properties":{"objectType":"annotation","classification":{"name":"LVI+","color":box_color}}} for index,box in enumerate(bboxes)]
            qupath_json_dict["features"].extend(box_features)
            if not remove_masks:
                masks = self.uniform_wsi_infer_results_pickle_data['masks']
                box_features = [{"type":"feature",
                                "id":f"mask_{index+1}",
                                "geometry":{"type":"Polygon","coordinates":self._convert_mask_to_qupath_mask(mask)},
                                "properties":{"objectType":"annotation","classification":{"name":"LVI+","color":mask_color}}} for index,mask in enumerate(masks)]
                qupath_json_dict["features"].extend(box_features)
        import json
        with open(qupath_json_save_path,'w') as f:
            json.dump(qupath_json_dict,f)
            print(f'qupath format json has been saved to {qupath_json_save_path}')
            
    def vis_pickle_results_on_wsi(self, wsi_path, vis_save_path, vis_level, 
                                  remove_bboxes = False, remove_masks = False, draw_score_above = 0,
                                  box_color = [185,203,90], mask_color = [100,203,70]):
        assert os.path.exists(wsi_path)
        assert vis_save_path.endswith(('.jpg','.png'))
        import openslide
        try:
            wsi = openslide.OpenSlide(wsi_path)
            assert vis_level <= wsi.level_count - 1
            if wsi.level_count > 1:
                level_scale_factor = wsi.level_downsamples[1]//wsi.level_downsamples[0] # level_scale_factor >= 1
            else:
                level_scale_factor = 1
        except:
            raise "error occurred because wsi process"
        from PIL import ImageDraw
        canvas = wsi.read_region((0, 0),vis_level, wsi.level_dimensions[vis_level])
        canvas = canvas.convert("RGB")
        drawer = ImageDraw.Draw(canvas)
        downsample_factor = level_scale_factor**vis_level # downsample_factor >= 1
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        if not remove_bboxes:
            bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes']
            bboxes = bboxes[[i for (i,score) in enumerate(scores) if score >= draw_score_above]]
            bboxes = np.array(bboxes)/downsample_factor
            bboxes = list(bboxes)
            for bbox in bboxes:
                drawer.rectangle(bbox, outline=box_color, width=2)
        if not remove_masks:
            masks = self.uniform_wsi_infer_results_pickle_data['masks']
            masks = masks[[i for (i,score) in enumerate(scores) if score >= draw_score_above]]
            masks = np.array(masks)/downsample_factor
            masks = list(masks)
            for mask in masks:
                drawer.polygon(mask, outline=mask_color, width=2)
        canvas.save(vis_save_path)

    @staticmethod
    def default_cut_fn(x_min,y_min,x_max,y_max) -> dict: 
        w = x_max - x_min
        h = y_max - y_min
        max_length = max(w,h)
        cut_size = max_length
        if max_length > 1024:
            resize_size = 1024
        elif max_length < 224:
            resize_size = 224
        else:
            resize_size = 512
        return {'x_min':int(x_min),'y_min':int(y_min),'cut_size':int(cut_size),'resize_size':int(resize_size)}
    
    @staticmethod
    def default_name_patch_fn(wsi_path,x_min,y_min,cut_size,resize_size):
        return f'{os.path.basename(wsi_path).splitext[0]}_Xmin_{x_min}_Ymin_{y_min}_Cutsize_{cut_size}_Resize_{resize_size}'
    
    def _cut_and_save_one_patch(self,wsi_path,wsi,cut_info:dict,patch_save_dir,patch_ext,name_patch_fn):

        x_min,y_min,cut_size,resize_size = cut_info['x_min'],cut_info['y_min'],cut_info['cut_size'],cut_info['resize_size']
        patch = wsi.read_region((x_min,y_min),0,(cut_size,cut_size)).convert('RGB').resize((resize_size,resize_size),Image.LANCZOS)
        save_path = os.path.join(patch_save_dir,name_patch_fn(wsi_path,x_min,y_min,cut_size,resize_size)+patch_ext)
        patch.save(save_path)

    @cal_time
    def cut_instances_to_patches(self, wsi_path, patch_save_dir, cut_fn = default_cut_fn, name_patch_fn = default_name_patch_fn,cut_score_above = 0, patch_ext = '.jpg', max_workers = 8):
        assert os.path.exists(wsi_path)
        assert patch_ext in ['.jpg','.png']
        import openslide
        try:
            wsi = openslide.OpenSlide(wsi_path)
        except:
            raise "error occurred because wsi process"
        bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes']
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        bboxes = bboxes[[i for (i,score) in enumerate(scores) if score >= cut_score_above]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for box in bboxes:
                cut_info = cut_fn(tuple(box))
                futures.append(
                    executor.submit(
                        self._cut_and_save_one_patch,
                        wsi_path,
                        wsi,
                        cut_info,
                        patch_save_dir,
                        patch_ext,
                        name_patch_fn))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result() 
                except Exception as e:
                    print(f"Error processing patch: {e}")

    def postprocess(self, func, safe = False):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not safe:
                print(f"warning: {func.__name__} is unsafe postprocess!")
            else:
                print(f"{func.__name__} is safe postprocess.")
            return result
        return wrapper

    @cal_time
    @postprocess(safe=True)
    def filter_contain_mask_and_save(self, save = False, save_path = None, inplace = True):
        if save:
            assert os.path.exists(save_path)
            assert save_path.endswith('.pickle')
        if (not save) and (not inplace):
            return
        masks = self.uniform_wsi_infer_results_pickle_data['masks']
        bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes'] 
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        sorted_indices = np.argsort(areas)[::-1]
        filtered_masks = []
        filtered_bboxes = []
        filtered_scores = []
        for idx in sorted_indices:
            current_mask = masks[idx]
            current_bbox = bboxes[idx]
            current_score = scores[idx]
            is_contained = False
            for filtered_mask, filtered_bbox in zip(filtered_masks, filtered_bboxes):
                if (filtered_bbox[0] <= current_bbox[0] and
                    filtered_bbox[1] <= current_bbox[1] and
                    filtered_bbox[2] >= current_bbox[2] and
                    filtered_bbox[3] >= current_bbox[3]):
                    if all(point in filtered_mask for point in current_mask):
                        is_contained = True
                        break
            if not is_contained:
                filtered_masks.append(current_mask)
                filtered_bboxes.append(current_bbox)
                filtered_scores.append(current_score)
        if inplace:
            self.uniform_wsi_infer_results_pickle_data['masks'] = filtered_masks
            self.uniform_wsi_infer_results_pickle_data['bboxes'] = filtered_bboxes
            self.uniform_wsi_infer_results_pickle_data['scores'] = filtered_scores
        if save:
            self.save_pickle(save_path)


    def _update_bbox(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        new_x_min = min(x1, x3)
        new_y_min = min(y1, y3)
        new_x_max = max(x2, x4)
        new_y_max = max(y2, y4)
        return [new_x_min, new_y_min, new_x_max, new_y_max]

    @cal_time
    @postprocess(safe=True)
    def merge_overlap_mask_and_save(self, save = False, save_path = None, inplace = True):
        if save:
            assert os.path.exists(save_path)
            assert save_path.endswith('.pickle')
        if (not save) and (not inplace):
            return
        masks = self.uniform_wsi_infer_results_pickle_data['masks']
        bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes'] 
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        n = len(masks)
        from .union_find import UnionFind
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if np.any(np.logical_and(np.array(masks[i]), np.array(masks[j]))):
                    uf.union(i, j)
        merged_masks = []
        merged_bboxes = []
        merged_scores = []
        processed = [False] * n
        for i in range(n):
            if processed[i]:
                continue
            root = uf.find(i)
            if processed[root]:
                continue
            processed[root] = True
            merged_mask = masks[root].copy()
            merged_bbox = bboxes[root].copy()
            merged_score = scores[root]
            for j in range(n):
                if uf.find(j) == root:
                    merged_mask = np.logical_or(merged_mask, masks[j])
                    merged_mask = [tuple(mask_point) for mask_point in merged_mask]
                    merged_bbox = self._update_bbox(tuple(merged_bbox), tuple(bboxes[j]))
                    merged_score = max(merged_score, scores[j])
            merged_masks.append(merged_mask)
            merged_bboxes.append(merged_bbox)
            merged_scores.append(merged_score)
        if inplace:
            self.uniform_wsi_infer_results_pickle_data['masks'] = merged_masks
            self.uniform_wsi_infer_results_pickle_data['bboxes'] = merged_bboxes
            self.uniform_wsi_infer_results_pickle_data['scores'] = merged_scores
        if save:
            self.save_pickle(save_path)

    @staticmethod
    def default_box_filter_fn(box:list) -> bool:
        return False
    
    @cal_time
    @postprocess(safe=False)
    def filter_pickle_data_use_box_and_save(self, box_filter_fn = default_box_filter_fn, save = False, save_path = None, inplace = True):
        if save:
            assert os.path.exists(save_path)
            assert save_path.endswith('.pickle')
        if (not save) and (not inplace):
            return
        masks = self.uniform_wsi_infer_results_pickle_data['masks']
        bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes'] 
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        filtered_bboxes = []
        filtered_masks = []
        filtered_scores = []
        for bbox, mask, score in zip(bboxes, masks, scores):
            if not box_filter_fn(bbox):  
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)
                filtered_scores.append(score)
        if inplace:
            self.data['bboxes'] = filtered_bboxes
            self.data['masks'] = filtered_masks
            self.data['scores'] = filtered_scores
        if save:
            self.save_pickle(save_path)

    def filter_pickle_data_use_score_and_save(self, score_filter_thresh = 0, save = False, save_path = None, inplace = True):
        if save:
            assert os.path.exists(save_path)
            assert save_path.endswith('.pickle')
        if (not save) and (not inplace):
            return
        masks = self.uniform_wsi_infer_results_pickle_data['masks']
        bboxes = self.uniform_wsi_infer_results_pickle_data['bboxes'] 
        scores = self.uniform_wsi_infer_results_pickle_data['scores']
        filtered_bboxes = []
        filtered_masks = []
        filtered_scores = []
        for bbox, mask, score in zip(bboxes, masks, scores):
            if score >= score_filter_thresh:
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)
                filtered_scores.append(score)
        if inplace:
            self.data['bboxes'] = filtered_bboxes
            self.data['masks'] = filtered_masks
            self.data['scores'] = filtered_scores
        if save:
            self.save_pickle(save_path)

    @cal_time
    @postprocess(safe = True)
    def roi_expand_and_save(self, wsi_path, model, save = False, save_path = None, inplace = True):
        pass
        
        


            



            
        
            


        



        

import random
from matplotlib import pyplot as plt
from skimage.io import imread
import re

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np

# reads images from path, transforms them to correct size and as a list of floats, also applies augmentation
class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        clss_t = None,
        clss_m = None,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
        classes=1,
        model_type=0

    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.class_tree = clss_t
        self.class_map = clss_m
        self.n_classes = classes
        self.model_type = model_type


    def _compute_masks_post(self, tree, target_mask, out):
        """
        Post-order: populate `out[name]` as 0/1 mask for every node in `tree`.
        Leaves: from target_mask via name2pix.
        Parents: bitwise OR of direct children (which are already in `out`).
        """
        H, W = target_mask.shape
        for name, subtree in tree.items():
            has_children = isinstance(subtree, dict) and len(subtree) > 0
            if has_children:
                # recurse into children first
                self._compute_masks_post(subtree, target_mask, out)
                # OR the direct children's masks
                agg = np.zeros((H, W), dtype=np.uint8)
                for child_name in subtree.keys():  # preserve insertion order
                    agg |= out[child_name]
                out[name] = agg
            else:
                # exact name match; raise if missing
                try:
                    pix = int(self.name2pix[name])
                except (AttributeError, KeyError):
                    # fallback path if name2pix isn't built
                    row = self.class_map.loc[self.class_map['class_name'] == name, 'pixel_val']
                    if row.empty:
                        raise KeyError(f"Class '{name}' not found in class_map.")
                    pix = int(row.iloc[0])
                out[name] = (target_mask == pix).astype(np.uint8)
    def _level_order_names(self, tree):
        """
        Level-order over the *forest* at the root of `tree`, preserving dict insertion order.
        Returns a list of node names in parent-before-children order.
        """
        from collections import deque
        q = deque()
        order = []
        # enqueue top-level nodes in order
        for name, subtree in tree.items():
            q.append((name, subtree))
        while q:
            name, subtree = q.popleft()
            order.append(name)
            if isinstance(subtree, dict) and len(subtree) > 0:
                for child_name, child_subtree in subtree.items():
                    q.append((child_name, child_subtree))
        return order
    def traverse_tree(self, tree, masks, target_mask):
        """
        Compute parent masks as sums of their children, returns masks in level-order.
        - Returns (masks, combined_mask) where combined_mask is OR of all top-level nodes.
        - Masks appended are 0/255 uint8 in the correct hierarchical order.
        """
        # 1) Build all node masks (0/1) by post-order DFS
        all_masks = {}  # name -> 0/1 uint8
        self._compute_masks_post(tree, target_mask, all_masks)
        # 2) Emit in level-order (parents before children), preserving insertion order
        names_in_order = self._level_order_names(tree)
        # If you only want leaves in flat mode, filter here:
        if getattr(self, 'model_type', 1) == 0:
            # keep only leaves in that same level-order walk
            def is_leaf(n):
                # walk the tree to see if `n` has children
                stack = [tree]
                while stack:
                    t = stack.pop()
                    for k, v in t.items():
                        if k == n:
                            return not (isinstance(v, dict) and len(v) > 0)
                        if isinstance(v, dict):
                            stack.append(v)
                return True  # default
            names_to_emit = [n for n in names_in_order if is_leaf(n)]
        else:
            # hierarchical: emit every node
            names_to_emit = names_in_order
        # Append masks in the chosen order
        for n in names_to_emit:
            masks.append((all_masks[n] * 255).astype(np.uint8))
        # Combined mask for the whole forest (useful if caller expects a return mask)
        H, W = target_mask.shape
        combined = np.zeros((H, W), dtype=np.uint8)
        for top_name in tree.keys():  # OR of top-level nodes
            combined |= all_masks[top_name]
        return masks, combined

    # def traverse_tree(self, tree, masks, target_mask):
    #     """
    #     depth first search over hierarchy tree (nested dict).
    #     For leaf nodes: create mask from target_mask using class mapping.
    #     For internal nodes: recursively accumulate children's masks (logical OR).
    #     In hierarchical mode (model_type==1), append each parent mask (sum/union of its children).
    #     Returns (masks, mask_this_subtree) where mask_this_subtree is 0/1 uint8.
    #     """
    #     # accumulate a boolean mask for the subtree
    #     agg_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    #     for key, subtree in tree.items():
    #         has_children = isinstance(subtree, dict) and len(subtree) > 0
    #         if has_children:
    #             # recursive call for depth first search
    #             masks, child_mask = self.traverse_tree(subtree, masks, target_mask)
    #             # child_mask accumulated into the parent mask
    #             agg_mask |= child_mask.astype(np.uint8)
    #         else:
    #             # exact matching first
    #             row = self.class_map.loc[self.class_map['class_name'] == key, 'pixel_val']
    #             # partial matching contingency
    #             if row.empty:
    #                 pattern = rf'(?:^|[^A-Za-z0-9_]){re.escape(key)}(?:[^A-Za-z0-9_]|$)'
    #                 row = self.class_map.loc[self.class_map['class_name'].str.contains(pattern, regex=True), 'pixel_val']
    #             if row.empty:
    #                 raise KeyError(f"Class '{key}' not found in class_map.")
    #             pix = int(row.iloc[0])
    #             # build leaf mask
    #             leaf01 = (target_mask == pix).astype(np.uint8)
    #             masks.append((leaf01 * 255).astype(np.uint8))
    #             # accumulate masks for parent
    #             agg_mask |= leaf01

    #     # if self.model_type == 1:
    #     masks.append((agg_mask * 255).astype(np.uint8))

    #     return masks, agg_mask
    

    # # traverse class_tree and extracts binary mask from the target mask. parent classes without any assigned pixel values are assigned masks of the sum of its child classes
    # def traverse_tree(self, tree, masks, target_mask):
    #     child_mask = np.zeros(target_mask.shape, dtype=np.int64) #, dtype=np.uint8
    #     for key, subtree in tree.items():
    #         if subtree:
    #             # inserts temp mask into masks
    #             if self.model_type == 1:
    #                 masks.append(np.zeros(target_mask.shape, dtype=np.int64)) # , dtype=np.uint8
    #             temp_mask_loc = len(masks) - 1
    #             masks, child_mask = self.traverse_tree(subtree, masks, target_mask)
    #             # changes temp mask to combined child mask
    #             if self.model_type == 1:
    #                 masks[temp_mask_loc] = child_mask
                
    #         else:
    #             mask = np.where(target_mask == int(self.class_map[self.class_map['class_name'].str.contains(key)]['pixel_val'].item()), 255, 0)
    #             masks.append(mask)
    #             # combines mask and child_mask
    #             child_mask = np.where(mask == 255, 255, child_mask)
    #         # if subtree:  # If there are nested dictionaries, traverse them
    #         #     self.traverse_tree(subtree, masks, target_mask)
    #     return masks, child_mask


    # traverses tree and processes child masks change values to -1 if the parent class is 0 
    # def process_ignore_values(self, tree, masks, parent=None, clss_index = 0):
    #     for key, subtree in tree.items():
    #         if subtree:
    #             # traverses child masks and processes them
    #             parent = masks[clss_index]
    #             masks, clss_index = self.process_ignore_values(subtree, masks, parent, clss_index + 1)
    #         else:
    #             # if the parent class is 0, change the child mask to -1
    #             if parent is not None:
    #                 # changes value in masks[clss_index] to -1 if parent is 0
    #                 masks[clss_index] = torch.where(parent == 0.0, -1.0, masks[clss_index])
    #             clss_index += 1
    #     return masks, clss_index






    def build_parent_map(self, tree, parent=None, out=None):
        """name -> direct_parent_name (or None for roots)."""
        if out is None:
            out = {}
        for name, subtree in tree.items():
            out[name] = parent
            if isinstance(subtree, dict) and len(subtree) > 0:
                self.build_parent_map(subtree, parent=name, out=out)
        return out
    def as_binary(self, t):
        """Convert mask to binary 0/1 float (accept 0/255, 0/1, any shape with batch/channel)."""
        # treat any positive value as 1
        return (t > 0).to(dtype=torch.float32)
    def broadcast_like(self, src, ref):
        """Broadcast src to ref's ndim by unsqueezing leading dims as needed."""
        while src.ndim < ref.ndim:
            src = src.unsqueeze(0)
        return src
    def process_ignore_values(self, tree, masks, name_to_index):
        """
        Convert per-class masks to ternary:
        roots:   1 on class, 0 elsewhere  (no -1 since no parent)
        non-roots:
            1 on class,
            0 inside direct parent's area but not class,
            -1 outside direct parent's area.
        Operates in-place on `masks`.
        """
        # 1) parent links
        parent_of = self.build_parent_map(tree)
        # 2) normalise all current masks to binary 0/1 for logical ops
        #    (works whether you passed in leaf-only or parent-union masks)
        bin_masks = {}
        for name, idx in name_to_index.items():
            bin_masks[name] = self.as_binary(masks[idx])
        # 3) write ternary masks back
        for name, idx in name_to_index.items():
            parent = parent_of.get(name, None)
            child = bin_masks[name]                   # 0/1 float
            # ensure we edit a float tensor that can hold -1
            out = masks[idx].to(dtype=torch.float32)
            if parent is None:
                # Root: no ignore region. Use standard 0/1.
                out = torch.where(child > 0,
                                torch.ones_like(out),
                                torch.zeros_like(out))
            else:
                p = self.broadcast_like(bin_masks[parent], child)   # 0/1 over same spatial dims as child
                p = p.to(dtype=child.dtype)
                # start with all -1
                out = torch.full_like(out, -1.0)
                # inside parent's area: set to 0
                out = torch.where(p > 0, torch.zeros_like(out), out)
                # child area: set to 1 (overrides the 0 inside parent)
                out = torch.where(child > 0, torch.ones_like(out), out)
            masks[idx] = out
        return masks





    # def process_ignore_values(self, tree, masks, name_to_index, parent_name=None):
    #     """
    #     Set each child's mask to -1 where its (direct) parent's mask is 0.
    #     - tree: nested dict hierarchy.
    #     - masks: list[Tensor], emitted in *any* order (BFS in your new code).
    #     Each tensor can be [H,W], [1,H,W], or [B,1,H,W].
    #     - name_to_index: dict: node name -> index in masks.
    #     Returns the same `masks` list with in-place modifications.
    #     """
    #     # Helper: broadcast parent's zeros to child's shape
    #     def parent_zero_mask(parent_t, child_t):
    #         # ensure comparable dtype & shape for where()
    #         if parent_t.dtype != child_t.dtype:
    #             parent_t = parent_t.to(child_t.dtype)
    #         # broadcast across leading dims if needed
    #         # Acceptable shapes: [H,W], [1,H,W], [B,1,H,W]
    #         while parent_t.ndim < child_t.ndim:
    #             parent_t = parent_t.unsqueeze(0)
    #         # now rely on broadcasting on channel/batch dims
    #         return (parent_t == 0)
    #     for name, subtree in tree.items():
    #         idx = name_to_index[name]
    #         parent_t = None
    #         if parent_name is not None:
    #             pidx = name_to_index[parent_name]
    #             parent_t = masks[pidx]
    #         has_children = isinstance(subtree, dict) and len(subtree) > 0
    #         if not has_children:
    #             if parent_t is not None:
    #                 # where parent==0, set child to -1 (keep child elsewhere)
    #                 zero_mask = parent_zero_mask(parent_t, masks[idx])
    #                 masks[idx] = torch.where(zero_mask,
    #                                         torch.full_like(masks[idx], -1),
    #                                         masks[idx])
    #         else:
    #             # Recurse to children with this node as their parent
    #             self.process_ignore_values(subtree, masks, name_to_index, parent_name=name)
    #     return masks


    # separate masks into binary masks for each class
    # this works by taking pixel 255 as the first class with 0 as background and 255 - 1 as the second class with 0 as background
    def separate_masks(self, target):
        # traverses class_tree indexes and branches and creates masks 
        masks = []
        # if self.model_type == 0:
        #     # if model_type is 0, use the class_map to create masks
        #     for i in range(self.n_classes):
        #         pixel_val = self.class_map[self.class_map['class_id'] == i]['pixel_val'].item()
        #         if not np.isnan(pixel_val):
        #             mask = np.where(target == int(pixel_val), 255, 0)
        #             masks.append(mask)
        # else:
        masks, _ = self.traverse_tree(self.class_tree, masks, target)
        # for i in range(len(masks)):
        #     # ues plt to show masks
        #     plt.imshow(masks[i])
        #     plt.show()
        
        # for i in range(self.n_classes):
        #     print(i)
        #     print(self.class_map)
        #     print(self.class_map["pixel_val"][i])
        #     print(self.class_tree)
        #     print(self.class_tree[self.class_map[i]])
        #     stop

        # stop
        
        
        # first_class = 255

        # masks = []
        # # extract background mask first
        # if self.include_background:
        #     masks.append(np.where(target == 0, 255, 0))
        #     n_classes -= 1

        # # loop the number of classes and create a new tensor with 0s and first class value
        # for i in range(n_classes):
        #     masks.append(np.where(target == first_class, 255, 0))
        #     first_class -= 1

        return masks
    # recombines masks into a single tensor (overwrites parent class masks with child class masks)
    # def combine_masks(self, target, n_classes):

    #     #background
    #     masks = torch.where(target[0] == 255, 0, 0)
    #     # traverse targets and combine into a single tensor as 0, 1, 2
    #     for i in range(n_classes - 1):
    #         masks = torch.where(target[i+1] == 255, i+1, masks)

    #     # else:
    #     #     masks = torch.where(target[0] == 255, 1, 0)

    #     #     # traverse targets and combine into a single tensor as 0, 1, 2
    #     #     for i in range(n_classes - 1):
    #     #         masks = torch.where(target[i+1] == 255, i+2, masks)

    #     # change 255 to 1 and 254 to 2
    #     # for i in range(n_classes-1):
    #     #     masks = np.where(masks == first_class, clss_num, masks)
    #     #     first_class -= 1
    #     #     clss_num += 1
            
    #     return masks

    def decode_segmap(self, temp):   # temp is HW np slice
        r = temp.copy() 
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))   # dummy tensor in np order HWC
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]
        # TODO: load model without hierarchy if args.model_type == 0
        x, y = imread(input_ID), imread(target_ID)

        # catch non values for composite masks
        # if y doesnt contain 42 as a value
        # if 42 not in np.unique(y):
        #     print(target_ID)
        #     print('no composite')
        # if 85 not in np.unique(y):
        #     print(target_ID)
        #     print('no enamel')

        y = self.separate_masks(y)
        # for i in y:
        #     print(np.unique(i, return_counts=True))
        # if x is single channel, converts it to 3 channels
        if len(x.shape) == 2:
            x = np.stack((x,)*3, axis=-1)
        x = self.transform_input(x)
        # transform all targets
        for i in range(len(y)):
            y[i] = self.transform_target(y[i])

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                # hflip all targets
                for i in range(len(y)):
                    y[i] = TF.hflip(y[i])

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                # vflip all targets
                for i in range(len(y)):
                    y[i] = TF.vflip(y[i])

        if self.affine:
            angle = random.uniform(-50.0, 50.0) # (-180.0, 180.0)
            h_trans = random.uniform(-20, 20) # (-352 / 8, 352 / 8)
            v_trans = random.uniform(-20, 20) # (-352 / 8, 352 / 8)
            scale = random.uniform(0.85, 1.15) # (0.5, 1.5)
            shear = random.uniform(-5, 5) # (-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            # affine all targets
            for i in range(len(y)):
                if i == 0:
                    # changes fill type to the largest value in the mask
                    fill_type = float(torch.max(y[i]).item())
                else:
                    fill_type = -1.0
                y[i] = TF.affine(y[i], angle, (h_trans, v_trans), scale, shear, fill=fill_type)

        # changes pixels to 0.0 if below 125.0 and 255.0 if above 125.0
        for i in range(len(y)):
            y[i] = torch.where(y[i] < 0.5, 0, 255)
            # y[i] = torch.where(y[i] < 125, 0, 255)
        
        # y = self.combine_masks(y, self.n_classes)
        # Convert y (list of tensors) to a single tensor of shape (batch, class, width, height)
        y = (torch.stack(y, dim=0)/ 255.0)  # Shape: (class, width, height)# Permute dimensions to Shape: (class, 1, width, height)
        if self.model_type == 1:
            # iterates over self.class_map items and sorts 'class_name' into list of strings in order or rows
            name_to_index = {row['class_name']: idx for idx, row in self.class_map.iterrows()}
            y= self.process_ignore_values(self.class_tree, y, name_to_index)
        y = y.permute(1,0,2,3)
        # traverses y[0] and shows images
        # for i in range(y.shape[1]):
        #     # changes 1 to 255 and -1 to 125
        #     y[0][i] = torch.where(y[0][i] == 1, 255, y[0][i])
        #     y[0][i] = torch.where(y[0][i] == -1, 125, y[0][i])
        #     plt.imshow(y[0][i])
        #     plt.show()
            
        # removes dim 0
        y = torch.squeeze(y, dim=0)
        return x.float(), y


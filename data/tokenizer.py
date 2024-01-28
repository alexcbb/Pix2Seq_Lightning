import numpy as np
import torch

class Tokenizer:
    def __init__(
            self, 
            num_classes: int, 
            num_bins: int, 
            width: int, 
            height: int, 
            tx: int = 0,
            ty: int = 0,
            tz: int = 0,
            rx :int = 0,
            ry: int = 0,
            rz: int = 0,
            max_len=500
        ):
        """
        Tokenizer that transforms objects classes and coordinates into tokens
        """
        # TODO : implement tokenization for 6D pose
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len

        self.BOS_code = num_classes + num_bins
        self.EOS_code = self.BOS_code + 1
        self.PAD_code = self.EOS_code + 1

        self.vocab_size = num_classes + num_bins + 3

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')
    
    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1)

    def __call__(
            self, 
            obj_class: list, 
            bboxes: list, 
            pose: list = None,
            shuffle=True
        ):
        assert len(obj_class) == len(bboxes), "labels and bboxes must have the same length"
        bboxes = np.array(bboxes)
        obj_class = np.array(obj_class)
        obj_class += self.num_bins
        obj_class = obj_class.astype('int')[:self.max_len]

        bboxes[:, 0] = bboxes[:, 0] / self.width
        bboxes[:, 2] = bboxes[:, 2] / self.width
        bboxes[:, 1] = bboxes[:, 1] / self.height
        bboxes[:, 3] = bboxes[:, 3] / self.height

        bboxes = self.quantize(bboxes)[:self.max_len]

        # TODO : quantize the 6D pose
        if pose is not None:
            pass

        if shuffle:
            rand_idxs = np.arange(0, len(bboxes))
            np.random.shuffle(rand_idxs)
            obj_class = obj_class[rand_idxs]
            bboxes = bboxes[rand_idxs]

        tokenized = [self.BOS_code]
        for obj, bbox in zip(obj_class, bboxes):
            tokens = list(bbox)
            tokens.append(obj)

            tokenized.extend(list(map(int, tokens)))
        tokenized.append(self.EOS_code)

        return tokenized    
    
    def decode(
            self, 
            tokens: torch.tensor
        ):
        """
        toekns: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        assert len(tokens) % 5 == 0, "invalid tokens"

        obj_class = []
        bboxes = []
        pose = []
        for i in range(4, len(tokens)+1, 5):
            obj = tokens[i]
            bbox = tokens[i-4: i]
            obj_class.append(int(obj))
            bboxes.append([int(item) for item in bbox])
        obj_class = np.array(obj_class) - self.num_bins

        # TODO : dequantize the 6D pose
        if len(pose) > 0:
            pass

        bboxes = np.array(bboxes)
        bboxes = self.dequantize(bboxes)
        
        bboxes[:, 0] = bboxes[:, 0] * self.width
        bboxes[:, 2] = bboxes[:, 2] * self.width
        bboxes[:, 1] = bboxes[:, 1] * self.height
        bboxes[:, 3] = bboxes[:, 3] * self.height
        
        return obj_class, bboxes
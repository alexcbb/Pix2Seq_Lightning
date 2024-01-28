class CFG:
    ycb_path = './dataset/ycbv/'
    ycb_train = ycb_path + 'train_real/'
    ycb_test = ycb_path + 'test/'
    
    max_len = 300
    img_size = 384
    num_bins = img_size
    
    batch_size = 512
    epochs = 30
    precision = 16
    gpus = 4
    nodes = 1
    strategy = "ddp"
    
    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    
    lr = 1e-4
    weight_decay = 1e-4
    
    num_classes = 21
    num_workers = 8
    
    generation_steps = 101
    run_eval = False
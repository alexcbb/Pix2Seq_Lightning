class CFG:
    ycb_path = './dataset/ycbv/'
    ycb_train = ycb_path + 'train_real/'
    ycb_test = ycb_path + 'test/'
    
    max_len = 300
    img_size = 384
    num_bins = img_size
    
    batch_size = 512
    epochs = 30
    precision = 32
    gpus = 4
    nodes = 1
    strategy = "ddp"
    
    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    
    lr = 1e-4
    weight_decay = 1e-4
    
    num_classes = 21
    num_workers = 0
    
    generation_steps = 101
    run_eval = False
    id2cls = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick"
    ]

    
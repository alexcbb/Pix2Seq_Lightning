class CFG:
    ycb_path = './dataset/ycbv/'
    ycb_train = ycb_path + 'train_real/'
    ycb_test = ycb_path + 'test/'
    
    max_len = 300
    img_size = 384
    num_bins = img_size
    
    batch_size = 256
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
    num_workers = 0
    
    generation_steps = 101
    run_eval = False
    id2cls = {
        "0": "002_master_chef_can",
        "1": "003_cracker_box",
        "2": "004_sugar_box",
        "3": "005_tomato_soup_can",
        "4": "006_mustard_bottle",
        "5": "007_tuna_fish_can",
        "6": "008_pudding_box",
        "7": "009_gelatin_box",
        "8": "010_potted_meat_can",
        "9": "011_banana",
        "10": "019_pitcher_base",
        "11": "021_bleach_cleanser",
        "12": "024_bowl",
        "13": "025_mug",
        "14": "035_power_drill",
        "15": "036_wood_block",
        "16": "037_scissors",
        "17": "040_large_marker",
        "18": "051_large_clamp",
        "19": "052_extra_large_clamp",
        "20": "061_foam_brick"
    }

    
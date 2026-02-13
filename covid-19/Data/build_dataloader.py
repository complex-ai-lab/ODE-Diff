import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    test_dataset = instantiate_from_config(config['dataloader']['test_dataset'])

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset
    }

    return dataload_info

def test_build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['sample_dataset']['params']['output_dir'] = args.save_dir
    sample_dataset = instantiate_from_config(config['dataloader']['sample_dataset'])

    dataloader = torch.utils.data.DataLoader(sample_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': sample_dataset
    }

    return dataload_info

def gt_build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['sample_dataset']['params']['output_dir'] = args.save_dir
    gt_dataset = instantiate_from_config(config['dataloader']['gt_dataset'])

    dataloader = torch.utils.data.DataLoader(gt_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': gt_dataset
    }

    return dataload_info

def expf_build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['sample_dataset']['params']['output_dir'] = args.save_dir
    expf_dataset = instantiate_from_config(config['dataloader']['expf_dataset'])

    dataloader = torch.utils.data.DataLoader(expf_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': expf_dataset
    }

    return dataload_info

def expc_build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['sample_dataset']['params']['output_dir'] = args.save_dir
    expc_dataset = instantiate_from_config(config['dataloader']['expc_dataset'])

    dataloader = torch.utils.data.DataLoader(expc_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': expc_dataset
    }

    return dataload_info

def build_dataloader_weight(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['weight_dataset']['params']['output_dir'] = args.save_dir
    weight_dataset = instantiate_from_config(config['dataloader']['weight_dataset'])

    dataloader = torch.utils.data.DataLoader(weight_dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': weight_dataset
    }

    return dataload_info



if __name__ == '__main__':
    pass


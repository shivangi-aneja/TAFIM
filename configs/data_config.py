from .transforms_config import EncodeTransforms, FaceSwapTransforms
from .paths_config import dataset_paths

DATASETS = {

    'ffhq_encode': {
        'transforms': EncodeTransforms,
        'train_source_root': dataset_paths['self_recon_train'],
        'train_target_root': dataset_paths['self_recon_train'],
        'val_source_root': dataset_paths['self_recon_val'],
        'val_target_root': dataset_paths['self_recon_val'],
        'test_source_root': dataset_paths['self_recon_test'],
        'test_target_root': dataset_paths['self_recon_test'],
    },

    'ffhq_stylemix': {
        'transforms': EncodeTransforms,
        'test_source_root': dataset_paths['style_mix_src'],
        'test_target_root': dataset_paths['style_mix_tgt'],
    },

    'ffhq_fs': {
        'transforms': FaceSwapTransforms,
        'train_source_root': dataset_paths['fs_train_src'],
        'train_target_root': dataset_paths['fs_train_tgt'],
        'val_source_root': dataset_paths['fs_val_src'],
        'val_target_root': dataset_paths['fs_val_tgt'],
        'test_source_root': dataset_paths['fs_test_src'],
        'test_target_root': dataset_paths['fs_test_tgt'],
    },


    'ffhq_all': {
        'transforms_pSp': EncodeTransforms,
        'transforms_fs': FaceSwapTransforms,

        'train_source_root': dataset_paths['self_recon_train'],
        'train_target_root': dataset_paths['self_recon_train'],
        'val_source_root': dataset_paths['self_recon_val'],
        'val_target_root': dataset_paths['self_recon_val'],
        'test_source_root': dataset_paths['self_recon_test'],
        'test_target_root': dataset_paths['self_recon_test'],
    },

}

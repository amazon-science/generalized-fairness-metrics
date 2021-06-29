
local DROPOUT = 0.1;
local EPOCHS = 10;
local BSIZE = 16;
local TRANSFORMER = "google/electra-base-discriminator";

{
    "dataset_reader": {
        "type": "flexible_reader",
        "reader": "", // to override
        "token_indexers": {
            "tokens": {
                 // wordpieces are combined via averaging before being used (as word embs) in the parser
                "type": "pretrained_transformer", 
                "model_name": TRANSFORMER
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": TRANSFORMER
        },
        "use_subtrees": true,
        "granularity": "", // to override
    },
    "validation_dataset_reader": self.dataset_reader + {
        "use_subtrees": false
    },
    "train_data_path": "", // to override
    "validation_data_path": "", // to override
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens":{
                    "type": "pretrained_transformer",
                    "model_name": TRANSFORMER,
                    "train_parameters": true
                }
            }
        },
        "seq2vec_encoder": {
            "type": "cls_pooler",
            "embedding_dim": 768,
            "cls_is_last_token": true
        },
        "feedforward": null, 
        "dropout": DROPOUT,
        "namespace": "tags", # this is important, otherwise at prediction the models get KeyError
        "num_labels": "", // to override
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "batch_size" : BSIZE
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "weight_decay": 0.1
        },
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": EPOCHS,
         //   "num_steps_per_epoch": 3088,
            "cut_frac": 0.06
        },
        "num_epochs": EPOCHS,
        "cuda_device": 0,
        "patience": 5
    }
}
// based on https://github.com/allenai/allennlp-models/blob/master/training_config/tagging/fgner_transformer.jsonnet
local EPOCHS = 10;
local BSIZE = 32;
local TRANSFORMER = "roberta-base";

{
    "dataset_reader": {
        "type": "conll2003",
        "tag_label": "ner",
        "coding_scheme": "BIOUL",
        "token_indexers": {
            "tokens": {
                 // wordpieces are combined via averaging before being used (as word embs) in the model
                "type": "pretrained_transformer_mismatched", 
                "model_name": TRANSFORMER,
                "max_length": 512
            }
        },
    },
    "train_data_path": "",
    "validation_data_path": "",
    "model": {
        "type": "crf_tagger",
        "label_encoding": "BIOUL",
        "dropout": 0.5,
        "include_start_end_transitions": false,
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "text_field_embedder": {
            "token_embedders": {
                "tokens":{
                    "type": "pretrained_transformer_mismatched",
                    "model_name": TRANSFORMER,
                    "train_parameters": true,
                    "max_length": 512
                }
            }
        },
        // "encoder": {
        //     "type": "lstm",
        //     "input_size": 768,
        //     "hidden_size": 200,
        //     "num_layers": 2,
        //     "dropout": 0.5,
        //     "bidirectional": true
        // },
        "encoder": {
            "type": "pass_through",
            "input_dim": 768,
        },

        "regularizer": {
            "regexes": [
                [
                    "scalar_parameters",
                    {
                        "type": "l2",
                        "alpha": 0.1
                    }
                ]
            ]
        }
    },
    // "data_loader": {
    //     "batch_size": BSIZE
    // },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": BSIZE
        }
    },

    "trainer": {
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0.01,
          "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
          "lr": 1e-5,
          "eps": 1e-8,
          "correct_bias": true,
        },
        "learning_rate_scheduler": {
          "type": "linear_with_warmup",
          "warmup_steps": 100,
        },
        // "grad_norm": 1.0,
        "num_epochs": EPOCHS,
        "validation_metric": "+f1-measure-overall",
        "patience": 3,
        "checkpointer": {
          "num_serialized_models_to_keep": 3,    
         },
    }

//     "trainer": {
//         "optimizer": {
//             "type": "huggingface_adamw",
//             "lr": 1e-5,
//             //  "weight_decay": 0.0,
//             "parameter_groups": [[
//                 ["text_field_embedder", "encoder", "tag_projection_layer", "crf"],
//                 {"weight_decay": 0}
//             ]],
//         },

//         "checkpointer": {
//             "num_serialized_models_to_keep": 3,
//         },
//         "validation_metric": "+f1-measure-overall",
//         "learning_rate_scheduler": {
//             "type": "slanted_triangular",
//             "num_epochs": EPOCHS,
//          //   "num_steps_per_epoch": 3088,
//             "cut_frac": 0.06
//         },
//         "num_epochs": EPOCHS,
//         "grad_norm": 5.0,
//         "patience": 5,
//   }
}
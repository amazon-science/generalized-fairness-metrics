
local EPOCHS = 40;
local BSIZE = 32;

{
  "dataset_reader": {
    "type": "flexible_reader",
    "reader": "", // to override
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
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
          "trainable": true
        }
      }
    },
    "seq2vec_encoder": {
       "type": "lstm",
       "input_size": 300,
       "hidden_size": 512,
       "num_layers": 2
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": EPOCHS,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
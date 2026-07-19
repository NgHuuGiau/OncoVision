import os, sys, json, time
os.environ["ONCOVISION_ALLOW_WEIGHT_DOWNLOAD"] = "1"
sys.path.insert(0, r"D:\OncoVision")
from medical.training import medical_training_paths, train_cnn_medical_model

t0 = time.time()
p = medical_training_paths()
print("START train", flush=True)
model_path = train_cnn_medical_model(
    p, prepare_dataset=False, verbose=False,
    settings_override={
        "cnn_backbone": "convnext_tiny",
        "cnn_image_size": 160,
        "cnn_batch_size": 32,
        "cnn_num_epochs": 6,
        "cnn_pretrained": True,
        "cnn_learning_rate": 0.0003,
        "cnn_mixed_precision": True,
        "cnn_gradient_accumulation_steps": 1,
        "cnn_early_stopping_patience": 3,
        "cnn_checkpoint_averaging": True,
        "cnn_tta": True,
    },
)
print("DONE train ->", model_path, "in", round(time.time() - t0, 1), "s", flush=True)

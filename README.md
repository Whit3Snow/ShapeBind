# ShapeBind

## Description

  * ## CS479 project 

    


## Environment
```
conda create --name shapebind python=3.8.18 -y
conda activate shapebind
pip install 
```

## Files 
```
models/


deepspeed_train_3d_text.py
deepspeed_train_3d_decoder.py
mgpu_train_3d_text.py
zeroshot_classification.py
zeroshot_classification_shapetalk.py
```

## Usage
* training 3d-text binding model
```
cd Imagebind_LoRA

python -u deepspeed_train_3d_text.py --batch_size 12 --max_epochs 64 --num_workers 4 --device cuda:0:1 --full_model_checkpoint_dir {directory} --full_model_checkpointing --datasets_dir {directory} --loggers tensorboard --loggers_dir {directory} --datasets shapetalk --temperature 0.07
```

* training decoder model
```
cd Imagebind_LoRA

python -u deepspeed_train_3d_decoder.py --batch_size 192 --max_epochs 350 --num_workers 1 --device cuda:0:1 --encoder_latent_dim 1024 --full_model_checkpoint_dir {directory} --full_model_checkpointing --datasets_dir {directory} --loggers tensorboard --loggers_dir {directory} --datasets embedding
```

* testing zeroshot classifier
```
cd Imagebind_LoRA
python zeroshot_classification_shapetalk.py
```

* Editing using Slerp
```
Imagebind_LoRA/interpolation.ipynb
```


## Citing Shapebind
code based on Imagebind_LoRA [Github](https://github.com/fabawi/ImageBind-LoRA)

models/utils.py based on [Github](https://github.com/openai/point-e/blob/main/point_e/models/util.py)

utils/ based on Imagebind code




## References

[1] Girdhar, Rohit, et al. "Imagebind: One embedding space to bind them all." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Achlioptas, Panos, et al. "ShapeTalk: A Language Dataset and Framework for 3D Shape Edits and Deformations." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[3] Yu, Xumin, et al. "Point-bert: Pre-training 3d point cloud transformers with masked point modeling." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

Acknowledgements
All three of us distributed the work evenly.


Traceback (most recent call last):
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 792, in <module>
    main(args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 747, in main
    trainVitExtractor(args.save_location, vit_model, train_data, val_data, args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 374, in trainVitExtractor
    transforms = model.transforms
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1688, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'ViT_FeatureExtractor' object has no attribute 'transforms'


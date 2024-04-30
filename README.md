Traceback (most recent call last):
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 792, in <module>
    main(args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 736, in main
    network = ViT_LSTM(512, num_classes, num_input_features, args.lstm_sequence_length, args.device)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/ViT_LSTM.py", line 44, in __init__
    self.vit_model = ViT_FeatureExtractor(num_output_features, num_classes)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/ViT_LSTM.py", line 25, in __init__
    self.linear1 = nn.Linear(self.vit.head.in_features, num_output_features)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1688, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'Identity' object has no attribute 'in_features'


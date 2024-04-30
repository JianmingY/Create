Traceback (most recent call last):
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 792, in <module>
    main(args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 747, in main
    trainVitExtractor(args.save_location, vit_model, train_data, val_data, args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 402, in trainVitExtractor
    outputs = model(inputs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/ViT_LSTM.py", line 53, in forward
    x = self.linear1(x)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x768 and 512x128)

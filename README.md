 File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 696, in <module>
    main(args)
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 654, in main
    trainResnet(foldDir, resnetModel,train_data,val_data,args,labelName=labelName)
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 214, in trainResnet
    train_dataset = CNNDataset(training_data, labelName, transforms,balance=args.balance_cnn,augmentations = args.augment_cnn)
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 24, in __init__
    best_frames_indices = self.selectBestFrames()
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 48, in selectBestFrames
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(frames)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/sklearn/base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py", line 1481, in fit
    X = self._validate_data(
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/sklearn/base.py", line 633, in _validate_data
    out = check_array(X, input_name="X", **check_params)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/sklearn/utils/validation.py", line 997, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/sklearn/utils/_array_api.py", line 521, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)

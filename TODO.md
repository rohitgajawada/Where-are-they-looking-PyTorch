1) Data Augmentation
2) Fix the metrics, its hard codes for 13 and I dont understand what we did exactly
2) Make sure all the scales for head bbox, eye pred, labels, etc are at right scale for data augmentation and euclid metric calculation
3) Shifted grids and average grid loss (grid offset, etc) (partially done)
4) Resize the test output with bicubic and return prediction coordinates
5) Check biases in network
6) AUC Score


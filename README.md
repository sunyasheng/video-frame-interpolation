# video-frame-interpolation
 In this project, we utilize a simplified tensorflow version of
 DAIN[1] with concise implementation as is shown below. 
 The optical flow is based on the pretrained pwc net[2].
 ![image](./assets/workflow.png =200x300)
 
 And the results demonstrate that this strategy is comparable to 
 state-of-art methods in terms of the computation efficiency and image quality.
 ![image](./assets/metrics_table.png =200x300)
 
 # Reference
 [1] https://github.com/baowenbo/DAIN
 
 [2] https://github.com/philferriere/tfoptflow

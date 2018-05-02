Features:
  1. Be able to have more than one label
  2. Add eta times to the features
Some bootstrap work:
  1. 
Validations:
  1. Check to make sure the directories that are specified are correct
  2. Single directory for all the data collected by the model
Front end work:
  1. Show improvements in performance given more labels
  2. Load next batch before current batch is complete
The evaluation is not correct, we need to simulate the max entropy sampling environment
Classify images according to evaluator
  1. Add extra tag to images labelled by model
    - We probably shouldn't train based on these labels?
  2. Do we need a minimum number of evaluation samples to be able to do this confidently?
  3. Go through the dataset and start classifying based on precision achieved

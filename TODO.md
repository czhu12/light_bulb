Add Skip
Stop Training When Training outpaces test

Features:
  2. Add eta times to the features
  3. Classify sequences
  4. Be able to upload a pretrained model. This one kind of works
Validations:
  1. Check to make sure the directories that are specified are correct
  2. Single directory for all the data collected by the model
  3. Make sure the output directory isn't already being used
  4. Make directories relative so that they don't change when computers change
  5. Theres a bug where we might label the same thing twice: since we sample next batch before current batch is labelled, we may end up sampling same items in next batch
  6. The evaluation is not correct, we need to simulate the max entropy sampling environment
Front end work:
  1. Show improvements in performance given more labels

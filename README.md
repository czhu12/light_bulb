Labelling tool.

Here is a basic configuration for how to set up a labelling tool.
```
task:
  title: Is this a cat?
  description: |
    <table style="width:50%;margin: 0 auto;text-align:center;">
      <tr>
        <th>YES</th>
        <th>NO</th>
      </tr>
      <tr>
        <td>Yes its a cat.</td>
        <td>No its not a cat.</td>
      </tr>
    </table>
dataset:
  directory: dataset/cat_or_dog
  data_type: images
  judgements_file: outputs/cat_or_dog/labels.csv
label:
  type: binary
model_directory: outputs/cat_or_dog/models/
user: chris
```

# Context_Prediction_NLP

In order to get the prediction, you first run process_data.py, which creates the pickled file needed for the prediction (otherwise there's a big loss of computing time for each prediction). You can run by writing the command:
````
python process_data.py
`````

As there's a lot of data (which can take forever to run), you can choose the number of contexts to consider with [--nb](by default it takes the whole dataset). You can also choose to run it on the train set or the dev set by choosing the right path with [--path] (by default it takes the training set). PS: if you choose the training path, and run the processing on it, but then want a prediction from the dev questions set, you need to run the processing again with the right path.
<br>
Then for the prediction, run:
````
python predict.py
`````

This command can also take 2 arguments, but you need to give it only one, otherwise it raises an exception: either you give it an index with [--index], which then predicts the context for the corresponding question from the dataset, or you give it a hand-made question (quote it) using [--question]. If you give it neither, it also raises an exception.

<br>

Finally, you can computer the accuracy over an amount of question using:
````
python accuracy.py
`````

This command can take one argument [--n], which is the number of questions taken into account. By default it takes all the dataset questions.
<br>

Additionally, you can find some of things I've tried to get a better prediction on draft_code.ipynb

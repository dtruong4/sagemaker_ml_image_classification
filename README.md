# Image Classification using AWS SageMaker

In this project, we will use AWS Sagemaker to train a pretrained model that can perform image classification by using various practices and tools in Sagemaker and PyTorch:
* Sagemaker profiling
* Sagemaker debugger
* Hyperparameter tuning/optimization (HPO)
* Integration with S3

For this project, we will be using this [dataset of dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

## Project Specifications
The notebook and endpoint were run in an AWS `ml.m5.large` instance, with the notebook additionally using `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`, while the HPO and Classification training jobs were run with an AWS `ml.g4dn.xlarge` and`ml.g4dn.2xlarge` instance.

## Key Files
* `train_and_deploy.ipynb` - The main notebook used to run commands and interact with SageMaker.
* `README.md` - Contains key information about set up and post-project analysis.
* `hpo.py` - The entry point for hyperparameter tuning job.
* `train_model.py` - The entry point for our training job to then run on our test data, includes debugging and profiling.
* `train_model_deploy.py` - The entry point of the model that will be deployed to a SageMaker inference endpoint, contains additional function to load the model and process image data.
* `profile-report.html`/`.pdf` - The HTML/PDF file discussing triggered rules, as well as suggestions to address them.

## Project Set Up and Installation
1. Enter AWS through the gateway in the course and open SageMaker Studio.
2. Download the starter notebook file.
3. Proceed by running the cells once in consecutive order.
4. The HPO process takes roughly 1 to 2 hours to run. If you prefer, you can just pass in the tuned hyperparameters provided in the `README.md` or `Python notebook` to test the model.
5. If the images are not rendering, or you need a closer look, you can check out all of them in the `./img` folder.

## Dataset
The provided dataset is the Udacity dog breed classification dataset, with 133 different breeds of dogs to classify. While this dataset has been provided, the project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete your own version of the project.

On a surface level, this dataset is:
* Labeled and divided into 133 breeds of dogs
* Already split into three `train`, `valid` (validation), and `test` folders
* Comprised of over 6500 images for training and over 800 images for validation and testing each.

More analysis is discussed in `train_and_deploy.ipynb`.

### Access
In order for SageMaker to have access to the data, we will need to upload the data to an S3 bucket through the AWS Gateway. This is handled in the `train_and_deploy.ipynb` notebook.

## Hyperparameter Tuning
For this project, I chose to tune the following hyperparameters:
* `batch-size`, the number of images that are processed before updating the model in training
* `lr`, the learning rate of the model's optimizer
* `epochs`, the number of times we will pass the training dataset through our model to update it.

I considered `batch-size` because this influenced how many times the model would be updated while training. It's important to view the updating process of the model as a method of finding an ideal "sweet spot" value for each of our model's parameters/weights. Usually, this "sweet spot" ends up being a maximum or minimum of some loss/gain functions. Each update is an attempt to shift the weight values up or down in the direction of the minimum/maximum (also known as gradient descent). This image below (from Adrian Rosebrock) is a good visual illustrating the goal we have in gradient descent. (Image: `train_val_loss_landscape.png`)

<img src="./img/train_val_loss_landscape.png">

Since one would only update a model after processing a batch of image, the smaller the batch size, the more times we would be able to update the classifier before going through all images. Processing less images in a batch can be less memory intensive, but it can also yield more volatile swings in the model's parameters (due to many more updates that overshoot past the maxima/minima). Typical batch sizes are powers of 2, usually starting with 32 or 64. I opted to keep the list of batch-size between the values of 32, 64, 128, and 256.

On the other hand, it is worth noting that we sometimes do _not_ want to set our weights to exactly minimum/maximum values of functions. This values might be optimal for features in our training photos, but don't want our model to just have weights in the optimal ranges of _just_ training images. This is exactly what overfitting is, and that is where the learning rate (`lr`) can help. In back-propagation, the gradients for each parameter in our model essentially are applied and shift the parameter values. The learning rate serves as a multiplier to dictate how much to "shift" by in the direction of the minima/maxima. A small learning rate can be useful when you want to curb the process of approaching these extrema, whereas a higher learning rate can get you there faster (at the risk of overshooting). In other words, while the `batch-size` controls how often the weights "jump", the `lr` controls how far to "jump". For my learning rate, I chose to tune the HP on a continuous range of values between 0.0001 and 0.1 (with common default learning rates being 0.1 or 0.01). This spans across 4 orders of magnitude when looking at the learning rate logarithmically.

Making sure that I tune the `epochs` hyperparameter was especially important to me, as it would dictate how many times the model would be trained with the same datasets. Pass the data through too few times, and the model won't be updated enough to properly weigh which features are important in classifying dog breeds, and which features are not. On the other hand, if the data is passed through the model too _many_ times, then we will have overfitted the data on the model and the model's trained weights will not be able to accurately reflect the features of unfamiliar images. I decided to examine a range of integer both 2 epochs less and more than 8 (a range of 6 - 10) in my tuning process.

In an ideal world, I would have the ability to performing HPO at least 160 times, a multiplicative product of:
* 4 different order of magnitude for the `lr`
* 4 different `batch-sizes`
* 10 different `epochs` lengths

However, we live in a world where time and money are real constraints. With that, I opted to perform 10 different jobs--with training jobs on `ml.g4dn.2xlarge` instances costing \\$0.94 an hour, the total cost of HPO tuning should be around \\$3.

In terms of selecting an objective metric, I opted to maximize the objective average validation accuracy, by determining how many validation images (images that do not account for adjusting the model weights) were correctly labeled.

After running 6 different training jobs for HPO, we can see that all of our training jobs completed successfully! (Image: `hpo_training_jobs_success.png`)

<img src="./img/hpo_training_jobs_success.png">

Additionally, we can see the model with the greatest measured objective measure (validation accuracy out of 835) of 675. (Image: `objective_metric.png`):

<img src="./img/objective_metric.png">

As an additional sidenote, I had actually tried to minimize the average validation loss when tuning my hyperparameters, but I noticed that the models with the smallest average validation loss also had the lowest accuracy, sometimes less than 10 correct out of 835 samples! Below is a screenshot of some log showing us a very inaccurate trained model from our HPO training job, even though the average validation loss is small. (Image: `hpo_logs_low_accuracy.png`)

<img src="./img/hpo_logs_low_accuracy.png">


Therefore, I decided to switch over to the using accuracy as my objective metric.

After getting a best model, as well as some additional modifications from debugging/profile analysis (such as reducing the number of `epochs`), I obtained the following hyperparameters:

```
{
   ...
   'batch-size': '32',
   'epochs': '7',
   'lr': '0.00037518045557353297',
   'num-classes': '133',
   ...
}
```

We can see these hyperparamters being passed into the tuning job via the `SM-HP-###` environment variables. We can also see being the resulting metrics being logged into CloudWatch.

<img src="./img/hpo_logs.png">

As we consider those model's hyperparameter to yield the best results, we'll pass them into our new `train_model.py` script, alongside some debugger and profile hooks.

## Debugging and Profiling
As mentioned in `train_and_deploy.ipynb`, the debugger helped in several ways. First, we were able to pass in some prebuilt `smdebug` rules in a config file that allowed for hooks to go in an observed the code being run and the variables they were passing. That allowed us to be able to be notified on any issues involving some key variables, such as gradients of running losses. Second, the debugger also allow statements to be printed directly into our notebook. This is a huge benefit compared to having to check out CloudWatch and query for logs related to our training job(s).

I included several rules that remained unviolated. For the issues that did come up, I used some suggestions from `profiler-report.html` (also accessible via `profiler-report.pdf`) to tweak my hyperparameters. Using the debugger and profiler allowed me to make these changes without blindly using more HPO trials (this can be very costly). More is explained in the following section and the `train_and_deploy.ipynb` notebook.

### Results
My model was overall successful, with many of the unviolated rules:

* VanishingGradient
* Overfit
* LossNotDecreasing
* ProfilerReport


Several debugging/profiling issues had come up while training my model:
* Overtraining
* LowGPUUtilization

To address the `Overtraining` rule, I looked at the Cross Entropy Loss curves to determine whether I could stop training at any earlier point. (Image: `loss_curves.png`)

<img src="./img/loss_curves.png">

We do see that both curves are not steeply decreasing, nor are they separate by the end of the training processing, so we can ensure there is no underfitting. The curves are also not diverging as the steps go on, so we have not reached the point of overfitting just yet. However, as the loss curve has mostly flattened out by the halfway point of training, it might explain why the `Overtraining` rule was triggered. To combat this, I just cut down the number of epochs proportionally to the number of steps where both the training and validation loss curves are stabilizing at a minimum.

Meanwhile, for the `LowGPUUtilization`, there were some efforts to address this, but they were mostly impractical or ineffective.

The profile report (`profiler-report.html`/`.pdf`) had suggested to _"check if there are bottlenecks, minimize blocking calls, change distributed training strategy, or increase the batch size."_

Looking at my code, there didn't seem to be any bottlenecks ot blocking calls. While I could have looked into distributed training strategy, this was outside the scope of this project.

I did try increasing the batch size to `64` and `128`, and I was able to get higher GPU utilization rates. Particularly, with a size of `128`, I was able to get rates that fluctated between 35-50% utilization with occasional spikes to 90% utilization. However, this still triggered the `LowGPUUtilization` profiler rule. On top of that, the accuracy of these model with _only_ an increased `batch_size` were lower as well, both of which were under 72% accuracy. This indicates that to increase the batch size, we will likely need to make some adjustments to the other hyperparameters as well, which is hard to accomplish without more HPO trials.

Admittedly, the most difficult part of this project was acknowledging the limitations we have on HPO. If we had more time, we would have been able to perform more runs with different hyperparameter combinations in our search space. We could have tried fixing the `batch_size` at higher values with different `learning_rate`'s, for instance.

Nonetheless, the finalized model that we achieved (`train_nodel_deploy.py`) was still very successful at inferring which breed out of 133 different classes an image belonged to. With a test accuracy of `77.272%`, we were able to send over several new images to our deployed model endpoint and yield satisfactory results. Even the incorrect inferences (such as the model misidentifying the right class of _retrievers_) were within reason.

## Model Deployment
The deployed model uses the `train_model_deploy.py` file as the entry point. This file is identical to `train_model.py` except for the following:
* Contains the modified hyperparameters (`epochs` from `10` to `7`) to address the issues noted by the
* The `smdebug` package was left out due to it being neither recognized nor installable in the saved model. A similar issue with more context can be found [here](https://stackoverflow.com/questions/60122070/aws-sagemaker-pytorch-no-module-named-sagemaker).
* It contains the `model_fn` and `predict_fn` functions to access the saved model and predictions
* It contains the functions `input_fn` and `output_fn` to pre/post-process the image and inference data.

Here is a screenshot of the deployed active endpoint in Sagemaker. (Image: `model_endpoint.png`)

<img src="./img/model_endpoint.png">

To query the endpoint with a sample input, go to the **Model Deploying** section in `train_and_deploy.ipynb` and perform the following steps:

1. Find an image of your choice. Make sure that it is of the JPEG format.
2. Save the image with a path of your choice. Make sure that you can access it via Sagemaker.
3. In the `image = ...` cell, change the string to your path. As an example, I set the path to `'./img/collie.jpg'`.
4. Run all the following cells up to where you check the dog breed with `df_breed_labels.loc[[predicted_label]]`

Example Code:
```
# Set the image path
image_path = './img/collie.jpg'

# Make any adjustments to the image, such as cropping
image = Image.open(image_path)
w, h = image.size
image = image.crop((0, 0.1 * h, 0.7 * w, h))

# Save image to new path
new_image_path_array = image_path.split('/')
new_image_path_array[-1] = f'cropped_{new_image_path_array[-1]}'
new_image_path = os.path.join(*new_image_path_array)

# Display cropped image if needed
image.save(new_image_path, format = 'JPEG')
display(image)

# Read the image file
with open(new_image_path, 'rb') as f:
    img_payload = f.read()

# Run an prediction on the endpoint
response = predictor.predict(img_payload)
response

# Determine and verify the predicted label
predicted_label = response[0].index(max(response[0])) + 1
df_breed_labels.loc[[predicted_label]]
```

## Additional Tasks
In this project, I provided a thorough pre-analysis of the data, allowing me to efficiently make decisions on what types of preprocessing/transformations would be needed (i.e. resizing, extracting labels).

## Sources:
* Template code and dataset images from Udacity's AWS course
* German Shepherd Image: https://en.wikipedia.org/wiki/German_Shepherd#/media/File:German-shepherd-4040871920.jpg
* Labrador Image: https://vgl.ucdavis.edu/panel/labrador-retriever-health-panel-2
* Example Gradient Descent Visualization: https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/

# Gresco: Generalizability Robustness Extensibility Score
Measuring the Generalizability, Robustness, and Extensibility of Computer Vision Models.

## Motivation
In the status quo, computer vision models report accuracy on test sets that are sampled from the same dataset as the train sets. However, these models in deployment encounter an environment with a higher variety of objects and scenes than in the datasets. This work proposes a method of transforming test images to test for generalizability, robustness, and elasticity of computer vision models to better predict the performance of these models deployed in unseen environments. 

Consider asking a question about an object in a scene.
* Generalizability = If we replace the object with another object of the same category, and the model correctly predicts the same answer.
* Robustness = If we replace the scene, and the model correctly predicts the same answer.
* Elasticity: If we replace the object with an object of a different category, and the model correctly predicts the new value.  

## Method
We invent a novel pipeline for transforming objects and scenes in images:
1. Prepare scene: 
    1. Obtain a segmentation mask over object to remove.
    2. Remove object using EdgeConnect. 
2. Prepare object: 
    1. Obtain a segmentation mask over object of interest. Carve it out.
    2. Optionally, we apply transformations to this object, such as changing the color, size, or rotation.
3. Add object into scene:
    1. Paste the object into scene.
    2. Apply GP-GAN blending to make the image natural.

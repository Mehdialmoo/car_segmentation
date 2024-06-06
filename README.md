
# Automatic Car Detection and Segmentation using SAM and CNN

### Abstract

This project aimed to automatically detect and segment cars in images using the Segment Anything Model (SAM) and a convolutional neural network trained on the CIFAR-10 dataset to classify cropped segments. The proposed solution leverages the powerful SAM model to generate masks, selects the mask classified as a car by the CIFAR-10 model, and fills in missing pixels from the background to produce a clean segmentation of the detected car. With its essential real-world implications in domains such as self-driving cars and traffic analysis, this report demonstrates the effectiveness of combining these automatic car detection and segmentation techniques.

##	Methodology 

Our proposed method consists of three main stages: 
1) Automatic object segmentation using SAM (Kirillov et al. 2023) 
2) Car classification using a CNN 
3) Image inpainting to extract the detected car (Chen et al. 2021)

###
In the first stage, we use a SAM model pre-trained on the SA-1B dataset, which contains over 1 billion masks on 11 million licensed images. The SAM 'Automatic Mask Generator' is a powerful tool that proposes masks for objects, parts of objects, and background regions in an image. This innovative feature allows us to obtain segmentation masks for all prominent objects in the scene without manual annotation, enhancing the efficiency and accuracy of our methodology.
Next, we train a simple CNN based on the LeNet architecture to classify image patches as "automobile" or "others". We train this on a subset of the CIFAR-10 dataset, using the "automobile" and "truck" classes as positive examples and the remaining 8 classes as negative examples. The trained classifier is then run on the crops of the input image corresponding to each SAM-generated mask. This allows us to identify which segmented objects are likely to be cars.
Finally, we use a simple image inpainting technique to remove the background and extract only the detected car. For each pixel in the mask region, we search nearby areas of the image for similar colours and textures to fill in the background. This is repeated several iterations to blend the inpainted region and remove artefacts.

##	Experiments and Results

We evaluated our pipeline on test images containing cars in road scenes with complex backgrounds. SAM could generally successfully segment the car objects, although it sometimes segmented cars into multiple parts. Our CNN classifier could robustly identify car segments with approximately 94.13% accuracy.
The image inpainting step usually cleanly separates the car from the background, although some artefacts, such as complex textures and hard edges, remain visible. 
Some failure cases for our method include shadows, very small or partially occluded cars, shiny surfaces and reflections that distract SAM, and some CNN misclassifications. However, the pipeline performs well on the majority of tested images.
The report figures include some sample results. The output car images have clean, soft masks despite the complex backgrounds in the inputs. The segmentation and inpainting stages generally preserve the car's edges and details.

## Conclusion and Future Work

The study presents an effective automatic car detection and extraction method using SAM's object segmentation capabilities and CNNs' classification power. It can accurately localize cars and separate them from background scenes without manual annotation. Future improvements include a more complex CNN architecture, dedicated automobile datasets, sophisticated image inpainting techniques, fine-tuning SAM on automotive datasets, and incorporating object tracking for video data. The study aims to inspire further research for niche computer vision applications.

## References

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P. and Girshick, R., 2023. Segment Anything. arXiv (Cornell University).

Chen, S., Cao, E., Koul, A., Ganju, S., Praveen, S. and Kasam, Meher Anand, 2021. Reducing effects of swath gaps in unsupervised machine learning. Committee on Space Research Machine Learning for Space Sciences Workshop.



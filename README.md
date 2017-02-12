

## Vehicle Detection Project
```
Udacity Self-car driving Nanodegree
Kwanghyun JUNG
12 FEB 2017
ihangulo@gmail.com
```

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./img/data_load.jpg
[image2]: ./img/hog_feature.jpg
[image21]:./img/HLS_chaannel1.png
[image3]: ./img/fullscreen_hog.jpg
[image4]: ./img/sliding_window_all.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./img/image_pipeline.jpg
[image7]: ./img/video_pipeline1.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

* prepare data

I used small set data from udacity project 5. But it's too small number, so I added GTI data (http://www.gti.ssr.upm.es/data/Vehicle_database.html). But this is only 'car' images. So there is so many unbalance number of car and non-car images.
I added non-car images capture from project video and test images. You can see "cell 2" module, how can I capture images. The images have various sizes and include car images. So I delete non-car image by hand and by own eye :) So I've got 9996 car / 9220 non-car images. ( You can see this Cell 6)

And when read data, I restrict number of cars data same with non-car data. So it is almost balanced.


![car / not car][image1]

* Hog features (Cell 7)

** Test version **

When I testing my functions, I use ordinary method, from `skimage.feature` hog function.

`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

I used the cv2.cvtColor to change color image to GRAYSCALE, and get Hog features. Here is example output.

![HOG features][image2]


####2. Explain how you settled on your final choice of HOG parameters.

** Video(Image) Pipeline **

When I use the test version, it must pass the image every time, and get hog features. But if I use this function on video pipeline, then it has overhead.

Because more than hundreds of sliding window must be use this function, it is useless repeated arithmetic operation. So I get a full-screen Hog features and get subset of it on every sliding window.

But it has some restrictions.

(1)The sliding windows coordinates must be the mutiple number of ``pixels_per_cell=8`` This is very simple, when I return sliding window, I selected all sizes and the start coordinates.

(2)Sliding window's size is not only 64px X 64px. When use above test version, then it's no problem. Before call Hog function resize the cell image to 64x64px. But if use full-screen hog features, then there is big difference the size of hog features. (when 64px it's shape is [7,7,2,2,9]) So, I got new idea, if the size of sliding window is multiple of 64px, then we can change ``pixels_per_cell`` parameter, then get [7,7,2,2,9] HOG feature.

![Fullscreen HOG features ][image3]

| window size | pixels_per_cell | HOG shape          | Reduced            |
|-------------|-----------------|--------------------|--------------------|
| 64x64       | 8               | (89, 159, 2, 2, 9) | (49, 159, 2, 2, 9) |
| 128x128     | 16              | (44, 79, 2, 2, 9)  | (24, 79, 2, 2, 9)  |
| 256x256     | 32              | (21, 39, 2, 2, 9)  | (11, 39, 2, 2, 9)  |
| 320x320     | 40              | (17, 31, 2, 2, 9)  | (9, 31, 2, 2, 9)   |

But almost half of screen is sky, so I cut the grayscale image 320px from top. So HOG shape is more reduced.

After that, it's so simple get hog features from full-screen HOG features. Below code will be help to understand

``Cell 10 / search_windows()``
```

  hog_feature = hog_one[ (window[0][1]-320) //pix_per_cell :
          (window[1][1]-320) //pix_per_cell-1
           window[0][0]//pix_per_cell: window[1][0]//pix_per_cell-1 ]

        hog_features = hog_feature.ravel()

```
Then hog_features.shape (7,7,2,2,2,9) and finally use .ravel() then change to (1764,)

Every frame, I make 4 full-screen HOG features and using sliding window. Every epoch it change pixels_per_cell parameter described upper table. I didn't change other paramenters. `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried various combinations of color space and etc. ``Cell 9 : Testing color models`` shows my experiments.

![HLS Channel][image21]
HLS Channel 0, 1, 2 : HLS channel 1 is best selection

After many time trying, I select ``HLS color space``, and when get HOG features using channel 1, i.e L channel.
** Select train and test set

** Process
Please see ``Cell 10 : make model `` / ``extract_features`` function.

1) Random filp : for more efficent training, 5% of training image will be filpped. But this is only car image. Because lack of non-car images, so I already inlcude flipped images, i mentioned above.

2) get feature image : using ``cv2.cvtColor`` change color space RGB to HLS color space.

3) get spatial_feature : get binned color features (32, 32)
Because 32x32px is enough to recognize the picture. It's simple routine like Below. : bin_spatial()
```
 features = cv2.resize(img, size).ravel()
```
4) get color histogram : hist_bins = 32 (from HLS image)
5) get HOG feature
6) concatenate all features above
7) scaled to zero mean and unit variance before training the classifier.

I choose Linear SVM classifier because, this is detecting 'Is this image is car or not?'. So SVM is best choice to get good result. If this project was determine 'car or man or lamp or..' then I will choose decesion tree.

```
# Parameters
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 1 # L channel # Can be 0, 1, 2, or "ALL"
spatial_size = (32,32) #(16, 16) # Spatial binning dimensions
hist_bins = 32  #16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

```


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
Code is shown at ``Cell 10 : Sliding window``

I select sliding window size the multiple of 64. Because if I use full-screen hog, then the for simple calculation need. And It's overlap value is one of (1, 0.75, 0.5, 0.25). Because the overlapped coordinates must be the multiple of 8. So it can be divide by 8 when we get Hog subset. (see above code) No.2)

I choose 320, 256, 128, 64 size of window. and overlaps 0.5 or 0.75. When it need more detailed search, then 0.75, or ordinary search, then 0.5 is best.


| size      | xy overlaps |
|-----------|-------------|
| (320x320) | (0.5,0.5)   |
| (256x256) | (0.75,0.5)  |
| (128,128) | (0.75,0.5) |
| (64,64)   | (0.5,0.5)   |

Below pictures are 320, 256, 128, 64 windows rectangle with overlapping. xy Overlaps are selected after 'trial-and-error' method.


![Sliding windows][image4]
(From top / 320px, 256px, 128px, 64px)

And I restrict x and y values, like mask-image, at the last time of return result of sliding window, I check many values and only valid-rect is returned.

```
# check if out of screen
if(endx <= x_start_stop[1]
    and startx > get_x_start_from_y(endy) # using my predefind xs
    and endy<= y_start_stop[1]  ) :

    # Append window position to list
    window_list.append(((startx, starty), (endx, endy)))
```





####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![video pipeline][image6]

I used 'full screen hog features', it's performance is more than 2 or 3 times increased.

I make balanced dataset, it means same data 'car' and 'non-car' images. And get features and using ``StandardScaler().fit(X)`` to normalize data.

After that, I split train set (80%) and test set(20%) with random shuffle.

after get model, I tried small test for this model. see ``Cell 12 : Model result test`` has various car, non-car images. and tested is OK.



---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

There is some trick, I stored 6 frames hotwindows and everytime added one more, delete the first one. To fast calculation, I saved recent result heatmap, and if I must extract oldest one, use ``minus_heat``. This routine is almost same with ``add_heat`` but it minus 1 which include in hot windows


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
### Here are five frames and their corresponding heatmaps at that time:
It shows almost same heatpmap, because every time heatmap is added and delete oldest one. And if it's value is under threshold, then its point is ignored that time. So I can get stable heatmaps.
![video pipeline][image7]
`scipy.ndimage.measurements.label()` on the integrated heatmap from all recent frames

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
- Result video is not perfect. It shows many error. Maybe the problem is 'series data set' and lack of 'non-car images'. And very small size car cannot be detected this version. It is also will be improved.
- The left and right boundary is some speed down factor. Because the useless region must be deleted when processing. I can just mask it like project1, but it is not good for most situation. There is always exceptions.
- Detect car pipeline is so slow, maybe if I reduce some calculation, then it reduce more time
- I use full 6 full heatmaps when determine false positives, but it is so many memory leak, so I'll change another way.
- It needs validation test of heatmap. I think the size of car and the location is most valid factors. Almost car has same width, so if some car is too big or too samll, then it's an error. then I can change that size.
- Or transform to bird-eye view, and can calculation the car size with easy. I'll try next time.

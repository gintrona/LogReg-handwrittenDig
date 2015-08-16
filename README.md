# LogReg-handwrittenDig
This code implements one-vs-all logistic regression to recognize hand-written digits, from zero to nine, and uses the GSL numerical library.
The file "example.txt" contains the data set (5000 images) to train the algorithm. Each image is a 20 pixel by 20 pixel grayscale image of a digit, and each pixel is represented by a floating point number that indicates the grayscale intensity at the corresponding location of the grid. The 20x20 grid of pixels is unrolled and transformed into a 400 vector, and each of these vectors represents a training element in the data set. 

Since we're dealing with a supervised algorithm, a digit is reported to each training example.

The file "example.txt" thus contains a 5000 x 401 matrix. While the first 400 columns allocate the image pixels, the last column indicates the corresponding digit. “0” digits are labeled as “10”



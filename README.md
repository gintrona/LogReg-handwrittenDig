# LogReg-handwrittenDig
This code implements one-vs-all logistic regression to recognize handwritten digits, from zero to nine, uses the GSL numerical library and supports multi-threading (a thread for each class/digit). It's essentially the C++ version of the corresponding algorithm presented in the online course by Andrew Ng, from Stanford University. In that course, the Octave language is used instead.

To train the algorithm, 5000 images are used.Each of them is a 20 pixel by 20 pixel grayscale image of a digit, and each pixel is represented by a floating point number that indicates the grayscale intensity at the corresponding location of the grid. The 20x20 grid of pixels is unrolled and transformed into a 400 vector that represents a training element in the data set, each of which is stored in a row of a matrix. This matrix is written in the file "example.txt". It contains 5000 rows and 401 columns, where the first 400 columns store the image pixels and the last column indicates the corresponding digit. It must be noted that “0” digits are labeled as “10”.

After having trained the algorithm, the program calculates the accuracy.

The accuracy obtained is 93.5%.



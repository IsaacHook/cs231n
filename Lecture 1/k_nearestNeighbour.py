# Rather than just looking at which training image the test image is closest
# to, we can look at the k images it is closest to.  This normally gives a better
# result than the nearest neighbour approach.

# But we need a way of choosing the parameters: the distance function and the value for k

# We do this by splitting the training data into two: training data and validation set
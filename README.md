# Binary-Index-Tree
A c++ data structure with logorithmic insertion and deletion time complexity. Excellent if you want to insert and erase into the middle of a vector and don't want to worry about slowdowns or finding the correct data structure for your task. Not for tasks that can be simplified into contiguous memory storage. If you want sorted order use a BST. This data structure does not sort. 

Implements the full std::vector interface except not custom allocators, and runtime complexity is different. Accesing elements takes logorithmic time, as do pushbacks. In return, inserts and deletes at arbitrary indices, even in bulk and in the center of the data structure, take linear time with respect to the number of elements inserted, and logorithmic time with respect to the number of elements already stored.

Constructing the BinaryIndexTree from a forward iterator takes linear time, and iterating through it takes amortized constant time per ++ of -- operation. 

iterators are not random access, though they do implement the full random access iterator interface. For efficient code writing they should be considered bidirectional iterators. Nevertheless, the iterators will function in code written for vector iterators.

Full usage instructions and code examples coming soon. BinaryIndexTree is a complete product, but hasn't undergone strenuous testing. Consider beta, and use at your own risk. Please report any bugs as Issues.

How it works:
A BinaryIndexTree is implemented as a Weight balanced binary tree. Each Node of the tree remembers its weight, the total number of nodes in its subtree. If a node is heavier than 3/4 of its parents weight, an apropriate rebalance is performed, thus keeping the weight balance and by extension the height reasonable.

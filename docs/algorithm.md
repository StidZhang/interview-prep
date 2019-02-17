# Algorithm

* [1. Time Analysis](#time-analysis)
    * [Mathematical Model](#mathematical-model)
    * [Precautions](#precautions)
    * [Three Sum Problem](#three-sum)
    * [Ratio Test](#ratio-test)
* [2. Sorting](#sorting)
    * [Selectionsort](#selectionsort)
    * [Bubblesort](#bubblesort)
    * [Insertionsort](#insertionsort)
    * [Shellsort](#shellsort)
    * [Mergesort](#mergesort)
    * [Quicksort](#quicksort)
    * [Heapsort](#heapsort)
* [3. Data Structure](#data-structure)
    * [Array](#array-1)
    * [Stack](#stack)
    * [Queue](#queue)
    * [Linked List](#linked-list)
    * [Tree](#tree)
    * [Heap](#heap)
    * [Hash Table](#hash-table)
    * [Graph](#graph-1)
    * [Disjoint set](#disjoint-set)
* [4. Algorithms](#algorithms)
    * [String](#string)
    * [Array](#array-2)
    * [Divide and Conquer](#divide-and-conquer)
    * [Dynamic Programming](#dynamic-programming)
    * [Backtracing](#backtracking)
    * [Graph](#graph-2)
    * [Mathematics](#mathematics)


# Time Analysis

## Mathematical Model

### Approximation

N<sup>3</sup>/6-N<sup>2</sup>/2+N/3 \~ N<sup>3</sup>/6. Uses \~f(N) to
represent all the functions such that as N increases, the function divided by
f(N) converges to 1.

### Big O Notation

N<sup>3</sup>/6-N<sup>2</sup>/2+N/3 has a run time
complexity of O(N<sup>3</sup>). Note that such model does not care about
the running environment of the algorithm.

### Running time within loop

The most frequently used command inside the loop decides the total running time
of the looping part of the algorithm.

### Cost model

Using cost model to evaluate the efficiency of algorithm. E.g. Total count of
visiting an array.

## Precautions

### Large Coefficients

When there're some large coefficients in the lower N terms, the approximation
result most likely will be incorrect.

### Cache

Cache is used in computer hardware so that visiting nearby elements in array
will be faster than visiting randomly.

### Worst case scenario

For some algorithm/software, we care most about the worst case scenario, such
as software that controls nuclear reactor.

### Equalization analysis

Split the cost by dividing the total operations. For example, push N times to
an empty stack would need to visit the array for N+4+8+16+...+2N=5N-4 times
(N times of writing element, all the rest are changing the array size, C++
implementation). Actual cost of a push would be 5N-4/N which is a constant.

## Three Sum
Three Sum problem is to find the total amount of unique triplets in an array
such that their sum is 0.

### ThreeSumSlow
```python
def three_sum_slow(nums):
    count = 0
    for i in nums:
        for j in nums[1:]:
            for k in nums[2:]:
                if (i + j + k == 0):
                    count += 1
    return count
```
The most frequently used command in the algorithm is `if (i + j + k == 0)`,
so total running time is N(N-1)(N-2) = N<sup>3</sup>/6-N<sup>2</sup>/2+N/3,
which approximates to \~N<sup>3</sup>/6 with O(N<sup>3</sup>) time complexity.

### ThreeSumBinarySearch
Sort the array first and use binary search to find the target=0-i-j. Note that
this algorithm only applies when all elements in the array are different
otherwise binary search might give incorrect result.
```python
from bisect import bisect_left

def binary_search(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1

def three_sum_binary_search(nums):
    nums.sort()
    count = 0
    for i in nums:
        for j in nums[1:]:
            target = - i - j
            index = binary_search(nums, target)
            if index != -1:
                count += 1
    return count
```
Time complexity is is O(N<sup>2</sup>logN).

### ThreeSumDoublePointer
Sort the array first and then use double pointer to find the target.
```python
def three_sum_double_pointer(nums):
    nums.sort()
    N = len(nums)
    count = 0
    for index, value in enumerate(nums[:-2]):
        if value > 0:
            break # Three sum of positives will never be 0
        l, h = index + 1,  N - 1
        if (index > 0 and value == nums[index-1]):
            continue # Skip if value is the same
        while (l < h):
            sum = value + nums[l] + nums[h]
            if sum == 0:
                count += 1
                while (l < h and nums[l] == nums[l + 1]):
                    l += 1 # Skip if same
                while (l < h and nums[h] == nums[h - 1]):
                    h -= 1 # Skip if same
                l += 1
                h -= 1
            elif sum < 0:
                l += 1
            else:
                h -= 1
    return count
```
Time complexity is O(N<sup>2</sup>).

## Ratio Test
If T(N) \~ aN<sup>b</sup>logN, then we haveT(2N)/T(N) \~ 2<sup>b</sup>.

# Sorting
We mostly care about the total amount of compare and swap operations in
sorting.
```python
def compare(a, b):
    return a < b

def swap(a, i, j):
    a[i], a[j] = a[j], a[i]
```

## Selectionsort
Pick the smallest element in array an swap it with first element. Then
pick second smallest from the rest of the array and swap with second element,
continues till the last element.
```python
def selectionsort(nums):
    for i, value_i in enumerate(nums):
        min_index = value_i
        for j, value_j in enumerate(nums[i+1:]):
            if compare(value_j, value_i):
                min_index = j
        swap(nums, i, min_index)
```
Time complexity is O(N<sup>2</sup>).

## Bubblesort
Swapping adjacent elements in order so that the largest element will appeared on
the right after one loop. Keep sorting until no swap needed which means the
array is sorted.
```python
def bubblesort(nums):
    sorted = False
    for i in range(0, len(nums), -1):
        if not sorted:
            sorted = True
            for j in range(i):
                if compare(nums[j + 1]), nums[j]):
                    sorted = False
                    swap(nums, j, j+1)

```
Time complexity is O(N<sup>2</sup>).

## Insertionsort
Find un-sorted element in the array and then insert it into the previous sorted
array.
```python
def insertionsort(nums):
    for index, value in enumerate(nums):
        for j in range(index-1, -1, -1):
            if compare(value, nums[j]):
                swap(nums, index, j)
            else:
                break
```
Time complexity is  O(N<sup>2</sup>).

## Shellsort
Apply insertion sort on array with gap as h, reduce h until h = 1.
```python
def shellsort(nums):
    h = len(nums) // 2
    while h > 0:
        for i in range(h, n):
            temp = nums[i]
            j = i
            while j >= h and nums[j-h] > temp:
                nums[j] = nums[j-h]
                j -= h
            nums[j] = temp
        h = h // 2
```
Time complexity is O(Nlog<sup>2</sup>N). For detail check
[here](https://en.wikipedia.org/wiki/Shellsort).

## Mergesort
Basically divide and conquer. Sort on sub array and then merge.
```python
def merge(nums, nums_a, nums_b):
    a_index = 0
    b_index = 0
    for index in range(nums):
        if a_index == len(nums_a):
            nums[i:] = nums_b[b_index:]
            return
        elif b_index == len(nums_b):
            nums[i:] = nums_a[a_index:]
            return
        elif compare(nums_a[a_index], nums_b[b_index]):
            nums[i] = nums_a[a_index]
            a_index += 1
        else:
            nums[i] = nums_b[b_index]
            b_index += 1


def mergesort(nums):
    if len(nums) = 1:
        return
    merge(nums,
        mergesort(nums[:len(nums) // 2]),
        mergesort(nums[len(nums) // 2:]))
```
Time complexity is O(NlogN).

## Quicksort
Pick a pivot in the array, then put everything less than the pivot on the left
and everything greater on the right. Recursively do it on all sub arrays.
```python
# Inplace implementation
def partition(nums, start, end):
    i = start - 1
    pivot_index = end
    pivot = nums[end]
    for j in range(start, end):
        if compare(nums[j], pivot):
            i += 1
            swap(nums, i, j)
    swap(nums, i+1, pivot_index)
    return i+1

def quicksort(nums, start=0, end=len(nums)-1):
    if start >= end:
        return
    pivot_index = partition(nums, start, end)
    quicksort(nums, start, pivot_index-1)
    quicksort(nums, pivot_index+1, end)
# Easy
def quicksort_easy(nums):
    if len(nums) == 1:
        return nums
    left,right = [], []
    pivot = nums[0]
    for value in nums[1:]:
        if compare(value, pivot):
            left.append(value)
        else:
            right.append(value)
    return quicksort_easy(left) + [pivot] + quicksort_easy(right)
```
Time complexity is O(NlogN) on average, worst case O(N<sup>2</sup>) when always
picking pivot that is max/min of the remaining array.

Furthermore, there're some modification for quicksort for special cases to
improve performance.
1. Use selection sort when imput size is small.
2. Randomly pick three numbers, and pick the medium as the pivot to avoid
extreme cases.
3. When array has a lot of duplicate values, split the array into 3 being
greater, lesser and equal. In this cases the running time may be down to O(N).

### Quick Selection
Quick selection is an algorithm based on quicksort's partition
which picks kth largest element in the array.
```python
def quickselection(nums, k):
    low, high = 0, len(nums) - 1
    while (low < high):
        j = partition(nums, low, high)
        if (j == k):
            return nums[k]
        elif (j > k):
            high = j - 1
        else:
            low = j + 1
    return nums[k]
```


## Heapsort
Build a max heap with the array, swap the first element with final element so
that the final element is the largest since it's maxheap. Then shift the new
first element down so that it becomes a new maxHeap with one less size.
Continue until heap size becomes one.
```python
def heapify(nums):
    for index in range(len(nums)):
        parent_index = (index - 1) // 2
        parent = nums[parent_index]
        if index == 0 or parent >= nums[index]:
            continue
        i = index
        while (nums[i] > nums[parent_index] and parent_index >= 0):
            nums[i], nums[parent_index] = nums[parent_index], nums[i]
            i = parent_index
            parent_index = (i - 1) // 2

def heapsort(nums):
    heapify(nums)
    for i in range(len(nums)-1, -1, -1):
        nums[0], nums[i] = nums[i], nums[0]
        j = 0
        left_index = 2*j+1
        right_index = 2*j+2
        while (left_index < i):
            left_child = nums[left_index]
            if right_index < i:
                right_child = nums[right_index]
                if left_child < right_child and nums[j] < right_child:
                    nums[j], nums[right_index] = nums[right_index], nums[j]
                    j = right_index
                    left_index = 2*j+1
                    right_index = 2*j+2
                    continue
            if nums[j] < left_child:
                nums[j], nums[left_index] = nums[left_index], nums[j]
                j = left_index
                left_index = 2*j+1
                right_index = 2*j+2
                continue
            break
```

# Data structure

## Array
Array size is normally fixed.

Indexing: O(1), Searching: O(n)

### Dynamic Array
Double the size whenever it hits current max size, so it no longer have a fixed
size. Also remove the memory allocation when size is smaller than some specific
value.

Indexing: O(1)

## Stack
Last in first out(LIFO).
Implementation can be done using Array or Linked List.

push: O(1), pop: O(1), search: O(n)

## Queue
Last in first out(FIFO).
Implementation can be done using Array or Linked List.

enqueue: O(1), dequeue: O(1), search: O(n)

### Priority Queue
A queue such that enqueues with priority. A dequeue will pop the element with
highest priority, and peek() will find such element but not pop it.

## Linked List
Indexing: O(n), Insert at beginning: O(1), Insert in middle: O(k) + O(1)

Extra pointer can be added so that insert at end can also be O(1).

## Tree
Nonlinear data structure with a node(root) and and zero/one/more subtrees or
empty.

Another definition is a connected graph without any circle.

### Binary Search Tree

Tree such that left child is smaller than parent node and right child is larger
than the parent node. When traversing in order, you will have a sorted array.

On average, search, insert and delete all take O(logN) time. In worst case,
when BST is one-sided, the tree becomes a linked list with search, insert and
delete in O(n) time.

In particular, deletion in BST:
1. If node is leaf, just remove.
2. Otherwise, substitute the node with the left most node in right subtree;
If such node is not leaf, replace the node with it's right subtree.

#### AVL Tree
A balanced BST such that difference in left/right subtree is maximum 1.

When inserting a node into AVL tree, do the same thing as BST, then if balance
factor |bf| > 1, do a rotation.

Similarly, for deletion, perform a normal deletion similar to BST, but perform a
rotation if |bf| > 1 for any node.

Since the tree is balanced, even in worst case search, insert and delete can
all be finished in O(logN).

### B-Tree
![B-Tree.svg](../pics/B-tree.svg)

A self-balancing tree data structure such that everything is in sorted order.

For any internal nodes, it can have a variable number of child nodes with in
some order k.

Used in storage systems and databases that read and write large blocks of data,
since it can act as an index that speed the search.

Insertion:
1. If node is not full yet, simply add it to the specific locaiton for the node.
2. Otherwise, split it into two nodes where median would remain in same node,
left will be anything smaller than median and right node will be greater than
median. Might need to do it recursively.

Deleteion:
1. If in leaf node, just remove and if underflow the node size, rebalancing
the tree.
2. Otherwise, take left most value in right subtree, then rebalance the whole
whole tree.

Operations normally requires O(logN) time on average.

#### 2-3 Tree
2-3 tree is B-tree with order 3. Any internal node will have either 1/2 values.

#### 2-3-4 Tree
A B-tree with order 4. Any internal node will have either 1/2/3 values.

### k-d Tree
Binary space paritioning such that cuts the space into two half-spaces, where
the cutting plane pass through any non-leaf node.

### Trie
Also called a prefix tree. All nodes with common prefix will be associated, so
often used in autocomplete dictionary/term indexing.

## Heap
A specific kind of complete tree such that parent always greater than the
child(or the opposite). Normally we discuss about binary heap, but __Fibonacci
heap__ could be used in priority queue implementation which allows operations such
as finding highest priority, insertion, decrease priority and merging to be
completed in O(1) while poping stays O(logN).

An array can act as a binary heap.

## Hash Table
Similar to dictionary. Can access key-value pair within O(1) time with decent
hashing function.

When multiple key had same hash value, we can have separate chaining(linked
list when getting hash collision) or open addressing.
Possible open addreesing approach includes: linear probing, quadratic probing,
double hashing and cuckoo hashing.

## Graph
Two types of graph exist: undirected or directed. It could have weight or not
(can be considered as weight 1 for all edges).

It can be represented by directly by linked nodes, adjacency list,
adjacency matrix or incidence matrix.

## Disjoint set

# Algorithms

## String

### KMP algorithm
A string searching algorithm with time complexity O(n + k).
Mainly include two steps:
1. Calculate failure array. 2. Find first match in the string.

```python
def failure_array(P):
    i, j = 1, 0
    F = [0]*len(P)
    while i < len(P):
        if P[i] == P[j]:
            F[i] = j + 1
            i += 1
            j += 1
        elif j > 0:
            j = F[j-1]
        else:
            F[i] = 0
            i = i + 1
    return F

def KMP(T, P):
    F = failure_array(P)
    i, j = 0, 0
    m, n = len(P), len(T)
    while i < n:
        if T[i] == P[j]:
            if j == m - 1:
                return i - j # Match and return index
            i += 1
            j += 1
        elif j > 0:
            j = F[j-1]
        else:
            i += 1
    return -1 # No match
```

### Boyer-Moore Algorithm
Based on reverse-order searching, bad character jumps and good suffix jumps.
It skip large parts of T.

```python
def last_occurrence(alphabet, P):
    valid_chars = {x:-1 for x in alphabet}
    for index, chr in enumerate(P):
        if index > valid_chars[chr]:
            valid_chars[chr] = index
    return valid_chars

def boyer_moore_match(text, pattern):
    """Find occurrence of pattern in text."""
    alphabet = set(text)
    last = last_occurrence(alphabet, pattern)
    m = len(pattern)
    n = len(text)
    i = m - 1  # text index
    j = m - 1  # pattern index
    while i < n:
        if text[i] == pattern[j]:
            if j == 0:
                return i
            else:
                i -= 1
                j -= 1
        else:
            l = last(text[i])
            i = i + m - min(j, 1+l)
            j = m - 1
    return -1
```

### Rabin-Karp Fingerprint Algorithm
Use hashing, and except first one all the rest hashing function can be finished
in O(1) with some specific hashing function.

### Suffix Tries and suffix trees
Used when there're multiple patterns P with the same fixed text T.

### Manacher's Algorithm
Used to find longest palindrome substring in a string.

```python
def manacher(s0 : str) -> list:
    T = '$#' + '#'.join(s0) + '#@'
    l = len(T)
    P = [0] * l
    R, C = 0, 0
    for i in range(1,l-1):
        if i < R:
            P[i] = min(P[2 * C - i], R - i)

        while T[i+(P[i]+1)] == T[i-(P[i]+1)]:
            P[i] += 1

        if P[i] + i > R:
            R = P[i] + i
            C = i
    return P
```

## Array

### Two/multiple pointers
Often used in searching pairs in sorted array. Normally there's a low/high and
gets closer and closer after iteration. e.g: 2sum
```python
def two_sum(nums, target):
    nums.sort()
    i, j = 0, len(nums) - 1
    while (i < j):
        current_sum = nums[i] + nums[j]
        if current_sum = target:
            return [nums[i], nums[j]]
        elif current_sum < target:
            i += 1
        else:
            j -= 1
    return -1
```

### Window sliding
Usually used to convert a nested loop to single loop. e.g: maximum subarray with
size k.
```python
def max_subarray_with_size(nums, k):
    max_sum = sum(nums[:k])
    window_sum = max_sum
    for i in range(k, n):
        window_sum += nums[i] - nums[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

## Greedy
Greedy algorithm finds local optimal choice at each stage with the intent of
finding a global optimum. It might not produce the best result, but it normally
can achieve a relatively optimal solution within a reasonable amount of time.

## Divide and Conquer
Basically multi-branched recursion, split the input so that a solution can be
found directly, then try to combine the answers together.
Best example would be mergesort.

### Master Theorem
Time complexity analysis for divide and conquer mostly can be solved with master theorem.
For any runtime recurrence can be written in
T(n) = aT(n/b) + O(n<sup>c</sup>), then we can have
- If c > logb(a), T(n) = Θ(n<sup>c</sup>)
- If c = logb(a), T(n) = Θ(n<sup>c</sup>log(n))
- If c < logb(a), T(n) = Θ(n<sup>logb(a)</sup>)

## Dynamic Programming
Solve sub-problem only once and save the result. Therefore when next time we
need the answer of the problem the result can be accessed with O(1) time.

E.g. Knapsack Problem: Given a set of items with weight and value, find the
maximum number of value while total weight is smaller or equal to a limit.

```python
def knapsack(weights: 'List[int]', value: 'List[int]', limit: 'int'): -> 'int'
    m = [0] * (limit+1) * len(weights)
    for i in range(1, len(weights)+1):
        for j in range(limit+1)):
            if weights[i] > j:
                m[i][j] = m[i-1][j]
            else:
                m[i][j] = max(m[i-1][j], m[i-1][j-weight[i]] + value[i])
    return m[-1][-1]
```
## Backtracking

## Graph

### Depth First Search

### Breath First Search

### Shortest Path

#### Dijkstra's algorithm

### Minimum spanning tree

### Traverse

## Search

### A* Search

## Mathematics

### GCD
GCD is to find greatest common divisor given a and b. With Euclidean division,
we can find the quotient and remainder.
```python
def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
```
A possible faster method involves half-GCD/matrices.

#### Bezout's Identity
For any interger a, b and m, __ax_by=c__ have integer solution for x and y
if and only if m is the multiple of the `gcd(a, b)`.

Alternatively, a __gcd__ is the smallest positive integer such that can be written in form __ax+by__.

### Fast multiplication

#### Karatsuba multiplication
Split the two numbers in half in digits (could be in bits) such that they become x1, x2, y1 and y2.
Then calculate x1y1, x2y2 and (x1+x2)(y1+y2).
This algorithm reduce the runtime for multiplication to
O(N<sup>log2(3)</sup>) = O(N<sup>1.58</sup>).

#### Toom-cook algorithm
Extend from above, we can split the numbers in k parts,
and the runtime will become O(N<sup>logk(2k-1)</sup>).

#### Schönhage–Strassen algorithm
A DFT can be finished in Θ(nlog(n)) using FFT,
as well as inverse DFT. The algorithm use such property
and makes multiplication to be finished in
O(nlog(n)log(log(n))).

### Gaussian elimination
Gaussian elimination is used for solving systems of
linear equations. It can be finished in O(n<sup>3</sup>).

### Division
Long division in elementary school is good enough. Maybe look at Newton-Raphson division.

### RSA
Exists because of the difficulty of integer factorization.
It works like this:
1. Alice send Bob public key (n, e) and keep d as private key and never distributed.
2. Bob convert message M to integer m, and calculate
`c = m^2 mod n` and send __c__ to Alice.
3. Alice decrypt __c__ with
`c^d = (m^e)^d = m mod n`.

### Prime test
Most simple way is sieve of Eratosthenes with running time
O(nlog(log(n))).
```python
def eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return {x for x in range(2, n + 1) if is_prime[x]}
```
Something else in polynomial time is found called AKS primality test, roughly finished in O(log<sup>6</sup>(n)).

### PageRank
An algorithm for search engine to rank results. It treats every webpage as a node with a weight.
Weights depend on the difference between link point to the
page and point away from the page.

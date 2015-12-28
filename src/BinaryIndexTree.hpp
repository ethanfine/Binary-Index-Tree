#pragma once
#ifndef Binary_Index_Tree
#define Binary_Index_Tree

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <stack>
#include <iterator>
#include <limits>
#include <type_traits>

template <typename T, typename Weight = std::size_t>
class BinaryIndexTree{

    static_assert(std::is_unsigned<Weight>::value,"Weight (the second template argument) must be an unsigned integral type");

    public: class const_iterator;
private:
    struct size_struct{
        Weight value;
        size_struct(Weight nvalue):value(nvalue){}

        size_struct(const_iterator& iter) {
            value = (Weight)iter;
        }

        size_struct(const_iterator&& iter) {
            value = (Weight)iter;
        }

        explicit operator Weight(){return value;}
    };

    struct Node{
        T value;
        Node* child[2];
        Weight weight;

        Weight cWeight(unsigned s) const{
            if(child[s]) return child[s]->weight;
            else return 0;
        }

        template<typename ...Args, typename = typename
            std::enable_if<std::is_constructible<T,Args...>::value>::type
        >
        Node(Args&&... args): value(std::forward<Args...>(args...)){
            weight = 1;
            child[0] = nullptr;
            child[1] = nullptr;
        }

        Node(Node const& node){
            value = node.value;
            weight = node.weight;
            if(node.child[0]) child[0] = new Node(*node.child[0]);
            else child[0] = nullptr;
            if(node.child[1]) child[1] = new Node(*node.child[1]);
            else child[1] = nullptr;
        }

        Node(Node && node){
            value = std::move(node.value);
            weight = node.weight;
            child[0] = node.child[0];
            child[1] = node.child[1];
            node.child[0] = nullptr;
            node.child[1] = nullptr;
        }

        ~Node(){
            if(child[0]) delete child[0];
            if(child[1]) delete child[1];
        }

        //test function
        bool verify(){
            return weight == 1+cWeight(0)+cWeight(1) && child[0]?child[0]->verify():true && child[1]?child[1]->verify():true;
        }

    };

    Node* m_root;

    //method to rebalance around a node given the side that may be too heavy.
    static bool rebalance(Node*& pivot, unsigned s){
        //s is the side that might be too heavy. Rebalance if child s is at least 3/4 as heavy as the parent.
        Node* b = pivot->child[s]; //b is the child that might be too heavy
        Node* d = pivot; //d is the pivot
        if(!b || b->weight<(d->weight)-(d->weight>>2)) return false;

        Node* a = b->child[s]; //a is the outer child
        Weight aweight = a?a->weight:0;
        Node* c = b->child[1-s]; //c is the inner child

        if(!c || c->weight <= d->cWeight(1-s)){
            //outer rotate
            pivot = b;

            b->child[1-s]=d;
            b->weight = d->weight;

            d->child[s]=c;
            d->weight -= aweight+1;
        }
        else{
            //inner rotate
            pivot = c;

            d->child[s] = c->child[1-s];
            c->child[1-s] = d;
            b->child[1-s] = c->child[s];
            c->child[s] = b;

            c->weight = d->weight;
            b->weight = 1 + b->cWeight(1-s) + aweight;
            d->weight -= 1+ b->weight;
        }
        return true;
    }

    //method to remove a node from the tree and delete it
    static void Delete(Node*& subtree){
        Node* root = subtree;
        if(!root->child[1]){
            //Replace the root with its left child, which is already balanced.
            subtree = root->child[0];
        } else{
            //replace the root with the leftmost child of its right subtree
            subtree = rDelete(root->child[1],0);
            subtree->weight = root->weight;
            subtree->child[0] = root->child[0];
            subtree->child[1] = root->child[1];
        }
        root->child[0] = nullptr;
        root->child[1] = nullptr;
        delete root;
    }

    //recursive method to find, remove from the tree, and return a pointer to a node in the tree.
    static Node* rDelete(Node*& subtree, Weight index){
        Node* returnvalue;
        Node* root = subtree;
        root->weight--;
        unsigned s = 0; //if a node is removed from a subtree, s is set to the side not removed from.
        if(index<root->cWeight(0)){
            //the node is removed from the left subtree
            returnvalue = rDelete(root->child[0],index); //rebalancing may be needed
            s=1;
        } else{
            index-=root->cWeight(0);
            if(index>0){
                //the node is removed from the right subtree
                returnvalue = rDelete(root->child[1],index-1); //rebalancing may be needed
                //s is already 0
            } else if(!root->child[1]){
                //the node removed is root. Replace it with its left child, which is already balanced.
                subtree = root->child[0];
                root->child[0] = nullptr;
                root->child[1] = nullptr;
                return root; //no rebalancing is needed
            } else{
                //the node to be removed is root. replace it with the leftmost child of its right subtree
                subtree = rDelete(root->child[1],0);
                subtree->weight = root->weight;
                subtree->child[0] = root->child[0];
                subtree->child[1] = root->child[1];
                //s is already 0
                root->child[0] = nullptr;
                root->child[1] = nullptr;
                returnvalue = root; //rebalancing may be needed
            }

        }

        rebalance(subtree,s);
        return returnvalue;

    }

    //recursively adds a node to a subtree
    static void rAdd(Node*& subtree, Weight index, Node* newNode){
        if(!subtree){
            subtree = newNode;
        } else{
            Node* root = subtree;
            root->weight++;
            Weight leftWeight = root->cWeight(0);
            unsigned s = index>leftWeight; // s is the side we add to.
            index = s?index-leftWeight-1:index;
            rAdd(root->child[s],index,newNode);
            rebalance(subtree,s);
        }
    }

    //recursively finds node in a subtree
    static Node* rGet(Node*  root, Weight index){
        Weight leftWeight = root->cWeight(0);
        if(index<leftWeight){
            return rGet(root->child[0],index);
        } else if(index == leftWeight){
            return root;
        } else{
            return rGet(root->child[1],index-leftWeight-1);
        }
    }

    //recursively makes a tree from an iterator
    template<class InputIterator, typename category = typename std::iterator_traits<InputIterator>::iterator_category>
    static Node* rMake(InputIterator &&start,Weight length){
        if(length==0){
            return nullptr;
        } else if(length==1){
            return new Node(*start++);
        } else{
            Node* lchild = rMake(std::forward<InputIterator>(start),length/2);
            Node* returnvalue = new Node(*start++);
            returnvalue->child[0] = lchild;
            returnvalue->child[1] = rMake(std::forward<InputIterator>(start),length-length/2-1);
            returnvalue->weight = length;
            return returnvalue;
        }
    }

    static Node* rMake(T const& val, Weight length){
        if(length==0){
            return nullptr;
        } else if(length==1){
            return new Node(val);
        } else{
            Node* returnvalue = new Node(val);
            returnvalue->child[0] = rMake(val,length/2);
            returnvalue->child[1] = rMake(val,length-length/2-1);
            returnvalue->weight = length;
            return returnvalue;
        }
    }

    template <class InputIterator,typename category = typename std::iterator_traits<InputIterator>::iterator_category>
    static Node* delegateMake(InputIterator first, InputIterator last){
        return delegateMake(first,last,category());
    }

    template <class InputIterator>
    static Node* delegateMake(InputIterator first, InputIterator last,std::input_iterator_tag delegate){
        static_assert(std::is_same<typename std::iterator_traits<InputIterator>::iterator_category,std::input_iterator_tag>::value, "Wrong delegateMake");
        std::vector<T> data(first,last);
        return rMake(data.begin(),last-first);
    }

    template <class InputIterator,typename category = typename std::iterator_traits<InputIterator>::iterator_category>
    static Node* delegateMake(InputIterator first, InputIterator last,category delegate){
        static_assert(std::is_same<typename std::iterator_traits<InputIterator>::iterator_category,category>::value, "Wrong delegateMake");
        Weight n = std::distance<InputIterator>(first,last);
        return rMake(std::forward<InputIterator>(first),n);
    }

    //TODO rrebalalance
    static void rRebalance(Node*& subtree){
        unsigned s = subtree->cWeight(1)>subtree->cWeight(0);
        if(rebalance(subtree,s)){
            rRebalance(subtree->child[1-s]);
        }
    }

    static void rMultiErase(Node*& subtree, Weight first, Weight last){
        if(last<=first){
            return;
        }
        subtree->weight -= last-first;
        if(!subtree->weight){
            delete subtree;
            subtree = nullptr;
            return; //no rebalance neccecary
        }
        Weight lweight = subtree->cWeight(0);
        Weight const& lfirst = first; //lfirst is the same as first. Will compile to an alias.
        Weight llast = std::min(lweight,last);
        Weight rfirst = first-std::min(first,lweight+1); //performs coordinate transform and mins at 0
        Weight rlast = last-std::min(last,lweight+1);
        rMultiErase(subtree->child[0],lfirst,llast);
        rMultiErase(subtree->child[1],rfirst,rlast);
        if(first<=lweight && last>lweight){
            Delete(subtree); //removes the node at a subtree, and deletes it, replacing it as apropriate.
        }

        rRebalance(subtree);
    }

    //recursively adds a tree newNode to a subtree
    static void rMultiAdd(Node*& subtree, Weight index, Node* newNode){
        if(!subtree){
            subtree = newNode;
        } else{
            Node* root = subtree;
            root->weight+= newNode->weight;
            Weight leftWeight = root->cWeight(0);
            unsigned s = index>leftWeight; // s is the side we add to.
            index = s?index-leftWeight-1:index;
            rMultiAdd(root->child[s],index,newNode);
            rRebalance(subtree);
        }
    }

public:

    typedef Weight size_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef ptrdiff_t difference_type;


    class const_iterator : public std::iterator<std::bidirectional_iterator_tag,T,ptrdiff_t,T*,T const&>{
    private:
        Node* current;
        Node* root;
        std::stack<Node*> ancestors;

        //crawl to the indexth element of the current subtree
        void crawldown(Weight index){
            Weight lweight = current->cWeight(0);
            if(index<lweight){
                ancestors.push(current);
                current = current->child[0];
                crawldown(index);
            } else if (index>lweight){
                ancestors.push(current);
                current = current->child[1];
                crawldown(index-lweight-1);
            }
        }

        //crawl up until after being a subtree on the s side
        void crawlup(unsigned s){
            while(!ancestors.empty()){
                Node* temp = current;
                current = ancestors.top();
                ancestors.pop();
                if(current->child[s]==temp){
                    return;
                }
            }
            current = nullptr;
        }

        //crawl distance in the s direction.
        const_iterator& crawl(unsigned s, Weight distance){
            if(distance){
                Weight sweight = current->cWeight(s);
                if(distance>sweight){
                    distance-=(sweight);
                    crawlup(1-s);
                    crawl(s,distance-1);
                } else{
                    ancestors.push(current);
                    current = current->child[s];
                    crawldown(s?distance-1:sweight-distance);
                }
            }
            return *this;
        }

    public:
        const_iterator(BinaryIndexTree const& tree, size_struct index){
            root = tree.m_root;
            current = root;
            if(root){
                if(root->weight>(Weight)index){
                    if(current) crawldown((Weight)index);
                }
            }
        }

        const_iterator(BinaryIndexTree const& tree){
            current = nullptr;
            root = tree.m_root;
        }

        Weight index() const{
            if(current==nullptr) return root?root->weight:0;
            std::stack<Node*> temp = ancestors;
            Node* child = current;
            Weight returnvalue = current->child[0]->weight;
            while(!temp.empty()){
                Node* parent = temp.top(); temp.pop();
                if(parent->child[1]==child){
                    returnvalue += parent->child[0]->weight+1;
                }
                child = parent;
            }
            return returnvalue;
        }

        T const& operator*() const{
            return current->value;
        }

        bool operator==(const_iterator const& other) const{
            return current == other.current;
        }

        bool operator!=(const_iterator const& other) const{
            return !(*this == other);
        }

        ptrdiff_t operator-(const_iterator const& other) const{
            return (ptrdiff_t)index() - other.index();
        }

        bool operator <(const_iterator const& other) const{
            return index() < other.index();
        }

        bool operator >(const_iterator const& other) const{
            return index() > other.index();
        }

        bool operator <=(const_iterator const& other) const{
            return index() <= other.index();
        }

        bool operator >=(const_iterator const& other) const{
            return index() >= other.index();
        }

        const_iterator& operator+=(ptrdiff_t index){
            unsigned dir = index<0;
            if(current==nullptr){
                if(dir){
                    index++;
                    current = root;
                    crawldown(root->weight-1);
                }else{
                    index--;
                    current = root;
                    crawldown(0);
                }

            }
            return crawl(1-dir,dir?-index:index);
        }

        const_iterator operator+(ptrdiff_t index) const{
            const_iterator returnvalue = *this;
            return returnvalue+=index;
        }

        const_iterator& operator-=(ptrdiff_t index){
            unsigned dir = index<0;
            if(current==nullptr){
                if(dir){
                    index++;
                    current = root;
                    crawldown(0);
                } else{
                    index--;
                    current = root;
                    crawldown(current->weight-1);
                }

            }
            return crawl(dir,dir?-index:index);
        }

        const_iterator operator-(ptrdiff_t index) const{
            const_iterator returnvalue = *this;
            return returnvalue-=index;
        }

        const_iterator operator++(int zero){
            const_iterator returnvalue = *this;
            *this+=1;
            return returnvalue;
        }

        const_iterator& operator++(){
            return *this+=1;
        }

        const_iterator operator--(int zero){
            const_iterator returnvalue = *this;
            *this-=1;
            return returnvalue;
        }

        const_iterator& operator--(){
            return *this-=1;
        }

        T const& operator[](ptrdiff_t index){
            return *(*this + index);
        }

        T const* operator->(){
            return current;
        }

        explicit operator Weight() const{
            return index();
        }

        friend const_iterator operator+(ptrdiff_t const lhs, const_iterator const& rhs){
            return rhs+lhs;
        }
    };

    class iterator : public const_iterator{
    private:
    public:
        iterator(BinaryIndexTree& tree, size_struct index): const_iterator(tree,index){}

        iterator(BinaryIndexTree& tree): const_iterator(tree){}

        iterator(const_iterator&& ci): const_iterator(ci){}

        bool operator==(iterator const& other) const{
            return (const_iterator)*this == other;
        }

        bool operator!=(iterator const& other) const{
            return !(*this == other);
        }

        T& operator*() const{
            return const_cast<T&>(*(static_cast<const_iterator>(*this)));
        }

        iterator& operator+=(ptrdiff_t index){
            const_iterator::operator+=( index);
            return *this;
        }

        iterator operator+(ptrdiff_t index) const{
            iterator returnvalue = *this;
            return returnvalue+=index;
        }

        iterator& operator-=(ptrdiff_t index){
            const_iterator::operator-= (index);
            return *this;
        }

        iterator operator-(ptrdiff_t index) const{
            iterator returnvalue = *this;
            return returnvalue-=index;
        }

        iterator operator++(int zero){
            iterator returnvalue = *this;
            *this+=1;
            return returnvalue;
        }

        iterator& operator++(){
            return *this+=1;
        }

        iterator operator--(int zero){
            iterator returnvalue = *this;
            *this-=1;
            return returnvalue;
        }

        iterator& operator--(){
            return *this-=1;
        }

        T& operator[](ptrdiff_t index){
            return *(*this + index);
        }

        T* operator->(){
            return const_iterator::operator->();
        }

        friend iterator operator+(ptrdiff_t const lhs, iterator const& rhs){
            return rhs+lhs;
        }
    };

    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;

    //default constructor
    BinaryIndexTree(){
        m_root = nullptr;
    }

    //copy constructor
    BinaryIndexTree(BinaryIndexTree<T,Weight> const& other){
        if(other.m_root){
            m_root = new Node(*other.m_root);
        } else{
            m_root = nullptr;
        }
    }

    //move constructor
    BinaryIndexTree(BinaryIndexTree<T,Weight>&& other){
        swap(other);
    }

    //construct from initializer list
    BinaryIndexTree(std::initializer_list<T> il){
        m_root = rMake(il.begin(),il.size());
    }

    //constructs length n default values
    BinaryIndexTree(Weight n){
        T val{};
        m_root = rMake(val,n);
    }

    //constructs length n copies of val
    BinaryIndexTree(Weight n, T const& val){
        m_root = rMake(val,n);
    }

    //ranged based constructor
    template<typename InputIterator,typename = typename std::iterator_traits<InputIterator>::iterator_category>
    BinaryIndexTree(InputIterator first,InputIterator last){
        m_root = nullptr;
        assign(first,last);
    }

    //ranged based constructor
    template<typename InputIterator,typename = typename std::iterator_traits<InputIterator>::iterator_category>
    BinaryIndexTree(Weight length,InputIterator first){
        m_root = nullptr;
        assign(first,length);
    }

    ~BinaryIndexTree(){
        if(m_root) delete m_root;
    }

    void swap(BinaryIndexTree& other){
        Node* temp = m_root;
        m_root = other.m_root;
        other.m_root = temp;
    }

    Weight size() const{
        if(m_root)
            return m_root->weight;
        else return 0;
    }

    //returns the theoretical maximum container size, although practically this may be unreachable depending on hardware.
    //If Weight is std::size_t it is assuredly unreachable.
    static Weight max_size(){
        return std::numeric_limits<Weight>::max();
    }

    //allows container to act like vector in generic code.
    std::size_t capacity(){
        return size() * 2;
    }

    //allows container to act like vector in generic code.
    void reserve(std::size_t useless){}

    //allows container to act like vector in generic code.
    void shrink_to_fit(){}

    //empties container
    void clear(){
        delete m_root;
        m_root = nullptr;
    }

    bool empty() const{
        return m_root == nullptr;
    }

    T& operator[](Weight index){
        Node* node = rGet(m_root,index);
        return node->value;
    }

    T& at(Weight index){
        if(0<=index && index<size()){
            return (*this)[index];
        } else throw std::out_of_range("Index out of bounds");
    }

    T& front(){
        return (*this)[0];
    }

    T& back(){
        return (*this)[size()-1];
    }

    T const& operator[](Weight index) const{
        Node* node = rGet(m_root,index);
        return node->value;
    }

    T const& at(Weight index) const{
        if(0<=index && index<size()){
            return (*this)[index];
        } else throw std::out_of_range("Index out of bounds");
    }

    T const& front() const{
        return (*this)[0];
    }

    T const& back() const{
        return (*this)[size()-1];
    }

    //single insert
    iterator insert(size_struct position, T const& val){
        Weight index = (Weight)position;
        Node* node = new Node(val);
        rAdd(m_root,index,node);
        return iterator(*this,index);
    }

    //inserts n copies of val
    iterator insert (size_struct position, Weight n, const T& val){
        rMultiAdd(m_root,(Weight)position,rMake(val,n));
        return iterator(*this,position);
    }

    //inserts the elements from first to last in order at position
    template <class InputIterator>
    iterator insert (size_struct position, InputIterator first, InputIterator last){
        rMultiAdd(m_root,(Weight)position,delegateMake(first,last));
        return iterator(*this,position);
    }

    //moves a value into position
    iterator insert (size_struct position, T&& val){
        Weight index = (Weight)position;
        Node* node = new Node(std::move(val));
        rAdd(m_root,index,node);
        return iterator(*this,index);
    }

    iterator insert (size_struct position, std::initializer_list<T> il){
        Node* node = rMake(il.begin(),il.size());
        rMultiAdd(m_root,(Weight)position,node);
        return iterator(*this,(Weight)position);
    }

    void insertAt(size_struct i, T const& value){
        Weight index = (Weight)i;
        if(0<=index && index<=size()){
            insert(index,value);
        } else throw std::out_of_range("Index out of range at insertAt.");
    }

    void push_back(T const& value){
        insert(size(),value);
    }

    void push_back(T && value){
        insert(size(),std::move(value));
    }

    void push_front(T const& value){
        insert(0,value);
    }

    void push_front(T && value){
        insert(0,std::move(value));
    }

    template<typename ... Args>
    void emplace(size_struct i,Args&&... args){
        Weight index = (Weight)i;
        Node* node = new Node(std::forward<Args...>(args...));
        rAdd(m_root,index,node);
    }

    template<typename ... Args>
    void emplaceAt(size_struct i,Args&&... args){
        Weight index = (Weight)i;
        if(0<=index && index<=size()){
            emplace(index,std::forward<Args...>(args...));
        } else throw std::out_of_range("Index out of bounds at emplaceAt");
    }

    template<typename ... Args>
    void emplace_back(Args&&... args){
        emplace(size(),std::forward<Args...>(args...));
    }

    template<typename ... Args>
    void emplace_front(Args&&... args){
        emplace(0,std::forward<Args...>(args...));
    }

    void assign(std::initializer_list<T> il){
        if(m_root) delete m_root;
        m_root = rMake(il.begin(),il.size());
    }

    void assign(size_struct n,T const& val){
        if(m_root) delete m_root;
        m_root = rMake(val,n);
    }

    void assign(size_struct n,T & val){
        if(m_root) delete m_root;
        m_root = rMake(val,n);
    }

    template <class InputIterator,typename = typename std::iterator_traits<InputIterator>::iterator_category>
    void assign(size_struct n, InputIterator first){
        if(m_root) delete m_root;
        m_root = rMake(first,n);
    }

    template <class InputIterator,typename category = typename std::iterator_traits<InputIterator>::iterator_category>
    void assign(InputIterator first, InputIterator last){
        if(m_root) delete m_root;
        m_root = delegateMake(first,last);
    }

    T pop(size_struct i){
        Weight index = (Weight)i;
        Node* node = rDelete(m_root,index);
        T value = node->value;
        delete node;
        return value;
    }

    T popAt(size_struct i){
        Weight index = (Weight)i;
        if(0<=index && index<size()){
            return pop(index);
        } else throw std::out_of_range("Index out of bounds at popAt");
    }

    iterator erase(size_struct index){
        pop(index);
        return iterator(*this,index);
    }

    iterator erase(size_struct first, size_struct last){
        rMultiErase(m_root,(Weight)first,(Weight)last);
        return iterator(*this,(Weight)first);
    }

    T pop_back(){
        Node* node = rDelete(m_root,size()-1);
        T value = node->value;
        delete node;
        return value;
    }

    T pop_front(){
        Node* node = rDelete(m_root,0);
        T value = node->value;
        delete node;
        return value;
    }

    void resize(Weight n, T const& val = T()){
        if(n<size()){
            erase(n,size());
        } else{
            insert(size(),n-size(),val);
        }
    }

    const_iterator cbegin() const{
        return const_iterator(*this,0);
    }

    const_iterator cend() const{
        return const_iterator(*this);
    }

    const_reverse_iterator crbegin() const{
        return const_reverse_iterator(cend());
    }

    const_reverse_iterator crend() const{
        return const_reverse_iterator(cbegin());
    }

    iterator begin(){
        return iterator(*this,0);
    }

    const_iterator begin() const{
        return const_iterator(*this,0);
    }

    iterator end(){
        return iterator(*this);
    }

    const_iterator end() const{
        return const_iterator(*this);
    }

    reverse_iterator rbegin(){
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const{
        return const_reverse_iterator(cend());
    }

    reverse_iterator rend(){
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const{
        return const_reverse_iterator(cbegin());
    }

    //deep copies a BinaryIndexTree
    BinaryIndexTree& operator=(BinaryIndexTree<T,Weight> const& other){
        if(m_root) delete m_root;
        if(other.m_root){
            m_root = new Node(*other.m_root);
        } else{
            m_root = nullptr;
        }
        return *this;
    }

    //moves a BinaryIndexTree
    BinaryIndexTree& operator=(BinaryIndexTree<T,Weight>&& other){
        if(m_root) delete m_root;
        m_root = other.m_root;
        other.m_root = nullptr;
        return *this;
    }

    //copies from an initializer list
    BinaryIndexTree& operator=(std::initializer_list<T> il){
        if(m_root) delete m_root;
        m_root = rMake(il.begin(),il.size());
    }

    bool operator==(BinaryIndexTree<T,Weight> const& other) const{
        return size() == other.size() && std::equal(cbegin(),cend(),other.cbegin());
    }

    bool operator!=(BinaryIndexTree<T,Weight> const& other) const{
        return ! (*this == other);
    }

    bool operator<(BinaryIndexTree<T,Weight> const& other) const{
        return std::lexicographical_compare(cbegin(),cend(),other.cbegin(),other.cend());
    }

    bool operator>(BinaryIndexTree<T,Weight> const& other) const{
        return other<*this;
    }

    bool operator<=(BinaryIndexTree<T,Weight> const& other) const{
        return !(*this>other);
    }

    bool operator>=(BinaryIndexTree<T,Weight> const& other) const{
        return !(*this<other);
    }

    /*===========================test functions====================================*/
    bool verify(){
        if(!m_root) return true;
        else return m_root->verify();
    }

    unsigned height(){
        return height(m_root);
    }

    unsigned height(Node*& subtree){
        if(!subtree) return 0;
        else return std::max(height(subtree->child[0]),height(subtree->child[1])) + 1;
    }

};

template <typename T,typename Weight>
  inline void std::swap (BinaryIndexTree<T,Weight>& x, BinaryIndexTree<T,Weight>& y){
    x.swap(y);
  }


#endif // Binary_Index_Tree

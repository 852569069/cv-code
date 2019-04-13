"""
在python中，一个变量保存的是该变量的内存地址。所以可以随意的指向。
c语言是静态语言，所以任何一个变量都需要先申明，才可以申请一部分内存。
而c语言中，该变量就是变量本身，并不是地址，所以不能随意的指向。

链表操作

"""
class Node(object):
    def __init__(self,elem=None):
        self.elem=elem
        self.next=None

class singlelink(object):
    def __init__(self,node=None):
        self.__head=node
        self.next=None
    #此处创建一个头节点，本身为空，不指向任何东西。
    #头节点本身就是只是用来指向地址的，不存储任何数值。

    #判断是否为空。
    def is_empty(self):
        self.__head=None#也就是说，头节点没有指向任何地址。所以列表为空。
        return True
    #长度
    def length(self):
        #整理一下思路。首先输入某个节点，
        curr=self.__head
        count=0
        while curr != None:
            count+=1
            curr=curr.next
        return count
    #
    def travel(self):
        #很佩服语言的严谨性，把这个地址称之为指针。数值输入时，指针就移到了链表的第一个node上。
        # 改变指针的位置通过当前节点node。next就可以实现。
        #历遍和计算长度比较类似。先找到指针的初始位置。
        curr=self.__head
        while curr.next==None:
            print(curr.elem,end=' ')
            curr=curr.next

    def append(self,item):
        #尾插法

        pass
    def add(self,item):
        #头插法。
        node=Node(item)
        prenode=self.__head
        self.__head=node
        node.next=prenode

    def insert(self,pos,item):
        #中间插入的方式。首先找到位置。
        curr=self.__head
        node=Node(item)
        po=0
        while po!=pos-1:
            curr=curr.next
            po+=1
        node.next=curr.next
        curr.next=node


    def remove(self,item):
        #用两个游标，或者说时指针。
        pre=None
        curr=self.__head
        while curr.elem!=item:
            pre=curr
            curr=curr.next
        pre.next=curr.next
        del curr

    def remove1(self,item):
        #只使用一个指针。
        curr=self.__head
        if curr.elem==item:
            self.__head=curr.next
            del curr
        else:
            while curr.next !=item:
                curr=curr.next
            curr=curr.next.next
            del curr.next
    def search(self,item):

        pass



node1=Node(3)
node2=Node(2)
node3=Node(4)

node1.next=node2
node2.next=node3

sll=singlelink(node1)




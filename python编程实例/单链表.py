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
    #此处创建一个头节点，本身为空，不指向任何东西。
    #头节点本身就是只是用来指向地址的，不存储任何数值。
    #所以考虑None的情况的时候，要细致的考虑清楚。

    #判断是否为空。
    def is_empty(self):
        return self.__head==None#也就是说，头节点没有指向任何地址。所以列表为空。

    #长度
    def length(self):
        #整理一下思路。首先输入某个节点，
        curr=self.__head
        count=0
        while curr != None:
            count+=1
            curr=curr.next
        print('length:',count)
    #
    def travel(self):
        #很佩服语言的严谨性，把这个地址称之为指针。数值输入时，指针就移到了链表的第一个node上。
        # 改变指针的位置通过当前节点node。next就可以实现。
        #历遍和计算长度比较类似。先找到指针的初始位置。
        curr=self.__head
        while curr!=None:
            print(curr.elem,end=' ')
            curr=curr.next

    def append(self,item):
        curr=self.__head
        node=Node(item)
        if curr == None:
            self.__head=node
        else:
            while curr.next!=None:
                curr=curr.next
            curr.next=node

    def add(self,item):
        node=Node(item)
        curr=self.__head
        node.next=curr
        self.__head=node



    def insert(self,pos,item):
        #中间插入的方式。首先找到位置。
        # curr=self.__head
        # node=Node(item)
        # po=0
        # while po!=pos-1:
        #     curr=curr.next
        #     po+=1
        # node.next=curr.next
        # curr.next=node
        node=Node(item)
        if self.is_empty():
            self.__head=node
        else:
            curr=self.__head
            count=0
            while count !=(pos-1):
                curr=curr.next
                count+=1
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
        curr=self.__head
        #需要搜索到尾部，如果只是单纯的看有没有值等于该数值，
        # 会出错，到最后一个值的时候还在跑。
        count=0
        while curr.next!=None:
            count+=1
            if curr.elem==item:
                return count
            else:
                curr=curr.next
        return -1




if __name__ == '__main__':
    sll=singlelink()
    sll.append(3)
    sll.append(6)
    sll.add(5)
    sll.append(8)
    sll.insert(2,5)
    sll.travel()



    print(sll.search(6))









# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:37:19 2019

@author: cherif
"""
import SerializedListNew_pb2

import lmdb
import time
import mmh3
import math
#from memory_profiler import profile
from line_profiler import LineProfiler
from memory_profiler import profile
import PageAnchorsList_pb2
# create environnment



my_list=SerializedListNew_pb2.SerializedListNew()

def indice_sup(s,score):
    """
    index of insertion using dichotomic search in a list of postings
    """
    a = 0
    b = len(s)-1
    m = (a+b)//2
    while a <= b :
        if int(s[m].score)<=score  :
            b=m-1
        else :
            a = m+1
        m = (a+b)//2
    return a
    
          
       
def insert_sorted(s,Score,DocId):
   """
   insertion in index postion returned by indice_sup
   """    
   mylist= SerializedListNew_pb2.SerializedListNew()  
   ind=indice_sup(s.Elements,Score)   
   #print('ind=',ind)
   mylist.Elements.extend(s.Elements[:ind])
   #print(mylist.Elements)
   mylist.Elements.add(score=Score,docId=DocId)
   #print(s[ind:])
   mylist.Elements.extend(s.Elements[ind:])
   #print('---------after insertion----------')
   #print(mylist.Elements)
   return mylist
    
def Go_forward(l1,l2,result,ind1,ind2,borne,c_row,n_rows,tt_Occ,ins_count):
     Sortie=False   
     finish=False
     while not Sortie and not finish: 
        if int(l1.Elements[ind1].docId)<int(l2.Elements[ind2].docId):
            ind1+=1
        else:
            Sortie=True
        finish=ind1>borne
            
           
     if not finish:    
      if int(l1.Elements[ind1].docId)==int(l2.Elements[ind2].docId):
         
         if l1.Elements[ind1].score<=l2.Elements[ind2].score:
             
             if c_row!=n_rows:
                 result.Elements.add(score=l1.Elements[ind1].score,docId=l1.Elements[ind1].docId)
             else :
                 result=insert_sorted(result,l1.Elements[ind1].score,l1.Elements[ind1].docId)   
                 tt_Occ+=l1.Elements[ind1].score
                 ins_count+=1
         else:
             if c_row!=n_rows:
                  result.Elements.add(score=l2.Elements[ind2].score,docId=l2.Elements[ind2].docId)
             else :
                  result=insert_sorted(result,l2.Elements[ind2].score,l2.Elements[ind2].docId)   
                  tt_Occ+=l2.Elements[ind2].score
                  ins_count+=1
         ind1+=1
         ind2+=1
         
      else:
         ind2+=1 
     return ind1,ind2,result,tt_Occ,ins_count      
    
def intersection(ll1,ll2,current_row,nb_rows):
    """
    Intersection of sorted lists
    """
    #print("ll1:",len(ll1.Elements))
    #print("ll2:",len(ll2.Elements))
    #print(ll1)
    #print(ll2)
    inl1=0
    inl2=0
    inter=SerializedListNew_pb2.SerializedListNew()
    #min_length=min(len(ll1.Elements),len(ll2.Elements))
    borne1=len(ll1.Elements)-1
    borne2=len(ll2.Elements)-1
    Total_Occurrence=0    
    inserted_count=0
    Kill=False
    while inl1<=borne1 and inl2<=borne2 and not Kill:   
      if int(ll1.Elements[inl1].docId)<int(ll2.Elements[inl2].docId):
        inl1,inl2,inter,Total_Occurrence,inserted_count=Go_forward(ll1,ll2,inter,inl1,inl2,borne1,current_row,nb_rows,Total_Occurrence,inserted_count)
      else:
        inl2,inl1,inter,Total_Occurrence,inserted_count=Go_forward(ll2,ll1,inter,inl2,inl1,borne2,current_row,nb_rows,Total_Occurrence,inserted_count) 
      Kill=current_row==nb_rows and inserted_count>49
    #print("list of intersection")  
    #print(inter)  
    return inter,Total_Occurrence

#@profile        
def Get_PostingsFromSketch(d,w,ngram,txn):
    l3=SerializedList_pb2.SerializedList()
    l2=SerializedList_pb2.SerializedList()
    l1=SerializedList_pb2.SerializedList()
    Total_Occurrence_ngram=0
    for r in range(d):
             # r is used as seed to generate same hash function for each row
             h=mmh3.hash(ngram,r)
             col=h%w
             posting_key=str(r)+"_"+str(col)
             #print(posting_key)
             val=txn.get(posting_key.encode('ascii'))
             my_list=SerializedList_pb2.SerializedList()
             if val!=None:
                        
                        if r==0 :       
                           l1=my_list.FromString(val) 
                        else : 
                           l3=my_list.FromString(val) 
                           l2,Total_Occurrence_ngram=intersection(l1,l3,r,d-1)
                            
                           #l2.Elements.extend([el for el in l1.Elements if el in l3.Elements])
                           l1.ClearField("Elements")
                           l1.Elements.extend(l2.Elements[:])
                           l3.ClearField("Elements")
                           l2.ClearField("Elements")
                           #l1.ClearField("Elements")
                           #♦print(l1)
             else:
                 print('ngram miss')
                 
    #for el in l1.Elements:
    #     q=db.PageIdToAnchors.find({'pageID':el.docId})
    #     for q1 in q:
    #        print('title',q1["title"]) 
    #        print('Page',q1["text"])
    #     time.sleep(5)
    #print(len(l1.Elements))   
    print('Intersection:',len(l1.Elements))
    #print(Total_Occurrence_ngram)
    return l1,Total_Occurrence_ngram

   
def Get_Postings(ngram,txn):
          tt_occr=0
          ll=SerializedListNew_pb2.SerializedListNew() 
          if ngram!='':   
             val=txn.get(ngram.encode())
             my_list=SerializedListNew_pb2.SerializedListNew()
  
             if val!=None:
                 #ll.Elements.extend((my_list.FromString(val)).Elements[:50])
                 ll=my_list.FromString(val)
                 #print("ll length:",len(ll.Elements))
                 for ii in range(len(ll.Elements)):
                     tt_occr+=ll.Elements[ii].score
             #else:
             #    print("ngram does not exist in inverted index")
                 #----------------print('ngram hit')    
             #----------else:
             #----------    print('ngram miss')    
             #print(ll)
          return ll,tt_occr
    
def Get_Page(PageId,txn,base):                 

     
         val=txn.get(PageId.encode(),default=None,db=base)
         if val!=None:
            elem=PageAnchorsList_pb2.PageAnchorsList()
            return (elem.FromString(val)).title,(elem.FromString(val)).Anchors
         else:
             print("Miss page")
             return "NoPage"
             
def Get_categories(Pageid,txn,base):
      val=txn.get(Pageid.encode(),default=None,db=base)    
      if val!=None:
          return val.decode("utf-8")
      else:
          return ''
          
def Get_infoxbox(Pageid,txn,base):
      val=txn.get(Pageid.encode(),default=None,db=base)    
      if val!=None:
          return val.decode()
      else:
          return '0'    
def IndexAccess(ngram,txn):    
   Delta=0.062
   epsilon=2.9*10**-6
   d=math.ceil(math.log(1/Delta))
   w=int(2/epsilon)
   #print('d=',d,'\n')
   #print('w=',w,'\n')
   Posting_List=SerializedListNew_pb2.SerializedListNew()
   Posting_List,TotalOcurr=Get_Postings(ngram,txn)
   #print(Posting_List)
  #------------------------ print('TotalOccur=',TotalOcurr)
   #print(Posting_List)
      #Posting_List,TotalOcurr=Get_Postings(d,w,ngram,txn)
      #print(Posting_List)
   return Posting_List,TotalOcurr  
      #lp = LineProfiler()
      #lp_wrapper = lp(Get_Postings)
      #lp_wrapper(d,w,ngram)
      #lp.print_stats()                    

       
if __name__=="__main__":
   ngram=input("Ngram=")
   Post_lists=SerializedListNew_pb2.SerializedListNew()
   Postings=lmdb.open('c:/appli/PostingsLast',readonly=True)#,max_dbs=5)
   with Postings.begin() as trxn:
    Post_lists,Occrs=IndexAccess(ngram,trxn)
    print(Post_lists.Elements)
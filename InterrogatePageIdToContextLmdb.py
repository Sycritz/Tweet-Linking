# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:33:00 2020

@author: cherif
"""
import lmdb
import DictionaryWithTitle_pb2 as dc

def get_Element(pageId,txnn):
    val=txnn.get(pageId.encode())
    if val!=None:
        elem=dc.Dico()
        elem=elem.FromString(val)
        Anchors={}
        Categories={}
        CalledPages={}
        OrgAnchors={} 
        for el in elem.OrgAnchors: 
            OrgAnchors[el]=elem.OrgAnchors[el]
        for el in elem.Anchors: 
            Anchors[el]=elem.Anchors[el]
        for el in elem.Categories:
            Categories[el]=elem.Categories[el]
        for el in elem.CalledPages:
            CalledPages[el]=elem.CalledPages[el]
        return elem.PageTitle,elem.length_anchors, elem.PageViews, elem.PageRank,OrgAnchors, Anchors,Categories,CalledPages
        #return Anchors,Categories,CalledPages
    else:
        return 0,0,0,0
    
enwikiNewlmdb=lmdb.open('d:/tels/PageIdToContexte2',map_size=16000000000)
with enwikiNewlmdb.begin() as txn:
  PageId=input("Donner une pages:")
  Title,anchors_len,views,rank,OrgAnchors,Anchors,Categories,CalledPages=get_Element(PageId,txn)
  print(Title)
  print(anchors_len)
  print(views)
  print(rank)
  nb=0
  for el in OrgAnchors:
      nb+=OrgAnchors[el]
  print("Orginial Anchors---------------------->", nb)
  print(OrgAnchors)
  
  print("Anchors Dictionary:------------------->")
  print(Anchors)
  print('Categories Dictionary:---------------->')
  print(Categories)
  print('Called Pages Dictionary:-------------->')
  print(CalledPages)
  

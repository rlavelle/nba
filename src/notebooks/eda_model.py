#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 19:47:29 2025

@author: rowanlavelle
"""

"""
%% compare modeling XGB and LM by player, and by general
    - can we at least predict when a player is going to shoot above average?
        * is this a classification problem better suited for logistic regression?
        * does standardizing data help with this?
        
    - is it worth while to try neural nets? if XGB dsnt outperform LM maybe signal is low
        
    
%% does modeling PPM and mutliplying by the minutes model have better accuracy?
    - does this provide a wider variance? how does it perform predicting above / below avg
    - predict ppm O/U mean ppm * minutes?
    - standardized data predicts std above mean? can use that for CI in prediction?
    
    
%% if logistic reg works, is it possible to use that as signal to predict points O/U mean?
    

%% write a framework to test jitter within n points to get distribution on model confidence


%% function (similar to scikit?) that can do train test for me (given train test frmwk)
    
"""

#%%

#%%